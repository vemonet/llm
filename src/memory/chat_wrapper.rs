//! Chat wrapper that adds memory capabilities to any ChatProvider.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    memory::{MemoryProvider, MessageCondition},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};

/// A wrapper that adds memory capabilities to any ChatProvider.
///
/// This wrapper implements all LLM provider traits and adds automatic memory
/// management to chat conversations without requiring backend modifications.
pub struct ChatWithMemory {
    provider: Arc<dyn LLMProvider>,
    memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    role: Option<String>,
    role_triggers: Vec<(String, MessageCondition)>,
}

impl ChatWithMemory {
    /// Create a new memory-enabled chat wrapper.
    ///
    /// # Arguments
    ///
    /// * `provider` - The underlying LLM provider
    /// * `memory` - Memory provider for conversation history
    /// * `role` - Optional role name for multi-agent scenarios
    /// * `role_triggers` - Role and condition pairs for reactive messaging
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
        role: Option<String>,
        role_triggers: Vec<(String, MessageCondition)>,
    ) -> Self {
        let wrapper = Self {
            provider,
            memory: memory.clone(),
            role,
            role_triggers: role_triggers.clone(),
        };

        if !role_triggers.is_empty() {
            wrapper.spawn_reactive_listener();
        }

        wrapper
    }

    /// Spawn a task to listen for reactive messages
    fn spawn_reactive_listener(&self) {
        if let Ok(memory_guard) = self.memory.try_read() {
            if let Some(mut receiver) = memory_guard.get_event_receiver() {
                let role_triggers = self.role_triggers.clone();
                let provider = self.provider.clone();
                let memory = self.memory.clone();
                let my_role = self.role.clone();

                tokio::spawn(async move {
                    while let Ok(event) = receiver.recv().await {
                        let should_trigger = role_triggers.iter().any(|(role, condition)| {
                            role == &event.role && condition.matches(&event)
                        });

                        if should_trigger {
                            let context = {
                                let memory_guard = memory.read().await;
                                memory_guard.recall("", None).await.unwrap_or_default()
                            };

                            let response_text = match provider.chat(&context).await {
                                Ok(response) => response.text().map(|s| s.to_string()),
                                Err(e) => {
                                    eprintln!("Reactive chat error: {}", e);
                                    None
                                }
                            };

                            if let (Some(text), Some(role)) = (response_text, &my_role) {
                                let formatted_text = format!("[{}] {}", role, text);
                                let msg = ChatMessage::assistant().content(formatted_text).build();
                                let mut memory_guard = memory.write().await;
                                if let Err(e) =
                                    memory_guard.remember_with_role(&msg, role.clone()).await
                                {
                                    eprintln!("Memory save error: {}", e);
                                }
                            }
                        }
                    }
                });
            }
        }
    }

    /// Get the underlying provider
    pub fn inner(&self) -> &dyn LLMProvider {
        self.provider.as_ref()
    }

    /// Get current memory contents for debugging
    pub async fn memory_contents(&self) -> Vec<crate::chat::ChatMessage> {
        let memory_guard = self.memory.read().await;
        memory_guard.recall("", None).await.unwrap_or_default()
    }
}

#[async_trait]
impl ChatProvider for ChatWithMemory {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut write = self.memory.write().await;

        for m in messages {
            let mut tagged = m.clone();
            if let Some(role) = &self.role {
                tagged.content = format!("[{}] User: {}", role, m.content);
                write.remember_with_role(&tagged, role.clone()).await?;
            } else {
                write.remember(&tagged).await?;
            }
        }

        let mut context = write.recall("", None).await?;
        let needs_summary = write.needs_summary();
        drop(write);

        if needs_summary {
            let summary = self.provider.summarize_history(&context).await?;
            let mut write = self.memory.write().await;
            write.replace_with_summary(summary);
            context = write.recall("", None).await?;
        }

        context.extend_from_slice(messages);
        let response = self.provider.chat_with_tools(&context, tools).await?;

        if let Some(text) = response.text() {
            let memory = self.memory.clone();
            let tag = self.role.clone();
            tokio::spawn(async move {
                let mut mem = memory.write().await;
                let formatted_text = match &tag {
                    Some(r) => format!("[{}] {}", r, text),
                    None => text.clone(),
                };
                let msg = ChatMessage::assistant().content(formatted_text).build();

                let result = if let Some(role) = tag {
                    mem.remember_with_role(&msg, role).await
                } else {
                    mem.remember(&msg).await
                };

                if let Err(e) = result {
                    eprintln!("Memory save error: {}", e);
                }
            });
        }

        Ok(response)
    }

    async fn memory_contents(&self) -> Option<Vec<crate::chat::ChatMessage>> {
        let memory_guard = self.memory.read().await;
        Some(memory_guard.recall("", None).await.unwrap_or_default())
    }
}

// Forward all other traits to the underlying provider
#[async_trait]
impl CompletionProvider for ChatWithMemory {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        self.provider.complete(request).await
    }
}

#[async_trait]
impl EmbeddingProvider for ChatWithMemory {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.provider.embed(input).await
    }
}

#[async_trait]
impl SpeechToTextProvider for ChatWithMemory {
    async fn transcribe(&self, audio_data: Vec<u8>) -> Result<String, LLMError> {
        self.provider.transcribe(audio_data).await
    }
}

#[async_trait]
impl TextToSpeechProvider for ChatWithMemory {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        self.provider.speech(text).await
    }
}

impl LLMProvider for ChatWithMemory {
    fn tools(&self) -> Option<&[Tool]> {
        self.provider.tools()
    }
}
