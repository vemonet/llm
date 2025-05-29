//! Chat wrapper that adds memory capabilities to any ChatProvider.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    memory::MemoryProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};

/// A wrapper that adds memory capabilities to any ChatProvider.
///
/// This wrapper implements all LLM provider traits and adds automatic memory
/// management to chat conversations without requiring backend modifications.
pub struct ChatWithMemory {
    provider: Box<dyn LLMProvider>,
    memory: Arc<Mutex<Box<dyn MemoryProvider>>>,
    role: Option<String>,
}

impl ChatWithMemory {
    /// Create a new memory-enabled chat wrapper.
    ///
    /// # Arguments
    ///
    /// * `provider` - The underlying LLM provider
    /// * `memory` - Memory provider for conversation history
    /// * `role` - Optional role name for multi-agent scenarios
    pub fn new(
        provider: Box<dyn LLMProvider>,
        memory: Arc<Mutex<Box<dyn MemoryProvider>>>,
        role: Option<String>,
    ) -> Self {
        Self {
            provider,
            memory,
            role,
        }
    }

    /// Get the underlying provider
    pub fn inner(&self) -> &dyn LLMProvider {
        self.provider.as_ref()
    }
}

#[async_trait]
impl ChatProvider for ChatWithMemory {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let previous_messages = {
            let memory_guard = self.memory.lock().await;
            memory_guard.recall("", None).await.unwrap_or_default()
        };

        {
            let mut memory_guard = self.memory.lock().await;
            for msg in messages {
                let mut memory_msg = msg.clone();
                if let Some(role) = &self.role {
                    memory_msg.content = format!("[{}] User: {}", role, msg.content);
                }
                let _ = memory_guard.remember(&memory_msg).await;
            }
        }

        let mut all_messages = previous_messages;
        all_messages.extend_from_slice(messages);

        let response = self.provider.chat_with_tools(&all_messages, tools).await?;

        let memory_clone = self.memory.clone();
        let role_clone = self.role.clone();
        let response_text = response.text();

        tokio::spawn(async move {
            if let Some(text) = response_text {
                let mut memory_guard = memory_clone.lock().await;
                let assistant_content = match &role_clone {
                    Some(role) => format!("[{}] Assistant: {}", role, text),
                    None => text,
                };

                let assistant_msg = ChatMessage::assistant().content(assistant_content).build();

                let _ = memory_guard.remember(&assistant_msg).await;
            }
        });

        Ok(response)
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
