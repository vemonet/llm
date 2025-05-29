//! Chat wrapper that adds memory capabilities to any ChatProvider.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

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
    memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
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
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
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
                tagged.content = format!("[{role}] User: {}", m.content);
            }
            write.remember(&tagged).await?;
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
            let tag    = self.role.clone();
            tokio::spawn(async move {
                let mut mem = memory.write().await;
                let text = match tag {
                    Some(r) => format!("[{r}] Assistant: {text}"),
                    None    => text,
                };
                mem.remember(&ChatMessage::assistant().content(text).build()).await.ok();
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
