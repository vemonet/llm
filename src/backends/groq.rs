//! Groq API client implementation for chat functionality.
//!
//! This module provides integration with Groq's LLM models through their API.

use crate::{
    providers::openai_compatible::{OpenAICompatibleConfig, OpenAICompatibleProvider}, chat::{StructuredOutputFormat, Tool, ToolChoice}, completion::{CompletionProvider, CompletionRequest, CompletionResponse}, embedding::EmbeddingProvider, error::LLMError, models::ModelsProvider, stt::SpeechToTextProvider, tts::TextToSpeechProvider, LLMProvider
};
use async_trait::async_trait;

/// Groq configuration for the generic provider
pub struct GroqConfig;

impl OpenAICompatibleConfig for GroqConfig {
    const PROVIDER_NAME: &'static str = "Groq";
    const DEFAULT_BASE_URL: &'static str = "https://api.groq.com/openai/v1/";
    const DEFAULT_MODEL: &'static str = "llama3-8b-8192";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type Groq = OpenAICompatibleProvider<GroqConfig>;

impl Groq {
    /// Creates a new Groq client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        _embedding_encoding_format: Option<String>,
        _embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        parallel_tool_calls: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<GroqConfig>::new(
            api_key,
            base_url,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning_effort,
            json_schema,
            None, // voice - not supported by Groq
            parallel_tool_calls,
            None, // embedding_encoding_format - not supported by Groq
            None, // embedding_dimensions - not supported by Groq
        )
    }
}


impl LLMProvider for Groq {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl CompletionProvider for Groq {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Groq completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Groq {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for Groq {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Groq does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Groq {}

#[async_trait]
impl ModelsProvider for Groq {}
