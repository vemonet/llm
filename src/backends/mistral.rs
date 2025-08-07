//! Mistral API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with Mistral's LLM models through their API.

use crate::providers::openai_compatible::{OpenAICompatibleConfig, OpenAICompatibleProvider};
use crate::{
    chat::{Tool, ToolChoice, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Mistral configuration for the generic provider
pub struct MistralConfig;

impl OpenAICompatibleConfig for MistralConfig {
    const PROVIDER_NAME: &'static str = "Mistral";
    const DEFAULT_BASE_URL: &'static str = "https://api.mistral.ai/v1/";
    const DEFAULT_MODEL: &'static str = "mistral-small-latest";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = true;
}

/// Type alias for Mistral client using the generic OpenAI-compatible provider
pub type Mistral = OpenAICompatibleProvider<MistralConfig>;

impl Mistral {
    /// Creates a new Mistral client with the specified configuration.
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
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        parallel_tool_calls: Option<bool>,
    ) -> Self {
        // Note: embedding params are stored but not used in the generic impl
        // They would need to be handled by specific Mistral-only implementations
        let _ = (embedding_encoding_format, embedding_dimensions);
        <OpenAICompatibleProvider<MistralConfig>>::new(
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
            None, // Not supported by Mistral
            parallel_tool_calls,
        )
    }
}

// Mistral-specific implementations that don't fit in the generic OpenAI-compatible provider

#[derive(Serialize)]
struct MistralEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
}

impl LLMProvider for Mistral {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl CompletionProvider for Mistral {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Mistral completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for Mistral {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError("Mistral does not support speech-to-text".into()))
    }

    async fn transcribe_file(&self, _file_path: &str) -> Result<String, LLMError> {
        Err(LLMError::ProviderError("Mistral does not support speech-to-text".into()))
    }
}

#[cfg(feature = "mistral")]
#[async_trait]
impl EmbeddingProvider for Mistral {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".into()));
        }

        // Note: This would need access to embedding-specific fields that aren't in the generic provider
        // For now, use defaults
        let body = MistralEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: Some("float".to_string()),
            dimensions: None,
        };

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: MistralEmbeddingResponse = resp.json().await?;
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl ModelsProvider for Mistral {
    async fn list_models(
        &self,
        _request: Option<&crate::models::ModelListRequest>,
    ) -> Result<Box<dyn crate::models::ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError("Mistral does not provide a models listing endpoint".into()))
    }
}


#[async_trait]
impl TextToSpeechProvider for Mistral {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError("Mistral does not support text-to-speech".into()))
    }
}

#[cfg(test)]
const LLM_API_KEY_ENV: &str = "MISTRAL_API_KEY";

#[tokio::test]
async fn test_mistral_chat() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };

    if std::env::var(LLM_API_KEY_ENV).is_err() {
        eprintln!("test test_mistral_chat ... ignored, {LLM_API_KEY_ENV} not set");
        return Ok(());
    }
    let api_key = std::env::var(LLM_API_KEY_ENV).unwrap();
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model("mistral-small-latest")
        .max_tokens(512)
        .temperature(0.7)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat(&messages).await {
        Ok(response) => {
            assert!(
                response.text().is_some(),
                "Expected response message, got None"
            );
            println!("Response: {:?}", response.text());
        }
        Err(e) => {
            eprintln!("Chat error: {e}");
            return Err(e.into());
        }
    }
    Ok(())
}


#[tokio::test]
async fn test_mistral_chat_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
        chat::ChatMessage,
    };

    // Skip test if MISTRAL_API_KEY environment variable is not set
    if std::env::var(LLM_API_KEY_ENV).is_err() {
        eprintln!("test test_mistral_chat_with_tools ... ignored, {LLM_API_KEY_ENV} not set");
        return Ok(());
    }
    let api_key = std::env::var(LLM_API_KEY_ENV).unwrap();
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model("mistral-small-latest")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user().content("You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available").build()];
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            let tool_calls = response.tool_calls();
            assert!(tool_calls.is_some(), "Expected tool calls to be present");
            let tool_calls = tool_calls.unwrap();
            assert_eq!(
                tool_calls.len(),
                1,
                "Expected exactly 1 tool call, got {}",
                tool_calls.len()
            );
            assert_eq!(
                tool_calls[0].function.name, "weather_function",
                "Expected function name 'weather_function'"
            );
            let usage = response.usage();
            assert!(usage.is_some(), "Expected usage information to be present");
            let usage = usage.unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => {
            eprintln!("Chat error: {e}");
            return Err(e.into());
        }
    }
    Ok(())
}


#[tokio::test]
async fn test_mistral_chat_stream_struct() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };
    use futures::StreamExt;

    if std::env::var(LLM_API_KEY_ENV).is_err() {
        eprintln!("test test_mistral_chat_stream_struct ... ignored, {LLM_API_KEY_ENV} not set");
        return Ok(());
    }
    let api_key = std::env::var(LLM_API_KEY_ENV).unwrap();
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model("mistral-small-latest")
        .max_tokens(512)
        .temperature(0.7)
        .stream(true)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            let mut usage_data = None;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                complete_text.push_str(content);
                            }
                        }
                        if let Some(usage) = stream_response.usage {
                            usage_data = Some(usage);
                        }
                    }
                    Err(e) => {
                        eprintln!("Stream error: {e}");
                        return Err(e.into());
                    }
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text"
            );
            if let Some(usage) = usage_data {
                assert!(
                    usage.prompt_tokens > 0,
                    "Expected prompt tokens > 0, got {}",
                    usage.prompt_tokens
                );
                assert!(
                    usage.completion_tokens > 0,
                    "Expected completion tokens > 0, got {}",
                    usage.completion_tokens
                );
                assert!(
                    usage.total_tokens > 0,
                    "Expected total tokens > 0, got {}",
                    usage.total_tokens
                );
                println!("Complete response: {complete_text}");
                println!("Usage: {usage:?}");
            } else {
                panic!("Expected usage data in response");
            }
        }
        Err(e) => {
            eprintln!("Chat stream struct error: {e}");
            return Err(e.into());
        }
    }
    Ok(())
}
