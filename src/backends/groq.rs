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
        let _ = (embedding_encoding_format, embedding_dimensions);
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
            None, // Not supported by Groq
            parallel_tool_calls,
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


#[cfg(test)]
const LLM_API_KEY_ENV: &str = "GROQ_API_KEY";

#[tokio::test]
async fn test_groq_chat() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };

    let api_key = match std::env::var(LLM_API_KEY_ENV) {
        Ok(key) => key,
        Err(_) => {
            eprintln!("test test_groq_chat ... ignored, {LLM_API_KEY_ENV} not set");
            return Ok(());
        }
    };
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Groq)
        .api_key(api_key)
        .model("llama3-8b-8192")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat(&messages).await {
        Ok(response) => {
            println!("LLM response: {response:?}");
            assert!(
                !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
            let usage = response.usage();
            assert!(usage.is_some(), "Expected usage information to be present");
            let usage = usage.unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens, got {}",
                usage.total_tokens
            );
        }
        Err(e) => {
            return Err(e.into());
        }
    }
    Ok(())
}

#[tokio::test]
async fn test_groq_chat_with_tools() -> Result<(), Box<dyn std::error::Error>> {
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
        .backend(LLMBackend::Groq)
        .api_key(api_key)
        .model("llama3-8b-8192")
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
async fn test_groq_chat_stream_struct() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };
    use futures::StreamExt;

    if std::env::var(LLM_API_KEY_ENV).is_err() {
        eprintln!("test test_groq_chat_stream_struct ... ignored, {LLM_API_KEY_ENV} not set");
        return Ok(());
    }
    let api_key = std::env::var(LLM_API_KEY_ENV).unwrap();
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Groq)
        .api_key(api_key)
        .model("llama3-8b-8192")
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
            println!("Complete response text: {complete_text}");
            // if let Some(usage) = usage_data {
            //     assert!(
            //         usage.prompt_tokens > 0,
            //         "Expected prompt tokens > 0, got {}",
            //         usage.prompt_tokens
            //     );
            //     assert!(
            //         usage.completion_tokens > 0,
            //         "Expected completion tokens > 0, got {}",
            //         usage.completion_tokens
            //     );
            //     assert!(
            //         usage.total_tokens > 0,
            //         "Expected total tokens > 0, got {}",
            //         usage.total_tokens
            //     );
            //     println!("Complete response: {complete_text}");
            //     println!("Usage: {usage:?}");
            // } else {
            //     panic!("Expected usage data in response");
            // }
        }
        Err(e) => {
            eprintln!("Chat stream struct error: {e}");
            return Err(e.into());
        }
    }
    Ok(())
}
