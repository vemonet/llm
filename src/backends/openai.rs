//! OpenAI API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::backends::openai_compatible::{OpenAICompatibleConfig, OpenAICompatibleProvider};
use crate::{
    chat::{Tool, ToolChoice, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// OpenAI configuration for the generic provider
pub struct OpenAIConfig;

impl OpenAICompatibleConfig for OpenAIConfig {
    const PROVIDER_NAME: &'static str = "OpenAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4.1-nano";
    const SUPPORTS_WEB_SEARCH: bool = true;
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Type alias for OpenAI client using the generic provider
pub type OpenAI = OpenAICompatibleProvider<OpenAIConfig>;

impl OpenAI {
    /// Creates a new OpenAI client with the specified configuration.
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
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        voice: Option<String>,
        enable_web_search: Option<bool>,
        web_search_context_size: Option<String>,
        web_search_user_location_type: Option<String>,
        web_search_user_location_approximate_country: Option<String>,
        web_search_user_location_approximate_city: Option<String>,
        web_search_user_location_approximate_region: Option<String>,
    ) -> Self {
        // Note: embedding params and voice are stored but not used in the generic impl
        // They would need to be handled by specific OpenAI-only implementations
        let _ = (embedding_encoding_format, embedding_dimensions, voice);

        <OpenAICompatibleProvider<OpenAIConfig>>::new(
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
            None, // voice - would need custom implementation
            enable_web_search,
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
            None, // parallel_tool_calls
        )
    }
}

// OpenAI-specific implementations that don't fit in the generic provider

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for OpenAIModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| chrono::DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAIModelListResponse {
    pub data: Vec<OpenAIModelEntry>,
}

use crate::builder::LLMBackend;

impl ModelListResponse for OpenAIModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|e| e.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::OpenAI
    }
}

#[async_trait]
impl CompletionProvider for OpenAI {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "OpenAI completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for OpenAI {
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LLMError> {
        let url = self
            .base_url
            .join("audio/transcriptions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let part = reqwest::multipart::Part::bytes(audio).file_name("audio.m4a");
        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "text")
            .part("file", part);

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        Ok(text)
    }

    async fn transcribe_file(&self, file_path: &str) -> Result<String, LLMError> {
        let url = self
            .base_url
            .join("audio/transcriptions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "text")
            .file("file", file_path)
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        Ok(text)
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        // Note: This would need access to embedding-specific fields that aren't in the generic provider
        // For now, use defaults
        let body = OpenAIEmbeddingRequest {
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

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl ModelsProvider for OpenAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .get(url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<OpenAIModelListResponse>().await?;
        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl TextToSpeechProvider for OpenAI {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let url = self
            .base_url
            .join("audio/speech")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        #[derive(Serialize)]
        struct SpeechRequest {
            model: String,
            input: String,
            voice: String,
        }

        let body = SpeechRequest {
            model: self.model.clone(),
            input: text.to_string(),
            voice: self.voice.clone().unwrap_or("alloy".to_string()),
        };

        let mut req = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(resp.bytes().await?.to_vec())
    }
}

#[cfg(test)]
const LLM_API_KEY_ENV: &str = "OPENAI_API_KEY";

// Tests remain the same, but would use the new generic implementation
#[tokio::test]
async fn test_openai_chat_stream_struct() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };
    use futures::StreamExt;

    let api_key = match std::env::var(LLM_API_KEY_ENV) {
        Ok(key) => key,
        Err(_) => {
            eprintln!("test test_openai_chat_stream_struct ... ignored, {LLM_API_KEY_ENV} not set");
            return Ok(());
        }
    };
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4.1-nano")
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
            } else {
                panic!("Expected usage data in response");
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

#[tokio::test]
async fn test_openai_chat_stream() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };
    use futures::StreamExt;

    let api_key = match std::env::var(LLM_API_KEY_ENV) {
        Ok(key) => key,
        Err(_) => {
            eprintln!("test test_openai_chat_stream_struct ... ignored, {LLM_API_KEY_ENV} not set");
            return Ok(());
        }
    };
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4.1-nano")
        .max_tokens(512)
        .temperature(0.7)
        .stream(true)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(content) => {
                        complete_text.push_str(&content);
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
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
