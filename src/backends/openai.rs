//! OpenAI API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::providers::openai_compatible::{
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    // OpenAICompatibleFunctionCall,
    // OpenAICompatibleFunctionPayload,
    OpenAICompatibleChatMessage,
    // MessageContent,
    // ImageUrlContent,
    // OpenAICompatibleChatResponse as OpenAIChatResponse,
    // OpenAICompatibleChatChoice as OpenAIChatChoice,
    // OpenAICompatibleChatMsg as OpenAIChatMsg,
    ResponseFormat as OpenAIResponseFormat,
};
use crate::{
    chat::{Tool, ToolChoice, StructuredOutputFormat, ChatMessage, ChatProvider, ChatResponse, StreamResponse},
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
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// OpenAI configuration for the generic provider
struct OpenAIConfig;

impl OpenAICompatibleConfig for OpenAIConfig {
    const PROVIDER_NAME: &'static str = "OpenAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4.1-nano";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

/// Client for OpenAI API
pub struct OpenAI {
    // Delegate to the generic provider for common functionality
    provider: OpenAICompatibleProvider<OpenAIConfig>,
    pub enable_web_search: Option<bool>,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
}

/// Web search options specific to OpenAI
#[derive(Deserialize, Debug, Serialize)]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct ApproximateLocation {
    pub country: String,
    pub city: String,
    pub region: String,
}

#[derive(Serialize, Debug)]
pub struct OpenAIChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<OpenAICompatibleChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<OpenAIWebSearchOptions>,
}

#[derive(Serialize, Debug)]
pub struct OpenAIWebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

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
        OpenAI {
            provider: <OpenAICompatibleProvider<OpenAIConfig>>::new(
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
                None, // parallel_tool_calls
            ),
            enable_web_search,
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        }
    }

    // /// Convert a ChatMessage to OpenAI-specific message format
    // fn chat_message_to_openai_message(&self, chat_msg: ChatMessage) -> OpenAIChatMessage {
    //     OpenAIChatMessage {
    //         role: match chat_msg.role {
    //             ChatRole::User => "user",
    //             ChatRole::Assistant => "assistant",
    //         },
    //         tool_call_id: None,
    //         content: match &chat_msg.message_type {
    //             MessageType::Text => Some(Right(chat_msg.content.clone())),
    //             MessageType::Image(_) => unreachable!(),
    //             MessageType::Pdf(_) => unimplemented!(),
    //             MessageType::ImageURL(url) => {
    //                 let owned_url = url.clone();
    //                 let url_str = Box::leak(owned_url.into_boxed_str());
    //                 Some(Left(vec![MessageContent {
    //                     message_type: Some("image_url"),
    //                     text: None,
    //                     image_url: Some(ImageUrlContent { url: url_str }),
    //                     tool_output: None,
    //                     tool_call_id: None,
    //                 }]))
    //             }
    //             MessageType::ToolUse(_) => None,
    //             MessageType::ToolResult(_) => None,
    //         },
    //         tool_calls: match &chat_msg.message_type {
    //             MessageType::ToolUse(calls) => {
    //                 let owned_calls: Vec<OpenAIFunctionCall> = calls
    //                     .iter()
    //                     .map(|c| OpenAIFunctionCall {
    //                         id: c.id.clone(),
    //                         content_type: "function",
    //                         function: OpenAIFunctionPayload {
    //                             name: c.function.name.clone(),
    //                             arguments: c.function.arguments.clone(),
    //                         },
    //                     })
    //                     .collect();
    //                 Some(owned_calls)
    //             }
    //             _ => None,
    //         },
    //     }
    // }
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

// #[async_trait]
// impl ChatProvider for OpenAI {
//     // async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
//     //     self.chat_with_tools(messages, None).await
//     // }

//     async fn chat_with_tools(
//         &self,
//         messages: &[ChatMessage],
//         tools: Option<&[Tool]>,
//     ) -> Result<Box<dyn ChatResponse>, LLMError> {
//         // If web search is not enabled, delegate to the generic provider
//         if !self.enable_web_search.unwrap_or(false) {
//             return self.provider.chat_with_tools(messages, tools).await;
//         }

//         // Full web search implementation
//         if self.api_key().is_empty() {
//             return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
//         }

//         // Clone the messages to have an owned mutable vector.
//         let messages = messages.to_vec();

//         let mut openai_msgs: Vec<OpenAIChatMessage> = vec![];

//         for msg in messages {
//             if let MessageType::ToolResult(ref results) = msg.message_type {
//                 for result in results {
//                     openai_msgs.push(
//                         // Clone strings to own them
//                         OpenAIChatMessage {
//                             role: "tool",
//                             tool_call_id: Some(result.id.clone()),
//                             tool_calls: None,
//                             content: Some(Right(result.function.arguments.clone())),
//                         },
//                     );
//                 }
//             } else {
//                 openai_msgs.push(self.chat_message_to_openai_message(msg))
//             }
//         }

//         if let Some(system) = &self.provider.system {
//             openai_msgs.insert(
//                 0,
//                 OpenAIChatMessage {
//                     role: "system",
//                     content: Some(Left(vec![MessageContent {
//                         message_type: Some("text"),
//                         text: Some(Box::leak(system.clone().into_boxed_str())),
//                         image_url: None,
//                         tool_call_id: None,
//                         tool_output: None,
//                     }])),
//                     tool_calls: None,
//                     tool_call_id: None,
//                 },
//             );
//         }

//         let response_format: Option<OpenAIResponseFormat> =
//             self.provider.json_schema.clone().map(|s| s.into());

//         let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.provider.tools.clone());

//         let request_tool_choice = if request_tools.is_some() {
//             self.provider.tool_choice.clone()
//         } else {
//             None
//         };

//         let web_search_options = if self.enable_web_search.unwrap_or(false) {
//             let loc_type_opt = self
//                 .web_search_user_location_type
//                 .as_ref()
//                 .filter(|t| matches!(t.as_str(), "exact" | "approximate"));

//             let country = self.web_search_user_location_approximate_country.as_ref();
//             let city = self.web_search_user_location_approximate_city.as_ref();
//             let region = self.web_search_user_location_approximate_region.as_ref();

//             let approximate = if [country, city, region].iter().any(|v| v.is_some()) {
//                 Some(ApproximateLocation {
//                     country: country.cloned().unwrap_or_default(),
//                     city: city.cloned().unwrap_or_default(),
//                     region: region.cloned().unwrap_or_default(),
//                 })
//             } else {
//                 None
//             };

//             let user_location = loc_type_opt.map(|loc_type| UserLocation {
//                 location_type: loc_type.clone(),
//                 approximate,
//             });

//             Some(OpenAIWebSearchOptions {
//                 search_context_size: self.web_search_context_size.clone(),
//                 user_location,
//             })
//         } else {
//             None
//         };

//         let body = OpenAIChatRequest {
//             model: &self.provider.model,
//             messages: openai_msgs,
//             max_tokens: self.provider.max_tokens,
//             temperature: self.provider.temperature,
//             stream: self.provider.stream.unwrap_or(false),
//             top_p: self.provider.top_p,
//             top_k: self.provider.top_k,
//             tools: request_tools,
//             tool_choice: request_tool_choice,
//             reasoning_effort: self.provider.reasoning_effort.clone(),
//             response_format,
//             web_search_options,
//         };

//         let url = self
//             .provider.base_url
//             .join("chat/completions")
//             .map_err(|e| LLMError::HttpError(e.to_string()))?;

//         let mut request = self.provider.client.post(url).bearer_auth(&self.provider.api_key).json(&body);

//         if log::log_enabled!(log::Level::Trace) {
//             if let Ok(json) = serde_json::to_string(&body) {
//                 log::trace!("OpenAI request payload: {}", json);
//             }
//         }

//         if let Some(timeout) = self.provider.timeout_seconds {
//             request = request.timeout(std::time::Duration::from_secs(timeout));
//         }

//         let response = request.send().await?;

//         log::debug!("OpenAI HTTP status: {}", response.status());

//         if !response.status().is_success() {
//             let status = response.status();
//             let error_text = response.text().await?;
//             return Err(LLMError::ResponseFormatError {
//                 message: format!("OpenAI API returned error status: {}", status),
//                 raw_response: error_text,
//             });
//         }

//         // Parse the successful response
//         let resp_text = response.text().await?;
//         let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
//             serde_json::from_str(&resp_text);

//         match json_resp {
//             Ok(response) => Ok(Box::new(response)),
//             Err(e) => Err(LLMError::ResponseFormatError {
//                 message: format!("Failed to decode OpenAI API response: {}", e),
//                 raw_response: resp_text,
//             }),
//         }
//     }
// }

// Delegate other provider traits to the internal provider
#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.provider.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        self.provider.chat_stream(messages).await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        self.provider.chat_stream_struct(messages).await
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
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "OpenAI speech-to-text not implemented in this wrapper.".into(),
        ))
    }

    async fn transcribe_file(&self, file_path: &str) -> Result<String, LLMError> {
        let url = self
            .base_url()
            .join("audio/transcriptions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let form = reqwest::multipart::Form::new()
            .text("model", self.model().to_string())
            .text("response_format", "text")
            .file("file", file_path)
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client()
            .post(url)
            .bearer_auth(self.api_key())
            .multipart(form);

        if let Some(t) = self.timeout_seconds() {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        Ok(text)
    }
}

#[async_trait]
impl TextToSpeechProvider for OpenAI {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "OpenAI text-to-speech not implemented in this wrapper.".into(),
        ))
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key().is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        // Note: This would need access to embedding-specific fields that aren't in the generic provider
        // For now, use defaults
        let body = OpenAIEmbeddingRequest {
            model: self.model().to_string(),
            input,
            encoding_format: Some("float".to_string()),
            dimensions: None,
        };

        let url = self
            .base_url()
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client()
            .post(url)
            .bearer_auth(self.api_key())
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
            .base_url()
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client()
            .get(url)
            .bearer_auth(self.api_key())
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<OpenAIModelListResponse>().await?;
        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {}

// Helper methods to access provider fields
impl OpenAI {
    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn base_url(&self) -> &reqwest::Url {
        &self.provider.base_url
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.provider.timeout_seconds
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.provider.client
    }

    pub fn tools(&self) -> Option<&[Tool]> {
        self.provider.tools.as_deref()
    }
}
