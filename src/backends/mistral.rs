//! Mistral API client implementation for chat and completion functionality.
//!
//! This module provides integration with Mistral's LLM models through their API.

use crate::chat::Usage;
#[cfg(feature = "mistral")]
use crate::{
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
#[cfg(feature = "mistral")]
use crate::{
    chat::{ChatResponse, ToolChoice},
    ToolCall,
};
use async_trait::async_trait;
use either::*;
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Client for interacting with Mistral's API.
///
/// Provides methods for chat and embedding requests using Mistral's models.
pub struct Mistral {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub reasoning_effort: Option<String>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    /// Whether to allow parallel tool usage
    pub parallel_tool_calls: Option<bool>,
    client: Client,
}

/// Individual message in a Mistral chat conversation.
#[derive(Serialize, Debug)]
struct MistralChatMessage<'a> {
    #[allow(dead_code)]
    role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<MistralMessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Payload for a function call in a Mistral chat message.
#[derive(Serialize, Debug)]
struct MistralFunctionPayload<'a> {
    name: &'a str,
    arguments: &'a str,
}

/// Representation of a function call (tool usage) in a Mistral chat message.
#[derive(Serialize, Debug)]
struct MistralFunctionCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    content_type: &'a str,
    function: MistralFunctionPayload<'a>,
}

/// Structured content for complex message types (e.g., image URLs, tool outputs) in Mistral.
#[derive(Serialize, Debug)]
struct MistralMessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<&'a str>,
}

/// Individual image URL content in a Mistral chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

/// Request payload for Mistral's chat API endpoint.
#[derive(Serialize, Debug)]
struct MistralChatRequest<'a> {
    model: &'a str,
    messages: Vec<MistralChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

/// Response from Mistral's chat API endpoint.
#[derive(Deserialize, Debug)]
struct MistralChatResponse {
    choices: Vec<MistralChatChoice>,
    usage: Option<Usage>,
}

/// Individual choice within a Mistral chat API response.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct MistralChatChoice {
    message: MistralChatMsg,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<u32>,
}

/// Message content within a Mistral chat API response.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct MistralChatMsg {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

/// SSE (Server-Sent Events) chunk parser for Mistral's streaming responses.
///
/// Parses an SSE data chunk and extracts any generated content.
fn parse_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    let mut collected_content = String::new();
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                return if collected_content.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(collected_content))
                };
            }
            match serde_json::from_str::<MistralChatStreamResponse>(data) {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    }
    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}

/// Response from Mistral's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct MistralChatStreamResponse {
    choices: Vec<MistralChatStreamChoice>,
}
#[derive(Deserialize, Debug)]
struct MistralChatStreamChoice {
    delta: MistralChatStreamDelta,
}
#[derive(Deserialize, Debug)]
struct MistralChatStreamDelta {
    content: Option<String>,
}

/// Output format type for structured responses in Mistral.
#[derive(Deserialize, Debug, Serialize)]
enum MistralResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

/// Configuration for forcing the model output format (e.g., JSON schema) in Mistral.
#[derive(Deserialize, Debug, Serialize)]
struct MistralResponseFormat {
    #[serde(rename = "type")]
    response_type: MistralResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

impl From<StructuredOutputFormat> for MistralResponseFormat {
    /// Converts a `StructuredOutputFormat` into Mistral's response format configuration.
    ///
    /// Ensures that JSON schema conforms to requirements (adds `"additionalProperties": false` if missing).
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        match structured_response_format.schema {
            None => MistralResponseFormat {
                response_type: MistralResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                // Ensure "additionalProperties": false in the schema if not present.
                if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                }
                MistralResponseFormat {
                    response_type: MistralResponseType::JsonSchema,
                    json_schema: Some(StructuredOutputFormat {
                        name: structured_response_format.name,
                        description: structured_response_format.description,
                        schema: Some(schema),
                        strict: structured_response_format.strict,
                    }),
                }
            }
        }
    }
}

impl ChatResponse for MistralChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

impl std::fmt::Display for MistralChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.text(), &self.tool_calls()) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl Mistral {
    /// Creates a new Mistral client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Mistral API key
    /// * `base_url` - Base URL for the API (defaults to "https://api.mistral.ai/v1/")
    /// * `model` - Model to use (defaults to "mistral-medium-2505")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt to guide the model
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    /// * `embedding_encoding_format` - Format for embedding outputs (`float` or `base64`)
    /// * `embedding_dimensions` - Dimensions for embedding vectors
    /// * `tools` - Function tools that the model can use
    /// * `tool_choice` - Determines how the model uses tools
    /// * `reasoning_effort` - Reasoning effort level (e.g., "low", "medium", "high")
    /// * `json_schema` - JSON schema for structured output (if required)
    /// * `parallel_tool_calls` - Whether to enable parallel tool usage
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        parallel_tool_calls: Option<bool>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(
                &base_url.unwrap_or_else(|| "https://api.mistral.ai/v1/".to_owned()),
            )
            .expect("Failed to parse base URL"),
            model: model.unwrap_or_else(|| "mistral-medium-2505".to_owned()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            reasoning_effort,
            json_schema,
            parallel_tool_calls,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Mistral {
    /// Sends a chat request to Mistral's API (optionally with tool usage).
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to include (overrides default tools if provided)
    ///
    /// # Returns
    ///
    /// The model's response or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".to_string()));
        }
        // Clone messages to own them for request payload
        let messages = messages.to_vec();
        let mut mistral_msgs: Vec<MistralChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                // Include tool results as messages with role "tool"
                for result in results {
                    mistral_msgs.push(MistralChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                mistral_msgs.push(chat_message_to_api_message(msg));
            }
        }

        // Prepend system prompt as a "system" role message if provided
        if let Some(system) = &self.system {
            mistral_msgs.insert(
                0,
                MistralChatMessage {
                    role: "system",
                    content: Some(Left(vec![MistralMessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        // Determine tools and tool_choice for this request
        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        // Build the response format object if a structured output schema is provided
        let response_format: Option<MistralResponseFormat> =
            self.json_schema.clone().map(|s| s.into());

        // Construct the request payload
        let body = MistralChatRequest {
            model: &self.model,
            messages: mistral_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
            parallel_tool_calls: self.parallel_tool_calls,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Mistral request payload: {}", json);
            }
        }

        // Prepare the request URL and HTTP request
        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(Duration::from_secs(timeout));
        }

        // Send the request
        let response = request.send().await?;
        log::debug!("Mistral HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Mistral API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        // Parse the successful response JSON
        let resp_text = response.text().await?;
        let json_resp: Result<MistralChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(res) => Ok(Box::new(res)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode Mistral API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Mistral's API.
    ///
    /// # Returns
    ///
    /// A stream of response text chunks or an error if the request fails.
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".to_string()));
        }
        // Clone messages for ownership
        let messages = messages.to_vec();
        let mut mistral_msgs: Vec<MistralChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    mistral_msgs.push(MistralChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                mistral_msgs.push(chat_message_to_api_message(msg));
            }
        }
        if let Some(system) = &self.system {
            mistral_msgs.insert(
                0,
                MistralChatMessage {
                    role: "system",
                    content: Some(Left(vec![MistralMessageContent {
                        message_type: Some("text"),
                        text: Some(system),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        // Use any configured tools (no override for streaming, uses self.tools)
        let body = MistralChatRequest {
            model: &self.model,
            messages: mistral_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: None,
            parallel_tool_calls: self.parallel_tool_calls,
        };
        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Mistral API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        // Return a Server-Sent Events stream of the response content
        Ok(crate::chat::create_sse_stream(response, parse_sse_chunk))
    }
}

/// Convert a `ChatMessage` into a `MistralChatMessage` with `'static` lifetime for API serialization.
fn chat_message_to_api_message(chat_msg: ChatMessage) -> MistralChatMessage<'static> {
    MistralChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        },
        tool_call_id: None,
        content: match &chat_msg.message_type {
            MessageType::Text => Some(Right(chat_msg.content.clone())),
            MessageType::Image(_) => unreachable!(),
            MessageType::Pdf(_) => unimplemented!(),
            MessageType::ImageURL(url) => {
                // Leak the URL string to achieve 'static lifetime
                let owned_url = url.clone();
                let url_str = Box::leak(owned_url.into_boxed_str());
                Some(Left(vec![MistralMessageContent {
                    message_type: Some("image_url"),
                    text: None,
                    image_url: Some(ImageUrlContent { url: url_str }),
                    tool_output: None,
                    tool_call_id: None,
                }]))
            }
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                // Convert each tool call to Mistral format with leaked string references
                let owned_calls: Vec<MistralFunctionCall<'static>> = calls
                    .iter()
                    .map(|c| {
                        let owned_id = c.id.clone();
                        let owned_name = c.function.name.clone();
                        let owned_args = c.function.arguments.clone();
                        let id_str = Box::leak(owned_id.into_boxed_str());
                        let name_str = Box::leak(owned_name.into_boxed_str());
                        let args_str = Box::leak(owned_args.into_boxed_str());
                        MistralFunctionCall {
                            id: id_str,
                            content_type: "function",
                            function: MistralFunctionPayload {
                                name: name_str,
                                arguments: args_str,
                            },
                        }
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

#[async_trait]
impl CompletionProvider for Mistral {
    /// Sends a completion request to Mistral's API (not supported for chat models).
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Mistral completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Mistral {
    /// Generates embeddings for the given input texts using Mistral's API.
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".into()));
        }
        // Determine output data type for embeddings (default to float)
        let dtype = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());
        // Prepare embedding request payload
        let body = MistralEmbeddingRequest {
            model: self.model.clone(),
            input,
            output_dtype: Some(dtype),
            output_dimension: self.embedding_dimensions,
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
        // Extract embeddings vectors from response
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

/// Request payload for Mistral's embedding API endpoint.
#[derive(Serialize)]
struct MistralEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "output_dtype")]
    output_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "output_dimension")]
    output_dimension: Option<u32>,
}

/// Response from Mistral's embedding API endpoint.
#[derive(Deserialize, Debug)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingData {
    embedding: Vec<f32>,
}

impl LLMProvider for Mistral {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl SpeechToTextProvider for Mistral {
    /// Transcribing audio is not supported by Mistral.
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Mistral does not implement speech-to-text.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Mistral {
    /// Text-to-speech conversion is not supported by Mistral.
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Text-to-speech not supported by Mistral.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Mistral {
    // Uses default implementation (listing models not supported by Mistral API)
}

#[tokio::test]
async fn test_mistral_chat_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
        chat::ChatMessage,
    };

    // Skip test if MISTRAL_API_KEY environment variable is not set
    if std::env::var("MISTRAL_API_KEY").is_err() {
        eprintln!("test test_mistral_chat_with_tools ... ignored, MISTRAL_API_KEY not set");
        return Ok(());
    }
    let api_key = std::env::var("MISTRAL_API_KEY").unwrap();
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
                    ParamBuilder::new("url")
                        .type_of("string")
                        .description("The url to get the weather from for the city"),
                )
                .required(vec!["url".to_string()]),
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
async fn test_mistral_chat() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        builder::{LLMBackend, LLMBuilder},
        chat::ChatMessage,
    };

    if std::env::var("MISTRAL_API_KEY").is_err() {
        eprintln!("test test_mistral_chat ... ignored, MISTRAL_API_KEY not set");
        return Ok(());
    }
    let api_key = std::env::var("MISTRAL_API_KEY").unwrap();
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model("mistral-small-latest")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat(&messages).await {
        Ok(response) => {
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
