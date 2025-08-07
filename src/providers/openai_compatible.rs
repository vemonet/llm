//! OpenAI-compatible API client base implementation
//!
//! This module provides a generic base for OpenAI-compatible APIs that can be reused
//! across multiple providers like OpenAI, Mistral, XAI, Groq, DeepSeek, etc.

use crate::{
    chat::{
        ChatMessage, ChatProvider, ChatRole, MessageType, StreamChoice, StreamDelta,
        StreamResponse, StructuredOutputFormat, Tool, ToolChoice, Usage,
    },
    chat::ChatResponse,
    ToolCall,
};
use crate::error::LLMError;
use async_trait::async_trait;
use either::*;
use futures::{stream::Stream, StreamExt};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Generic OpenAI-compatible provider
///
/// This struct provides a base implementation for any OpenAI-compatible API.
/// Different providers can customize behavior by implementing the `OpenAICompatibleConfig` trait.
pub struct OpenAICompatibleProvider<T: OpenAICompatibleConfig> {
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
    pub reasoning_effort: Option<String>,
    pub json_schema: Option<StructuredOutputFormat>,
    pub voice: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub client: Client,
    _phantom: PhantomData<T>,
}

/// Configuration trait for OpenAI-compatible providers
///
/// This trait allows different providers to customize behavior while reusing
/// the common OpenAI-compatible implementation.
pub trait OpenAICompatibleConfig: Send + Sync {
    /// The name of the provider (e.g., "OpenAI", "Mistral", "XAI")
    const PROVIDER_NAME: &'static str;

    /// Default base URL for the provider
    const DEFAULT_BASE_URL: &'static str;

    /// Default model for the provider
    const DEFAULT_MODEL: &'static str;

    /// Chat completions endpoint path (usually "chat/completions")
    const CHAT_ENDPOINT: &'static str = "chat/completions";

    /// Whether this provider supports reasoning effort
    const SUPPORTS_REASONING_EFFORT: bool = false;

    /// Whether this provider supports structured output
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;

    /// Whether this provider supports parallel tool calls
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;

    /// Whether this provider supports stream options (like include_usage)
    const SUPPORTS_STREAM_OPTIONS: bool = false;

    /// Custom headers to add to requests
    fn custom_headers() -> Option<Vec<(String, String)>> {
        None
    }

    /// Transform the request before sending (for provider-specific modifications)
    fn transform_request(request: &mut OpenAICompatibleChatRequest) -> Result<(), LLMError> {
        let _ = request;
        Ok(())
    }

    /// Transform the response after receiving (for provider-specific processing)
    fn transform_response(response: &mut OpenAICompatibleChatResponse) -> Result<(), LLMError> {
        let _ = response;
        Ok(())
    }
}

/// Generic OpenAI-compatible chat message
#[derive(Serialize, Debug)]
pub struct OpenAICompatibleChatMessage<'a> {
    pub role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    pub content: Option<Either<Vec<MessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAICompatibleFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct OpenAICompatibleFunctionPayload<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug)]
pub struct OpenAICompatibleFunctionCall<'a> {
    pub id: &'a str,
    #[serde(rename = "type")]
    pub content_type: &'a str,
    pub function: OpenAICompatibleFunctionPayload<'a>,
}

#[derive(Serialize, Debug)]
pub struct MessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    pub tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    pub tool_output: Option<&'a str>,
}

#[derive(Serialize, Debug)]
pub struct ImageUrlContent<'a> {
    pub url: &'a str,
}

/// Generic OpenAI-compatible chat request
#[derive(Serialize, Debug)]
pub struct OpenAICompatibleChatRequest<'a> {
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
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Generic OpenAI-compatible chat response
#[derive(Deserialize, Debug)]
pub struct OpenAICompatibleChatResponse {
    pub choices: Vec<OpenAICompatibleChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAICompatibleChatChoice {
    pub message: OpenAICompatibleChatMsg,
}

#[derive(Deserialize, Debug)]
pub struct OpenAICompatibleChatMsg {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug, Serialize)]
pub enum ResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub response_type: ResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<StructuredOutputFormat>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// Streaming response structures
#[derive(Deserialize, Debug)]
pub struct ChatStreamChunk {
    pub choices: Vec<ChatStreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct ChatStreamChoice {
    pub delta: ChatStreamDelta,
}

#[derive(Deserialize, Debug)]
pub struct ChatStreamDelta {
    pub content: Option<String>,
}

impl From<StructuredOutputFormat> for ResponseFormat {
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        match structured_response_format.schema {
            None => ResponseFormat {
                response_type: ResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                schema = if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                    schema
                } else {
                    schema
                };

                ResponseFormat {
                    response_type: ResponseType::JsonSchema,
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

impl ChatResponse for OpenAICompatibleChatResponse {
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

impl std::fmt::Display for OpenAICompatibleChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (
            &self.choices.first().unwrap().message.content,
            &self.choices.first().unwrap().message.tool_calls,
        ) {
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

impl<T: OpenAICompatibleConfig> OpenAICompatibleProvider<T> {
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
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        voice: Option<String>,
        parallel_tool_calls: Option<bool>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }

        Self {
            api_key: api_key.into(),
            base_url: Url::parse(
                &base_url.unwrap_or_else(|| T::DEFAULT_BASE_URL.to_owned()),
            )
            .expect("Failed to parse base URL"),
            model: model.unwrap_or_else(|| T::DEFAULT_MODEL.to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning_effort,
            json_schema,
            voice,
            parallel_tool_calls,
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T: OpenAICompatibleConfig> ChatProvider for OpenAICompatibleProvider<T> {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!("Missing {} API key", T::PROVIDER_NAME)));
        }

        let messages = messages.to_vec();
        let mut openai_msgs: Vec<OpenAICompatibleChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    openai_msgs.push(OpenAICompatibleChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAICompatibleChatMessage {
                    role: "system",
                    content: Some(Left(vec![MessageContent {
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

        let response_format: Option<ResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            self.json_schema.clone().map(|s| s.into())
        } else {
            None
        };
        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };
        let reasoning_effort = if T::SUPPORTS_REASONING_EFFORT {
            self.reasoning_effort.clone()
        } else {
            None
        };
        let parallel_tool_calls = if T::SUPPORTS_PARALLEL_TOOL_CALLS {
            self.parallel_tool_calls
        } else {
            None
        };

        let mut body = OpenAICompatibleChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort,
            response_format,
            stream_options: None,
            parallel_tool_calls,
        };

        // Allow provider-specific request transformation
        T::transform_request(&mut body)?;

        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        // Add custom headers if provider specifies them
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("{} request payload: {}", T::PROVIDER_NAME, json);
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("{} HTTP status: {}", T::PROVIDER_NAME, response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }

        let resp_text = response.text().await?;
        let json_resp: Result<OpenAICompatibleChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(mut response) => {
                // Allow provider-specific response transformation
                T::transform_response(&mut response)?;
                Ok(Box::new(response))
            }
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode {} API response: {e}", T::PROVIDER_NAME),
                raw_response: resp_text,
            }),
        }
    }

    /// Perform a chat request without tool calls
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Stream chat responses as a stream of strings
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let struct_stream = self.chat_stream_struct(messages).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    if let Some(choice) = stream_response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                return Some(Ok(content.clone()));
                            }
                        }
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    /// Stream chat responses as `ChatMessage` structured objects, including usage information
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!("Missing {} API key", T::PROVIDER_NAME)));
        }
        let messages = messages.to_vec();
        let mut openai_msgs: Vec<OpenAICompatibleChatMessage> = vec![];
        for msg in messages {
            if let MessageType::ToolResult(ref results) = msg.message_type {
                for result in results {
                    openai_msgs.push(OpenAICompatibleChatMessage {
                        role: "tool",
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }
        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAICompatibleChatMessage {
                    role: "system",
                    content: Some(Left(vec![MessageContent {
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
        let mut body = OpenAICompatibleChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: if T::SUPPORTS_REASONING_EFFORT {
                self.reasoning_effort.clone()
            } else {
                None
            },
            response_format: None,
            stream_options: if T::SUPPORTS_STREAM_OPTIONS {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            parallel_tool_calls: if T::SUPPORTS_PARALLEL_TOOL_CALLS {
                self.parallel_tool_calls
            } else {
                None
            },
        };

        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        T::transform_request(&mut body)?;
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }
        Ok(create_struct_sse_stream(response))
    }
}

/// Create an owned `OpenAICompatibleChatMessage` that doesn't borrow from any temporary variables
pub fn chat_message_to_api_message(chat_msg: ChatMessage) -> OpenAICompatibleChatMessage<'static> {
    OpenAICompatibleChatMessage {
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
                let owned_url = url.clone();
                let url_str = Box::leak(owned_url.into_boxed_str());
                Some(Left(vec![MessageContent {
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
                let owned_calls: Vec<OpenAICompatibleFunctionCall<'static>> = calls
                    .iter()
                    .map(|c| {
                        let id_str = Box::leak(c.id.clone().into_boxed_str());
                        let name_str = Box::leak(c.function.name.clone().into_boxed_str());
                        let args_str = Box::leak(c.function.arguments.clone().into_boxed_str());
                        OpenAICompatibleFunctionCall {
                            id: id_str,
                            content_type: "function",
                            function: OpenAICompatibleFunctionPayload {
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

/// Creates a structured SSE stream that returns `StreamResponse` objects
pub fn create_struct_sse_stream(
    response: reqwest::Response,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .map(move |chunk| match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                parse_sse_chunk(&text)
            }
            Err(e) => Err(LLMError::HttpError(e.to_string())),
        })
        .filter_map(|result| async move {
            match result {
                Ok(Some(response)) => Some(Ok(response)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });
    Box::pin(stream)
}

/// Parse SSE chunk and convert to `StreamResponse` format
pub fn parse_sse_chunk(chunk: &str) -> Result<Option<StreamResponse>, LLMError> {
    let mut collected_content = String::new();
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                if collected_content.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some(StreamResponse {
                        choices: vec![StreamChoice {
                            delta: StreamDelta {
                                content: Some(collected_content),
                            },
                        }],
                        usage: None,
                    }));
                }
            }
            match serde_json::from_str::<ChatStreamChunk>(data) {
                Ok(response) => {
                    if let Some(usage) = response.usage {
                        return Ok(Some(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta { content: None },
                            }],
                            usage: Some(usage),
                        }));
                    }
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
        Ok(Some(StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: Some(collected_content),
                },
            }],
            usage: None,
        }))
    }
}
