use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use uuid::Uuid;

use super::types::{ChatRequest, ChatResponse, Choice, Message};
use super::ServerState;
use crate::chat::{ChatMessage, ChatRole};

/// Handles chat completion requests to the API server
///
/// # Arguments
/// * `state` - Server state containing LLM registry and auth configuration
/// * `headers` - HTTP request headers for authentication
/// * `req` - Chat request containing messages and model specification
///
/// # Returns
/// * `Ok(Json<ChatResponse>)` - Successful chat completion response
/// * `Err((StatusCode, String))` - Error response with status code and message
///
/// # Authentication
/// If server has auth_key configured, validates Bearer token in Authorization header
///
/// # Request Format
/// Model must be specified as "provider:model_name" (e.g. "openai:gpt-4")
///
/// # Response Format
/// Returns standardized chat completion response with:
/// - Unique ID
/// - Timestamp
/// - Model name
/// - Generated message
pub async fn handle_chat(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    if let Some(key) = &state.auth_key {
        let auth_header = headers.get("Authorization").ok_or((
            StatusCode::UNAUTHORIZED,
            "Missing authorization".to_string(),
        ))?;

        let auth_str = auth_header.to_str().map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                "Invalid authorization header".to_string(),
            )
        })?;

        if !auth_str.starts_with("Bearer ") || &auth_str[7..] != key {
            return Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string()));
        }
    }

    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|msg| ChatMessage {
            role: match msg.role.as_str() {
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                _ => ChatRole::User,
            },
            content: msg.content,
        })
        .collect();

    if let Some((provider_id, model_name)) = req.model.split_once(':') {
        let provider = state.llms.get(provider_id).ok_or((
            StatusCode::BAD_REQUEST,
            format!("Unknown provider: {}", provider_id),
        ))?;

        let response = provider
            .chat(&messages)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        return Ok(Json(ChatResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: model_name.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: response,
                },
                finish_reason: "stop".to_string(),
            }],
        }));
    } else {
        return Err((StatusCode::BAD_REQUEST, "Invalid model format".to_string()));
    }
}
