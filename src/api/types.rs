use serde::{Deserialize, Serialize};

/// Request payload for chat completion API endpoint
#[derive(Deserialize)]
pub struct ChatRequest {
    /// List of messages in the conversation
    pub messages: Vec<Message>,
    /// Model identifier in format "provider:model_name"
    pub model: String,
}

/// Single message in a chat conversation
#[derive(Deserialize, Serialize)]
pub struct Message {
    /// Role of the message sender ("user" or "assistant")
    pub role: String,
    /// Content of the message
    pub content: String,
}

/// Response payload from chat completion API endpoint
#[derive(Serialize)]
pub struct ChatResponse {
    /// Unique identifier for this completion
    pub id: String,
    /// Object type identifier
    pub object: String,
    /// Unix timestamp when response was created
    pub created: u64,
    /// Name of the model that generated the completion
    pub model: String,
    /// List of completion choices generated
    pub choices: Vec<Choice>,
}

/// Single completion choice in a chat response
#[derive(Serialize)]
pub struct Choice {
    /// Index of this choice in the list
    pub index: usize,
    /// Generated message for this choice
    pub message: Message,
    /// Reason why the model stopped generating
    pub finish_reason: String,
}
