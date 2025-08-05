//! LLM (Rust LLM) is a unified interface for interacting with Large Language Model providers.
//!
//! # Overview
//! This crate provides a consistent API for working with different LLM backends by abstracting away
//! provider-specific implementation details. It supports:
//!
//! - Chat-based interactions
//! - Text completion
//! - Embeddings generation
//! - Multiple providers (OpenAI, Anthropic, etc.)
//! - Request validation and retry logic
//!
//! # Architecture
//! The crate is organized into modules that handle different aspects of LLM interactions:

// Re-export for convenience
pub use async_trait::async_trait;

use chat::Tool;
use serde::{Deserialize, Serialize};

/// Backend implementations for supported LLM providers like OpenAI, Anthropic, etc.
pub mod backends;

/// Builder pattern for configuring and instantiating LLM providers
pub mod builder;

/// Chain multiple LLM providers together for complex workflows
pub mod chain;

/// Chat-based interactions with language models (e.g. ChatGPT style)
pub mod chat;

/// Text completion capabilities (e.g. GPT-3 style completion)
pub mod completion;

/// Vector embeddings generation for text
pub mod embedding;

/// Error types and handling
pub mod error;

/// Validation wrapper for LLM providers with retry capabilities
pub mod validated_llm;

/// Evaluator for LLM providers
pub mod evaluator;

/// Speech-to-text support
pub mod stt;

/// Text-to-speech support
pub mod tts;

/// Secret store for storing API keys and other sensitive information
pub mod secret_store;

/// Listing models support
pub mod models;

/// Memory providers for storing and retrieving conversation history
#[macro_use]
pub mod memory;

#[cfg(feature = "agent")]
pub mod agent;

#[cfg(feature = "api")]
pub mod api;

#[inline]
/// Initialize logging using env_logger if the "logging" feature is enabled.
/// This is a no-op if the feature is not enabled.
pub fn init_logging() {
    #[cfg(feature = "logging")]
    {
        let _ = env_logger::try_init();
    }
}

/// Core trait that all LLM providers must implement, combining chat, completion
/// and embedding capabilities into a unified interface
pub trait LLMProvider:
    chat::ChatProvider
    + completion::CompletionProvider
    + embedding::EmbeddingProvider
    + stt::SpeechToTextProvider
    + tts::TextToSpeechProvider
    + models::ModelsProvider
{
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

/// Tool call represents a function call that an LLM wants to make.
/// This is a standardized structure used across all providers.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool call (defaults to "function" if not provided).
    #[serde(rename = "type", default = "default_call_type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// Default value for call_type field in ToolCall
fn default_call_type() -> String {
    "function".to_string()
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}

impl std::fmt::Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"id\": \"{}\",\n  \"type\": \"{}\",\n  \"function\": {}\n}}",
            self.id, self.call_type, self.function
        )
    }
}

impl std::fmt::Display for FunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"name\": \"{}\",\n  \"arguments\": {}\n}}",
            self.name, self.arguments
        )
    }
}
