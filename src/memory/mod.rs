//! Memory module for storing and retrieving conversation history.
//!
//! This module provides various memory implementations for LLM conversations:
//! - SlidingWindowMemory: Simple FIFO memory with size limit
//! - EmbeddingMemory: Semantic search using embeddings (future)
//! - RAGMemory: Document-based retrieval (future)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{chat::ChatMessage, error::LLMError};

pub mod chat_wrapper;
pub mod shared_memory;
pub mod sliding_window;

pub use chat_wrapper::ChatWithMemory;
pub use shared_memory::SharedMemory;
pub use sliding_window::{SlidingWindowMemory, TrimStrategy};

/// Types of memory implementations available
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Simple sliding window that keeps the N most recent messages
    SlidingWindow,
}

/// Trait for memory providers that can store and retrieve conversation history.
///
/// Memory providers enable LLMs to maintain context across conversations by:
/// - Storing messages as they are exchanged
/// - Retrieving relevant past messages based on queries
/// - Managing memory size and cleanup
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Store a message in memory.
    ///
    /// # Arguments
    ///
    /// * `message` - The chat message to store
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the message was stored successfully
    /// * `Err(LLMError)` if storage failed
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError>;

    /// Retrieve relevant messages from memory based on a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to search for relevant messages
    /// * `limit` - Optional maximum number of messages to return
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<ChatMessage>)` containing relevant messages
    /// * `Err(LLMError)` if retrieval failed
    async fn recall(&self, query: &str, limit: Option<usize>)
        -> Result<Vec<ChatMessage>, LLMError>;

    /// Clear all stored messages from memory.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if memory was cleared successfully
    /// * `Err(LLMError)` if clearing failed
    async fn clear(&mut self) -> Result<(), LLMError>;

    /// Get the type of this memory provider.
    ///
    /// # Returns
    ///
    /// The memory type enum variant
    fn memory_type(&self) -> MemoryType;

    /// Get the current number of stored messages.
    ///
    /// # Returns
    ///
    /// The number of messages currently in memory
    fn size(&self) -> usize;

    /// Check if the memory is empty.
    ///
    /// # Returns
    ///
    /// `true` if no messages are stored, `false` otherwise
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Check if memory needs summarization
    fn needs_summary(&self) -> bool {
        false
    }

    /// Mark memory as needing summarization
    fn mark_for_summary(&mut self) {}

    /// Replace all messages with a summary
    fn replace_with_summary(&mut self, _summary: String) {}
}
