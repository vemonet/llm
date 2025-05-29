//! Simple sliding window memory implementation.
//!
//! This module provides a basic FIFO (First In, First Out) memory that maintains
//! a fixed-size window of the most recent conversation messages.

use std::collections::VecDeque;

use async_trait::async_trait;

use crate::{chat::ChatMessage, error::LLMError};

use super::{MemoryProvider, MemoryType};

/// Simple sliding window memory that keeps the N most recent messages.
///
/// This implementation uses a FIFO strategy where old messages are automatically
/// removed when the window size limit is reached. It's suitable for:
/// - Simple conversation contexts
/// - Memory-constrained environments
/// - Cases where only recent context matters
///
/// # Examples
///
/// ```rust
/// use llm::memory::SlidingWindowMemory;
/// use llm::chat::ChatMessage;
///
/// let mut memory = SlidingWindowMemory::new(3);
///
/// // Store messages - only the last 3 will be kept
/// memory.remember(&ChatMessage::user().content("Hello").build()).await?;
/// memory.remember(&ChatMessage::assistant().content("Hi there!").build()).await?;
/// memory.remember(&ChatMessage::user().content("How are you?").build()).await?;
/// memory.remember(&ChatMessage::assistant().content("I'm doing well!").build()).await?;
///
/// // Only the last 3 messages are kept
/// assert_eq!(memory.size(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct SlidingWindowMemory {
    messages: VecDeque<ChatMessage>,
    window_size: usize,
}

impl SlidingWindowMemory {
    /// Create a new sliding window memory with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of messages to keep in memory
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm::memory::SlidingWindowMemory;
    ///
    /// let memory = SlidingWindowMemory::new(10);
    /// assert_eq!(memory.window_size(), 10);
    /// assert!(memory.is_empty());
    /// ```
    pub fn new(window_size: usize) -> Self {
        if window_size == 0 {
            panic!("Window size must be greater than 0");
        }

        Self {
            messages: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Get the configured window size.
    ///
    /// # Returns
    ///
    /// The maximum number of messages this memory can hold
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get all stored messages in chronological order.
    ///
    /// # Returns
    ///
    /// A vector containing all messages from oldest to newest
    pub fn messages(&self) -> Vec<ChatMessage> {
        Vec::from(self.messages.clone())
    }

    /// Get the most recent N messages.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of recent messages to return
    ///
    /// # Returns
    ///
    /// A vector containing the most recent messages, up to `limit`
    pub fn recent_messages(&self, limit: usize) -> Vec<ChatMessage> {
        let len = self.messages.len();
        let start = len.saturating_sub(limit);
        self.messages.range(start..).cloned().collect()
    }
}

#[async_trait]
impl MemoryProvider for SlidingWindowMemory {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        if self.messages.len() >= self.window_size {
            self.messages.pop_front();
        }
        self.messages.push_back(message.clone());
        Ok(())
    }

    async fn recall(
        &self,
        _query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let limit = limit.unwrap_or(self.messages.len());
        Ok(self.recent_messages(limit))
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        self.messages.clear();
        Ok(())
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        self.messages.len()
    }
}
