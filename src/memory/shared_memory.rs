use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{MemoryProvider, MemoryType};
use crate::{chat::ChatMessage, error::LLMError};

#[derive(Clone)]
pub struct SharedMemory<T: MemoryProvider> {
    inner: Arc<RwLock<T>>,
}

impl<T: MemoryProvider> SharedMemory<T> {
    pub fn new(provider: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(provider)),
        }
    }
}

#[async_trait]
impl<T: MemoryProvider> MemoryProvider for SharedMemory<T> {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        let mut guard = self.inner.write().await;
        guard.remember(message).await
    }

    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let guard = self.inner.read().await;
        guard.recall(query, limit).await
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        let mut guard = self.inner.write().await;
        guard.clear().await
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        0
    }

    fn needs_summary(&self) -> bool {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let guard = self.inner.read().await;
                guard.needs_summary()
            })
        })
    }

    fn mark_for_summary(&mut self) {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut guard = self.inner.write().await;
                guard.mark_for_summary();
            })
        })
    }

    fn replace_with_summary(&mut self, summary: String) {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut guard = self.inner.write().await;
                guard.replace_with_summary(summary);
            })
        })
    }
}
