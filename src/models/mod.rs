use crate::{builder::LLMBackend, error::LLMError};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;

pub trait ModelListResponse: Send + Sync {
    fn get_models(&self) -> Vec<String>;
    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>>;
    fn get_backend(&self) -> LLMBackend;
}

pub trait ModelListRawEntry: Debug {
    fn get_id(&self) -> String;
    fn get_created_at(&self) -> DateTime<Utc>;
    fn get_raw(&self) -> serde_json::Value;
}

#[derive(Debug, Clone, Default)]
pub struct ModelListRequest {
    pub filter: Option<String>,
}

/// Standard model entry structure used by OpenAI-compatible providers
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StandardModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for StandardModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

/// Standard model list response structure used by OpenAI-compatible providers
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StandardModelListResponse {
    pub data: Vec<StandardModelEntry>,
}

impl StandardModelListResponse {
    pub fn new(data: Vec<StandardModelEntry>) -> Self {
        Self { data }
    }
}

// Note: Each provider will implement ModelListResponse for StandardModelListResponse
// with their specific backend type

/// Trait for providers that support listing and retrieving model information.
#[async_trait]
pub trait ModelsProvider {
    /// Asynchronously retrieves the list of available models ID's from the provider.
    ///
    /// # Arguments
    ///
    /// * `_request` - Optional filter by model ID
    ///
    /// # Returns
    ///
    /// List of model ID's or error
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "List Models not supported".to_string(),
        ))
    }
}
