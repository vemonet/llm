use crate::error::LLMError;

pub trait EmbeddingProvider {
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError>;
}
