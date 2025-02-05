//! Example demonstrating how to serve multiple LLM backends through a REST API
//!
//! This example shows how to:
//! 1. Initialize multiple LLM backends (OpenAI, Anthropic, DeepSeek)
//! 2. Create a registry to manage multiple backends
//! 3. Start a REST API server to expose the LLM backends
//! 4. Handle requests through a standardized API interface

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chain::LLMRegistryBuilder,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenAI backend with API key and model settings
    let openai_llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-OPENAI".into()))
        .model("gpt-4o")
        .build()?;

    // Initialize Anthropic backend with API key and model settings
    let anthro_llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
        .model("claude-3-5-sonnet-20240620")
        .build()?;

    // Initialize DeepSeek backend with API key and model settings
    let deepseek_llm = LLMBuilder::new()
        .backend(LLMBackend::DeepSeek)
        .api_key(std::env::var("DEEPSEEK_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("deepseek-chat")
        .build()?;

    // Initialize Groq backend with API key and model settings
    let groq_llm = LLMBuilder::new()
        .backend(LLMBackend::Groq)
        .api_key(std::env::var("GROQ_API_KEY").unwrap_or("gsk-YOUR_API_KEY".into()))
        .model("deepseek-r1-distill-llama-70b")
        .build()?;

    // Create registry to manage multiple backends
    let registry = LLMRegistryBuilder::new()
        .register("openai", openai_llm)
        .register("anthro", anthro_llm)
        .register("deepseek", deepseek_llm)
        .register("groq", groq_llm)
        .build();

    // Start REST API server on localhost port 3000
    registry.serve("127.0.0.1:3000").await?;
    Ok(())
}
