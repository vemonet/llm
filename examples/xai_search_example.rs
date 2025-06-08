//! Example demonstrating X.AI search functionality
//!
//! This example shows how to use X.AI's search parameters to:
//! 1. Enable search mode
//! 2. Set maximum search results
//! 3. Specify date ranges for search
//! 4. Configure search sources with exclusions

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole, MessageType},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Basic search with auto mode and result limit
    let llm_basic = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(std::env::var("XAI_API_KEY").unwrap_or("xai-test-key".into()))
        .model("grok-3-latest")
        .xai_search_mode("auto")
        .xai_max_search_results(10)
        .build()?;

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "What are some recently discovered alternative DNA shapes?".to_string(),
    }];

    println!("=== Basic Search Example ===");
    let response = llm_basic.chat(&messages).await?;
    println!("Response: {}", response.text().unwrap_or_default());

    // Example 2: Search with date range
    let llm_dated = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(std::env::var("XAI_API_KEY").unwrap_or("xai-test-key".into()))
        .model("grok-3-latest")
        .xai_search_mode("auto")
        .xai_search_date_range("2022-01-01", "2022-12-31")
        .xai_max_search_results(5)
        .build()?;

    let dated_messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "What were the major AI breakthroughs in 2022?".to_string(),
    }];

    println!("\n=== Date Range Search Example ===");
    let dated_response = llm_dated.chat(&dated_messages).await?;
    println!("Response: {}", dated_response.text().unwrap_or_default());

    // Example 3: Search with source exclusions
    let llm_filtered = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(std::env::var("XAI_API_KEY").unwrap_or("xai-test-key".into()))
        .model("grok-3-latest")
        .xai_search_mode("auto")
        .xai_search_source("web", Some(vec!["wikipedia.org".to_string()]))
        .xai_search_source("news", Some(vec!["bbc.co.uk".to_string()]))
        .xai_max_search_results(8)
        .xai_search_from_date("2023-01-01")
        .build()?;

    let filtered_messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "What are the latest developments in quantum computing?".to_string(),
    }];

    println!("\n=== Filtered Sources Search Example ===");
    let filtered_response = llm_filtered.chat(&filtered_messages).await?;
    println!("Response: {}", filtered_response.text().unwrap_or_default());

    Ok(())
}
