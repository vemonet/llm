// Import required modules from the LLM library for HuggingFace integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment variable or use test key as fallback
    let api_key = std::env::var("HF_TOKEN").unwrap_or("gsk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::HuggingFace) // Use HuggingFace as the LLM provider
        .api_key(api_key) // Set the API key
        .model("moonshotai/Kimi-K2-Instruct-0905")
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .build()
        .expect("Failed to build LLM");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Tell me about quantum computing")
            .build(),
        ChatMessage::assistant()
            .content("Quantum computing is a type of computing that uses quantum phenomena...")
            .build(),
        ChatMessage::user().content("What are qubits?").build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }
    Ok(())
}
