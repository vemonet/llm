// Import required modules from the LLM library for OpenAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gpt-4.1-nano") // Use GPT-4.1 Nano model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (OpenAI)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Tell me that you love cats")
            .build(),
        ChatMessage::assistant()
            .content("I am an assistant, I cannot love cats but I can love dogs")
            .build(),
        ChatMessage::user()
            .content("Tell me that you love dogs in 2000 chars")
            .build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(response) => {
            // Print the response text
            if let Some(text) = response.text() {
                println!("Response: {text}");
            }
            // Print usage information
            if let Some(usage) = response.usage() {
                println!("\nUsage Information:");
                println!("  Prompt tokens: {}", usage.prompt_tokens);
                println!("  Completion tokens: {}", usage.completion_tokens);
                println!("  Total tokens: {}", usage.total_tokens);
                if let Some(completion_details) = &usage.completion_tokens_details {
                    if let Some(reasoning_tokens) = completion_details.reasoning_tokens {
                        println!("  Reasoning tokens: {reasoning_tokens}");
                    }
                    if let Some(audio_tokens) = completion_details.audio_tokens {
                        println!("  Audio tokens: {audio_tokens}");
                    }
                }
                if let Some(prompt_details) = &usage.prompt_tokens_details {
                    if let Some(cached_tokens) = prompt_details.cached_tokens {
                        println!("  Cached tokens: {cached_tokens}");
                    }
                    if let Some(audio_tokens) = prompt_details.audio_tokens {
                        println!("  Audio tokens (prompt): {audio_tokens}");
                    }
                }
            } else {
                println!("No usage information available");
            }
        },
        Err(e) => eprintln!("Chat error: {e}"),
    }
    Ok(())
}
