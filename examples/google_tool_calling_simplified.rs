// A simplified example for Google tool calling that follows Google's API strictly
use llm::{
    builder::LLMBackend,
    builder::LLMBuilder,
    chat::ChatMessage,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable
    let api_key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set");

    // Build a Google client manually with JSON schema for structured output
    // This is an alternative to direct tool calling for Gemini
    let schema = json!({
        "type": "object",
        "properties": {
            "weather": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "temperature": { "type": "number" },
                    "description": { "type": "string" }
                },
                "required": ["location", "temperature", "description"]
            }
        }
    });
    
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(api_key)
        .model("gemini-1.5-flash")
        .max_tokens(512)
        .temperature(0.7)
        .schema(schema)
        .build()
        .expect("Failed to build LLM");

    // Create a simplified message that doesn't explicitly request tool use
    let messages = vec![
        ChatMessage::user()
            .content("What is the current weather in Tokyo, Japan?")
            .build()
    ];

    // Send the request
    match llm.chat(&messages).await {
        Ok(response) => {
            println!("Response: {}", response);
        },
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}