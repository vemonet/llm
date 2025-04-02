// Simplified debug example for OpenAI tool calling
use llm::{
    builder::{LLMBuilder, LLMBackend, FunctionBuilder, ParamBuilder},
    chat::ChatMessage,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Create the OpenAI client using the builder
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-3.5-turbo")
        .max_tokens(512)
        .temperature(0.7)
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a given location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city and state, e.g. San Francisco, CA")
                )
                .required(vec!["location".to_string()])
        )
        .build()
        .expect("Failed to build LLM");

    // Create a chat message that should trigger the function call
    let messages = vec![
        ChatMessage::user()
            .content("What's the weather like in Miami right now?")
            .build()
    ];

    println!("Sending chat request with function tools...");
    
    // Make the API call
    match llm.chat(&messages).await {
        Ok(response) => {
            println!("Success! Response text: {}", response.text().unwrap_or_default());
            
            if let Some(tool_calls) = response.tool_calls() {
                println!("Tool calls ({}):", tool_calls.len());
                for call in tool_calls {
                    println!("  Function: {}", call.function.name);
                    println!("  Arguments: {}", call.function.arguments);
                }
            } else {
                println!("No tool calls detected in the response");
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}