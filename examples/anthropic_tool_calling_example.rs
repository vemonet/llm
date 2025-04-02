// Import required modules for Anthropic tool calling functionality
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::ChatMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Anthropic API key from environment variable
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY environment variable not set");

    // Initialize and configure the LLM client with function tools
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(api_key)
        // Claude 3 Opus/Sonnet/Haiku supports tool use
        .model("claude-3-opus-20240229")
        .max_tokens(1024)
        .temperature(0.7)
        .system("You are a helpful assistant that can access external tools when needed.")
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a specific location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city and state, e.g. San Francisco, CA"),
                )
                .required(vec!["location".to_string()]),
        )
        .function(
            FunctionBuilder::new("get_current_time")
                .description("Get the current time in a specific time zone")
                .param(
                    ParamBuilder::new("timezone")
                        .type_of("string")
                        .description("The timezone, e.g. EST, PST, UTC, etc."),
                )
                .required(vec!["timezone".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    // Prepare conversation with a query that should trigger tool use
    let messages = vec![
        ChatMessage::user()
            .content("What's the current weather and time in Tokyo?")
            .build()
    ];

    println!("Sending query to Claude about weather and time in Tokyo...");
    
    // Send chat request with tools
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            println!("Claude response:\n{}", response);
            
            // Display tool calls if any were made
            if let Some(tool_calls) = response.tool_calls() {
                println!("\nTool calls made by Claude:");
                for (i, call) in tool_calls.iter().enumerate() {
                    println!("Tool call #{}:", i+1);
                    println!("  Function: {}", call.function.name);
                    println!("  Arguments: {}", call.function.arguments);
                }
            }
        },
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}