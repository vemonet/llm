// Import required modules for demonstrating tool choice options
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ToolChoice},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Run examples with each tool choice mode
    println!("=== Testing ToolChoice::Auto (default) ===");
    run_example(
        &api_key, 
        ToolChoice::Auto
    ).await?;

    println!("\n=== Testing ToolChoice::Any (forces tool use) ===");
    run_example(
        &api_key, 
        ToolChoice::Any
    ).await?;

    println!("\n=== Testing ToolChoice::Tool (forces specific tool) ===");
    run_example(
        &api_key, 
        ToolChoice::Tool("get_weather".to_string())
    ).await?;

    Ok(())
}

// Helper function to run an example with the given tool choice mode
async fn run_example(
    api_key: &str, 
    tool_choice: ToolChoice,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build the LLM with the given tool choice and functions
    let builder = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4")
        .max_tokens(1024)
        .temperature(0.7)
        .tool_choice(tool_choice)
        // Add weather function
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a specific location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city and state, e.g. San Francisco, CA"),
                )
                .required(vec!["location".to_string()])
        )
        // Add time function
        .function(
            FunctionBuilder::new("get_current_time")
                .description("Get the current time in a specific time zone")
                .param(
                    ParamBuilder::new("timezone")
                        .type_of("string")
                        .description("The timezone, e.g. EST, PST, UTC, etc."),
                )
                .required(vec!["timezone".to_string()])
        );

    let llm = builder.build()?;

    // Create a query that could use either tool
    let messages = vec![
        ChatMessage::user()
            .content("What's the weather and time in Tokyo right now?")
            .build()
    ];

    // Send the request
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            println!("Response text:\n{}", response);
            
            // Show which tools were called
            match response.tool_calls() {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    println!("\nTools called:");
                    for call in tool_calls {
                        println!("- {}", call.function.name);
                    }
                },
                _ => println!("\nNo tools were called"),
            }
        },
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}