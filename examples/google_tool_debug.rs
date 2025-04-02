// Detailed debugging example for Google tool calling
use llm::{
    backends::google::Google,
    chat::{ChatMessage, ChatProvider, Tool, FunctionTool, ParametersSchema},
    LLMProvider, // Add the trait import
};
use serde_json::json;
use std::{env, collections::HashMap};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable
    let api_key = env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set");

    // Define a test function for the weather using the public API
    let weather_function = Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: "get_weather".to_string(),
            description: "Get the current weather in a given location".to_string(),
            parameters: ParametersSchema {
                schema_type: "object".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "location".to_string(),
                        llm::chat::ParameterProperty {
                            property_type: "string".to_string(),
                            description: "The city and state, e.g. San Francisco, CA".to_string(),
                            items: None,
                            enum_list: None,
                        },
                    );
                    props
                },
                required: vec!["location".to_string()],
            },
        },
    };

    // Create the Google client directly to have more control
    let client = Google::new(
        api_key,
        Some("gemini-1.5-flash".to_string()),
        Some(512),  // max_tokens
        Some(0.7),  // temperature
        Some(30),   // timeout_seconds
        None,       // system
        None,       // stream
        None,       // top_p
        None,       // top_k
        None,       // json_schema
        Some(vec![weather_function]),
    );

    // Create a chat message that should trigger the function call
    let messages = vec![
        ChatMessage::user()
            .content("What's the weather like in Miami right now?")
            .build()
    ];

    println!("Sending chat request with function tools...");
    
    // Get the tools from the client using the LLMProvider trait method
    let tools = client.tools();
    
    // Make the API call
    match client.chat_with_tools(&messages, tools).await {
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