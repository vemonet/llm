// Detailed debugging example for Google tool calling
use llm::{
    backends::google::Google,
    chat::{ChatMessage, ChatProvider, Tool, FunctionTool, ParametersSchema, MessageType},
    LLMProvider, FunctionCall, ToolCall,
};
use serde_json::{json, Value};
use std::{env, collections::HashMap};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable
    let api_key = env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set");

    // Define theater search function
    let theaters_function = Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: "find_theaters".to_string(),
            description: "Find theaters based on location and optionally movie title which are currently playing in theaters".to_string(),
            parameters: ParametersSchema {
                schema_type: "object".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "location".to_string(),
                        llm::chat::ParameterProperty {
                            property_type: "string".to_string(),
                            description: "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616".to_string(),
                            items: None,
                            enum_list: None,
                        },
                    );
                    props.insert(
                        "movie".to_string(),
                        llm::chat::ParameterProperty {
                            property_type: "string".to_string(),
                            description: "Any movie title".to_string(),
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
        Some("gemini-1.5-pro".to_string()),
        Some(512),  // max_tokens
        Some(0.7),  // temperature
        Some(30),   // timeout_seconds
        None,       // system
        None,       // stream
        None,       // top_p
        None,       // top_k
        None,       // json_schema
        Some(vec![theaters_function]),
    );

    // Create a chat message that should trigger the function call
    let messages = vec![
        ChatMessage::user()
            .content("Which theaters in Mountain View show Barbie movie?")
            .build()
    ];

    println!("Sending chat request with function tools...");
    
    // Get the tools from the client using the LLMProvider trait method
    let tools = client.tools();
    
    // Make the API call
    let response = client.chat_with_tools(&messages, tools).await?;
    println!("Success! Response text: {}", response.text().unwrap_or_default());
    
    let tool_calls = match response.tool_calls() {
        Some(calls) => {
            println!("Tool calls ({}):", calls.len());
            for call in &calls {
                println!("  Function: {}", call.function.name);
                println!("  Arguments: {}", call.function.arguments);
            }
            calls
        },
        None => {
            println!("No tool calls detected in the response");
            return Ok(());
        }
    };

    println!("\nNow passing the tool results back...");

    // Process the tool call and generate simulated responses
    let mut tool_results = Vec::new();
    
    for tool_call in &tool_calls {
        // Generate mock result for the theater search
        let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
        let location = args["location"].as_str().unwrap_or("Mountain View, CA");
        let movie = args["movie"].as_str().unwrap_or("Barbie");
        
        let result = json!({
            "movie": movie,
            "theaters": [
                {
                    "name": "AMC Mountain View 16",
                    "address": "2000 W El Camino Real, Mountain View, CA 94040"
                },
                {
                    "name": "Regal Edwards 14",
                    "address": "245 Castro St, Mountain View, CA 94040"
                }
            ]
        });
        
        // Add to tool results for the next message
        let result_string = serde_json::to_string(&result)?;
        tool_results.push(ToolCall {
            id: tool_call.id.clone(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: tool_call.function.name.clone(),
                arguments: result_string,
            },
        });
    }
    
    // Build a follow-up message with the tool results
    let mut follow_up_messages = messages.clone();
    
    // Add the assistant's message with tool calls
    follow_up_messages.push(
        ChatMessage::assistant()
            .tool_use(tool_calls.clone())
            .content("")
            .build(),
    );
    
    // Add tool results as the user message
    follow_up_messages.push(
        ChatMessage::user()
            .tool_result(tool_results)
            .content("")
            .build(),
    );
    
    // Now get the final response incorporating the tool results
    let final_response = client.chat_with_tools(&follow_up_messages, tools).await?;
    
    println!("\nFinal response with tool results incorporated:");
    println!("{}", final_response.text().unwrap_or_default());

    Ok(())
}