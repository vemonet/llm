// Import required modules for advanced tool calling functionality
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ToolChoice},
    FunctionCall, ToolCall,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Define multiple tools for the model to use
    let weather_tool = FunctionBuilder::new("get_weather")
        .description("Get the current weather in a specific location")
        .param(
            ParamBuilder::new("location")
                .type_of("string")
                .description("The city and state, e.g. San Francisco, CA"),
        )
        .required(vec!["location".to_string()]);

    let time_tool = FunctionBuilder::new("get_current_time")
        .description("Get the current time in a specific time zone")
        .param(
            ParamBuilder::new("timezone")
                .type_of("string")
                .description("The timezone, e.g. EST, PST, UTC, etc."),
        )
        .required(vec!["timezone".to_string()]);

    // Initialize and configure the LLM client with multiple tools
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4") // Using a more capable model for tool calling
        .max_tokens(1024)
        .temperature(0.7)
        .function(weather_tool) // Add the weather tool
        .function(time_tool)   // Add the time tool
        // Force the model to use tools by setting toolChoice to "any"
        .tool_choice(ToolChoice::Any)
        .build()
        .expect("Failed to build LLM");

    // First interaction: Ask about weather and time
    let initial_message = vec![
        ChatMessage::user()
            .content("What's the weather in Tokyo right now? And what time is it there?")
            .build()
    ];

    println!("Sending initial query about weather and time in Tokyo...");
    
    // Send the request and get the response with tool calls
    let response = llm.chat_with_tools(&initial_message, llm.tools()).await?;
    println!("Model response with tool calls:\n{}", response);

    // Extract tool calls from the response
    let tool_calls = response.tool_calls().unwrap_or_default();
    println!("\nExtracted {} tool calls", tool_calls.len());

    // Prepare simulated tool responses
    let mut tool_results = Vec::new();
    for tool_call in &tool_calls {
        // In a real application, you would actually call external APIs here
        match tool_call.function.name.as_str() {
            "get_weather" => {
                println!("Processing weather tool call with args: {}", tool_call.function.arguments);
                
                // Parse the arguments to extract the location
                let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;
                let location = args["location"].as_str().unwrap_or("Tokyo");
                
                // Create a simulated weather report
                let weather_data = json!({
                    "location": location,
                    "temperature": 22,
                    "units": "celsius",
                    "forecast": "Sunny with some clouds",
                    "humidity": 65
                });
                
                // Add the simulated tool result
                tool_results.push(ToolCall {
                    id: tool_call.id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: weather_data.to_string(),
                    },
                });
            },
            "get_current_time" => {
                println!("Processing time tool call with args: {}", tool_call.function.arguments);
                
                // Parse the arguments to extract the timezone
                let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;
                let timezone = args["timezone"].as_str().unwrap_or("JST");
                
                // Create a simulated time response
                let time_data = json!({
                    "timezone": timezone,
                    "current_time": "15:30",
                    "date": "2025-04-01"
                });
                
                // Add the simulated tool result
                tool_results.push(ToolCall {
                    id: tool_call.id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_current_time".to_string(),
                        arguments: time_data.to_string(),
                    },
                });
            },
            _ => {
                println!("Unknown tool call: {}", tool_call.function.name);
            }
        }
    }

    // Build a follow-up message with the tool results
    let mut follow_up_messages = initial_message.clone();
    
    // Add the assistant's response with tool usage
    follow_up_messages.push(
        ChatMessage::assistant()
            .tool_use(tool_calls)
            .content("")
            .build()
    );
    
    // Add the tool results
    follow_up_messages.push(
        ChatMessage::user()
            .tool_result(tool_results)
            .content("")
            .build()
    );
    
    // Demonstrate using tool_choice to specify a particular tool
    println!("\nSending follow-up query with tool results...");
    
    // Now let's call chat without tool_choice to get a response based on the tool results
    let final_response = llm.chat(&follow_up_messages).await?;
    println!("Final response with tool results incorporated:\n{}", final_response);

    Ok(())
}