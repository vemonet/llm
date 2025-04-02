// OpenAI comprehensive tool calling example with multiple tools and parallel execution
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ToolChoice},
    FunctionCall, ToolCall,
};
use serde_json::{json, Value};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // We'll define tools directly in the builder
    
    // Initialize LLM client with multiple tools
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        // GPT-4 has excellent function calling capabilities
        .model("gpt-4")
        .max_tokens(1024)
        .temperature(0.7)
        // Add all the tools we defined
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get weather information for a specific location and time period")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city and country, e.g., 'Tokyo, Japan'"),
                )
                .param(
                    ParamBuilder::new("time_period")
                        .type_of("string")
                        .description("When you want weather info for, e.g., 'May', 'next week', 'tomorrow'"),
                )
                .required(vec!["location".to_string()])
        )
        .function(
            FunctionBuilder::new("search_flights")
                .description("Search for flight information between two locations")
                .param(
                    ParamBuilder::new("origin")
                        .type_of("string")
                        .description("Departure city or airport code"),
                )
                .param(
                    ParamBuilder::new("destination")
                        .type_of("string")
                        .description("Arrival city or airport code"),
                )
                .param(
                    ParamBuilder::new("date")
                        .type_of("string")
                        .description("Travel date or month, e.g., 'May 15, 2025', 'May 2025'"),
                )
                .required(vec!["origin".to_string(), "destination".to_string()])
        )
        .function(
            FunctionBuilder::new("search_hotels")
                .description("Search for hotels in a specified location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("City, district, or area where you want to stay"),
                )
                .param(
                    ParamBuilder::new("check_in")
                        .type_of("string")
                        .description("Check-in date or month"),
                )
                .param(
                    ParamBuilder::new("price_range")
                        .type_of("string")
                        .description("Optional price range, e.g., 'budget', 'luxury', '$100-$200'"),
                )
                .required(vec!["location".to_string()])
        )
        .function(
            FunctionBuilder::new("convert_currency")
                .description("Get current exchange rates between currencies")
                .param(
                    ParamBuilder::new("from_currency")
                        .type_of("string")
                        .description("The currency to convert from, e.g., 'USD', 'EUR'"),
                )
                .param(
                    ParamBuilder::new("to_currency")
                        .type_of("string")
                        .description("The currency to convert to, e.g., 'JPY', 'GBP'"),
                )
                .required(vec!["from_currency".to_string(), "to_currency".to_string()])
        )
        // Automatically decide whether to use tools
        .tool_choice(ToolChoice::Auto)
        .build()
        .expect("Failed to build LLM");

    println!("=== OpenAI Tool Calling Example ===");
    
    // Create a complex travel planning query that will require multiple tools
    let query = "I'm planning a trip to Tokyo in May. I need to know: \
                 1) What's the weather like then? \
                 2) What are typical flight prices from New York? \
                 3) Can you find a good hotel in Shinjuku? \
                 4) What's the current exchange rate for USD to JPY?";
    
    let messages = vec![
        ChatMessage::user().content(query).build()
    ];

    println!("Sending query about travel planning to Tokyo...");
    
    // Send the request to get tool calls
    let response = llm.chat_with_tools(&messages, llm.tools()).await?;
    println!("Initial response with tool calls:\n{}", response);
    
    // Extract tool calls from the response
    let tool_calls = response.tool_calls();
    
    if let Some(tool_calls) = tool_calls {
        println!("\nTools called: {}", tool_calls.len());
        
        // Process each tool call and generate simulated responses
        let mut tool_results = Vec::new();
        let mut tool_response_map = HashMap::new();
        
        for tool_call in &tool_calls {
            let result = process_tool_call(tool_call)?;
            
            // Store the result for later use in the conversation
            tool_response_map.insert(tool_call.function.name.clone(), result.clone());
            
            // Add to tool results for the next message
            tool_results.push(ToolCall {
                id: tool_call.id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tool_call.function.name.clone(),
                    arguments: serde_json::to_string(&result)?,
                },
            });
            
            println!("Processed tool call: {}", tool_call.function.name);
        }
        
        // Build a follow-up message with the tool results
        let mut follow_up_messages = messages.clone();
        
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
        
        // Now let's get a final response that incorporates all tool results
        println!("\nSending follow-up with tool results...");
        let final_response = llm.chat(&follow_up_messages).await?;
        println!("\nFinal response with integrated tool results:\n{}", final_response);
    } else {
        println!("No tool calls were made in the response.");
    }

    // Example of forcing a specific tool to be used
    println!("\n\n=== Example: Forcing Specific Tool ===");
    let weather_messages = vec![
        ChatMessage::user().content("Tell me about the weather in Paris.").build()
    ];
    
    // Create a new LLM instance that forces the use of the weather tool
    let forced_llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set"))
        .model("gpt-4")
        .max_tokens(1024)
        .temperature(0.7)
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get weather information for a specific location and time period")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city and country")
                )
                .required(vec!["location".to_string()])
        )
        .tool_choice(ToolChoice::Tool("get_weather".to_string()))
        .build()
        .expect("Failed to build forced LLM");
    
    let forced_response = forced_llm.chat_with_tools(&weather_messages, forced_llm.tools()).await?;
    println!("Response with forced weather tool:\n{}", forced_response);

    Ok(())
}

// Placeholder function to keep the original structure but not used in updated code
fn define_travel_tools() {
    // This is just a placeholder - we're now defining the tools inline
}

// Process a tool call and generate a simulated response
fn process_tool_call(tool_call: &ToolCall) -> Result<Value, Box<dyn std::error::Error>> {
    let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
    
    match tool_call.function.name.as_str() {
        "get_weather" => {
            let location = args["location"].as_str().unwrap_or("unknown location");
            let time_period = args["time_period"].as_str().unwrap_or("unknown time");
            
            Ok(json!({
                "location": location,
                "time_period": time_period,
                "conditions": "Mostly sunny with occasional light rain",
                "average_temp_celsius": 22,
                "average_humidity": "65%",
                "precipitation_chance": "30%",
                "note": "May is generally pleasant in Tokyo with mild temperatures"
            }))
        },
        "search_flights" => {
            let origin = args["origin"].as_str().unwrap_or("unknown origin");
            let destination = args["destination"].as_str().unwrap_or("unknown destination");
            
            Ok(json!({
                "origin": origin,
                "destination": destination,
                "options": [
                    {
                        "airline": "Japan Airlines",
                        "flight_number": "JL5",
                        "price_usd": 950,
                        "duration_hours": 14.5,
                        "stopovers": 0
                    },
                    {
                        "airline": "United Airlines",
                        "flight_number": "UA837",
                        "price_usd": 875,
                        "duration_hours": 15.2,
                        "stopovers": 1
                    },
                    {
                        "airline": "ANA",
                        "flight_number": "NH9",
                        "price_usd": 1100,
                        "duration_hours": 13.8,
                        "stopovers": 0
                    }
                ],
                "average_price": "$900-$1100 USD"
            }))
        },
        "search_hotels" => {
            let location = args["location"].as_str().unwrap_or("unknown location");
            let _price_range = args["price_range"].as_str().unwrap_or("moderate");
            
            Ok(json!({
                "location": location,
                "available_hotels": [
                    {
                        "name": "Hotel Gracery Shinjuku",
                        "stars": 4,
                        "price_per_night": "$180 USD",
                        "highlights": "Close to train station, Godzilla statue on roof, modern rooms"
                    },
                    {
                        "name": "Shinjuku Granbell Hotel",
                        "stars": 3.5,
                        "price_per_night": "$150 USD",
                        "highlights": "Artistic design, rooftop bar, walking distance to nightlife"
                    },
                    {
                        "name": "Hyatt Regency Tokyo",
                        "stars": 4.5,
                        "price_per_night": "$250 USD",
                        "highlights": "Luxury property, multiple restaurants, excellent service"
                    }
                ],
                "recommended": "Hotel Gracery Shinjuku offers good value and an excellent location"
            }))
        },
        "convert_currency" => {
            let from = args["from_currency"].as_str().unwrap_or("USD");
            let to = args["to_currency"].as_str().unwrap_or("JPY");
            
            Ok(json!({
                "from_currency": from,
                "to_currency": to,
                "exchange_rate": 153.24,
                "example_conversion": "100 USD = 15,324 JPY",
                "last_updated": "2025-04-01"
            }))
        },
        _ => Ok(json!({
            "error": "Unknown function",
            "function_name": tool_call.function.name
        }))
    }
}