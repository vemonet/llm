// Import required modules for advanced tool calling functionality
use llm::{
    backends::openai::OpenAI,
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ChatProvider, ChatResponse, Tool, ToolChoice},
    FunctionCall, LLMProvider, ToolCall,
};
use serde_json::{json, Value};
use std::io::{self, Write};

// Function to process a tool call and generate simulated response
fn process_tool_call(tool_call: &ToolCall) -> Result<Value, Box<dyn std::error::Error>> {
    match tool_call.function.name.as_str() {
        "get_weather" => {
            println!(
                "Processing weather tool call: {}",
                tool_call.function.arguments
            );

            // Parse the arguments to extract the location
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let location = args["location"].as_str().unwrap_or("unknown location");

            // Simulate different weather based on location
            let (temp, forecast) = match location.to_lowercase().as_str() {
                loc if loc.contains("tokyo") => (22, "Sunny with some clouds"),
                loc if loc.contains("london") => (14, "Rainy and foggy"),
                loc if loc.contains("new york") => (18, "Partly cloudy"),
                loc if loc.contains("sydney") => (25, "Clear skies"),
                _ => (20, "Fair weather"),
            };

            Ok(json!({
                "location": location,
                "temperature": temp,
                "units": "celsius",
                "forecast": forecast,
                "humidity": 65
            }))
        }
        "get_current_time" => {
            println!(
                "Processing time tool call: {}",
                tool_call.function.arguments
            );

            // Parse the arguments to extract the timezone
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let timezone = args["timezone"].as_str().unwrap_or("UTC");

            // Simulate different times based on timezone
            let (time, date) = match timezone.to_uppercase().as_str() {
                "JST" => ("15:30", "2025-04-01"),
                "GMT" => ("06:30", "2025-04-01"),
                "EST" => ("01:30", "2025-04-01"),
                "PST" => ("22:30", "2025-03-31"),
                _ => ("12:00", "2025-04-01"),
            };

            Ok(json!({
                "timezone": timezone,
                "current_time": time,
                "date": date
            }))
        }
        "search_restaurants" => {
            println!(
                "Processing restaurant search: {}",
                tool_call.function.arguments
            );

            // Parse the arguments
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let location = args["location"].as_str().unwrap_or("unknown");
            let cuisine = args["cuisine"].as_str().unwrap_or("any");

            // Simulated restaurant data based on location and cuisine
            let restaurants = match (
                location.to_lowercase().as_str(),
                cuisine.to_lowercase().as_str(),
            ) {
                (loc, cui) if loc.contains("tokyo") && cui.contains("sushi") => vec![
                    json!({"name": "Sukiyabashi Jiro", "rating": 4.9, "price": "$$$$", "specialty": "Omakase"}),
                    json!({"name": "Sushi Saito", "rating": 4.8, "price": "$$$$", "specialty": "Seasonal fish"}),
                    json!({"name": "Tsukiji Fish Market Sushi", "rating": 4.5, "price": "$$", "specialty": "Fresh market sushi"}),
                ],
                (loc, cui) if loc.contains("tokyo") && cui.contains("ramen") => vec![
                    json!({"name": "Ichiran Ramen", "rating": 4.7, "price": "$$", "specialty": "Tonkotsu ramen"}),
                    json!({"name": "Afuri", "rating": 4.6, "price": "$$", "specialty": "Yuzu ramen"}),
                ],
                (loc, _) if loc.contains("tokyo") => vec![
                    json!({"name": "Robot Restaurant", "rating": 4.3, "price": "$$$", "specialty": "Entertainment dining"}),
                    json!({"name": "Gonpachi", "rating": 4.1, "price": "$$$", "specialty": "Traditional Japanese"}),
                ],
                _ => vec![
                    json!({"name": "Local Restaurant", "rating": 4.0, "price": "$$", "specialty": "Local cuisine"}),
                ],
            };

            Ok(json!({
                "location": location,
                "cuisine": cuisine,
                "results": restaurants
            }))
        }
        "convert_currency" => {
            println!(
                "Processing currency conversion: {}",
                tool_call.function.arguments
            );

            // Parse the arguments
            let args: Value = serde_json::from_str(&tool_call.function.arguments)?;
            let from = args["from_currency"].as_str().unwrap_or("USD");
            let to = args["to_currency"].as_str().unwrap_or("JPY");
            let amount = args["amount"].as_f64().unwrap_or(1.0);

            // Simulate exchange rates
            let rate = match (from.to_uppercase().as_str(), to.to_uppercase().as_str()) {
                ("USD", "JPY") => 153.24,
                ("JPY", "USD") => 0.0065,
                ("USD", "EUR") => 0.92,
                ("EUR", "USD") => 1.09,
                ("EUR", "JPY") => 166.02,
                ("JPY", "EUR") => 0.006,
                _ => 1.0,
            };

            let converted = amount * rate;

            Ok(json!({
                "from_currency": from,
                "to_currency": to,
                "amount": amount,
                "converted_amount": converted,
                "exchange_rate": rate,
                "last_updated": "2025-04-01"
            }))
        }
        _ => Ok(json!({
            "error": "Unknown function",
            "function_name": tool_call.function.name
        })),
    }
}

// Function to handle a single turn of conversation with tool execution
async fn handle_turn(
    llm: &Box<dyn LLMProvider>,
    messages: &mut Vec<ChatMessage>,
    user_query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Add user's message to conversation
    messages.push(ChatMessage::user().content(user_query).build());

    println!("\n--- Sending user query to model ---");
    println!("User: {}", user_query);

    // Get model's response, which may include tool calls
    let response = llm.chat_with_tools(messages, llm.tools()).await?;

    // Check if model wants to use tools
    if let Some(tool_calls) = response.tool_calls() {
        println!("\n--- Model is using {} tools ---", tool_calls.len());

        // Add the assistant's tool use to the conversation
        messages.push(
            ChatMessage::assistant()
                .tool_use(tool_calls.clone())
                .content("")
                .build(),
        );

        // Process each tool call and collect results
        let mut tool_results = Vec::new();
        for tool_call in &tool_calls {
            let result = process_tool_call(tool_call)?;

            tool_results.push(ToolCall {
                id: tool_call.id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: tool_call.function.name.clone(),
                    arguments: result.to_string(),
                },
            });
        }

        // Add tool results to the conversation
        messages.push(
            ChatMessage::user()
                .tool_result(tool_results)
                .content("")
                .build(),
        );

        // Get final response after tool use
        let final_response = llm.chat(messages).await?;

        // Add the assistant's final response to the conversation
        messages.push(
            ChatMessage::assistant()
                .content(final_response.text().unwrap_or_default())
                .build(),
        );

        println!("\n--- Assistant response after using tools ---");
        println!("Assistant: {}", final_response.text().unwrap_or_default());
    } else {
        // No tools used, just add the direct response
        let response_text = response.text().unwrap_or_default();

        messages.push(
            ChatMessage::assistant()
                .content(response_text.clone())
                .build(),
        );

        println!("\n--- Assistant direct response ---");
        println!("Assistant: {}", response_text);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    // Define tools for the model to use
    let weather_tool = FunctionBuilder::new("get_weather")
        .description("Get the current weather in a specific location")
        .param(
            ParamBuilder::new("location")
                .type_of("string")
                .description("The city and state/country, e.g. San Francisco, CA or Tokyo, Japan"),
        )
        .required(vec!["location".to_string()]);

    let time_tool = FunctionBuilder::new("get_current_time")
        .description("Get the current time in a specific time zone")
        .param(
            ParamBuilder::new("timezone")
                .type_of("string")
                .description("The timezone, e.g. EST, PST, UTC, JST, etc."),
        )
        .required(vec!["timezone".to_string()]);

    let restaurant_tool = FunctionBuilder::new("search_restaurants")
        .description("Search for restaurants in a specific location")
        .param(
            ParamBuilder::new("location")
                .type_of("string")
                .description("The city or neighborhood to search in"),
        )
        .param(
            ParamBuilder::new("cuisine")
                .type_of("string")
                .description("Type of cuisine, e.g. Italian, Japanese, etc."),
        )
        .required(vec!["location".to_string()]);

    let currency_tool = FunctionBuilder::new("convert_currency")
        .description("Convert an amount from one currency to another")
        .param(
            ParamBuilder::new("from_currency")
                .type_of("string")
                .description("Source currency code (e.g., USD, EUR, JPY)"),
        )
        .param(
            ParamBuilder::new("to_currency")
                .type_of("string")
                .description("Target currency code (e.g., USD, EUR, JPY)"),
        )
        .param(
            ParamBuilder::new("amount")
                .type_of("number")
                .description("Amount to convert"),
        )
        .required(vec!["from_currency".to_string(), "to_currency".to_string()]);

    // Import the LLMBuilder again
    use llm::builder::LLMBuilder;

    // Initialize OpenAI client with the builder pattern
    let llm = LLMBuilder::new()
        .backend(llm::builder::LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4")
        .max_tokens(1024)
        .temperature(0.7)
        .function(weather_tool)
        .function(time_tool)
        .function(restaurant_tool)
        .function(currency_tool)
        .tool_choice(ToolChoice::Auto)
        .build()
        .expect("Failed to build LLM");

    println!("=== Multi-Turn Tool Calling Example ===");
    println!("This example demonstrates maintaining conversation context across multiple turns");
    println!("with tool use and tool results included in the conversation history.\n");

    // Initialize conversation history
    let mut conversation = Vec::new();

    // First turn: Ask about weather and time
    handle_turn(
        &llm,
        &mut conversation,
        "I'm planning a trip to Tokyo. What's the weather like there right now? And what time is it?"
    ).await?;

    // Second turn: Ask about restaurants, building on the previous context
    handle_turn(
        &llm,
        &mut conversation,
        "Can you recommend some good sushi restaurants in Tokyo?",
    )
    .await?;

    // Third turn: Ask about currency conversion, building on the entire conversation
    handle_turn(
        &llm,
        &mut conversation,
        "If I have 100 USD, how much is that in Japanese Yen?",
    )
    .await?;

    // Fourth turn: Ask a follow-up question that requires memory of the entire conversation
    handle_turn(
        &llm,
        &mut conversation,
        "Based on everything we've discussed, what would be a good time to visit the restaurants you mentioned?"
    ).await?;

    println!("\n=== Interactive Mode ===");
    println!("You can now chat with the assistant. Type 'exit' to quit.");

    // Interactive mode for user to continue the conversation
    loop {
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;

        let user_input = user_input.trim();
        if user_input.to_lowercase() == "exit" {
            break;
        }

        handle_turn(&llm, &mut conversation, user_input).await?;
    }

    Ok(())
}
