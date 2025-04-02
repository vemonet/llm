// Import required modules from the LLM library for Google integration
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ParameterProperty}, // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable
    let api_key =
        std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set");

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google) // Use Google as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gemini-1.5-flash") // Use Gemini model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .function(
            FunctionBuilder::new("schedule_meeting")
                .description(
                    "Schedules a meeting with specified attendees at a given time and date.",
                )
                .param(
                    ParamBuilder::new("attendees")
                        .type_of("string")
                        .description("Attendee names"),
                )
                .param(
                    ParamBuilder::new("date")
                        .type_of("string")
                        .description("Date of the meeting (e.g., '2024-07-29')"),
                )
                .param(
                    ParamBuilder::new("time")
                        .type_of("string")
                        .description("Time of the meeting (e.g., '15:00')"),
                )
                .param(
                    ParamBuilder::new("topic")
                        .type_of("string")
                        .description("The subject or topic of the meeting."),
                )
                .required(vec![
                    "attendees".to_string(),
                    "date".to_string(),
                    "time".to_string(),
                    "topic".to_string(),
                ]),
        )
        .build()
        .expect("Failed to build LLM");

    // Prepare conversation history with example messages - make it explicitly match the provided tool
    let messages = vec![
        ChatMessage::user()
            .content("Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM about the Q3 planning.")
            .build()
    ];

    // Send chat request and handle the response
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            println!("Chat response:\n{}", response);

            // Check if any tools were called
            if let Some(tool_calls) = response.tool_calls() {
                println!("\nTool calls found:");
                for call in tool_calls {
                    println!("- Function: {}", call.function.name);
                    println!("  Arguments: {}", call.function.arguments);
                }
            } else {
                println!("\nNo tool calls were made.");
            }
        }
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}
