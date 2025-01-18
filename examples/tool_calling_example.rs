use std::collections::HashMap;

// Import required modules from the LLM library for OpenAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole, FunctionTool, ParameterProperty, ParametersSchema, Tool}, // Chat-related structures
};

fn main() {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gpt-3.5-turbo") // Use GPT-3.5 Turbo model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (OpenAI)");

 

    let tool = Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: "weather_function".to_string(),
            description: "Use this tool to get the weather in a specific city".to_string(),
            parameters: ParametersSchema {
                schema_type: "object".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "url".to_string(),
                        ParameterProperty {
                            property_type: "string".to_string(),
                            description: "The url to get the weather from for the city".to_string(),
                            items: None,
                            enum_list: None,
                        },
                    );
                    props
                },

                required: vec!["url".to_string()],
            },
        },
    };

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available".into(),
    }];

    // Send chat request and handle the response
    // this returns the response as a string. The tool call is also returned as a serialized string. We can deserialize if needed.
    match llm.chat_with_tools(&messages, Some(&[tool])) {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }
}
