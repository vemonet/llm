// Debugging tool for API tool calling across different providers
use llm::{
    builder::{LLMBuilder, LLMBackend, FunctionBuilder, ParamBuilder},
    chat::{ChatMessage, ToolChoice},
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check command-line arguments for backend type
    let args: Vec<String> = std::env::args().collect();
    let backend_type = if args.len() > 1 {
        match args[1].to_lowercase().as_str() {
            "openai" => LLMBackend::OpenAI,
            "google" => LLMBackend::Google,
            "anthropic" => LLMBackend::Anthropic,
            "ollama" => LLMBackend::Ollama,
            _ => {
                eprintln!("Unknown backend: {}. Using OpenAI as default.", args[1]);
                LLMBackend::OpenAI
            }
        }
    } else {
        println!("No backend specified, using OpenAI. Run with 'openai', 'google', 'anthropic', or 'ollama'.");
        LLMBackend::OpenAI
    };

    // Get API key based on backend
    let api_key = match &backend_type {
        LLMBackend::OpenAI => env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
        LLMBackend::Google => env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY not set"),
        LLMBackend::Anthropic => env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
        LLMBackend::Ollama => String::new(), // No API key needed for Ollama
        _ => panic!("Unsupported backend"),
    };

    let model = match &backend_type {
        LLMBackend::OpenAI => "gpt-3.5-turbo",
        LLMBackend::Google => "gemini-1.5-flash",
        LLMBackend::Anthropic => "claude-3-opus-20240229",
        LLMBackend::Ollama => "llama3", // Or whichever model you have in Ollama
        _ => "unknown",
    };

    println!("Using backend: {:?} with model: {}", backend_type, model);

    // Create a builder with common settings
    let mut builder = LLMBuilder::new()
        .backend(backend_type.clone())
        .model(model)
        .max_tokens(512)
        .temperature(0.7);

    // Add API key if needed
    if !api_key.is_empty() {
        builder = builder.api_key(api_key);
    }

    // Add Ollama base URL if needed
    if matches!(backend_type, LLMBackend::Ollama) {
        builder = builder.base_url("http://localhost:11434");
    }

    // Add a simple weather function that should work across providers
    builder = builder.function(
        FunctionBuilder::new("get_weather")
            .description("Get the current weather in a given location")
            .param(
                ParamBuilder::new("location")
                    .type_of("string")
                    .description("The city and state, e.g. San Francisco, CA")
            )
            .required(vec!["location".to_string()])
    );

    // Add a tool choice to auto
    builder = builder.tool_choice(ToolChoice::Auto);

    // Build the LLM
    let llm = builder.build()?;

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
            println!("\nSuccess! Response text: {}", response.text().unwrap_or_default());
            
            if let Some(tool_calls) = response.tool_calls() {
                println!("\nTool calls ({}):", tool_calls.len());
                for call in tool_calls {
                    println!("  Function: {}", call.function.name);
                    println!("  Arguments: {}", call.function.arguments);
                }
            } else {
                println!("\nNo tool calls detected in the response");
            }
        }
        Err(e) => {
            eprintln!("\nError: {}", e);
        }
    }

    Ok(())
}