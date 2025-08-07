use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::ChatMessage,
};
use futures::StreamExt;

// Backend configuration struct to hold backend-specific settings
#[derive(Debug, Clone)]
struct BackendConfig {
    backend: LLMBackend,
    env_key: &'static str,
    model: &'static str,
    backend_name: &'static str,
}

// Define all backend configurations
fn get_backend_configs() -> Vec<BackendConfig> {
    vec![
        BackendConfig {
            backend: LLMBackend::OpenAI,
            env_key: "OPENAI_API_KEY",
            model: "gpt-4o-mini",
            backend_name: "openai",
        },
        BackendConfig {
            backend: LLMBackend::Mistral,
            env_key: "MISTRAL_API_KEY",
            model: "mistral-small-latest",
            backend_name: "mistral",
        },
        BackendConfig {
            backend: LLMBackend::Google,
            env_key: "GOOGLE_API_KEY",
            model: "gemini-2.5-flash-lite",
            backend_name: "google",
        },
        BackendConfig {
            backend: LLMBackend::Groq,
            env_key: "GROQ_API_KEY",
            model: "llama3-8b-8192",
            backend_name: "groq",
        },
    ]
}

// Generic test function for chat functionality
async fn test_chat_generic(config: &BackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {

            eprintln!(
                "test test_{}_chat ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return Ok(());
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat(&messages).await {
        Ok(response) => {
            assert!(
                response.text().is_some() && !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
            let usage = response.usage();
            assert!(usage.is_some(), "Expected usage information to be present");
            let usage = usage.unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => {
            eprintln!("Chat error for {}: {e}", config.backend_name);
            return Err(e.into());
        }
    }
    Ok(())
}

// Generic test function for streaming chat functionality
async fn test_chat_stream_struct_generic(config: &BackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream_struct ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return Ok(());
        }
    };

    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(0.7)
        .stream(true)
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user().content("Hello.").build()];

    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            let mut usage_data = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                complete_text.push_str(content);
                            }
                        }
                        if let Some(usage) = stream_response.usage {
                            usage_data = Some(usage);
                        }
                    }
                    Err(e) => {
                        eprintln!("Stream error for {}: {e}", config.backend_name);
                        return Err(e.into());
                    }
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text"
            );
            // if let Some(usage) = usage_data {
            //     assert!(
            //         usage.prompt_tokens > 0,
            //         "Expected prompt tokens > 0, got {}",
            //         usage.prompt_tokens
            //     );
            //     assert!(
            //         usage.completion_tokens > 0,
            //         "Expected completion tokens > 0, got {}",
            //         usage.completion_tokens
            //     );
            //     assert!(
            //         usage.total_tokens > 0,
            //         "Expected total tokens > 0, got {}",
            //         usage.total_tokens
            //     );
            // } else {
            //     // Some backends might not provide usage in streaming mode
            //     println!("Warning: No usage data provided for {}", config.backend_name);
            // }
        }
        Err(e) => {
            eprintln!("Stream error for {}: {e}", config.backend_name);
            return Err(e.into());
        }
    }
    Ok(())
}

async fn test_chat_stream(config: &BackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream_struct ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return Ok(());
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(0.7)
        .stream(true)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(content) => {
                        complete_text.push_str(&content);
                    }
                    Err(e) => {
                        eprintln!("Stream error: {e}");
                        return Err(e.into());
                    }
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text for {}",
                config.backend_name
            );
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

// Generic test function for chat with tools functionality
async fn test_chat_with_tools_generic(config: &BackendConfig) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_with_tools ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return Ok(());
        }
    };

    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user()
        .content("You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available")
        .build()];

    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            let tool_calls = response.tool_calls();
            assert!(tool_calls.is_some(), "Expected tool calls to be present");
            let tool_calls = tool_calls.unwrap();
            assert_eq!(
                tool_calls.len(),
                1,
                "Expected exactly 1 tool call, got {}",
                tool_calls.len()
            );
            assert_eq!(
                tool_calls[0].function.name, "weather_function",
                "Expected function name 'weather_function'"
            );
            let usage = response.usage();
            assert!(usage.is_some(), "Expected usage information to be present");
            let usage = usage.unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => {
            eprintln!("Chat with tools error for {}: {e}", config.backend_name);
            return Err(e.into());
        }
    }
    Ok(())
}

// Macro to generate individual test functions for each backend
macro_rules! generate_backend_tests {
    ($($backend_name:ident),* $(,)?) => {
        $(
            paste::paste! {
                #[tokio::test]
                async fn [<test_ $backend_name _chat>]() -> Result<(), Box<dyn std::error::Error>> {
                    let configs = get_backend_configs();
                    let config = configs.iter()
                        .find(|c| c.backend_name == stringify!($backend_name))
                        .expect(&format!("Backend config not found for {}", stringify!($backend_name)));
                    test_chat_generic(config).await
                }

                #[tokio::test]
                async fn [<test_ $backend_name _chat_stream_struct>]() -> Result<(), Box<dyn std::error::Error>> {
                    let configs = get_backend_configs();
                    let config = configs.iter()
                        .find(|c| c.backend_name == stringify!($backend_name))
                        .expect(&format!("Backend config not found for {}", stringify!($backend_name)));
                    test_chat_stream_struct_generic(config).await
                }

                #[tokio::test]
                async fn [<test_ $backend_name _chat_with_tools>]() -> Result<(), Box<dyn std::error::Error>> {
                    let configs = get_backend_configs();
                    let config = configs.iter()
                        .find(|c| c.backend_name == stringify!($backend_name))
                        .expect(&format!("Backend config not found for {}", stringify!($backend_name)));
                    test_chat_with_tools_generic(config).await
                }

                #[tokio::test]
                async fn [<test_ $backend_name _chat_stream>]() -> Result<(), Box<dyn std::error::Error>> {
                    let configs = get_backend_configs();
                    let config = configs.iter()
                        .find(|c| c.backend_name == stringify!($backend_name))
                        .expect(&format!("Backend config not found for {}", stringify!($backend_name)));
                    test_chat_stream(config).await
                }
            }
        )*
    };
}

// Generate tests for each backend
generate_backend_tests!(openai, mistral, google, groq);
