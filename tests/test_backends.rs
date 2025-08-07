use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::ChatMessage,
};
use futures::StreamExt;
use rstest::rstest;

#[derive(Debug, Clone)]
struct BackendTestConfig {
    backend: LLMBackend,
    env_key: &'static str,
    model: &'static str,
    backend_name: &'static str,
}

const BACKEND_CONFIGS: &[BackendTestConfig] = &[
    BackendTestConfig {
        backend: LLMBackend::OpenAI,
        env_key: "OPENAI_API_KEY",
        model: "gpt-4o-mini",
        backend_name: "openai",
    },
    BackendTestConfig {
        backend: LLMBackend::Mistral,
        env_key: "MISTRAL_API_KEY",
        model: "mistral-small-latest",
        backend_name: "mistral",
    },
    BackendTestConfig {
        backend: LLMBackend::Google,
        env_key: "GOOGLE_API_KEY",
        model: "gemini-2.5-flash-lite",
        backend_name: "google",
    },
    BackendTestConfig {
        backend: LLMBackend::Groq,
        env_key: "GROQ_API_KEY",
        model: "llama3-8b-8192",
        backend_name: "groq",
    },
    BackendTestConfig {
        backend: LLMBackend::Cohere,
        env_key: "COHERE_API_KEY",
        model: "command-r7b-12-2024",
        backend_name: "cohere",
    },
    // BackendTestConfig {
    //     backend: LLMBackend::Anthropic,
    //     env_key: "ANTHROPIC_API_KEY",
    //     model: "claude-3-5-haiku-20241022",
    //     backend_name: "anthropic",
    // },
];

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
// #[case::anthropic(&BACKEND_CONFIGS[5])]
#[tokio::test]
async fn test_chat(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
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
            panic!("Chat error for {}: {e}", config.backend_name);
        }
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
// #[case::anthropic(&BACKEND_CONFIGS[5])]
#[tokio::test]
async fn test_chat_with_tools(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_with_tools ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
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
            panic!("Chat with tools error for {}: {e}", config.backend_name);
        }
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[tokio::test]
async fn test_chat_stream_struct(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream_struct ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
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
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                complete_text.push_str(content);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("Stream error for {}: {e}", config.backend_name);
                    }
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text"
            );
        }
        Err(e) => {
            panic!("Stream error for {}: {e}", config.backend_name);
        }
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
// #[case::anthropic(&BACKEND_CONFIGS[5])]
#[tokio::test]
async fn test_chat_stream(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
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
                        panic!("Stream error: {e}");
                    }
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text for {}",
                config.backend_name
            );
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[tokio::test]
async fn test_embedding(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_embedding ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    // Use embedding-specific models for each backend
    let embedding_model = match config.backend {
        LLMBackend::OpenAI => "text-embedding-3-small",
        LLMBackend::Mistral => "mistral-embed",
        LLMBackend::Cohere => "embed-english-v3.0",
        _ => config.model,
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(embedding_model)
        .build()
        .expect("Failed to build LLM");
    let input_texts = vec!["Test sentence for embedding generation".to_string()];
    match llm.embed(input_texts).await {
        Ok(embeddings) => {
            assert!(
                !embeddings.is_empty(),
                "Expected at least one embedding, got empty vector for {}",
                config.backend_name
            );
            let first_embedding = &embeddings[0];
            assert!(
                !first_embedding.is_empty(),
                "Expected non-empty embedding vector for {}",
                config.backend_name
            );
            // Check that embeddings have reasonable dimensions
            let embedding_dim = first_embedding.len();
            assert!(
                embedding_dim > 100,
                "Expected embedding dimension > 100, got {embedding_dim} for {}",
                config.backend_name
            );
            // Verify the embedding values are not all zeros
            let non_zero_count = first_embedding.iter().filter(|&&x| x != 0.0).count();
            assert!(
                non_zero_count > 0,
                "Expected some non-zero embedding values for {}",
                config.backend_name
            );
        }
        Err(e) => {
            panic!("Embedding error for {}: {e}", config.backend_name);
        }
    }
}
