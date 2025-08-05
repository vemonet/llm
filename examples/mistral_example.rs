use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("MISTRAL_API_KEY").unwrap_or("your-mistral-api-key".into());
    // Configure Mistral LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model("mistral-small-latest")         // default model
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .build()
        .expect("Failed to build LLM (Mistral)");
    // Prepare conversation
    let messages = vec![
        ChatMessage::user().content("Hello, what is Mistral AI?").build(),
        ChatMessage::assistant().content("Mistral AI is ...").build(),
        ChatMessage::user().content("Does it support function calling?").build(),
    ];
    // Send chat request
    match llm.chat(&messages).await {
        Ok(response) => {
            println!("Chat response:\n{response}");
            println!("Chat response:\n{:?}", response.usage())
        },
        Err(e) => eprintln!("Chat error: {e}"),
    }
    Ok(())
}
