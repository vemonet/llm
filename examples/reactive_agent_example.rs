// Memory sharing example
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    memory::{MessageCondition, SharedMemory, SlidingWindowMemory},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shared_memory = SharedMemory::new_reactive(SlidingWindowMemory::new(10));

    let proposer = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-3.5-turbo")
        .role("assistant")
        .on_message_from_with_trigger("reviewer", MessageCondition::Contains("REJECT".to_string()))
        .system("You are a proposer agent. Answer user questions accurately and concisely. When you receive REJECT from a reviewer, correct your previous response.")
        .memory(shared_memory.clone())
        .build()?;

    let _ = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
        .model("claude-sonnet-4-20250514")
        .role("reviewer")
        .system("You are a reviewer. Your ONLY job is to check if the assistant's answer is correct. Do NOT solve the problem yourself. Just evaluate the assistant's response and reply with ONLY the word ACCEPT (if correct) or REJECT (if wrong). Nothing else.")
        .on_message_from("assistant")
        .max_tokens(512)
        .temperature(0.7)
        .memory(shared_memory.clone())
        .build()?;

    let _ = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-4o")
        .role("resumer")
        .on_message_from_with_trigger("reviewer", MessageCondition::NotContains("REJECT".to_string()))
        .on_message_from_with_trigger("reviewer", MessageCondition::Eq("ACCEPT".to_string()))
        .system("You are a resumer agent. Summarize the conversation between the proposer and the reviewer.")
        .memory(shared_memory.clone())
        .build()?;

    let task = ChatMessage::user()
        .content("how much R in the word strawberry ?")
        .build();
    _ = proposer.chat(&[task]).await;

    let mut receiver = shared_memory.subscribe();
    while let Ok(evt) = receiver.recv().await {
        println!("{} said: {}", evt.role, evt.msg.content);
    }

    Ok(())
}
