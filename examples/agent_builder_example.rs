//! Portage de `reactive_agent_example.rs` vers le nouveau `AgentBuilder`.
//!
//! Trois agents coopèrent via une mémoire partagée réactive :
//! 1. proposer (assistant) – répond aux questions de l’utilisateur.
//! 2. reviewer – juge la réponse (ACCEPT / REJECT).
//! 3. resumer  – résume la discussion quand la réponse est acceptée.

use llm::{
    agent::AgentBuilder,
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    memory::{MessageCondition, SharedMemory, SlidingWindowMemory},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Mémoire partagée (réactive).
    let shared_memory = SharedMemory::new_reactive(SlidingWindowMemory::new(10));

    // Proposer – corrige sa réponse quand le reviewer dit REJECT.
    let proposer = AgentBuilder::new()
        .role("assistant")
        .on_message_from_with_trigger("reviewer", MessageCondition::Contains("REJECT".to_string()))
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("gpt-3.5-turbo")
                .system("You are a proposer agent. Answer user questions accurately and concisely. When you receive REJECT from a reviewer, correct your previous response.")
        )
        .stt(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("whisper-1")
        )
        .tts(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("tts-1")
                .voice("alloy")
        )
        .memory(shared_memory.clone())
        .build()?;

    // Reviewer – répond uniquement ACCEPT / REJECT.
    let _reviewer = AgentBuilder::new()
        .role("reviewer")
        .on_message_from("assistant")
        .debounce(800)
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("o3")
                .system("Respond with a single word: ACCEPT (if the assistant is correct) or REJECT (if wrong). No explanation..")
        )
        .memory(shared_memory.clone())
        .build()?;

    // Resumer – résume la conversation quand ACCEPT et pas REJECT.
    let _resumer = AgentBuilder::new()
        .role("resumer")
        // Le résumé se déclenche quand le reviewer envoie ACCEPT.
        .on_message_from_with_trigger("reviewer", MessageCondition::Contains("ACCEPT".to_string()))
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("gpt-4o")
                .system("You are a resumer agent. Summarize the conversation between the proposer and the reviewer."),
        )
        .memory(shared_memory.clone())
        .build()?;

    // L'utilisateur pose sa question.
    let task = ChatMessage::user()
        .content("how much R in the word strawberry ?")
        .build();
    _ = proposer.chat(&[task]).await;

    // Observer la conversation.
    let mut receiver = shared_memory.subscribe();
    while let Ok(evt) = receiver.recv().await {
        println!("{} said: {}", evt.role, evt.msg.content);
    }

    Ok(())
}
