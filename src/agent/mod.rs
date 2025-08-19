//! Agent module for building reactive, memory-enabled LLM agents.

#[cfg(feature = "agent")]
pub mod builder;

#[cfg(feature = "agent")]
pub mod mcp;

#[cfg(feature = "agent")]
pub use builder::AgentBuilder;

#[cfg(feature = "agent")]
pub use mcp::{McpAgent, McpAgentBuilder, McpAgentConfig};
