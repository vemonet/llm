//! MCP (Model Context Protocol) agent implementation.
//!
//! This module provides an MCP agent that can connect to MCP servers,
//! discover available tools, and use them with LLM providers through the tool calling interface.

use crate::{
    builder::LLMBuilder,
    chat::{ChatMessage, ChatResponse, FunctionTool, Tool, Usage},
    error::LLMError,
    LLMProvider, ToolCall,
};
use rmcp::{
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    service::RunningService,
    transport::StreamableHttpClientTransport,
    RoleClient, ServiceExt as _,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// MCP transport type for different connection methods
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// HTTP/WebSocket transport to a URL
    Http(String),
    // Note: Stdio transport removed due to rmcp API changes
}

/// Builder for creating MCP agents with a fluent interface
pub struct McpAgentBuilder {
    transport: Option<McpTransport>,
    llm_builder: LLMBuilder,
    client_name: String,
    client_version: String,
}

impl Default for McpAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl McpAgentBuilder {
    /// Create a new MCP agent builder
    pub fn new() -> Self {
        Self {
            transport: None,
            llm_builder: LLMBuilder::new(),
            client_name: "LLM MCP Agent".to_string(),
            client_version: "1.0.0".to_string(),
        }
    }

    /// Set the MCP server URL for HTTP transport
    pub fn mcp_url<S: Into<String>>(mut self, url: S) -> Self {
        self.transport = Some(McpTransport::Http(url.into()));
        self
    }

    /// Set the MCP server command for stdio transport
    pub fn mcp_cmd<S: Into<String>>(self, _command: S) -> Self {
        // let cmd = command.into();
        // let parts: Vec<String> = cmd.split_whitespace().map(|s| s.to_string()).collect();
        // if parts.is_empty() {
        //     self.transport = Some(McpTransport::Stdio(String::new(), Vec::new()));
        // } else {
        //     let command = parts[0].clone();
        //     let args = parts[1..].to_vec();
        //     self.transport = Some(McpTransport::Stdio(command, args));
        // }
        self
    }

    /// Set the MCP server command with explicit arguments for stdio transport
    pub fn mcp_cmd_with_args<S: Into<String>>(self, _command: S, _args: Vec<String>) -> Self {
        // self.transport = Some(McpTransport::Stdio(command.into(), args));
        self
    }

    /// Set the underlying LLM configuration using LLMBuilder
    pub fn llm(mut self, llm_builder: LLMBuilder) -> Self {
        self.llm_builder = llm_builder;
        self
    }

    /// Set the client name for MCP identification
    pub fn client_name<S: Into<String>>(mut self, name: S) -> Self {
        self.client_name = name.into();
        self
    }

    /// Set the client version for MCP identification
    pub fn client_version<S: Into<String>>(mut self, version: S) -> Self {
        self.client_version = version.into();
        self
    }

    /// Build the MCP agent
    pub async fn build(self) -> Result<McpAgent, LLMError> {
        let transport = self.transport.ok_or_else(|| {
            LLMError::Generic("MCP transport not specified. Use mcp_url() or mcp_cmd()".to_string())
        })?;

        McpAgent::new(
            self.llm_builder,
            transport,
            self.client_name,
            self.client_version,
        )
        .await
    }
}

/// Configuration for the MCP agent (for backwards compatibility)
#[derive(Debug, Clone)]
pub struct McpAgentConfig {
    /// The MCP server URI to connect to
    pub server_uri: String,
    /// Client name for identification
    pub client_name: String,
    /// Client version for identification
    pub client_version: String,
}

impl Default for McpAgentConfig {
    fn default() -> Self {
        Self {
            server_uri: "http://localhost:8000/mcp".to_string(),
            client_name: "LLM MCP Agent".to_string(),
            client_version: "1.0.0".to_string(),
        }
    }
}

/// Information about a tool execution in the MCP agent
#[derive(Debug, Clone)]
pub struct McpToolExecution {
    /// The original tool call request
    pub tool_call: ToolCall,
    /// The result of the tool execution
    pub result: String,
}

/// Response from MCP agent's chat_with_tools that includes conversation history and tool executions
#[derive(Debug)]
pub struct McpChatResponse {
    /// The final conversation messages including tool calls and results
    pub messages: Vec<ChatMessage>,
    /// Information about tool calls that were executed
    pub tool_executions: Vec<McpToolExecution>,
    /// The final response from the LLM
    pub final_response: Box<dyn ChatResponse>,
}

impl ChatResponse for McpChatResponse {
    fn text(&self) -> Option<String> {
        self.final_response.text()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.final_response.tool_calls()
    }

    fn thinking(&self) -> Option<String> {
        self.final_response.thinking()
    }

    fn usage(&self) -> Option<Usage> {
        self.final_response.usage()
    }
}

impl std::fmt::Display for McpChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display tool executions first
        for tool_execution in &self.tool_executions {
            writeln!(
                f,
                "Tool executed: {} -> {}",
                tool_execution.tool_call.function.name, tool_execution.result
            )?;
        }
        // Then display the final response
        write!(f, "{}", self.final_response)
    }
}

/// An agent that uses MCP (Model Context Protocol) to discover and execute tools
/// in conjunction with an LLM provider.
pub struct McpAgent {
    /// The underlying LLM provider
    llm: Box<dyn LLMProvider>,
    /// MCP client connection
    pub mcp_client:
        rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>,
    /// Client identification
    client_name: String,
    client_version: String,
    /// Cached tools from the MCP server
    cached_tools: Arc<RwLock<Option<Vec<Tool>>>>,
}

pub const ADDRESS: &str = "0.0.0.0:8000";

impl McpAgent {
    /// Create a new MCP agent with specified transport
    async fn new(
        llm_builder: LLMBuilder,
        transport: McpTransport,
        client_name: String,
        client_version: String,
    ) -> Result<Self, LLMError> {
        let McpTransport::Http(transport_uri) = transport;
        let transport = StreamableHttpClientTransport::from_uri(transport_uri);
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: client_name.clone(),
                version: client_version.clone(),
            },
        };
        let mcp_client = client_info
            .serve(transport)
            .await
            .map_err(|e| LLMError::Generic(format!("MCP client initialization failed: {e}")))?;

        // Build the LLM from the builder
        let llm = llm_builder.build()?;
        Ok(Self {
            llm,
            mcp_client,
            client_name,
            client_version,
            cached_tools: Arc::new(RwLock::new(None)),
        })
    }

    /// Get available tools from the MCP server
    pub async fn get_mcp_tools(&self) -> Result<Vec<Tool>, LLMError> {
        // Check if we have cached tools
        {
            let cached = self.cached_tools.read().await;
            if let Some(ref tools) = *cached {
                return Ok(tools.clone());
            }
        }
        // Fetch tools from MCP server
        let tools_response = self
            .mcp_client
            .list_tools(Default::default())
            .await
            .map_err(|e| LLMError::Generic(format!("Failed to list MCP tools: {e}")))?;
        let mut tools = Vec::new();
        for tool in &tools_response.tools {
            // Convert MCP tool to our Tool format
            let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());

            // Create a simple function tool structure
            let function_tool = FunctionTool {
                name: tool.name.to_string(),
                description: tool.description.as_deref().unwrap_or("").to_string(),
                parameters: schema_value,
            };
            let converted_tool = Tool {
                tool_type: "function".to_string(),
                function: function_tool,
            };
            tools.push(converted_tool);
        }
        // Cache the tools
        {
            let mut cached = self.cached_tools.write().await;
            *cached = Some(tools.clone());
        }
        Ok(tools)
    }

    /// Execute a tool call through the MCP server
    async fn execute_mcp_tool(&self, tool_call: &ToolCall) -> Result<String, LLMError> {
        // Execute the tool using the client
        let arguments =
            match serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) {
                Ok(value) => value.as_object().cloned(),
                Err(_) => None,
            };
        let tool_results = self
            .mcp_client
            .call_tool(CallToolRequestParam {
                name: tool_call.function.name.clone().into(),
                arguments,
            })
            .await
            .map_err(|e| LLMError::InvalidRequest(format!("Tool execution failed: {e}")))?;

        // Extract text content from the tool result
        let tool_result_text = tool_results
            .content
            .iter()
            .filter_map(|annotated| match &annotated.raw {
                rmcp::model::RawContent::Text(text_content) => Some(text_content.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        Ok(tool_result_text)
    }

    /// Process a conversation with MCP tool execution
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    ///
    /// # Returns
    ///
    /// The response containing the final conversation, tool executions, and LLM response
    pub async fn chat_with_mcp_tools(
        &self,
        messages: &[ChatMessage],
    ) -> Result<McpChatResponse, LLMError> {
        // Get available tools from MCP server
        let mcp_tools = self.get_mcp_tools().await?;
        if mcp_tools.is_empty() {
            // No tools available, just do regular chat
            let response = self.llm.chat(messages).await?;
            return Ok(McpChatResponse {
                messages: messages.to_vec(),
                tool_executions: vec![],
                final_response: response,
            });
        }

        // First, ask the LLM if it needs to use any tools
        let response = self.llm.chat_with_tools(messages, Some(&mcp_tools)).await?;
        // Check if the LLM wants to use tools
        if let Some(tool_calls) = response.tool_calls() {
            let mut conversation = messages.to_vec();
            let mut tool_executions = Vec::new();
            // Add the assistant's response with tool calls
            conversation.push(
                ChatMessage::assistant()
                    .tool_use(tool_calls.clone())
                    .content(response.text().unwrap_or_default())
                    .build(),
            );
            // Execute each tool call through MCP
            let mut tool_results = Vec::new();
            for tool_call in &tool_calls {
                let tool_result = self.execute_mcp_tool(tool_call).await?;
                // Store the tool execution information
                tool_executions.push(McpToolExecution {
                    tool_call: tool_call.clone(),
                    result: tool_result.clone(),
                });
                // Create a tool result for the conversation
                tool_results.push(ToolCall {
                    id: tool_call.id.clone(),
                    call_type: tool_call.call_type.clone(),
                    function: crate::FunctionCall {
                        name: tool_call.function.name.clone(),
                        arguments: tool_result,
                    },
                });
            }
            // Add tool results to conversation
            conversation.push(
                ChatMessage::user()
                    .tool_result(tool_results)
                    .content("")
                    .build(),
            );

            // Get final response from LLM with tool results
            let final_response = self
                .llm
                .chat_with_tools(&conversation, Some(&mcp_tools))
                .await?;
            Ok(McpChatResponse {
                messages: conversation,
                tool_executions,
                final_response,
            })
        } else {
            // No tools needed, return the response
            Ok(McpChatResponse {
                messages: messages.to_vec(),
                tool_executions: vec![],
                final_response: response,
            })
        }
    }

    /// Get the underlying LLM provider
    pub fn llm(&self) -> &dyn LLMProvider {
        &*self.llm
    }

    /// Get the client name
    pub fn client_name(&self) -> &str {
        &self.client_name
    }

    /// Get the client version
    pub fn client_version(&self) -> &str {
        &self.client_version
    }

    /// Clear the cached tools, forcing a refresh on next use
    pub async fn refresh_tools(&self) {
        let mut cached = self.cached_tools.write().await;
        *cached = None;
    }
}
