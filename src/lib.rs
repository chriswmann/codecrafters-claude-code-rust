use std::fmt;
use std::path::Path;

use async_openai::types::chat::FinishReason;
use serde::{Deserialize, Serialize};

#[allow(clippy::doc_markdown, clippy::doc_link_with_quotes)]
/// Model a data response like this:
/// {
///  "choices": [
///    {
///      "index": 0,
///      "message": {
///        "role": "assistant",
///        "content": null,
///        "tool_calls": [
///          {
///            "id": "call_abc123",
///            "type": "function",
///            "function": {
///              "name": "Read",
///              "arguments": "{\"file_path\": \"/path/to/file.txt\"}"
///            }
///          }
///        ]
///      },
///      "finish_reason": "tool_calls"
///    }
///  ]
///}

#[derive(Serialize, Deserialize)]
pub struct Response {
    pub(crate) choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize)]
pub struct Choice {
    index: u32,
    message: Message,
    finish_reason: FinishReason,
}

#[derive(Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Serialize, Deserialize)]
pub struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: Function,
}

#[derive(Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

/// # Errors
///
/// Will return `Err` if `path` does not exist or the user does not have permissions
/// to read it.
pub fn read_file_to_string(path: impl AsRef<Path>) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

#[derive(Debug)]
pub enum AppError {
    ArgumentError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            AppError::ArgumentError(msg) => write!(f, "ArgumentError: {msg}"),
        }
    }
}

impl std::error::Error for AppError {}
