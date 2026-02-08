use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process};

use codecrafters_claude_code::{AppError, ToolCall, read_file_to_string};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
            "model": "anthropic/claude-haiku-4.5",
            "tools": [
                {
                    "type": "function",
                    "function": {
                      "name": "Read",
                      "description": "Read and return the contents of a file",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                          }
                        },
                        "required": ["file_path"]
                      }
                    }
                }

        ]
        }))
        .await?;

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
        // Only print if the content is not empt A
        if !content.trim().is_empty() {
            println!("{content}");
        }
    }

    if let Some(tool_calls_array) = response["choices"][0]["message"]["tool_calls"].as_array() {
        for tool_call in tool_calls_array {
            let tool_call: ToolCall = serde_json::from_value(tool_call.clone())?;
            let function_arguments = tool_call.function.arguments;
            let args: Value = serde_json::from_str(&function_arguments)?;
            let file_path = args["file_path"]
                .as_str()
                .ok_or(AppError::ArgumentError("Missing 'file_path'".to_string()))?;
            let file_contents = read_file_to_string(file_path)?;
            println!("{file_contents}");
        }
    }
    Ok(())
}
