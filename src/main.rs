use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionTool, CreateChatCompletionRequestArgs,
    FunctionObjectArgs,
};
use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::path::Path;
use std::{env, process};

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
    let model = "anthropic/claude-haiku-4.5";
    let user_prompt = args.prompt;

    let mut request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(128_u32)
        .model(model)
        .messages(ChatCompletionRequestUserMessage::from(user_prompt.clone()))
        .tools(ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name("Read")
                .description("Read and return the contents of a file")
                .parameters(json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    }
                }))
                .strict(true)
                .build()?,
        })
        .build()?;

    loop {
        let response_message = client
            .chat()
            .create(request.clone())
            .await?
            .choices
            .first()
            .ok_or("No choices")?
            .message
            .clone();

        if let Some(tool_calls) = response_message.tool_calls {
            let mut function_responses = Vec::new();
            for tool_call_enum in tool_calls {
                // Extract the function tool call from the enum
                if let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call_enum {
                    let name = tool_call.function.name.clone();
                    eprintln!("Calling {name} function.");
                    let args = tool_call.function.arguments.clone();
                    let args: Value = serde_json::from_str(&args).unwrap();
                    let file_path = args["file_path"].as_str().unwrap();
                    let file_contents = read_file_to_string(file_path);
                    function_responses.push((tool_call.clone(), file_contents));
                }
            }
            // Assemble the messages ready to pass to the request on the next iteration
            // let mut messages: Vec<ChatCompletionRequestMessage> =
            //     ChatCompletionRequestUserMessage::from(user_prompt.clone()).into();
            // Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCalls enum
            let tool_calls: Vec<ChatCompletionMessageToolCalls> = function_responses
                .iter()
                .map(|(tool_call, _response_content)| {
                    ChatCompletionMessageToolCalls::from(tool_call.clone())
                })
                .collect();
            let assistant_messages: ChatCompletionRequestMessage =
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls)
                    .build()?
                    .into();

            let tool_messages: Vec<ChatCompletionRequestMessage> = function_responses
                .iter()
                .map(|(tool_call, response_content)| {
                    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                        content: response_content.to_string().into(),
                        tool_call_id: tool_call.id.clone(),
                    })
                })
                .collect();

            request.messages.push(assistant_messages);
            request.messages.extend(tool_messages);
        } else {
            if let Some(message) = response_message.content {
                println!("{message}");
            }
            break;
        }
    }
    Ok(())
}

/// # Errors
///
/// Will return `Err` if `path` does not exist or the user does not have permissions
/// to read it.
pub fn read_file_to_string(path: impl AsRef<Path>) -> Value {
    let file_contents = std::fs::read_to_string(&path).unwrap_or_default();
    eprintln!("Read file: {}", path.as_ref().display());
    eprintln!("File contents: {}", file_contents);
    json!({
        "content": file_contents
    })
}
