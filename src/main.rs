use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestUserMessage, ChatCompletionTool,
    ChatCompletionTools, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    FunctionObjectArgs,
};
use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::io::Write;
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

    let read_tool = ChatCompletionTool {
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
    };

    let write_tool = ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name("Write")
            .description("Write contents to a file")
            .parameters(json!({
                      "type": "object",
            "required": ["file_path", "content"],
            "properties": {
              "file_path": {
                "type": "string",
                "description": "The path of the file to write to"
              },
              "content": {
                "type": "string",
                "description": "The content to write to the file"
              }
            }
                  }))
            .strict(true)
            .build()?,
    };
    let tools = vec![
        ChatCompletionTools::Function(read_tool),
        ChatCompletionTools::Function(write_tool),
    ];
    let mut request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(128_u32)
        .model(model)
        .messages(ChatCompletionRequestUserMessage::from(user_prompt.clone()))
        .tools(tools)
        .build()?;

    loop {
        let response = client.chat().create(request.clone()).await?;
        let response_message = response.choices.first().ok_or("No choices")?;

        if let Some(ref tool_calls) = response_message.message.tool_calls {
            let mut function_responses = Vec::new();
            for tool_call_enum in tool_calls {
                // Extract the function tool call from the enum.
                // At the moment we only have the Read tool.
                if let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call_enum {
                    let name = tool_call.function.name.clone();
                    eprintln!("Calling {name} function.");
                    let args = tool_call.function.arguments.clone();
                    let args: Value =
                        serde_json::from_str(&args).expect("Should have some arguments");
                    eprintln!("{args:?}");
                    let file_path = args["file_path"]
                        .as_str()
                        .expect("Should have a `file_path` argument.");
                    match name.as_str() {
                        "Read" => {
                            let file_contents = read_file_to_string(file_path)?;
                            eprintln!("file contents: {file_contents}");
                            let file_contents = json!(&file_contents);
                            function_responses.push((tool_call, file_contents));
                        }
                        "Write" => {
                            let content = args["content"]
                                .as_str()
                                .expect("Should have a `content` argument");
                            let new_file_contents = write_to_file(file_path, content)?;
                            let new_file_value = json!(new_file_contents.as_str());
                            function_responses.push((tool_call, new_file_value));
                        }
                        _ => unimplemented!("We only have read and write tools at this stage."),
                    }
                }
            }
            append_tool_responses(&mut request, &function_responses)?;
        } else if let Some(message) = &response_message.message.content {
            println!("{message}");
            break;
        }
    }
    Ok(())
}

fn append_tool_responses(
    request: &mut CreateChatCompletionRequest,
    function_responses: &[(&ChatCompletionMessageToolCall, Value)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCalls enum
    let tool_calls: Vec<ChatCompletionMessageToolCalls> = function_responses
        .iter()
        .map(|(tool_call, _response_content)| {
            ChatCompletionMessageToolCalls::from((*tool_call).clone())
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
    Ok(())
}

/// # Errors
///
/// Will return `Err` if `path` does not exist or the user does not have permissions
/// to read it.
fn read_file_to_string(path: impl AsRef<Path>) -> Result<String, Box<dyn std::error::Error>> {
    let file_contents = std::fs::read_to_string(&path).unwrap_or_default();
    eprintln!("Read file: {}", path.as_ref().display());
    eprintln!("File contents: {file_contents}");
    Ok(file_contents)
}

fn write_to_file(
    path: impl AsRef<Path>,
    contents: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(contents.as_bytes())?;
    Ok(contents.into())
}
