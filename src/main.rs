use anyhow::{Context, Result, bail};
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
async fn main() -> Result<()> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY")?;

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);
    let model = "anthropic/claude-haiku-4.5";

    let read_tool_parameters = json!(include_str!("tool_definitions/read_tool_params.json"));
    let read_tool = tool_definition_factory(
        "Read",
        "Read and return the contents of a file. Takes a `file_path` argument.",
        read_tool_parameters,
    )?;

    let write_tool_parameters = json!(include_str!("tool_definitions/write_tool_params.json"));
    let write_tool = tool_definition_factory(
        "Write",
        "Write contents to a file. Takes `file_path` and `content` arguments.",
        write_tool_parameters,
    )?;

    let bash_tool_parameters = json!(include_str!("tool_definitions/bash_tool_params.json"));
    let bash_tool = tool_definition_factory(
        "Bash",
        "Execute a shell command. Takes a `command` argument.",
        bash_tool_parameters,
    )?;
    let tools = vec![
        ChatCompletionTools::Function(read_tool),
        ChatCompletionTools::Function(write_tool),
        ChatCompletionTools::Function(bash_tool),
    ];
    let user_prompt = args.prompt;
    let mut request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(128_u32)
        .model(model)
        .messages(ChatCompletionRequestUserMessage::from(user_prompt.clone()))
        .tools(tools)
        .build()?;

    loop {
        let response = client.chat().create(request.clone()).await?;
        let response_message = response.choices.first().context("No choices")?;

        if let Some(ref tool_calls) = response_message.message.tool_calls {
            let mut function_responses = Vec::new();
            for tool_call_enum in tool_calls {
                // Extract the function tool call from the enum.
                if let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call_enum {
                    let name = tool_call.function.name.as_str();
                    eprintln!("Calling {name} function.");
                    let args = tool_call.function.arguments.as_str();
                    let args: Value = serde_json::from_str(&args)?;
                    eprintln!("{args:?}");
                    match name {
                        "Read" => call_read_tool(&tool_call, &args, &mut function_responses)?,
                        "Write" => call_write_tool(&tool_call, &args, &mut function_responses)?,
                        "Bash" => call_bash_tool(&tool_call, &args, &mut function_responses)?,
                        _ => {
                            let err_msg = format!("Unknown tool: {name}");
                            function_responses.push((&tool_call, json!(err_msg)));
                        }
                    }
                }
            }
            append_tool_responses(&mut request, &function_responses)?;
        } else if let Some(message) = &response_message.message.content {
            println!("{message}");
            break;
        } else {
            bail!("Response had neither tool calls nor content");
        }
    }
    Ok(())
}

fn append_tool_responses(
    request: &mut CreateChatCompletionRequest,
    function_responses: &[(&ChatCompletionMessageToolCall, Value)],
) -> Result<()> {
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

fn call_read_tool<'tool_call>(
    tool_call: &'tool_call ChatCompletionMessageToolCall,
    args: &Value,
    function_responses: &mut Vec<(&'tool_call ChatCompletionMessageToolCall, Value)>,
) -> Result<()> {
    let file_path = args["file_path"].as_str().context(format!(
        "Should have a `file_path` argument. Args were: {args:#?}"
    ))?;
    let file_contents = read_file_to_string(file_path)?;
    eprintln!("file contents: {file_contents}");
    let file_contents = json!(&file_contents);
    function_responses.push((tool_call, file_contents));
    Ok(())
}

fn call_write_tool<'tool_call>(
    tool_call: &'tool_call ChatCompletionMessageToolCall,
    args: &Value,
    function_responses: &mut Vec<(&'tool_call ChatCompletionMessageToolCall, Value)>,
) -> Result<()> {
    let file_path = args["file_path"]
        .as_str()
        .context("Should have a `file_path` argument.")?;
    let content = args["content"].as_str().context(format!(
        "Should have a `content` argument. Args were: {args:#?}"
    ))?;
    write_to_file(file_path, content)?;
    let new_file_value = json!(content);
    function_responses.push((tool_call, new_file_value));
    Ok(())
}

fn call_bash_tool<'tool_call>(
    tool_call: &'tool_call ChatCompletionMessageToolCall,
    args: &Value,
    function_responses: &mut Vec<(&'tool_call ChatCompletionMessageToolCall, Value)>,
) -> Result<()> {
    let command = args["command"].as_str().context(format!(
        "Should have a `command` argument. Args were: {args:#?}"
    ))?;
    let output = execute_bash_command(command)?;
    function_responses.push((tool_call, json!(output)));
    Ok(())
}

fn read_file_to_string(path: impl AsRef<Path>) -> Result<String> {
    let file_contents = std::fs::read_to_string(&path)?;
    eprintln!("Read file: {}", path.as_ref().display());
    Ok(file_contents)
}

fn write_to_file(path: impl AsRef<Path>, contents: &str) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

fn execute_bash_command(command: &str) -> Result<String> {
    let output = process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(format!("{stdout} {stderr}"))
}

fn tool_definition_factory(
    name: &str,
    description: &str,
    parameters: Value,
) -> Result<ChatCompletionTool> {
    let chat_completion_tool = ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name(name)
            .description(description)
            .parameters(parameters)
            .strict(true)
            .build()?,
    };
    Ok(chat_completion_tool)
}
