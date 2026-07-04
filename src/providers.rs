//! Provider-neutral message types and LLM API clients.
//!
//! Request building and response parsing are pure functions over JSON values
//! so they can be tested without a network; `complete` is a thin HTTP shim.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    /// A tool result, keyed to the assistant's tool call by `tool_call_id`.
    Tool,
}

/// Token accounting for one completion, as reported by the provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// One turn in the conversation, in provider-neutral form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Present on assistant messages parsed from an API response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            is_error: false,
            usage: None,
        }
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
            is_error: false,
            usage: None,
        }
    }

    pub fn tool_result(id: impl Into<String>, output: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::Tool,
            content: output.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(id.into()),
            is_error,
            usage: None,
        }
    }
}

#[derive(Debug)]
pub struct ProviderError {
    pub message: String,
    /// Worth retrying: rate limits, overload, network failures. Bad requests
    /// and auth failures are not.
    pub retryable: bool,
}

impl ProviderError {
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }

    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provider error: {}", self.message)
    }
}

impl std::error::Error for ProviderError {}

/// A chat-completion backend: takes the conversation, returns the next
/// assistant message (text and/or tool calls).
pub trait Provider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
    ) -> Result<Message, ProviderError>;

    /// Like `complete`, but emits text chunks through `on_text` as they
    /// arrive. The default falls back to one chunk after the full response —
    /// mocks and future providers work unchanged.
    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let message = self.complete(system, messages, tools)?;
        if !message.content.is_empty() {
            on_text(&message.content);
        }
        Ok(message)
    }
}

fn send(
    url: &str,
    headers: &[(String, String)],
    body: Value,
) -> Result<ureq::Response, ProviderError> {
    let mut request = ureq::post(url);
    for (name, value) in headers {
        request = request.set(name, value);
    }
    match request.send_json(body) {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            let message = format!("HTTP {code}: {detail}");
            if matches!(code, 408 | 429 | 500..=599) {
                Err(ProviderError::transient(message))
            } else {
                Err(ProviderError::fatal(message))
            }
        }
        // Transport-level failures (DNS, refused, reset) are worth retrying.
        Err(e) => Err(ProviderError::transient(e.to_string())),
    }
}

fn post(url: &str, headers: &[(String, String)], body: Value) -> Result<Value, ProviderError> {
    send(url, headers, body)?
        .into_json()
        .map_err(|e| ProviderError::fatal(format!("invalid JSON response: {e}")))
}

fn parse_usage(body: &Value, input_key: &str, output_key: &str) -> Option<Usage> {
    let usage = body.get("usage")?;
    Some(Usage {
        input_tokens: usage[input_key].as_u64().unwrap_or(0),
        output_tokens: usage[output_key].as_u64().unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// Anthropic Messages API
// ---------------------------------------------------------------------------

pub struct AnthropicProvider {
    pub api_key: String,
    /// OAuth/gateway bearer token; used instead of the API key when set.
    pub auth_token: Option<String>,
    pub model: String,
    pub base_url: String,
    pub max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            auth_token: None,
            model: model.into(),
            base_url: "https://api.anthropic.com".into(),
            max_tokens: 8192,
        }
    }

    /// Auth + version headers for every request.
    pub fn headers(&self) -> Vec<(String, String)> {
        let auth = match &self.auth_token {
            Some(token) => ("Authorization".into(), format!("Bearer {token}")),
            None => ("x-api-key".into(), self.api_key.clone()),
        };
        vec![auth, ("anthropic-version".into(), "2023-06-01".into())]
    }
}

/// Convert neutral messages to Anthropic content-block form, merging
/// consecutive same-role entries (parallel tool results must share one
/// user message, and roles must alternate).
pub fn anthropic_build_request(
    model: &str,
    max_tokens: u32,
    system: &str,
    messages: &[Message],
    tools: &[Value],
) -> Value {
    let mut out: Vec<Value> = Vec::new();
    for msg in messages {
        let (role, blocks) = match msg.role {
            Role::User => ("user", vec![json!({"type": "text", "text": msg.content})]),
            Role::Assistant => {
                let mut blocks = Vec::new();
                if !msg.content.is_empty() {
                    blocks.push(json!({"type": "text", "text": msg.content}));
                }
                for call in &msg.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.args,
                    }));
                }
                ("assistant", blocks)
            }
            Role::Tool => (
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                    "is_error": msg.is_error,
                })],
            ),
        };
        match out.last_mut() {
            Some(last) if last["role"] == role => {
                let content = last["content"].as_array_mut().unwrap();
                content.extend(blocks);
            }
            _ => out.push(json!({"role": role, "content": blocks})),
        }
    }
    // System and tools are stable across a session; cache_control markers
    // let the API reuse them instead of re-processing every request.
    let mut tools = tools.to_vec();
    if let Some(last) = tools.last_mut() {
        last["cache_control"] = json!({"type": "ephemeral"});
    }
    json!({
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": out,
        "tools": tools,
    })
}

pub fn anthropic_parse_response(body: &Value) -> Result<Message, ProviderError> {
    let blocks = body["content"]
        .as_array()
        .ok_or_else(|| ProviderError::fatal(format!("unexpected response shape: {body}")))?;
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => content.push_str(block["text"].as_str().unwrap_or_default()),
            Some("tool_use") => tool_calls.push(ToolCall {
                id: block["id"].as_str().unwrap_or_default().to_string(),
                name: block["name"].as_str().unwrap_or_default().to_string(),
                args: block["input"].clone(),
            }),
            _ => {}
        }
    }
    let mut message = Message::assistant(content, tool_calls);
    message.usage = parse_usage(body, "input_tokens", "output_tokens");
    Ok(message)
}

impl Provider for AnthropicProvider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
    ) -> Result<Message, ProviderError> {
        let body = anthropic_build_request(&self.model, self.max_tokens, system, messages, tools);
        let response = post(
            &format!("{}/v1/messages", self.base_url),
            &self.headers(),
            body,
        )?;
        anthropic_parse_response(&response)
    }

    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let mut body =
            anthropic_build_request(&self.model, self.max_tokens, system, messages, tools);
        body["stream"] = Value::Bool(true);
        let response = send(
            &format!("{}/v1/messages", self.base_url),
            &self.headers(),
            body,
        )?;
        let mut acc = crate::streaming::AnthropicAccumulator::default();
        crate::streaming::each_sse_data(std::io::BufReader::new(response.into_reader()), |data| {
            let event: Value = serde_json::from_str(data)
                .map_err(|e| ProviderError::fatal(format!("invalid stream event: {e}")))?;
            if event["type"] == "error" {
                return Err(ProviderError::transient(
                    event["error"]["message"].to_string(),
                ));
            }
            acc.handle(&event, &mut *on_text);
            Ok(())
        })?;
        Ok(acc.finish())
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible Chat Completions API (OpenAI, Ollama, vLLM, …)
// ---------------------------------------------------------------------------

pub struct OpenAIProvider {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".into(),
        }
    }
}

pub fn openai_build_request(
    model: &str,
    system: &str,
    messages: &[Message],
    tools: &[Value],
) -> Value {
    let mut out = vec![json!({"role": "system", "content": system})];
    for msg in messages {
        out.push(match msg.role {
            Role::User => json!({"role": "user", "content": msg.content}),
            Role::Assistant => {
                let mut entry = json!({"role": "assistant", "content": msg.content});
                if !msg.tool_calls.is_empty() {
                    entry["tool_calls"] = msg
                        .tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.args.to_string(),
                                },
                            })
                        })
                        .collect();
                }
                entry
            }
            Role::Tool => json!({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            }),
        });
    }
    let functions: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            })
        })
        .collect();
    let mut body = json!({"model": model, "messages": out});
    if !functions.is_empty() {
        body["tools"] = json!(functions);
    }
    body
}

pub fn openai_parse_response(body: &Value) -> Result<Message, ProviderError> {
    let message = &body["choices"][0]["message"];
    if message.is_null() {
        return Err(ProviderError::fatal(format!(
            "unexpected response shape: {body}"
        )));
    }
    let content = message["content"].as_str().unwrap_or_default().to_string();
    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    let arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
                    ToolCall {
                        id: call["id"].as_str().unwrap_or_default().to_string(),
                        name: call["function"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        args: serde_json::from_str(arguments).unwrap_or(json!({})),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let mut message = Message::assistant(content, tool_calls);
    message.usage = parse_usage(body, "prompt_tokens", "completion_tokens");
    Ok(message)
}

impl Provider for OpenAIProvider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
    ) -> Result<Message, ProviderError> {
        let body = openai_build_request(&self.model, system, messages, tools);
        let response = post(
            &format!("{}/chat/completions", self.base_url),
            &[("Authorization".into(), format!("Bearer {}", self.api_key))],
            body,
        )?;
        openai_parse_response(&response)
    }

    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Value],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let mut body = openai_build_request(&self.model, system, messages, tools);
        body["stream"] = Value::Bool(true);
        body["stream_options"] = json!({"include_usage": true});
        let response = send(
            &format!("{}/chat/completions", self.base_url),
            &[("Authorization".into(), format!("Bearer {}", self.api_key))],
            body,
        )?;
        let mut acc = crate::streaming::OpenAIAccumulator::default();
        crate::streaming::each_sse_data(std::io::BufReader::new(response.into_reader()), |data| {
            let chunk: Value = serde_json::from_str(data)
                .map_err(|e| ProviderError::fatal(format!("invalid stream chunk: {e}")))?;
            acc.handle(&chunk, &mut *on_text);
            Ok(())
        })?;
        Ok(acc.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![json!({
            "name": "read",
            "description": "Read a file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        })]
    }

    fn history() -> Vec<Message> {
        vec![
            Message::user("hi"),
            Message::assistant(
                "reading",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    args: json!({"path": "a"}),
                }],
            ),
            Message::tool_result("t1", "file data", false),
        ]
    }

    #[test]
    fn anthropic_request_shapes_history_into_content_blocks() {
        let body = anthropic_build_request("m", 100, "sys", &history(), &tools());
        // System and the last tool carry cache markers for prompt caching.
        assert_eq!(body["system"][0]["text"], "sys");
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(body["tools"][0]["name"], "read");
        assert_eq!(
            body["tools"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["content"], json!([{"type": "text", "text": "hi"}]));
        assert_eq!(
            msgs[1]["content"],
            json!([
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a"}},
            ])
        );
        assert_eq!(
            msgs[2],
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "file data",
                    "is_error": false,
                }],
            })
        );
    }

    #[test]
    fn anthropic_request_merges_parallel_tool_results_into_one_user_message() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant(
                "",
                vec![
                    ToolCall {
                        id: "t1".into(),
                        name: "read".into(),
                        args: json!({"path": "a"}),
                    },
                    ToolCall {
                        id: "t2".into(),
                        name: "read".into(),
                        args: json!({"path": "b"}),
                    },
                ],
            ),
            Message::tool_result("t1", "one", false),
            Message::tool_result("t2", "two", true),
        ];
        let body = anthropic_build_request("m", 100, "sys", &messages, &tools());
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        let results = msgs[2]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[1]["is_error"], json!(true));
    }

    #[test]
    fn anthropic_parses_text_and_tool_use_blocks() {
        let reply = anthropic_parse_response(&json!({
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.txt"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 120, "output_tokens": 45},
        }))
        .unwrap();
        assert_eq!(reply.role, Role::Assistant);
        assert_eq!(reply.content, "reading");
        assert_eq!(
            reply.usage,
            Some(Usage {
                input_tokens: 120,
                output_tokens: 45
            })
        );
        assert_eq!(
            reply.tool_calls,
            vec![ToolCall {
                id: "toolu_1".into(),
                name: "read".into(),
                args: json!({"path": "a.txt"})
            }]
        );
    }

    #[test]
    fn anthropic_rejects_malformed_response() {
        assert!(anthropic_parse_response(&json!({"error": "nope"})).is_err());
    }

    #[test]
    fn openai_request_shapes_history_and_tools() {
        let body = openai_build_request("m", "sys", &history(), &tools());
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[1], json!({"role": "user", "content": "hi"}));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(
            msgs[3],
            json!({"role": "tool", "content": "file data", "tool_call_id": "t1"})
        );
        assert_eq!(body["tools"][0]["function"]["name"], "read");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn openai_parses_text_and_tool_calls() {
        let reply = openai_parse_response(&json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\": \"a.txt\"}"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20},
        }))
        .unwrap();
        assert_eq!(reply.content, "");
        assert_eq!(
            reply.usage,
            Some(Usage {
                input_tokens: 80,
                output_tokens: 20
            })
        );
        assert_eq!(
            reply.tool_calls,
            vec![ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                args: json!({"path": "a.txt"})
            }]
        );
    }

    #[test]
    fn openai_rejects_malformed_response() {
        assert!(openai_parse_response(&json!({"error": "nope"})).is_err());
    }

    #[test]
    fn anthropic_headers_prefer_bearer_token() {
        let mut provider = AnthropicProvider::new("sk-key", "m");
        assert!(provider
            .headers()
            .contains(&("x-api-key".into(), "sk-key".into())));

        provider.auth_token = Some("oauth-tok".into());
        let headers = provider.headers();
        assert!(headers.contains(&("Authorization".into(), "Bearer oauth-tok".into())));
        assert!(!headers.iter().any(|(name, _)| name == "x-api-key"));
    }

    #[test]
    fn message_serde_round_trips() {
        for msg in history() {
            let encoded = serde_json::to_string(&msg).unwrap();
            let decoded: Message = serde_json::from_str(&encoded).unwrap();
            assert_eq!(msg, decoded);
        }
    }
}
