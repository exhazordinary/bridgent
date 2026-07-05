//! Provider-neutral message types and LLM API clients.
//!
//! Request building and response parsing are pure functions over JSON values
//! so they can be tested without a network; `complete` is a thin HTTP shim.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::ToolSchema;

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

    /// An assistant message as parsed from an API response, usage included.
    pub fn assistant_with_usage(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        usage: Option<Usage>,
    ) -> Self {
        Self {
            usage,
            ..Self::assistant(content, tool_calls)
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
        tools: &[ToolSchema],
    ) -> Result<Message, ProviderError>;

    /// Like `complete`, but emits text chunks through `on_text` as they
    /// arrive. The default falls back to one chunk after the full response —
    /// mocks and future providers work unchanged.
    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let message = self.complete(system, messages, tools)?;
        if !message.content.is_empty() {
            on_text(&message.content);
        }
        Ok(message)
    }
}

/// Wraps any provider with retry-on-transient-error and exponential backoff,
/// so every consumer (agent loop, refine engine) gets resilience from the
/// transport instead of re-implementing it. Streaming calls retry only while
/// nothing has been emitted — retrying after text reached the user would
/// duplicate output.
pub struct RetryingProvider {
    pub inner: Box<dyn Provider>,
    /// Total attempts per call (1 = no retry).
    pub max_attempts: u32,
    /// Base delay between retries; doubles each attempt.
    pub retry_delay: std::time::Duration,
}

impl RetryingProvider {
    pub fn new(inner: Box<dyn Provider>) -> Self {
        Self {
            inner,
            max_attempts: 3,
            retry_delay: std::time::Duration::from_secs(1),
        }
    }

    /// Run `call` until it succeeds, is fatal, aborts, or attempts run out.
    /// `call` returns the result plus whether retrying is no longer safe.
    fn retry<T>(
        &self,
        mut call: impl FnMut() -> (Result<T, ProviderError>, bool),
    ) -> Result<T, ProviderError> {
        let mut delay = self.retry_delay;
        for attempt in 1.. {
            if attempt > 1 {
                std::thread::sleep(delay);
                delay *= 2;
            }
            let (result, abort) = call();
            match result {
                Ok(value) => return Ok(value),
                Err(e) if !e.retryable || abort || attempt >= self.max_attempts => return Err(e),
                Err(_) => {}
            }
        }
        unreachable!("retry loop always returns")
    }
}

impl Provider for RetryingProvider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
    ) -> Result<Message, ProviderError> {
        self.retry(|| (self.inner.complete(system, messages, tools), false))
    }

    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        self.retry(|| {
            let mut emitted = false;
            let result = self
                .inner
                .complete_stream(system, messages, tools, &mut |text| {
                    emitted = true;
                    on_text(text);
                });
            (result, emitted)
        })
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

pub(crate) fn parse_usage(body: &Value, input_key: &str, output_key: &str) -> Option<Usage> {
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

    fn url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
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
    tools: &[ToolSchema],
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
    let mut tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect();
    // System and tools are stable across a session; cache_control markers
    // let the API reuse them instead of re-processing every request.
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
    Ok(Message::assistant_with_usage(
        content,
        tool_calls,
        parse_usage(body, "input_tokens", "output_tokens"),
    ))
}

impl Provider for AnthropicProvider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
    ) -> Result<Message, ProviderError> {
        let body = anthropic_build_request(&self.model, self.max_tokens, system, messages, tools);
        let response = post(&self.url(), &self.headers(), body)?;
        anthropic_parse_response(&response)
    }

    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let mut body =
            anthropic_build_request(&self.model, self.max_tokens, system, messages, tools);
        body["stream"] = Value::Bool(true);
        let response = send(&self.url(), &self.headers(), body)?;
        crate::streaming::drive_stream(
            std::io::BufReader::new(response.into_reader()),
            crate::streaming::AnthropicAccumulator::default(),
            on_text,
        )
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

    pub fn headers(&self) -> Vec<(String, String)> {
        vec![("Authorization".into(), format!("Bearer {}", self.api_key))]
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

pub fn openai_build_request(
    model: &str,
    system: &str,
    messages: &[Message],
    tools: &[ToolSchema],
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
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
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
    Ok(Message::assistant_with_usage(
        content,
        tool_calls,
        parse_usage(body, "prompt_tokens", "completion_tokens"),
    ))
}

impl Provider for OpenAIProvider {
    fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
    ) -> Result<Message, ProviderError> {
        let body = openai_build_request(&self.model, system, messages, tools);
        let response = post(&self.url(), &self.headers(), body)?;
        openai_parse_response(&response)
    }

    fn complete_stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSchema],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message, ProviderError> {
        let mut body = openai_build_request(&self.model, system, messages, tools);
        body["stream"] = Value::Bool(true);
        body["stream_options"] = json!({"include_usage": true});
        let response = send(&self.url(), &self.headers(), body)?;
        crate::streaming::drive_stream(
            std::io::BufReader::new(response.into_reader()),
            crate::streaming::OpenAIAccumulator::default(),
            on_text,
        )
    }
}

#[cfg(test)]
pub mod test_support {
    use super::*;
    use std::cell::RefCell;

    /// Replays a script of canned responses and records every call's
    /// message history — the one `Provider` mock shared by all test modules.
    pub struct ScriptedProvider {
        script: RefCell<Vec<Result<Message, ProviderError>>>,
        pub calls: RefCell<Vec<Vec<Message>>>,
    }

    impl ScriptedProvider {
        pub fn new(script: Vec<Result<Message, ProviderError>>) -> Self {
            Self {
                script: RefCell::new(script),
                calls: RefCell::new(Vec::new()),
            }
        }

        /// Plain-text assistant replies, in order.
        pub fn texts(replies: &[&str]) -> Self {
            Self::new(
                replies
                    .iter()
                    .map(|reply| Ok(Message::assistant(*reply, vec![])))
                    .collect(),
            )
        }
    }

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            _system: &str,
            messages: &[Message],
            _tools: &[ToolSchema],
        ) -> Result<Message, ProviderError> {
            self.calls.borrow_mut().push(messages.to_vec());
            let mut script = self.script.borrow_mut();
            if script.is_empty() {
                return Err(ProviderError::fatal("script exhausted"));
            }
            script.remove(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<ToolSchema> {
        vec![ToolSchema {
            name: "read".into(),
            description: "Read a file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
        }]
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

    fn retrying(script: Vec<Result<Message, ProviderError>>) -> RetryingProvider {
        let mut provider =
            RetryingProvider::new(Box::new(test_support::ScriptedProvider::new(script)));
        provider.retry_delay = std::time::Duration::from_millis(1);
        provider
    }

    #[test]
    fn retrying_provider_retries_transient_errors() {
        let provider = retrying(vec![
            Err(ProviderError::transient("HTTP 529: overloaded")),
            Ok(Message::assistant("recovered", vec![])),
        ]);
        let reply = provider
            .complete("sys", &[Message::user("hi")], &[])
            .unwrap();
        assert_eq!(reply.content, "recovered");
    }

    #[test]
    fn retrying_provider_gives_up_on_fatal_errors() {
        let provider = retrying(vec![
            Err(ProviderError::fatal("HTTP 401: bad key")),
            Ok(Message::assistant("never reached", vec![])),
        ]);
        let error = provider
            .complete("sys", &[Message::user("hi")], &[])
            .unwrap_err();
        assert!(error.message.contains("bad key"));
    }

    #[test]
    fn retrying_provider_returns_the_last_error_when_exhausted() {
        let provider = retrying(vec![
            Err(ProviderError::transient("boom 1")),
            Err(ProviderError::transient("boom 2")),
            Err(ProviderError::transient("boom 3")),
        ]);
        let error = provider
            .complete("sys", &[Message::user("hi")], &[])
            .unwrap_err();
        assert!(error.message.contains("boom 3"));
    }

    #[test]
    fn retrying_provider_never_retries_after_text_was_emitted() {
        // The scripted mock's default complete_stream emits the full text on
        // success only; simulate a mid-stream failure with a custom inner.
        struct EmitsThenFails;
        impl Provider for EmitsThenFails {
            fn complete(
                &self,
                _system: &str,
                _messages: &[Message],
                _tools: &[ToolSchema],
            ) -> Result<Message, ProviderError> {
                unreachable!("streaming only")
            }

            fn complete_stream(
                &self,
                _system: &str,
                _messages: &[Message],
                _tools: &[ToolSchema],
                on_text: &mut dyn FnMut(&str),
            ) -> Result<Message, ProviderError> {
                on_text("partial output");
                Err(ProviderError::transient("stream cut off"))
            }
        }
        let mut provider = RetryingProvider::new(Box::new(EmitsThenFails));
        provider.retry_delay = std::time::Duration::from_millis(1);
        let mut streamed = String::new();
        let error = provider
            .complete_stream("sys", &[Message::user("hi")], &[], &mut |t| {
                streamed.push_str(t)
            })
            .unwrap_err();
        assert!(error.message.contains("cut off"));
        assert_eq!(streamed, "partial output"); // exactly once — no retry
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
