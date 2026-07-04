//! The agent loop: call the model, run requested tools, feed results back,
//! repeat until the model answers without tool calls.
//!
//! No step limits, no hidden context injection. Tool failures are returned
//! to the model as error results — it sees exactly what went wrong and can
//! correct course. Transient provider failures are retried with backoff.

use std::time::Duration;

use crate::providers::{Message, Provider, ProviderError, ToolCall};
use crate::session::Session;
use crate::tools::{ToolRegistry, ToolResult};

/// Progress notifications emitted while a turn runs, for the UI layer.
pub enum Event<'a> {
    AssistantText(&'a str),
    ToolStart(&'a ToolCall),
    ToolEnd(&'a ToolCall, &'a ToolResult),
}

pub struct Agent<'a> {
    pub provider: &'a dyn Provider,
    pub tools: &'a ToolRegistry,
    pub system: String,
    /// Total attempts per model call (1 = no retry).
    pub max_attempts: u32,
    /// Base delay between retries; doubles each attempt.
    pub retry_delay: Duration,
}

impl<'a> Agent<'a> {
    pub fn new(provider: &'a dyn Provider, tools: &'a ToolRegistry, system: String) -> Self {
        Self {
            provider,
            tools,
            system,
            max_attempts: 3,
            retry_delay: Duration::from_secs(1),
        }
    }

    /// Run one user turn to completion. Appends every message (user,
    /// assistant, tool results) to the session as it happens, emits events
    /// for the UI, and returns the model's final text answer.
    pub fn run_turn(
        &self,
        session: &mut Session,
        user_input: &str,
        mut on_event: impl FnMut(Event),
    ) -> Result<String, ProviderError> {
        session
            .append(Message::user(user_input))
            .map_err(|e| ProviderError::fatal(format!("cannot persist session: {e}")))?;
        loop {
            let schemas = self.tools.schemas();
            let reply = self.complete_with_retry(&session.messages, &schemas)?;
            if !reply.content.is_empty() {
                on_event(Event::AssistantText(&reply.content));
            }
            let tool_calls = reply.tool_calls.clone();
            let final_text = reply.content.clone();
            session
                .append(reply)
                .map_err(|e| ProviderError::fatal(format!("cannot persist session: {e}")))?;
            if tool_calls.is_empty() {
                return Ok(final_text);
            }
            for call in &tool_calls {
                on_event(Event::ToolStart(call));
                let result = self.tools.run(&call.name, &call.args);
                on_event(Event::ToolEnd(call, &result));
                session
                    .append(Message::tool_result(
                        &call.id,
                        &result.output,
                        result.is_error,
                    ))
                    .map_err(|e| ProviderError::fatal(format!("cannot persist session: {e}")))?;
            }
        }
    }

    fn complete_with_retry(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<Message, ProviderError> {
        let mut delay = self.retry_delay;
        let mut last_error = ProviderError::fatal("no attempts made");
        for attempt in 0..self.max_attempts {
            if attempt > 0 {
                std::thread::sleep(delay);
                delay *= 2;
            }
            match self.provider.complete(&self.system, messages, tools) {
                Ok(reply) => return Ok(reply),
                // Retrying a fatal error (bad request, auth) wastes time
                // and money; only transient failures get another attempt.
                Err(e) if !e.retryable => return Err(e),
                Err(e) => last_error = e,
            }
        }
        Err(last_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::cell::RefCell;
    use tempfile::TempDir;

    /// A provider that replays a script of canned responses.
    struct ScriptedProvider {
        script: RefCell<Vec<Result<Message, ProviderError>>>,
        calls: RefCell<Vec<Vec<Message>>>,
    }

    impl ScriptedProvider {
        fn new(script: Vec<Result<Message, ProviderError>>) -> Self {
            Self {
                script: RefCell::new(script),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            _system: &str,
            messages: &[Message],
            _tools: &[Value],
        ) -> Result<Message, ProviderError> {
            self.calls.borrow_mut().push(messages.to_vec());
            self.script.borrow_mut().remove(0)
        }
    }

    fn agent_fixture<'a>(
        provider: &'a ScriptedProvider,
        tools: &'a crate::tools::ToolRegistry,
    ) -> Agent<'a> {
        let mut agent = Agent::new(provider, tools, "sys".into());
        agent.retry_delay = Duration::from_millis(1);
        agent
    }

    #[test]
    fn text_only_reply_ends_the_turn() {
        let dir = TempDir::new().unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![Ok(Message::assistant("done", vec![]))]);
        let mut session = Session::create(dir.path()).unwrap();

        let answer = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "hello", |_| {})
            .unwrap();

        assert_eq!(answer, "done");
        assert_eq!(session.messages.len(), 2); // user + assistant
        assert_eq!(provider.calls.borrow().len(), 1);
    }

    #[test]
    fn tool_calls_are_executed_and_fed_back() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "file contents").unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Ok(Message::assistant(
                "reading",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    args: json!({"path": "a.txt"}),
                }],
            )),
            Ok(Message::assistant("the file says: file contents", vec![])),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        let mut events = Vec::new();
        let answer = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "read a.txt", |event| {
                events.push(match event {
                    Event::AssistantText(t) => format!("text:{t}"),
                    Event::ToolStart(c) => format!("start:{}", c.name),
                    Event::ToolEnd(c, r) => format!("end:{}:{}", c.name, r.output),
                });
            })
            .unwrap();

        assert_eq!(answer, "the file says: file contents");
        // user, assistant(tool call), tool result, assistant(final)
        assert_eq!(session.messages.len(), 4);
        assert_eq!(session.messages[2].content, "file contents");
        assert_eq!(
            events,
            vec![
                "text:reading",
                "start:read",
                "end:read:file contents",
                "text:the file says: file contents",
            ]
        );
        // Second model call must include the tool result.
        let second_call = &provider.calls.borrow()[1];
        assert_eq!(
            second_call.last().unwrap().tool_call_id.as_deref(),
            Some("t1")
        );
    }

    #[test]
    fn tool_errors_are_fed_back_not_fatal() {
        let dir = TempDir::new().unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Ok(Message::assistant(
                "",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    args: json!({"path": "ghost.txt"}),
                }],
            )),
            Ok(Message::assistant("that file doesn't exist", vec![])),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        let answer = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "read ghost.txt", |_| {})
            .unwrap();

        assert_eq!(answer, "that file doesn't exist");
        assert!(session.messages[2].is_error);
    }

    #[test]
    fn parallel_tool_calls_run_in_order() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "aaa").unwrap();
        std::fs::write(dir.path().join("b.txt"), "bbb").unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Ok(Message::assistant(
                "",
                vec![
                    ToolCall {
                        id: "t1".into(),
                        name: "read".into(),
                        args: json!({"path": "a.txt"}),
                    },
                    ToolCall {
                        id: "t2".into(),
                        name: "read".into(),
                        args: json!({"path": "b.txt"}),
                    },
                ],
            )),
            Ok(Message::assistant("done", vec![])),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        agent_fixture(&provider, &tools)
            .run_turn(&mut session, "read both", |_| {})
            .unwrap();

        assert_eq!(session.messages[2].content, "aaa");
        assert_eq!(session.messages[3].content, "bbb");
    }

    #[test]
    fn transient_provider_errors_are_retried() {
        let dir = TempDir::new().unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Err(ProviderError::transient("HTTP 529: overloaded")),
            Ok(Message::assistant("recovered", vec![])),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        let answer = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "hi", |_| {})
            .unwrap();
        assert_eq!(answer, "recovered");
    }

    #[test]
    fn fatal_provider_errors_are_not_retried() {
        let dir = TempDir::new().unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Err(ProviderError::fatal("HTTP 401: bad key")),
            Ok(Message::assistant("never reached", vec![])),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        let error = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "hi", |_| {})
            .unwrap_err();
        assert!(error.message.contains("bad key"));
        assert_eq!(provider.calls.borrow().len(), 1); // no second attempt
    }

    #[test]
    fn exhausted_retries_return_the_last_error() {
        let dir = TempDir::new().unwrap();
        let tools = crate::tools::default_registry(dir.path());
        let provider = ScriptedProvider::new(vec![
            Err(ProviderError::transient("boom 1")),
            Err(ProviderError::transient("boom 2")),
            Err(ProviderError::transient("boom 3")),
        ]);
        let mut session = Session::create(dir.path()).unwrap();

        let error = agent_fixture(&provider, &tools)
            .run_turn(&mut session, "hi", |_| {})
            .unwrap_err();
        assert!(error.message.contains("boom 3"));
    }
}
