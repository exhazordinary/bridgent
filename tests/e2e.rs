//! End-to-end tests over real HTTP: an in-process fake OpenAI-compatible
//! server exercises the transport path (`post`, headers, error mapping)
//! that the pure request/parse unit tests can't reach.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

use serde_json::{json, Value};

use bridgent::agent::Agent;
use bridgent::providers::{Message, OpenAIProvider, Provider};
use bridgent::session::Session;
use bridgent::tools::default_registry;

/// One canned HTTP exchange: a plain JSON body or an SSE event stream.
enum Reply {
    Json(Value),
    Sse(Vec<Value>),
}

/// Serve canned replies (one per request) on an ephemeral port. Returns the
/// base URL. The server thread exits after the last reply.
fn fake_server(replies: Vec<Reply>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut content_length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim();
                if let Some(len) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = len.trim().parse().unwrap();
                }
                if line.is_empty() {
                    break;
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).unwrap();
            let (content_type, payload) = match &reply {
                Reply::Json(value) => ("application/json", value.to_string()),
                Reply::Sse(events) => (
                    "text/event-stream",
                    events
                        .iter()
                        .map(|e| format!("data: {e}\n\n"))
                        .chain(std::iter::once("data: [DONE]\n\n".into()))
                        .collect(),
                ),
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
    format!("http://{addr}/v1")
}

fn text_response(content: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": content}}]})
}

fn provider_for(base_url: String) -> OpenAIProvider {
    let mut provider = OpenAIProvider::new("test-key", "fake-model");
    provider.base_url = base_url;
    provider
}

#[test]
fn provider_round_trips_over_real_http() {
    let base_url = fake_server(vec![Reply::Json(text_response("hello over http"))]);
    let reply = provider_for(base_url)
        .complete("sys", &[Message::user("hi")], &[])
        .unwrap();
    assert_eq!(reply.content, "hello over http");
}

#[test]
fn anthropic_streaming_works_over_real_http() {
    let base_url = fake_server(vec![Reply::Sse(vec![
        json!({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
        json!({"type": "content_block_start", "content_block": {"type": "text"}}),
        json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "streamed"}}),
        json!({"type": "content_block_stop"}),
        json!({"type": "message_delta", "usage": {"output_tokens": 3}}),
    ])]);
    let mut provider = bridgent::providers::AnthropicProvider::new("test-key", "fake-model");
    provider.base_url = base_url.trim_end_matches("/v1").to_string();

    let mut streamed = String::new();
    let reply = provider
        .complete_stream("sys", &[Message::user("hi")], &[], &mut |t| {
            streamed.push_str(t)
        })
        .unwrap();

    assert_eq!(streamed, "streamed");
    assert_eq!(reply.content, "streamed");
    assert_eq!(
        reply.usage.map(|u| (u.input_tokens, u.output_tokens)),
        Some((12, 3))
    );
}

#[test]
fn agent_loop_streams_and_executes_tools_end_to_end() {
    let base_url = fake_server(vec![
        // First turn: a streamed tool call, split across SSE chunks.
        Reply::Sse(vec![
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "write", "arguments": "{\"path\": \"hello.txt\","},
            }]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": " \"content\": \"from the agent\"}"},
            }]}}]}),
        ]),
        // Second turn: streamed text.
        Reply::Sse(vec![
            json!({"choices": [{"delta": {"content": "wrote "}}]}),
            json!({"choices": [{"delta": {"content": "the file"}}]}),
        ]),
    ]);
    let dir = tempfile::TempDir::new().unwrap();
    let provider = provider_for(base_url);
    let tools = default_registry(dir.path());
    let agent = Agent::new(&provider, &tools, "sys".into());
    let mut session = Session::create(dir.path()).unwrap();

    let mut deltas = Vec::new();
    let answer = agent
        .run_turn(&mut session, "create hello.txt", |event| {
            if let bridgent::agent::Event::AssistantDelta(text) = event {
                deltas.push(text.to_string());
            }
        })
        .unwrap();

    assert_eq!(answer, "wrote the file");
    assert_eq!(deltas, vec!["wrote ", "the file"]); // streamed in two chunks
    assert_eq!(
        std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
        "from the agent"
    );
    // user, assistant(tool call), tool result, assistant(final)
    assert_eq!(session.messages.len(), 4);
}
