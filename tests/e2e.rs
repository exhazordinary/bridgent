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

/// Serve canned JSON responses (one per request) on an ephemeral port.
/// Returns the base URL. The server thread exits after the last response.
fn fake_server(responses: Vec<Value>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for response in responses {
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
            let payload = response.to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(reply.as_bytes()).unwrap();
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
    let base_url = fake_server(vec![text_response("hello over http")]);
    let reply = provider_for(base_url)
        .complete("sys", &[Message::user("hi")], &[])
        .unwrap();
    assert_eq!(reply.content, "hello over http");
}

#[test]
fn agent_loop_executes_tools_end_to_end() {
    let base_url = fake_server(vec![
        json!({"choices": [{"message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": "{\"path\": \"hello.txt\", \"content\": \"from the agent\"}",
                },
            }],
        }}]}),
        text_response("wrote the file"),
    ]);
    let dir = tempfile::TempDir::new().unwrap();
    let provider = provider_for(base_url);
    let tools = default_registry(dir.path());
    let agent = Agent::new(&provider, &tools, "sys".into());
    let mut session = Session::create(dir.path()).unwrap();

    let answer = agent
        .run_turn(&mut session, "create hello.txt", |_| {})
        .unwrap();

    assert_eq!(answer, "wrote the file");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
        "from the agent"
    );
    // user, assistant(tool call), tool result, assistant(final)
    assert_eq!(session.messages.len(), 4);
}
