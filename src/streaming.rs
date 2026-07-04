//! Server-sent-events streaming: parse each provider's delta events into
//! the same neutral `Message` the non-streaming path produces, emitting
//! text chunks as they arrive.
//!
//! The accumulators are pure state machines over parsed JSON events, so
//! every provider quirk (split tool-call JSON, usage in the final frame)
//! is tested without a network.

use std::io::BufRead;

use serde_json::Value;

use crate::providers::{Message, ProviderError, ToolCall, Usage};

/// Read an SSE body line by line, invoking `on_data` for each `data:`
/// payload. Ignores comments, event names, and keep-alive blank lines.
pub fn each_sse_data(
    reader: impl BufRead,
    mut on_data: impl FnMut(&str) -> Result<(), ProviderError>,
) -> Result<(), ProviderError> {
    for line in reader.lines() {
        let line = line.map_err(|e| ProviderError::transient(format!("stream cut off: {e}")))?;
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                on_data(data)?;
            }
        }
    }
    Ok(())
}

/// Builds a `Message` from Anthropic `message_start` / `content_block_*` /
/// `message_delta` events.
#[derive(Default)]
pub struct AnthropicAccumulator {
    content: String,
    tool_calls: Vec<ToolCall>,
    /// JSON fragments for the tool call currently being streamed.
    pending_json: String,
    usage: Usage,
}

impl AnthropicAccumulator {
    pub fn handle(&mut self, event: &Value, mut on_text: impl FnMut(&str)) {
        match event["type"].as_str() {
            Some("message_start") => {
                self.usage.input_tokens = event["message"]["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap_or(0);
            }
            Some("content_block_start") => {
                let block = &event["content_block"];
                if block["type"] == "tool_use" {
                    self.pending_json.clear();
                    self.tool_calls.push(ToolCall {
                        id: block["id"].as_str().unwrap_or_default().to_string(),
                        name: block["name"].as_str().unwrap_or_default().to_string(),
                        args: Value::Null,
                    });
                }
            }
            Some("content_block_delta") => match event["delta"]["type"].as_str() {
                Some("text_delta") => {
                    let text = event["delta"]["text"].as_str().unwrap_or_default();
                    self.content.push_str(text);
                    on_text(text);
                }
                Some("input_json_delta") => {
                    self.pending_json
                        .push_str(event["delta"]["partial_json"].as_str().unwrap_or_default());
                }
                _ => {}
            },
            Some("content_block_stop") => {
                if let Some(call) = self.tool_calls.last_mut() {
                    if call.args.is_null() {
                        call.args = serde_json::from_str(&self.pending_json)
                            .unwrap_or_else(|_| serde_json::json!({}));
                    }
                }
            }
            Some("message_delta") => {
                if let Some(tokens) = event["usage"]["output_tokens"].as_u64() {
                    self.usage.output_tokens = tokens;
                }
            }
            _ => {}
        }
    }

    pub fn finish(self) -> Message {
        let mut message = Message::assistant(self.content, self.tool_calls);
        message.usage = Some(self.usage);
        message
    }
}

/// Builds a `Message` from OpenAI chat-completion chunks.
#[derive(Default)]
pub struct OpenAIAccumulator {
    content: String,
    tool_calls: Vec<ToolCall>,
    /// Argument JSON accumulates per tool call, keyed by chunk index.
    pending_args: Vec<String>,
    usage: Option<Usage>,
}

impl OpenAIAccumulator {
    pub fn handle(&mut self, chunk: &Value, mut on_text: impl FnMut(&str)) {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(text) = delta["content"].as_str() {
            self.content.push_str(text);
            on_text(text);
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                let index = call["index"].as_u64().unwrap_or(0) as usize;
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(ToolCall {
                        id: String::new(),
                        name: String::new(),
                        args: Value::Null,
                    });
                    self.pending_args.push(String::new());
                }
                if let Some(id) = call["id"].as_str() {
                    self.tool_calls[index].id = id.to_string();
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    self.tool_calls[index].name = name.to_string();
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    self.pending_args[index].push_str(args);
                }
            }
        }
        // With stream_options.include_usage, the final chunk carries usage.
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(Usage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
    }

    pub fn finish(mut self) -> Message {
        for (call, args) in self.tool_calls.iter_mut().zip(&self.pending_args) {
            call.args = serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({}));
        }
        let mut message = Message::assistant(self.content, self.tool_calls);
        message.usage = self.usage;
        message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collect_stream<A>(
        events: &[Value],
        mut handle: impl FnMut(&mut A, &Value, &mut dyn FnMut(&str)),
        mut acc: A,
    ) -> (A, String) {
        let mut streamed = String::new();
        for event in events {
            handle(&mut acc, event, &mut |t| streamed.push_str(t));
        }
        (acc, streamed)
    }

    #[test]
    fn sse_lines_extract_data_payloads() {
        let body = "event: ping\n\ndata: {\"a\":1}\n\ndata: [DONE]\n\n: comment\ndata: {\"b\":2}\n";
        let mut seen = Vec::new();
        each_sse_data(body.as_bytes(), |data| {
            seen.push(data.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec!["{\"a\":1}", "{\"b\":2}"]);
    }

    #[test]
    fn anthropic_accumulates_text_and_usage() {
        let events = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 50}}}),
            json!({"type": "content_block_start", "content_block": {"type": "text"}}),
            json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}}),
            json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_stop"}),
            json!({"type": "message_delta", "usage": {"output_tokens": 7}}),
        ];
        let (acc, streamed) = collect_stream(
            &events,
            |a: &mut AnthropicAccumulator, e, f| a.handle(e, f),
            AnthropicAccumulator::default(),
        );
        assert_eq!(streamed, "hello");
        let message = acc.finish();
        assert_eq!(message.content, "hello");
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: 50,
                output_tokens: 7
            })
        );
    }

    #[test]
    fn anthropic_reassembles_split_tool_call_json() {
        let events = [
            json!({"type": "content_block_start",
                   "content_block": {"type": "tool_use", "id": "t1", "name": "read"}}),
            json!({"type": "content_block_delta",
                   "delta": {"type": "input_json_delta", "partial_json": "{\"pa"}}),
            json!({"type": "content_block_delta",
                   "delta": {"type": "input_json_delta", "partial_json": "th\": \"a.txt\"}"}}),
            json!({"type": "content_block_stop"}),
        ];
        let (acc, streamed) = collect_stream(
            &events,
            |a: &mut AnthropicAccumulator, e, f| a.handle(e, f),
            AnthropicAccumulator::default(),
        );
        assert_eq!(streamed, "");
        let message = acc.finish();
        assert_eq!(
            message.tool_calls,
            vec![ToolCall {
                id: "t1".into(),
                name: "read".into(),
                args: json!({"path": "a.txt"})
            }]
        );
    }

    #[test]
    fn openai_accumulates_text_tool_calls_and_usage() {
        let chunks = [
            json!({"choices": [{"delta": {"content": "wor"}}]}),
            json!({"choices": [{"delta": {"content": "king"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{\"pa"}}
            ]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "th\": \"b.txt\"}"}}
            ]}}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}),
        ];
        let (acc, streamed) = collect_stream(
            &chunks,
            |a: &mut OpenAIAccumulator, e, f| a.handle(e, f),
            OpenAIAccumulator::default(),
        );
        assert_eq!(streamed, "working");
        let message = acc.finish();
        assert_eq!(message.content, "working");
        assert_eq!(
            message.tool_calls,
            vec![ToolCall {
                id: "c1".into(),
                name: "read".into(),
                args: json!({"path": "b.txt"})
            }]
        );
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: 9,
                output_tokens: 4
            })
        );
    }

    #[test]
    fn malformed_tool_json_degrades_to_empty_args() {
        let events = [
            json!({"type": "content_block_start",
                   "content_block": {"type": "tool_use", "id": "t1", "name": "read"}}),
            json!({"type": "content_block_delta",
                   "delta": {"type": "input_json_delta", "partial_json": "{broken"}}),
            json!({"type": "content_block_stop"}),
        ];
        let (acc, _) = collect_stream(
            &events,
            |a: &mut AnthropicAccumulator, e, f| a.handle(e, f),
            AnthropicAccumulator::default(),
        );
        assert_eq!(acc.finish().tool_calls[0].args, json!({}));
    }
}
