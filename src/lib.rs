//! bridle — a minimal, provider-agnostic coding agent harness.
//!
//! An agent is a model plus a harness. This crate is the harness: the four
//! core tools (read, write, edit, bash), an agent loop that runs until the
//! model stops calling tools, provider-neutral message types, and persistent
//! JSONL sessions.

pub mod providers;
pub mod session;
pub mod tools;
