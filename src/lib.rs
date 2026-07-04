//! bridle — a minimal, provider-agnostic coding agent harness.
//!
//! An agent is a model plus a harness. This crate is the harness: the four
//! core tools (read, write, edit, bash), an agent loop that runs until the
//! model stops calling tools, provider-neutral message types, and persistent
//! JSONL sessions.

pub mod agent;
pub mod config;
pub mod context;
pub mod process;
pub mod providers;
pub mod refine;
pub mod session;
pub mod tools;
