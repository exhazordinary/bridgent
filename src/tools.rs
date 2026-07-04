//! The four core tools: read, write, edit, bash.
//!
//! Frontier models are heavily RL-trained on coding-agent workflows; these
//! four tools are sufficient for effective coding work. Anything else the
//! agent needs it can get through bash.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::{json, Value};

use crate::process::run_with_timeout;

/// Outcome of a tool invocation, fed back to the model verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }

    pub fn err(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
        }
    }
}

/// A callable capability exposed to the model.
pub trait Tool {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// JSON schema for the tool's arguments.
    fn parameters(&self) -> Value;
    fn run(&self, args: &Value) -> ToolResult;
}

/// Holds the tools available to one agent and dispatches calls to them.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Tool schemas in provider-neutral form (name, description, input_schema).
    pub fn schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.parameters(),
                })
            })
            .collect()
    }

    pub fn run(&self, name: &str, args: &Value) -> ToolResult {
        match self.tools.iter().find(|t| t.name() == name) {
            Some(tool) => tool.run(args),
            None => ToolResult::err(format!("Unknown tool: {name}")),
        }
    }
}

/// The standard four-tool registry rooted at `workdir`.
pub fn default_registry(workdir: &Path) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry.register(Box::new(ReadTool::new(workdir)));
    registry.register(Box::new(WriteTool::new(workdir)));
    registry.register(Box::new(EditTool::new(workdir)));
    registry.register(Box::new(BashTool::new(workdir)));
    registry
}

fn resolve(workdir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        workdir.join(p)
    }
}

fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolResult> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolResult::err(format!("Missing required argument: {key}")))
}

pub struct ReadTool {
    workdir: PathBuf,
}

impl ReadTool {
    pub const MAX_LINES: usize = 2000;

    pub fn new(workdir: &Path) -> Self {
        Self {
            workdir: workdir.to_path_buf(),
        }
    }
}

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read a file. Returns its contents. Use offset (1-based line number) \
         and limit to page through large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative or absolute"},
                "offset": {"type": "integer", "description": "1-based first line to read"},
                "limit": {"type": "integer", "description": "Max number of lines to read"},
            },
            "required": ["path"],
        })
    }

    fn run(&self, args: &Value) -> ToolResult {
        let path = match required_str(args, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let content = match std::fs::read_to_string(resolve(&self.workdir, path)) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("Cannot read {path}: {e}")),
        };
        let lines: Vec<&str> = content.split_inclusive('\n').collect();
        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(Self::MAX_LINES as u64) as usize;
        let start = offset - 1;
        let selected: String = lines.iter().skip(start).take(limit).copied().collect();
        let remaining = lines.len().saturating_sub(start + limit);
        if remaining > 0 {
            let plural = if remaining == 1 { "" } else { "s" };
            return ToolResult::ok(format!(
                "{selected}\n[truncated: {remaining} more line{plural}, re-read with offset={}]",
                offset + limit
            ));
        }
        ToolResult::ok(selected)
    }
}

pub struct WriteTool {
    workdir: PathBuf,
}

impl WriteTool {
    pub fn new(workdir: &Path) -> Self {
        Self {
            workdir: workdir.to_path_buf(),
        }
    }
}

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn description(&self) -> &'static str {
        "Write content to a file, creating it (and parent directories) or overwriting it."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative or absolute"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        })
    }

    fn run(&self, args: &Value) -> ToolResult {
        let (path, content) = match (required_str(args, "path"), required_str(args, "content")) {
            (Ok(p), Ok(c)) => (p, c),
            (Err(e), _) | (_, Err(e)) => return e,
        };
        let target = resolve(&self.workdir, path);
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::err(format!("Cannot create directories for {path}: {e}"));
            }
        }
        match std::fs::write(&target, content) {
            Ok(()) => ToolResult::ok(format!("Wrote {} bytes to {path}", content.len())),
            Err(e) => ToolResult::err(format!("Cannot write {path}: {e}")),
        }
    }
}

pub struct EditTool {
    workdir: PathBuf,
}

impl EditTool {
    pub fn new(workdir: &Path) -> Self {
        Self {
            workdir: workdir.to_path_buf(),
        }
    }
}

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Replace old_string with new_string in a file. old_string must match \
         exactly one location unless replace_all is true."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative or absolute"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence"},
            },
            "required": ["path", "old_string", "new_string"],
        })
    }

    fn run(&self, args: &Value) -> ToolResult {
        let path = match required_str(args, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let (old, new) = match (
            required_str(args, "old_string"),
            required_str(args, "new_string"),
        ) {
            (Ok(o), Ok(n)) => (o, n),
            (Err(e), _) | (_, Err(e)) => return e,
        };
        let target = resolve(&self.workdir, path);
        let content = match std::fs::read_to_string(&target) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("Cannot read {path}: {e}")),
        };
        let count = content.matches(old).count();
        if count == 0 {
            return ToolResult::err(format!("old_string not found in {path}"));
        }
        let replace_all = args
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if count > 1 && !replace_all {
            return ToolResult::err(format!(
                "old_string matches {count} locations in {path}; add more context \
                 to make it unique, or set replace_all"
            ));
        }
        match std::fs::write(&target, content.replace(old, new)) {
            Ok(()) => ToolResult::ok(format!("Replaced {count} occurrence(s) in {path}")),
            Err(e) => ToolResult::err(format!("Cannot write {path}: {e}")),
        }
    }
}

pub struct BashTool {
    workdir: PathBuf,
}

impl BashTool {
    pub const MAX_OUTPUT_CHARS: usize = 50_000;
    pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

    pub fn new(workdir: &Path) -> Self {
        Self {
            workdir: workdir.to_path_buf(),
        }
    }
}

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the working directory. Returns combined \
         stdout/stderr. Use for anything the other tools don't cover: \
         search, git, tests, package managers, process control."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120)"},
            },
            "required": ["command"],
        })
    }

    fn run(&self, args: &Value) -> ToolResult {
        let command = match required_str(args, "command") {
            Ok(c) => c,
            Err(e) => return e,
        };
        let timeout = Duration::from_secs(
            args.get("timeout")
                .and_then(Value::as_u64)
                .unwrap_or(Self::DEFAULT_TIMEOUT_SECS),
        );
        let result = run_with_timeout(
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.workdir),
            None,
            timeout,
        );
        let process = match result {
            Ok(process) => process,
            Err(e) => return ToolResult::err(e.to_string()),
        };
        let mut output = process.stdout;
        output.push_str(&process.stderr);
        if output.len() > Self::MAX_OUTPUT_CHARS {
            let mut end = Self::MAX_OUTPUT_CHARS;
            while !output.is_char_boundary(end) {
                end -= 1;
            }
            output.truncate(end);
            output.push_str("\n[truncated]");
        }
        if output.is_empty() {
            output.push_str("[no output]");
        }
        match process.exit_code {
            Some(0) => ToolResult::ok(output),
            code => ToolResult::err(format!(
                "{output}\n[exit code {}]",
                code.map_or_else(|| "killed by signal".to_string(), |c| c.to_string())
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn workdir() -> TempDir {
        TempDir::new().unwrap()
    }

    #[test]
    fn registry_dispatches_and_reports_unknown_tools() {
        let dir = workdir();
        let registry = default_registry(dir.path());
        let names: Vec<String> = registry
            .schemas()
            .iter()
            .map(|s| s["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["read", "write", "edit", "bash"]);

        let result = registry.run("nope", &json!({}));
        assert!(result.is_error);
        assert!(result.output.contains("nope"));
    }

    #[test]
    fn read_returns_file_contents() {
        let dir = workdir();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\n").unwrap();
        let result = ReadTool::new(dir.path()).run(&json!({"path": "a.txt"}));
        assert!(!result.is_error);
        assert_eq!(result.output, "hello\nworld\n");
    }

    #[test]
    fn read_missing_file_is_error() {
        let dir = workdir();
        let result = ReadTool::new(dir.path()).run(&json!({"path": "nope.txt"}));
        assert!(result.is_error);
        assert!(result.output.contains("nope.txt"));
    }

    #[test]
    fn read_offset_and_limit_page_through_lines() {
        let dir = workdir();
        std::fs::write(dir.path().join("a.txt"), "l1\nl2\nl3\nl4\n").unwrap();
        let result =
            ReadTool::new(dir.path()).run(&json!({"path": "a.txt", "offset": 2, "limit": 2}));
        assert!(result.output.starts_with("l2\nl3\n"));
        assert!(result.output.contains("1 more line"));
        assert!(result.output.contains("offset=4"));
    }

    #[test]
    fn read_truncates_long_files_with_notice() {
        let dir = workdir();
        std::fs::write(dir.path().join("big.txt"), "x\n".repeat(5000)).unwrap();
        let result = ReadTool::new(dir.path()).run(&json!({"path": "big.txt"}));
        assert!(result.output.contains("truncated"));
        assert!(result.output.contains("3000 more lines"));
    }

    #[test]
    fn write_creates_parents_and_overwrites() {
        let dir = workdir();
        let tool = WriteTool::new(dir.path());
        let result = tool.run(&json!({"path": "a/b/c.txt", "content": "deep"}));
        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "deep"
        );

        tool.run(&json!({"path": "a/b/c.txt", "content": "new"}));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "new"
        );
    }

    #[test]
    fn edit_replaces_unique_string() {
        let dir = workdir();
        std::fs::write(dir.path().join("f.py"), "a = 1\nb = 2\n").unwrap();
        let result = EditTool::new(dir.path())
            .run(&json!({"path": "f.py", "old_string": "b = 2", "new_string": "b = 3"}));
        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.py")).unwrap(),
            "a = 1\nb = 3\n"
        );
    }

    #[test]
    fn edit_rejects_missing_and_ambiguous_matches() {
        let dir = workdir();
        std::fs::write(dir.path().join("f.py"), "x\nx\n").unwrap();
        let tool = EditTool::new(dir.path());

        let missing = tool.run(&json!({"path": "f.py", "old_string": "zzz", "new_string": "y"}));
        assert!(missing.is_error);
        assert!(missing.output.contains("not found"));

        let ambiguous = tool.run(&json!({"path": "f.py", "old_string": "x", "new_string": "y"}));
        assert!(ambiguous.is_error);
        assert!(ambiguous.output.contains("2 locations"));
    }

    #[test]
    fn edit_replace_all_replaces_every_occurrence() {
        let dir = workdir();
        std::fs::write(dir.path().join("f.py"), "x\nx\n").unwrap();
        let result = EditTool::new(dir.path()).run(
            &json!({"path": "f.py", "old_string": "x", "new_string": "y", "replace_all": true}),
        );
        assert!(!result.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.py")).unwrap(),
            "y\ny\n"
        );
    }

    #[test]
    fn bash_runs_in_workdir_and_captures_output() {
        let dir = workdir();
        std::fs::write(dir.path().join("marker.txt"), "here").unwrap();
        let result = BashTool::new(dir.path()).run(&json!({"command": "ls"}));
        assert!(!result.is_error);
        assert!(result.output.contains("marker.txt"));
    }

    #[test]
    fn bash_nonzero_exit_is_error_with_code() {
        let dir = workdir();
        let result = BashTool::new(dir.path()).run(&json!({"command": "exit 3"}));
        assert!(result.is_error);
        assert!(result.output.contains("exit code 3"));
    }

    #[test]
    fn bash_timeout_kills_command() {
        let dir = workdir();
        let result = BashTool::new(dir.path()).run(&json!({"command": "sleep 5", "timeout": 1}));
        assert!(result.is_error);
        assert!(result.output.contains("timed out"));
    }

    #[test]
    fn bash_output_is_capped() {
        let dir = workdir();
        let result = BashTool::new(dir.path()).run(&json!({"command": "yes | head -c 1000000"}));
        assert!(result.output.len() <= BashTool::MAX_OUTPUT_CHARS + 100);
        assert!(result.output.contains("[truncated]"));
    }

    #[test]
    fn missing_required_argument_is_error() {
        let dir = workdir();
        let result = ReadTool::new(dir.path()).run(&json!({}));
        assert!(result.is_error);
        assert!(result.output.contains("path"));
    }
}
