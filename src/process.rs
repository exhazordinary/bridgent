//! Shared subprocess execution with a hard timeout.
//!
//! Pipes are drained on background threads so a child producing more output
//! than the OS pipe buffer can never deadlock against the timeout wait.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use wait_timeout::ChildExt;

#[derive(Debug)]
pub struct ProcessOutput {
    pub stdout: String,
    pub stderr: String,
    /// Exit code; `None` when the child was killed by a signal.
    pub exit_code: Option<i32>,
}

#[derive(Debug, PartialEq)]
pub enum ProcessError {
    Spawn(String),
    TimedOut(Duration),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "cannot spawn command: {e}"),
            Self::TimedOut(t) => write!(f, "command timed out after {}s", t.as_secs()),
        }
    }
}

/// Run `command` to completion, optionally feeding `stdin_data`, killing it
/// after `timeout`.
pub fn run_with_timeout(
    command: &mut Command,
    stdin_data: Option<&[u8]>,
    timeout: Duration,
) -> Result<ProcessOutput, ProcessError> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
    let mut child = command
        .spawn()
        .map_err(|e| ProcessError::Spawn(e.to_string()))?;

    if let (Some(data), Some(mut stdin)) = (stdin_data, child.stdin.take()) {
        // Ignore broken pipes: the child may exit before reading everything.
        let _ = stdin.write_all(data);
    }

    let drain = |pipe: Option<Box<dyn std::io::Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_string(&mut buf);
            }
            buf
        })
    };
    let stdout_thread = drain(child.stdout.take().map(|p| Box::new(p) as _));
    let stderr_thread = drain(child.stderr.take().map(|p| Box::new(p) as _));

    let status = match child.wait_timeout(timeout) {
        Ok(Some(status)) => status,
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ProcessError::TimedOut(timeout));
        }
        Err(e) => return Err(ProcessError::Spawn(e.to_string())),
    };
    Ok(ProcessOutput {
        stdout: stdout_thread.join().unwrap_or_default(),
        stderr: stderr_thread.join().unwrap_or_default(),
        exit_code: status.code(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_stdout_stderr_and_exit_code() {
        let out = run_with_timeout(
            Command::new("sh")
                .arg("-c")
                .arg("echo out; echo err >&2; exit 3"),
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.stdout, "out\n");
        assert_eq!(out.stderr, "err\n");
        assert_eq!(out.exit_code, Some(3));
    }

    #[test]
    fn feeds_stdin() {
        let out = run_with_timeout(
            &mut Command::new("cat"),
            Some(b"hello stdin"),
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.stdout, "hello stdin");
    }

    #[test]
    fn times_out_and_kills() {
        let err = run_with_timeout(Command::new("sleep").arg("5"), None, Duration::from_secs(1))
            .unwrap_err();
        assert_eq!(err, ProcessError::TimedOut(Duration::from_secs(1)));
    }

    #[test]
    fn large_output_does_not_deadlock() {
        let out = run_with_timeout(
            Command::new("sh").arg("-c").arg("yes | head -c 1000000"),
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.stdout.len(), 1_000_000);
    }
}
