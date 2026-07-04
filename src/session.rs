//! Persistent sessions: append-only JSONL, one message per line.
//!
//! Sessions live in `.bridle/sessions/` under the working directory. The
//! format is trivially greppable and survives crashes — every message is
//! flushed as soon as it is appended.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::providers::Message;

pub struct Session {
    pub path: PathBuf,
    pub messages: Vec<Message>,
}

impl Session {
    /// Start a fresh session under `workdir/.bridle/sessions/`.
    pub fn create(workdir: &Path) -> std::io::Result<Self> {
        let dir = sessions_dir(workdir);
        fs::create_dir_all(&dir)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_millis();
        let path = dir.join(format!("{stamp}.jsonl"));
        File::create(&path)?;
        Ok(Self {
            path,
            messages: Vec::new(),
        })
    }

    /// Load an existing session file, skipping lines that fail to parse so
    /// a corrupt tail never blocks a resume.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        let messages = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            messages,
        })
    }

    /// Resume the most recent session in `workdir`, if any exists.
    pub fn latest(workdir: &Path) -> Option<std::io::Result<Self>> {
        let mut paths: Vec<PathBuf> = fs::read_dir(sessions_dir(workdir))
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        paths.sort();
        paths.pop().map(|path| Self::open(&path))
    }

    /// Append a message to memory and disk (flushed immediately).
    pub fn append(&mut self, message: Message) -> std::io::Result<()> {
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        serde_json::to_writer(&mut file, &message)?;
        file.write_all(b"\n")?;
        self.messages.push(message);
        Ok(())
    }
}

fn sessions_dir(workdir: &Path) -> PathBuf {
    workdir.join(".bridle").join("sessions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{Message, ToolCall};
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn create_append_open_round_trips() {
        let dir = TempDir::new().unwrap();
        let mut session = Session::create(dir.path()).unwrap();
        session.append(Message::user("hi")).unwrap();
        session
            .append(Message::assistant(
                "reading",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "read".into(),
                    args: json!({"path": "a"}),
                }],
            ))
            .unwrap();
        session
            .append(Message::tool_result("t1", "data", false))
            .unwrap();

        let reopened = Session::open(&session.path).unwrap();
        assert_eq!(reopened.messages, session.messages);
        assert_eq!(reopened.messages.len(), 3);
    }

    #[test]
    fn open_skips_corrupt_lines() {
        let dir = TempDir::new().unwrap();
        let mut session = Session::create(dir.path()).unwrap();
        session.append(Message::user("hi")).unwrap();
        fs::write(
            &session.path,
            format!(
                "{}\nnot json at all\n",
                serde_json::to_string(&Message::user("hi")).unwrap()
            ),
        )
        .unwrap();
        let reopened = Session::open(&session.path).unwrap();
        assert_eq!(reopened.messages.len(), 1);
    }

    #[test]
    fn latest_returns_most_recent_session() {
        let dir = TempDir::new().unwrap();
        assert!(Session::latest(dir.path()).is_none());

        let mut first = Session::create(dir.path()).unwrap();
        first.append(Message::user("first")).unwrap();
        // Session files are named by millisecond timestamp; ensure distinct names.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let mut second = Session::create(dir.path()).unwrap();
        second.append(Message::user("second")).unwrap();

        let latest = Session::latest(dir.path()).unwrap().unwrap();
        assert_eq!(latest.messages[0].content, "second");
    }
}
