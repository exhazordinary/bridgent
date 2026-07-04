//! System prompt construction.
//!
//! Frontier models are RL-trained as coding agents; they don't need pages of
//! instructions. The base prompt stays under 200 words and project-specific
//! guidance comes from the project's own AGENTS.md (or CLAUDE.md) file —
//! nothing is injected that the user can't see in their own repo.

use std::path::Path;

const BASE_PROMPT: &str = "\
You are a coding agent operating in the user's working directory.

Rules:
- Prefer edit over write for existing files; read a file before editing it.
- Use bash for search (grep, find), git, builds, and tests.
- Verify your work: run the relevant build or test command before declaring \
a task done, and report failures honestly.
- Keep answers short. Don't narrate tool output the user can already see.
- Never revert or destroy user changes you didn't make.";

/// Project guidance files, in priority order. Only the first match is used.
const PROJECT_FILES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// Build the system prompt for a session rooted at `workdir`.
pub fn system_prompt(workdir: &Path) -> String {
    let mut prompt = format!(
        "{BASE_PROMPT}\n\nWorking directory: {}\nPlatform: {}",
        workdir.display(),
        std::env::consts::OS,
    );
    if let Some((name, content)) = project_guidance(workdir) {
        prompt.push_str(&format!("\n\nProject instructions from {name}:\n{content}"));
    }
    prompt
}

fn project_guidance(workdir: &Path) -> Option<(&'static str, String)> {
    PROJECT_FILES.iter().find_map(|name| {
        std::fs::read_to_string(workdir.join(name))
            .ok()
            .map(|content| (*name, content.trim().to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn base_prompt_includes_workdir_and_stays_minimal() {
        let dir = TempDir::new().unwrap();
        let prompt = system_prompt(dir.path());
        assert!(prompt.contains(&dir.path().display().to_string()));
        // The whole point: a lean prompt. Fail loudly if it bloats.
        assert!(
            prompt.len() < 1500,
            "base prompt grew to {} chars",
            prompt.len()
        );
    }

    #[test]
    fn agents_md_is_appended_when_present() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "Always run cargo fmt.\n").unwrap();
        let prompt = system_prompt(dir.path());
        assert!(prompt.contains("Project instructions from AGENTS.md"));
        assert!(prompt.contains("Always run cargo fmt."));
    }

    #[test]
    fn claude_md_is_fallback_and_agents_md_wins() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude rules").unwrap();
        assert!(system_prompt(dir.path()).contains("claude rules"));

        std::fs::write(dir.path().join("AGENTS.md"), "agents rules").unwrap();
        let prompt = system_prompt(dir.path());
        assert!(prompt.contains("agents rules"));
        assert!(!prompt.contains("claude rules"));
    }
}
