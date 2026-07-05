//! System prompt construction.
//!
//! Frontier models are RL-trained as coding agents; they don't need pages of
//! instructions. The base prompt stays under 200 words and project-specific
//! guidance comes from the project's own AGENTS.md (or CLAUDE.md) files —
//! nothing is injected that the user can't see on disk.

use std::path::{Path, PathBuf};

const BASE_PROMPT: &str = "\
You are a coding agent operating in the user's working directory.

Rules:
- Prefer edit over write for existing files; read a file before editing it.
- Use bash for search (grep, find), git, builds, and tests.
- Verify your work: run the relevant build or test command before declaring \
a task done, and report failures honestly.
- Keep answers short. Don't narrate tool output the user can already see.
- Never revert or destroy user changes you didn't make.";

/// Project guidance files, in priority order. Only the first match per
/// directory is used.
const PROJECT_FILES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// Build the system prompt for a session rooted at `workdir`.
pub fn system_prompt(workdir: &Path) -> String {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    system_prompt_with(workdir, home.as_deref())
}

/// Testable core: `home` is injectable so tests don't read the real user's
/// global guidance.
fn system_prompt_with(workdir: &Path, home: Option<&Path>) -> String {
    let mut prompt = format!(
        "{BASE_PROMPT}\n\nWorking directory: {}\nPlatform: {}",
        workdir.display(),
        std::env::consts::OS,
    );
    // Broadest first, so more specific guidance can override it: global
    // (~/.config/bridgent/), then repo root down to the working directory.
    let global = home.map(|h| h.join(".config/bridgent"));
    for dir in global
        .iter()
        .map(PathBuf::as_path)
        .chain(guidance_dirs(workdir))
    {
        if let Some((path, content)) = guidance_in(dir) {
            prompt.push_str(&format!(
                "\n\nProject instructions from {}:\n{content}",
                path.display()
            ));
        }
    }
    prompt
}

/// Directories that may hold guidance: the git repo root (or filesystem
/// walk-up limit) down to `workdir` itself, outermost first.
fn guidance_dirs(workdir: &Path) -> impl Iterator<Item = &Path> {
    let mut dirs: Vec<&Path> = Vec::new();
    let mut dir = workdir;
    loop {
        dirs.push(dir);
        // The repo root is the outermost directory worth reading; stop there.
        if dir.join(".git").exists() {
            break;
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    dirs.into_iter().rev()
}

fn guidance_in(dir: &Path) -> Option<(PathBuf, String)> {
    PROJECT_FILES.iter().find_map(|name| {
        let path = dir.join(name);
        std::fs::read_to_string(&path)
            .ok()
            .map(|content| (path, content.trim().to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn prompt_no_home(workdir: &Path) -> String {
        system_prompt_with(workdir, None)
    }

    #[test]
    fn base_prompt_includes_workdir_and_stays_minimal() {
        let dir = TempDir::new().unwrap();
        let prompt = prompt_no_home(dir.path());
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
        let prompt = prompt_no_home(dir.path());
        assert!(prompt.contains("Project instructions from"));
        assert!(prompt.contains("Always run cargo fmt."));
    }

    #[test]
    fn claude_md_is_fallback_and_agents_md_wins() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude rules").unwrap();
        assert!(prompt_no_home(dir.path()).contains("claude rules"));

        std::fs::write(dir.path().join("AGENTS.md"), "agents rules").unwrap();
        let prompt = prompt_no_home(dir.path());
        assert!(prompt.contains("agents rules"));
        assert!(!prompt.contains("claude rules"));
    }

    #[test]
    fn guidance_walks_up_to_the_git_root() {
        let repo = TempDir::new().unwrap();
        std::fs::create_dir_all(repo.path().join(".git")).unwrap();
        std::fs::write(repo.path().join("AGENTS.md"), "root rules").unwrap();
        let sub = repo.path().join("crates/web");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("AGENTS.md"), "web rules").unwrap();

        let prompt = prompt_no_home(&sub);
        // Both levels included, root first so the nearer file can override.
        let root_at = prompt.find("root rules").unwrap();
        let web_at = prompt.find("web rules").unwrap();
        assert!(root_at < web_at);
    }

    #[test]
    fn walk_up_stops_at_the_git_root() {
        let outer = TempDir::new().unwrap();
        std::fs::write(outer.path().join("AGENTS.md"), "outside the repo").unwrap();
        let repo = outer.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();

        let prompt = prompt_no_home(&repo);
        assert!(!prompt.contains("outside the repo"));
    }

    #[test]
    fn global_guidance_comes_before_project_guidance() {
        let home = TempDir::new().unwrap();
        std::fs::create_dir_all(home.path().join(".config/bridgent")).unwrap();
        std::fs::write(
            home.path().join(".config/bridgent/AGENTS.md"),
            "global rules",
        )
        .unwrap();
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "project rules").unwrap();

        let prompt = system_prompt_with(dir.path(), Some(home.path()));
        let global_at = prompt.find("global rules").unwrap();
        let project_at = prompt.find("project rules").unwrap();
        assert!(global_at < project_at);
    }
}
