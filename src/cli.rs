//! Shared command-line plumbing for the bridgent binaries: flag parsing
//! helpers, the provider-selection flags every binary accepts, and the
//! standard exit wrapper.

use std::process::ExitCode;

/// Take the value following a flag, erroring with the flag's name.
pub fn flag_value(iter: &mut std::slice::Iter<String>, name: &str) -> Result<String, String> {
    iter.next()
        .cloned()
        .ok_or_else(|| format!("{name} requires a value"))
}

/// Provider selection flags (`--provider`, `--model`, `--base-url`,
/// `--max-tokens`) shared by every binary, in the shape `Config::from_env`
/// consumes.
#[derive(Debug, Default)]
pub struct ProviderFlags {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub max_tokens: Option<String>,
}

impl ProviderFlags {
    /// Consume `arg` if it is a provider flag; returns whether it was.
    pub fn parse(
        &mut self,
        arg: &str,
        iter: &mut std::slice::Iter<String>,
    ) -> Result<bool, String> {
        match arg {
            "--provider" => self.provider = Some(flag_value(iter, "--provider")?),
            "--model" => self.model = Some(flag_value(iter, "--model")?),
            "--base-url" => self.base_url = Some(flag_value(iter, "--base-url")?),
            "--max-tokens" => self.max_tokens = Some(flag_value(iter, "--max-tokens")?),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Standard binary exit: errors print as `name: message` and fail.
pub fn exit(name: &str, result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{name}: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn provider_flags_consume_their_flags_and_skip_others() {
        let args = argv(&["--model", "m1", "--verbose", "--provider", "openai"]);
        let mut iter = args.iter();
        let mut flags = ProviderFlags::default();
        assert!(flags.parse(iter.next().unwrap(), &mut iter).unwrap());
        assert!(!flags.parse(iter.next().unwrap(), &mut iter).unwrap());
        assert!(flags.parse(iter.next().unwrap(), &mut iter).unwrap());
        assert_eq!(flags.model.as_deref(), Some("m1"));
        assert_eq!(flags.provider.as_deref(), Some("openai"));
        assert_eq!(flags.base_url, None);
    }

    #[test]
    fn missing_flag_value_names_the_flag() {
        let args = argv(&["--model"]);
        let mut iter = args.iter();
        let mut flags = ProviderFlags::default();
        let error = flags.parse(iter.next().unwrap(), &mut iter).unwrap_err();
        assert!(error.contains("--model"));
    }
}
