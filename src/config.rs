//! Configuration from environment variables and CLI flags.
//!
//! Env vars: `BRIDLE_PROVIDER` (anthropic | openai), `BRIDLE_MODEL`,
//! `BRIDLE_BASE_URL`, and the provider key (`ANTHROPIC_API_KEY` or
//! `OPENAI_API_KEY`). Flags override env; env overrides defaults.

use crate::providers::{AnthropicProvider, OpenAIProvider, Provider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAI,
}

impl ProviderKind {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAI),
            other => Err(format!(
                "unknown provider '{other}' (expected anthropic or openai)"
            )),
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-6",
            Self::OpenAI => "gpt-5.2",
        }
    }

    fn key_var(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAI => "OPENAI_API_KEY",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub provider: ProviderKind,
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
}

impl Config {
    /// Resolve config from an environment lookup (injectable for tests) and
    /// optional flag overrides.
    pub fn resolve(
        env: impl Fn(&str) -> Option<String>,
        provider_flag: Option<&str>,
        model_flag: Option<&str>,
        base_url_flag: Option<&str>,
    ) -> Result<Self, String> {
        let provider_name = provider_flag
            .map(str::to_string)
            .or_else(|| env("BRIDLE_PROVIDER"))
            .unwrap_or_else(|| "anthropic".into());
        let provider = ProviderKind::parse(&provider_name)?;
        let model = model_flag
            .map(str::to_string)
            .or_else(|| env("BRIDLE_MODEL"))
            .unwrap_or_else(|| provider.default_model().into());
        let base_url = base_url_flag
            .map(str::to_string)
            .or_else(|| env("BRIDLE_BASE_URL"));
        // Local OpenAI-compatible servers (ollama, vllm) don't need a real key.
        let api_key = match env(provider.key_var()) {
            Some(key) => key,
            None if base_url.is_some() => String::new(),
            None => return Err(format!("{} is not set", provider.key_var())),
        };
        Ok(Self {
            provider,
            model,
            api_key,
            base_url,
        })
    }

    pub fn build_provider(&self) -> Box<dyn Provider> {
        match self.provider {
            ProviderKind::Anthropic => {
                let mut p = AnthropicProvider::new(&self.api_key, &self.model);
                if let Some(url) = &self.base_url {
                    p.base_url = url.clone();
                }
                Box::new(p)
            }
            ProviderKind::OpenAI => {
                let mut p = OpenAIProvider::new(&self.api_key, &self.model);
                if let Some(url) = &self.base_url {
                    p.base_url = url.clone();
                }
                Box::new(p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn defaults_to_anthropic_with_key_from_env() {
        let config =
            Config::resolve(env_of(&[("ANTHROPIC_API_KEY", "sk-ant")]), None, None, None).unwrap();
        assert_eq!(config.provider, ProviderKind::Anthropic);
        assert_eq!(config.model, "claude-sonnet-4-6");
        assert_eq!(config.api_key, "sk-ant");
        assert_eq!(config.base_url, None);
    }

    #[test]
    fn missing_api_key_is_a_clear_error() {
        let error = Config::resolve(env_of(&[]), None, None, None).unwrap_err();
        assert!(error.contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn flags_override_env() {
        let env = env_of(&[
            ("BRIDLE_PROVIDER", "anthropic"),
            ("BRIDLE_MODEL", "env-model"),
            ("OPENAI_API_KEY", "sk-oai"),
        ]);
        let config = Config::resolve(env, Some("openai"), Some("flag-model"), None).unwrap();
        assert_eq!(config.provider, ProviderKind::OpenAI);
        assert_eq!(config.model, "flag-model");
        assert_eq!(config.api_key, "sk-oai");
    }

    #[test]
    fn openai_env_provider_gets_openai_default_model() {
        let env = env_of(&[("BRIDLE_PROVIDER", "openai"), ("OPENAI_API_KEY", "k")]);
        let config = Config::resolve(env, None, None, None).unwrap();
        assert_eq!(config.model, "gpt-5.2");
    }

    #[test]
    fn local_base_url_needs_no_api_key() {
        let config = Config::resolve(
            env_of(&[]),
            Some("openai"),
            Some("qwen3"),
            Some("http://localhost:11434/v1"),
        )
        .unwrap();
        assert_eq!(config.api_key, "");
        assert_eq!(
            config.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn unknown_provider_is_rejected() {
        let error = Config::resolve(env_of(&[]), Some("gemini"), None, None).unwrap_err();
        assert!(error.contains("gemini"));
    }
}
