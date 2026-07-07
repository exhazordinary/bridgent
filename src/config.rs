//! Configuration from environment variables and CLI flags.
//!
//! Env vars: `BRIDGENT_PROVIDER` (anthropic | openai), `BRIDGENT_MODEL`,
//! `BRIDGENT_BASE_URL`, and the provider key (`ANTHROPIC_API_KEY` or
//! `OPENAI_API_KEY`). Flags override env; env overrides defaults.

use crate::providers::{AnthropicProvider, OpenAIProvider, Provider, RetryingProvider};

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

    /// Env var carrying a bearer token that replaces the API key, if the
    /// provider supports one.
    fn auth_token_var(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("ANTHROPIC_AUTH_TOKEN"),
            Self::OpenAI => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub provider: ProviderKind,
    pub model: String,
    pub api_key: String,
    /// Bearer token (`ANTHROPIC_AUTH_TOKEN`) for gateways and other
    /// OAuth-issued credentials; takes precedence over the API key.
    pub auth_token: Option<String>,
    pub base_url: Option<String>,
    /// Max output tokens per response; falls back to the provider default.
    pub max_tokens: Option<u32>,
}

impl Config {
    /// Resolve config from the process environment and parsed CLI flags.
    pub fn from_env(flags: &crate::cli::ProviderFlags) -> Result<Self, String> {
        Self::resolve(|key| std::env::var(key).ok(), flags)
    }

    /// Resolve config from an environment lookup (injectable for tests) and
    /// flag overrides.
    pub fn resolve(
        env: impl Fn(&str) -> Option<String>,
        flags: &crate::cli::ProviderFlags,
    ) -> Result<Self, String> {
        let provider_name = flags
            .provider
            .clone()
            .or_else(|| env("BRIDGENT_PROVIDER"))
            .unwrap_or_else(|| "anthropic".into());
        let provider = ProviderKind::parse(&provider_name)?;
        let model = flags
            .model
            .clone()
            .or_else(|| env("BRIDGENT_MODEL"))
            .unwrap_or_else(|| provider.default_model().into());
        let base_url = flags.base_url.clone().or_else(|| env("BRIDGENT_BASE_URL"));
        let max_tokens = flags
            .max_tokens
            .clone()
            .or_else(|| env("BRIDGENT_MAX_TOKENS"))
            .map(|value| {
                value
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| format!("invalid max tokens '{value}' (expected an integer)"))
            })
            .transpose()?;
        let auth_token = provider.auth_token_var().and_then(&env);
        // Local OpenAI-compatible servers (ollama, vllm) don't need a real
        // key, and a bearer token replaces the API key entirely.
        let api_key = match env(provider.key_var()) {
            Some(key) => key,
            None if base_url.is_some() || auth_token.is_some() => String::new(),
            None => {
                let hint = provider
                    .auth_token_var()
                    .map(|var| format!(" ({var} also works)"))
                    .unwrap_or_default();
                return Err(format!("{} is not set{hint}", provider.key_var()));
            }
        };
        Ok(Self {
            provider,
            model,
            api_key,
            auth_token,
            base_url,
            max_tokens,
        })
    }

    /// The configured provider, wrapped with transient-error retry.
    pub fn build_provider(&self) -> Box<dyn Provider> {
        let inner: Box<dyn Provider> = match self.provider {
            ProviderKind::Anthropic => {
                let mut p = AnthropicProvider::new(&self.api_key, &self.model);
                p.auth_token = self.auth_token.clone();
                if let Some(url) = &self.base_url {
                    p.base_url = url.clone();
                }
                if let Some(max_tokens) = self.max_tokens {
                    p.max_tokens = max_tokens;
                }
                Box::new(p)
            }
            ProviderKind::OpenAI => {
                let mut p = OpenAIProvider::new(&self.api_key, &self.model);
                if let Some(url) = &self.base_url {
                    p.base_url = url.clone();
                }
                p.max_tokens = self.max_tokens;
                Box::new(p)
            }
        };
        Box::new(RetryingProvider::new(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ProviderFlags;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| map.get(key).cloned()
    }

    fn flags(provider: Option<&str>, model: Option<&str>, base_url: Option<&str>) -> ProviderFlags {
        ProviderFlags {
            provider: provider.map(String::from),
            model: model.map(String::from),
            base_url: base_url.map(String::from),
            max_tokens: None,
        }
    }

    #[test]
    fn defaults_to_anthropic_with_key_from_env() {
        let config = Config::resolve(
            env_of(&[("ANTHROPIC_API_KEY", "sk-ant")]),
            &flags(None, None, None),
        )
        .unwrap();
        assert_eq!(config.provider, ProviderKind::Anthropic);
        assert_eq!(config.model, "claude-sonnet-4-6");
        assert_eq!(config.api_key, "sk-ant");
        assert_eq!(config.base_url, None);
        assert_eq!(config.max_tokens, None);
    }

    #[test]
    fn missing_api_key_is_a_clear_error() {
        let error = Config::resolve(env_of(&[]), &flags(None, None, None)).unwrap_err();
        assert!(error.contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn flags_override_env() {
        let env = env_of(&[
            ("BRIDGENT_PROVIDER", "anthropic"),
            ("BRIDGENT_MODEL", "env-model"),
            ("OPENAI_API_KEY", "sk-oai"),
        ]);
        let config =
            Config::resolve(env, &flags(Some("openai"), Some("flag-model"), None)).unwrap();
        assert_eq!(config.provider, ProviderKind::OpenAI);
        assert_eq!(config.model, "flag-model");
        assert_eq!(config.api_key, "sk-oai");
    }

    #[test]
    fn openai_env_provider_gets_openai_default_model() {
        let env = env_of(&[("BRIDGENT_PROVIDER", "openai"), ("OPENAI_API_KEY", "k")]);
        let config = Config::resolve(env, &flags(None, None, None)).unwrap();
        assert_eq!(config.model, "gpt-5.2");
    }

    #[test]
    fn local_base_url_needs_no_api_key() {
        let config = Config::resolve(
            env_of(&[]),
            &flags(
                Some("openai"),
                Some("qwen3"),
                Some("http://localhost:11434/v1"),
            ),
        )
        .unwrap();
        assert_eq!(config.api_key, "");
        assert_eq!(
            config.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn bearer_token_replaces_api_key_for_anthropic() {
        let config = Config::resolve(
            env_of(&[("ANTHROPIC_AUTH_TOKEN", "oauth-tok")]),
            &flags(None, None, None),
        )
        .unwrap();
        assert_eq!(config.auth_token.as_deref(), Some("oauth-tok"));
        assert_eq!(config.api_key, "");
    }

    #[test]
    fn bearer_token_is_ignored_for_openai() {
        let env = env_of(&[("ANTHROPIC_AUTH_TOKEN", "tok"), ("OPENAI_API_KEY", "k")]);
        let config = Config::resolve(env, &flags(Some("openai"), None, None)).unwrap();
        assert_eq!(config.auth_token, None);
    }

    #[test]
    fn unknown_provider_is_rejected() {
        let error = Config::resolve(env_of(&[]), &flags(Some("gemini"), None, None)).unwrap_err();
        assert!(error.contains("gemini"));
    }

    #[test]
    fn max_tokens_resolves_from_flag_or_env_and_rejects_garbage() {
        let env = env_of(&[("ANTHROPIC_API_KEY", "k"), ("BRIDGENT_MAX_TOKENS", "16000")]);
        let config = Config::resolve(&env, &flags(None, None, None)).unwrap();
        assert_eq!(config.max_tokens, Some(16000));

        let mut with_flag = flags(None, None, None);
        with_flag.max_tokens = Some("32000".into());
        let config = Config::resolve(&env, &with_flag).unwrap();
        assert_eq!(config.max_tokens, Some(32000)); // flag beats env

        with_flag.max_tokens = Some("lots".into());
        let error = Config::resolve(&env, &with_flag).unwrap_err();
        assert!(error.contains("lots"));
    }
}
