//! Evolutionary refinement: generate candidates, verify them
//! deterministically, feed scores and failure feedback back to the model,
//! and evolve the best candidates across rounds.
//!
//! This is the harness pattern behind the strongest ARC-AGI results
//! (Greenblatt's sample-and-verify, Berman's evolutionary test-time compute,
//! Pang's efficient evolutionary program synthesis), generalized: it works
//! for anything with a deterministic verifier.

use crate::providers::{Message, Provider, ProviderError};

/// Deterministic judgment of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// Primary score in `0.0..=1.0`; `1.0` means solved.
    pub score: f64,
    /// Tie-breaker for equal primary scores (e.g. cell-level accuracy).
    pub secondary: f64,
    /// Shown to the model when this candidate seeds the next round.
    pub feedback: String,
}

pub trait Verifier {
    fn verify(&self, candidate: &str) -> Verdict;
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub content: String,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Copy)]
pub struct RefineConfig {
    /// Evolution rounds (each seeds prompts with the previous best).
    pub rounds: usize,
    /// Candidates sampled per round.
    pub per_round: usize,
    /// How many top candidates seed the next round's prompt.
    pub keep_top: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            rounds: 3,
            per_round: 5,
            keep_top: 3,
        }
    }
}

/// Progress notifications for the UI layer.
pub enum RefineEvent<'a> {
    Sampled(&'a Candidate),
    RoundDone { round: usize, best_score: f64 },
}

/// Run the generate–verify–revise loop. Returns all candidates, best first.
/// Stops early as soon as a candidate scores `1.0`.
pub fn refine(
    provider: &dyn Provider,
    base_prompt: &str,
    verifier: &dyn Verifier,
    config: RefineConfig,
    mut on_event: impl FnMut(RefineEvent),
) -> Result<Vec<Candidate>, ProviderError> {
    let mut pool: Vec<Candidate> = Vec::new();
    for round in 0..config.rounds {
        let prompt = build_prompt(base_prompt, &pool, config.keep_top);
        for _ in 0..config.per_round {
            let reply = provider.complete("", &[Message::user(prompt.clone())], &[])?;
            let content = extract_code(&reply.content);
            let verdict = verifier.verify(&content);
            let solved = verdict.score >= 1.0;
            let candidate = Candidate { content, verdict };
            on_event(RefineEvent::Sampled(&candidate));
            pool.push(candidate);
            if solved {
                rank(&mut pool);
                return Ok(pool);
            }
        }
        rank(&mut pool);
        on_event(RefineEvent::RoundDone {
            round,
            best_score: pool.first().map_or(0.0, |c| c.verdict.score),
        });
    }
    Ok(pool)
}

fn rank(pool: &mut [Candidate]) {
    pool.sort_by(|a, b| {
        (b.verdict.score, b.verdict.secondary)
            .partial_cmp(&(a.verdict.score, a.verdict.secondary))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// First round: the task alone. Later rounds: the task plus the strongest
/// prior attempts with their scores and failure feedback (the evolutionary
/// step — the model revises rather than starting over).
fn build_prompt(base: &str, pool: &[Candidate], keep_top: usize) -> String {
    if pool.is_empty() {
        return base.to_string();
    }
    let mut prompt = format!(
        "{base}\n\nPrevious attempts, best first. Study why they fail and \
         produce a better solution — revise the strongest attempt or take a \
         new approach if they are all wrong.\n"
    );
    for (i, candidate) in pool.iter().take(keep_top).enumerate() {
        prompt.push_str(&format!(
            "\n--- Attempt {} (score {:.2}) ---\n{}\nFeedback:\n{}\n",
            i + 1,
            candidate.verdict.score,
            candidate.content,
            candidate.verdict.feedback,
        ));
    }
    prompt
}

/// Pull the last fenced code block out of a reply; models wrap code in
/// markdown fences and often lead with reasoning. Falls back to the whole
/// reply when there is no fence.
pub fn extract_code(reply: &str) -> String {
    let mut blocks = Vec::new();
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after_fence = &rest[start + 3..];
        let body_start = after_fence.find('\n').map_or(after_fence.len(), |i| i + 1);
        let Some(end) = after_fence[body_start..].find("```") else {
            break;
        };
        blocks.push(after_fence[body_start..body_start + end].trim().to_string());
        rest = &after_fence[body_start + end + 3..];
    }
    blocks.pop().unwrap_or_else(|| reply.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::test_support::ScriptedProvider;

    /// The prompt sent on the i-th model call.
    fn prompt(provider: &ScriptedProvider, i: usize) -> String {
        provider.calls.borrow()[i][0].content.clone()
    }

    /// Scores a candidate by exact match against a target string.
    struct ExactVerifier(&'static str);

    impl Verifier for ExactVerifier {
        fn verify(&self, candidate: &str) -> Verdict {
            if candidate == self.0 {
                Verdict {
                    score: 1.0,
                    secondary: 1.0,
                    feedback: "correct".into(),
                }
            } else {
                Verdict {
                    score: 0.0,
                    secondary: candidate.len().min(10) as f64 / 10.0,
                    feedback: format!("expected `{}`", self.0),
                }
            }
        }
    }

    #[test]
    fn stops_early_when_a_candidate_solves_the_task() {
        let provider = ScriptedProvider::texts(&["wrong", "right", "never sampled"]);
        let config = RefineConfig {
            rounds: 3,
            per_round: 2,
            keep_top: 2,
        };
        let pool = refine(&provider, "task", &ExactVerifier("right"), config, |_| {}).unwrap();
        assert_eq!(pool[0].content, "right");
        assert_eq!(pool[0].verdict.score, 1.0);
        assert_eq!(pool.len(), 2); // third reply never requested
    }

    #[test]
    fn later_rounds_see_prior_attempts_and_feedback() {
        let provider = ScriptedProvider::texts(&["alpha", "beta", "gamma", "delta"]);
        let config = RefineConfig {
            rounds: 2,
            per_round: 2,
            keep_top: 1,
        };
        refine(&provider, "task", &ExactVerifier("zzz"), config, |_| {}).unwrap();

        assert_eq!(prompt(&provider, 0), "task"); // round one: base prompt only
        assert_eq!(prompt(&provider, 1), "task");
        // round two: includes the best prior attempt and its feedback
        assert!(prompt(&provider, 2).contains("Previous attempts"));
        assert!(prompt(&provider, 2).contains("expected `zzz`"));
        // keep_top=1: only the single best attempt is included
        assert_eq!(prompt(&provider, 2).matches("--- Attempt").count(), 1);
    }

    #[test]
    fn pool_is_ranked_by_score_then_secondary() {
        // "longlonglong" ties "beta" on score 0.0 but wins on secondary.
        let provider = ScriptedProvider::texts(&["ab", "longlonglong"]);
        let config = RefineConfig {
            rounds: 1,
            per_round: 2,
            keep_top: 2,
        };
        let pool = refine(&provider, "task", &ExactVerifier("zzz"), config, |_| {}).unwrap();
        assert_eq!(pool[0].content, "longlonglong");
    }

    #[test]
    fn events_report_samples_and_rounds() {
        let provider = ScriptedProvider::texts(&["a", "b"]);
        let config = RefineConfig {
            rounds: 2,
            per_round: 1,
            keep_top: 1,
        };
        let mut log = Vec::new();
        refine(&provider, "task", &ExactVerifier("zzz"), config, |event| {
            log.push(match event {
                RefineEvent::Sampled(c) => format!("sample:{}", c.content),
                RefineEvent::RoundDone { round, best_score } => {
                    format!("round:{round}:{best_score}")
                }
            });
        })
        .unwrap();
        assert_eq!(log, vec!["sample:a", "round:0:0", "sample:b", "round:1:0"]);
    }

    #[test]
    fn provider_errors_propagate() {
        let provider = ScriptedProvider::texts(&[]);
        let config = RefineConfig::default();
        assert!(refine(&provider, "task", &ExactVerifier("x"), config, |_| {}).is_err());
    }

    #[test]
    fn extract_code_takes_the_last_fenced_block() {
        let reply = "Reasoning first.\n```python\nold = 1\n```\nBetter:\n```python\nnew = 2\n```";
        assert_eq!(extract_code(reply), "new = 2");
    }

    #[test]
    fn extract_code_falls_back_to_whole_reply() {
        assert_eq!(extract_code("  plain code  "), "plain code");
    }

    #[test]
    fn extract_code_handles_unterminated_fence() {
        assert_eq!(
            extract_code("```python\nno closing fence"),
            "```python\nno closing fence"
        );
    }
}
