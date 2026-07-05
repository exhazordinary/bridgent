//! ARC-AGI solver harness: evolve Python `transform` programs against a
//! task's training pairs, then apply the winner to the test inputs.
//!
//! The design follows the strongest published ARC approaches: sample
//! candidate programs, execute them against the demonstrations (the
//! deterministic verifier), and feed exact failure diffs back for revision
//! via the generic `refine` engine.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;

use crate::process::run_with_timeout;
use crate::providers::Provider;
use crate::refine::{refine, Candidate, RefineConfig, RefineEvent, Verdict, Verifier};

pub type Grid = Vec<Vec<u8>>;

#[derive(Debug, Clone, Deserialize)]
pub struct Pair {
    pub input: Grid,
    /// Absent on hidden-evaluation test pairs.
    #[serde(default)]
    pub output: Option<Grid>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArcTask {
    pub train: Vec<Pair>,
    pub test: Vec<Pair>,
}

pub fn load_task(path: &Path) -> Result<ArcTask, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("invalid ARC task JSON: {e}"))
}

fn render_grid(grid: &Grid) -> String {
    let rows: Vec<String> = grid.iter().map(|row| format!("{row:?}")).collect();
    format!(
        "{}x{}\n{}",
        grid.len(),
        grid.first().map_or(0, Vec::len),
        rows.join("\n")
    )
}

/// The task prompt: every training pair, rendered as dimensions plus rows,
/// with instructions to return one Python `transform` function.
pub fn base_prompt(task: &ArcTask) -> String {
    let mut prompt = String::from(
        "You are solving an ARC-AGI puzzle. Each training pair shows an input \
         grid transformed into an output grid by one consistent rule. Cells \
         are integers 0-9 (colors). Infer the rule.\n",
    );
    for (i, pair) in task.train.iter().enumerate() {
        prompt.push_str(&format!("\n## Training pair {}\nInput ", i + 1));
        prompt.push_str(&render_grid(&pair.input));
        if let Some(output) = &pair.output {
            prompt.push_str("\nOutput ");
            prompt.push_str(&render_grid(output));
        }
        prompt.push('\n');
    }
    prompt.push_str(
        "\nReason about the rule first, then write a Python function\n\n\
         def transform(grid: list[list[int]]) -> list[list[int]]:\n\n\
         implementing it. Standard library only. Return exactly one fenced \
         python code block containing the complete function.",
    );
    prompt
}

/// Executes a candidate program against a list of input grids.
pub struct PythonRunner {
    pub timeout: Duration,
}

impl Default for PythonRunner {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
        }
    }
}

impl PythonRunner {
    /// Run `code`'s `transform` over every input; returns one grid per input.
    pub fn run(&self, code: &str, inputs: &[Grid]) -> Result<Vec<Grid>, String> {
        let program = format!(
            "{code}\n\n\
             import json, sys\n\
             _inputs = json.load(sys.stdin)\n\
             print(json.dumps([transform(g) for g in _inputs]))\n"
        );
        // Unique per invocation: concurrent runs (e.g. parallel tests) must
        // not overwrite each other's program files.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("bridgent-arc-{}-{serial}.py", std::process::id()));
        std::fs::write(&path, &program).map_err(|e| format!("cannot write program: {e}"))?;
        let stdin = serde_json::to_vec(&json!(inputs)).expect("grids serialize");
        let result = run_with_timeout(
            Command::new("python3").arg(&path),
            Some(&stdin),
            self.timeout,
        );
        let _ = std::fs::remove_file(&path);
        let output = result.map_err(|e| e.to_string())?;
        if output.exit_code != Some(0) {
            // The tail of stderr carries the actual exception.
            let tail: Vec<&str> = output.stderr.lines().rev().take(5).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!("program crashed:\n{}", tail.join("\n")));
        }
        serde_json::from_str(output.stdout.trim())
            .map_err(|e| format!("program printed invalid grids: {e}"))
    }
}

/// Fraction of matching cells between two grids; shape mismatches score by
/// the overlapping region against the larger area.
pub fn cell_accuracy(expected: &Grid, got: &Grid) -> f64 {
    let total = (expected.len() * expected.first().map_or(0, Vec::len))
        .max(got.len() * got.first().map_or(0, Vec::len));
    if total == 0 {
        return 0.0;
    }
    let mut matching = 0;
    for (expected_row, got_row) in expected.iter().zip(got) {
        matching += expected_row
            .iter()
            .zip(got_row)
            .filter(|(a, b)| a == b)
            .count();
    }
    matching as f64 / total as f64
}

/// Verifies candidate programs against the task's training pairs.
pub struct TrainVerifier<'a> {
    pub train: &'a [Pair],
    pub runner: PythonRunner,
}

impl Verifier for TrainVerifier<'_> {
    fn verify(&self, candidate: &str) -> Verdict {
        let inputs: Vec<Grid> = self.train.iter().map(|p| p.input.clone()).collect();
        let outputs = match self.runner.run(candidate, &inputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                return Verdict {
                    score: 0.0,
                    secondary: 0.0,
                    feedback: e,
                }
            }
        };
        let mut correct = 0;
        let mut cell_total = 0.0;
        let mut feedback = String::new();
        for (i, pair) in self.train.iter().enumerate() {
            let expected = pair.output.as_ref().expect("training pairs have outputs");
            let got = outputs.get(i);
            if got == Some(expected) {
                correct += 1;
                cell_total += 1.0;
            } else if let Some(got) = got {
                cell_total += cell_accuracy(expected, got);
                feedback.push_str(&format!(
                    "Training pair {}: WRONG.\nExpected {}\nGot {}\n",
                    i + 1,
                    render_grid(expected),
                    render_grid(got),
                ));
            }
        }
        if feedback.is_empty() && correct == self.train.len() {
            feedback = "all training pairs correct".into();
        }
        Verdict {
            score: correct as f64 / self.train.len().max(1) as f64,
            secondary: cell_total / self.train.len().max(1) as f64,
            feedback,
        }
    }
}

/// Compare predictions against the task's known test outputs. `None` when
/// the task ships without published outputs (hidden evaluation).
pub fn score_predictions(task: &ArcTask, predictions: &[Grid]) -> Option<bool> {
    let expected: Vec<&Grid> = task.test.iter().filter_map(|p| p.output.as_ref()).collect();
    if expected.len() != task.test.len() {
        return None;
    }
    Some(
        predictions.len() == expected.len()
            && expected.iter().zip(predictions).all(|(e, g)| *e == g),
    )
}

pub struct Solution {
    /// Predicted output grids, one per test input.
    pub predictions: Vec<Grid>,
    /// The winning program.
    pub program: String,
    /// Training score of the winning program (1.0 = all pairs solved).
    pub train_score: f64,
    pub candidates_tried: usize,
}

/// Full solve: evolve programs against the training pairs, apply the best
/// one to the test inputs.
pub fn solve(
    provider: &dyn Provider,
    task: &ArcTask,
    config: RefineConfig,
    on_event: impl FnMut(RefineEvent),
) -> Result<Solution, String> {
    let verifier = TrainVerifier {
        train: &task.train,
        runner: PythonRunner::default(),
    };
    let pool = refine(provider, &base_prompt(task), &verifier, config, on_event)
        .map_err(|e| e.to_string())?;
    let best: &Candidate = pool.first().ok_or("no candidates generated")?;
    let test_inputs: Vec<Grid> = task.test.iter().map(|p| p.input.clone()).collect();
    let predictions = PythonRunner::default().run(&best.content, &test_inputs)?;
    Ok(Solution {
        predictions,
        program: best.content.clone(),
        train_score: best.verdict.score,
        candidates_tried: pool.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{Message, ProviderError};
    use crate::tools::ToolSchema;
    use std::cell::RefCell;

    /// Identity task: output equals input.
    fn identity_task() -> ArcTask {
        serde_json::from_value(json!({
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5]], "output": [[5]]},
            ],
            "test": [{"input": [[7, 8]]}],
        }))
        .unwrap()
    }

    #[test]
    fn load_task_parses_arc_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("task.json");
        std::fs::write(
            &path,
            r#"{"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]]}]}"#,
        )
        .unwrap();
        let task = load_task(&path).unwrap();
        assert_eq!(task.train.len(), 1);
        assert_eq!(task.test[0].input, vec![vec![3]]);
        assert!(task.test[0].output.is_none());
    }

    #[test]
    fn base_prompt_contains_pairs_and_contract() {
        let prompt = base_prompt(&identity_task());
        assert!(prompt.contains("[1, 2]"));
        assert!(prompt.contains("2x2"));
        assert!(prompt.contains("def transform"));
    }

    #[test]
    fn python_runner_executes_transform() {
        let outputs = PythonRunner::default()
            .run("def transform(grid):\n    return grid", &[vec![vec![1, 2]]])
            .unwrap();
        assert_eq!(outputs, vec![vec![vec![1, 2]]]);
    }

    #[test]
    fn python_runner_reports_crashes() {
        let error = PythonRunner::default()
            .run(
                "def transform(grid):\n    raise ValueError('bad rule')",
                &[vec![vec![1]]],
            )
            .unwrap_err();
        assert!(error.contains("bad rule"));
    }

    #[test]
    fn python_runner_times_out_on_infinite_loops() {
        let runner = PythonRunner {
            timeout: Duration::from_secs(1),
        };
        let error = runner
            .run(
                "def transform(grid):\n    while True: pass",
                &[vec![vec![1]]],
            )
            .unwrap_err();
        assert!(error.contains("timed out"));
    }

    #[test]
    fn cell_accuracy_scores_partial_matches() {
        let expected = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(cell_accuracy(&expected, &expected), 1.0);
        assert_eq!(
            cell_accuracy(&expected, &vec![vec![1, 2], vec![3, 9]]),
            0.75
        );
        // Shape mismatch: overlap counted against the larger area.
        assert_eq!(cell_accuracy(&expected, &vec![vec![1, 2]]), 0.5);
        assert_eq!(cell_accuracy(&expected, &vec![]), 0.0);
    }

    #[test]
    fn train_verifier_scores_and_builds_diff_feedback() {
        let task = identity_task();
        let verifier = TrainVerifier {
            train: &task.train,
            runner: PythonRunner::default(),
        };

        let perfect = verifier.verify("def transform(grid):\n    return grid");
        assert_eq!(perfect.score, 1.0);

        let wrong =
            verifier.verify("def transform(grid):\n    return [[0 for _ in row] for row in grid]");
        assert!(wrong.score < 1.0);
        assert!(wrong.feedback.contains("WRONG"));
        assert!(wrong.feedback.contains("Expected"));

        let crash = verifier.verify("not python at all");
        assert_eq!(crash.score, 0.0);
        assert!(crash.feedback.contains("crashed") || crash.feedback.contains("invalid"));
    }

    #[test]
    fn score_predictions_checks_known_outputs() {
        let task: ArcTask = serde_json::from_value(json!({
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[2]], "output": [[2]]}],
        }))
        .unwrap();
        assert_eq!(score_predictions(&task, &[vec![vec![2]]]), Some(true));
        assert_eq!(score_predictions(&task, &[vec![vec![9]]]), Some(false));
        assert_eq!(score_predictions(&task, &[]), Some(false));
        // Hidden outputs: nothing to score against.
        assert_eq!(
            score_predictions(&identity_task(), &[vec![vec![7, 8]]]),
            None
        );
    }

    struct ScriptedProvider(RefCell<Vec<String>>);

    impl Provider for ScriptedProvider {
        fn complete(
            &self,
            _system: &str,
            _messages: &[Message],
            _tools: &[ToolSchema],
        ) -> Result<Message, ProviderError> {
            Ok(Message::assistant(self.0.borrow_mut().remove(0), vec![]))
        }
    }

    #[test]
    fn solve_evolves_to_a_working_program_and_predicts_test_output() {
        // First sample is wrong (all zeros), second sample solves it.
        let provider = ScriptedProvider(RefCell::new(vec![
            "```python\ndef transform(grid):\n    return [[0 for _ in row] for row in grid]\n```"
                .into(),
            "```python\ndef transform(grid):\n    return grid\n```".into(),
        ]));
        let config = RefineConfig {
            rounds: 2,
            per_round: 1,
            keep_top: 2,
        };
        let solution = solve(&provider, &identity_task(), config, |_| {}).unwrap();
        assert_eq!(solution.train_score, 1.0);
        assert_eq!(solution.predictions, vec![vec![vec![7, 8]]]);
        assert_eq!(solution.candidates_tried, 2);
    }
}
