//! bridgent-arc: solve ARC-AGI tasks by evolving Python programs.
//!
//! Single task: prints the predicted grids as JSON. Multiple tasks (several
//! paths or --dir): evaluation mode — solves each, scores against known
//! outputs, and prints an accuracy report. Benchmarks are meaningless
//! without the harness disclosed, so the report states the full sampling
//! configuration.

use std::path::PathBuf;
use std::process::ExitCode;

use bridgent::arc::{load_task, score_predictions, solve, ArcTask, Solution};
use bridgent::config::Config;
use bridgent::providers::Provider;
use bridgent::refine::{RefineConfig, RefineEvent};

const USAGE: &str = "\
bridgent-arc — ARC-AGI solver: evolve Python programs against a task's
training pairs, apply the winner to the test inputs.

Usage:
  bridgent-arc [OPTIONS] <task.json>          solve one task, print predictions
  bridgent-arc [OPTIONS] <a.json> <b.json>…   evaluate several tasks
  bridgent-arc [OPTIONS] --dir <tasks/>       evaluate every .json in a directory

Options:
      --dir <PATH>        evaluate all .json tasks in a directory
      --rounds <N>        evolution rounds (default 3)
      --samples <N>       candidates per round (default 5)
      --keep <N>          top attempts seeding the next round (default 3)
      --provider <NAME>   anthropic (default) or openai
      --model <MODEL>     model id (default per provider)
      --base-url <URL>    override API base URL
  -h, --help              show this help

Output: predictions (single task) or an accuracy report (evaluation) on
stdout; progress on stderr. Task format: ARC JSON with train/test grids.";

struct Args {
    tasks: Vec<PathBuf>,
    refine: RefineConfig,
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut refine = RefineConfig::default();
    let mut provider = None;
    let mut model = None;
    let mut base_url = None;
    let mut tasks: Vec<PathBuf> = Vec::new();
    let mut iter = argv.iter();
    while let Some(arg) = iter.next() {
        let mut flag_value = |name: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        let parse_count = |name: &str, value: String| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{name} must be a number"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(None);
            }
            "--dir" => tasks.extend(tasks_in_dir(&flag_value("--dir")?)?),
            "--rounds" => refine.rounds = parse_count("--rounds", flag_value("--rounds")?)?,
            "--samples" => refine.per_round = parse_count("--samples", flag_value("--samples")?)?,
            "--keep" => refine.keep_top = parse_count("--keep", flag_value("--keep")?)?,
            "--provider" => provider = Some(flag_value("--provider")?),
            "--model" => model = Some(flag_value("--model")?),
            "--base-url" => base_url = Some(flag_value("--base-url")?),
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            path => tasks.push(PathBuf::from(path)),
        }
    }
    if tasks.is_empty() {
        return Err("no task files (see --help)".into());
    }
    Ok(Some(Args {
        tasks,
        refine,
        provider,
        model,
        base_url,
    }))
}

fn tasks_in_dir(dir: &str) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {dir}: {e}"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .json tasks in {dir}"));
    }
    Ok(paths)
}

fn print_progress(event: RefineEvent) {
    match event {
        RefineEvent::Sampled(c) => {
            eprintln!("\x1b[2m  candidate score {:.2}\x1b[0m", c.verdict.score);
        }
        RefineEvent::RoundDone { round, best_score } => {
            eprintln!(
                "\x1b[2m  round {} done · best {:.2}\x1b[0m",
                round + 1,
                best_score
            );
        }
    }
}

fn solve_one(
    provider: &dyn Provider,
    path: &std::path::Path,
    refine: RefineConfig,
) -> Result<(ArcTask, Solution), String> {
    let task = load_task(path)?;
    let solution = solve(provider, &task, refine, print_progress)?;
    Ok((task, solution))
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(args) = parse_args(&argv)? else {
        return Ok(());
    };
    let config = Config::resolve(
        |key| std::env::var(key).ok(),
        args.provider.as_deref(),
        args.model.as_deref(),
        args.base_url.as_deref(),
    )?;
    let provider = config.build_provider();

    // Single task: predictions on stdout, ready to pipe.
    if let [path] = args.tasks.as_slice() {
        eprintln!(
            "solving {} · {} rounds × {} samples · {}",
            path.display(),
            args.refine.rounds,
            args.refine.per_round,
            config.model,
        );
        let (_, solution) = solve_one(provider.as_ref(), path, args.refine)?;
        eprintln!(
            "best program (train score {:.2}, {} candidates tried):\n{}",
            solution.train_score, solution.candidates_tried, solution.program,
        );
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!(solution.predictions))
                .expect("grids serialize")
        );
        return Ok(());
    }

    // Evaluation mode: solve every task, score what can be scored, and
    // disclose the harness configuration alongside the numbers.
    println!(
        "harness: bridgent-arc {} · model {} · {} rounds × {} samples · keep {}",
        env!("CARGO_PKG_VERSION"),
        config.model,
        args.refine.rounds,
        args.refine.per_round,
        args.refine.keep_top,
    );
    let (mut solved, mut scoreable, mut candidates) = (0usize, 0usize, 0usize);
    for path in &args.tasks {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        eprintln!("── {name}");
        match solve_one(provider.as_ref(), path, args.refine) {
            Ok((task, solution)) => {
                candidates += solution.candidates_tried;
                let verdict = match score_predictions(&task, &solution.predictions) {
                    Some(true) => {
                        solved += 1;
                        scoreable += 1;
                        "SOLVED"
                    }
                    Some(false) => {
                        scoreable += 1;
                        "failed"
                    }
                    None => "no known outputs",
                };
                println!(
                    "{name}: {verdict} · train {:.2} · {} candidates",
                    solution.train_score, solution.candidates_tried,
                );
            }
            Err(e) => println!("{name}: error · {e}"),
        }
    }
    println!(
        "result: {solved}/{scoreable} solved ({} tasks total, {candidates} candidates sampled)",
        args.tasks.len(),
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("bridgent-arc: {message}");
            ExitCode::FAILURE
        }
    }
}
