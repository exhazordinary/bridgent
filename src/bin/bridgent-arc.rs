//! bridgent-arc: solve an ARC-AGI task by evolving Python programs.

use std::path::PathBuf;
use std::process::ExitCode;

use bridgent::arc::{load_task, solve};
use bridgent::config::Config;
use bridgent::refine::{RefineConfig, RefineEvent};

const USAGE: &str = "\
bridgent-arc — ARC-AGI solver: evolve Python programs against a task's
training pairs, apply the winner to the test inputs.

Usage:
  bridgent-arc [OPTIONS] <task.json>

Options:
      --rounds <N>        evolution rounds (default 3)
      --samples <N>       candidates per round (default 5)
      --keep <N>          top attempts seeding the next round (default 3)
      --provider <NAME>   anthropic (default) or openai
      --model <MODEL>     model id (default per provider)
      --base-url <URL>    override API base URL
  -h, --help              show this help

Output: predicted test grid(s) as JSON on stdout; progress on stderr.
Task format: ARC JSON with train/test input/output grids.";

struct Args {
    task: PathBuf,
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
    let mut task = None;
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
            "--rounds" => refine.rounds = parse_count("--rounds", flag_value("--rounds")?)?,
            "--samples" => refine.per_round = parse_count("--samples", flag_value("--samples")?)?,
            "--keep" => refine.keep_top = parse_count("--keep", flag_value("--keep")?)?,
            "--provider" => provider = Some(flag_value("--provider")?),
            "--model" => model = Some(flag_value("--model")?),
            "--base-url" => base_url = Some(flag_value("--base-url")?),
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            path => task = Some(PathBuf::from(path)),
        }
    }
    let task = task.ok_or("missing task file (see --help)")?;
    Ok(Some(Args {
        task,
        refine,
        provider,
        model,
        base_url,
    }))
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
    let task = load_task(&args.task)?;
    eprintln!(
        "solving {} · {} train pairs · {} rounds × {} samples · {}",
        args.task.display(),
        task.train.len(),
        args.refine.rounds,
        args.refine.per_round,
        config.model,
    );
    let provider = config.build_provider();
    let solution = solve(provider.as_ref(), &task, args.refine, |event| match event {
        RefineEvent::Sampled(c) => {
            eprintln!("\x1b[2m  candidate score {:.2}\x1b[0m", c.verdict.score);
        }
        RefineEvent::RoundDone { round, best_score } => {
            eprintln!("round {} done · best {:.2}", round + 1, best_score);
        }
    })?;
    eprintln!(
        "best program (train score {:.2}, {} candidates tried):\n{}",
        solution.train_score, solution.candidates_tried, solution.program,
    );
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!(solution.predictions)).expect("grids serialize")
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
