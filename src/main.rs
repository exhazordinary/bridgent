//! bridgent CLI: one-shot prompts, an interactive REPL, and session resume.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use bridgent::agent::{Agent, Event};
use bridgent::config::Config;
use bridgent::context::system_prompt;
use bridgent::session::Session;
use bridgent::tools::default_registry;

const USAGE: &str = "\
bridgent — a minimal, provider-agnostic coding agent

Usage:
  bridgent [OPTIONS]              interactive session in the current directory
  bridgent [OPTIONS] <PROMPT>     run one prompt, print the answer, exit

Options:
  -c, --continue          resume the most recent session
      --provider <NAME>   anthropic (default) or openai
      --model <MODEL>     model id (default per provider)
      --base-url <URL>    override API base URL (local models, proxies)
  -h, --help              show this help
  -V, --version           show version

Environment:
  ANTHROPIC_API_KEY / OPENAI_API_KEY   provider credentials
  BRIDGENT_PROVIDER, BRIDGENT_MODEL, BRIDGENT_BASE_URL   defaults for the flags";

struct Args {
    prompt: Option<String>,
    resume: bool,
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut args = Args {
        prompt: None,
        resume: false,
        provider: None,
        model: None,
        base_url: None,
    };
    let mut words: Vec<String> = Vec::new();
    let mut iter = argv.iter();
    while let Some(arg) = iter.next() {
        let mut flag_value = |name: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("bridgent {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-c" | "--continue" => args.resume = true,
            "--provider" => args.provider = Some(flag_value("--provider")?),
            "--model" => args.model = Some(flag_value("--model")?),
            "--base-url" => args.base_url = Some(flag_value("--base-url")?),
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            word => words.push(word.to_string()),
        }
    }
    if !words.is_empty() {
        args.prompt = Some(words.join(" "));
    }
    Ok(Some(args))
}

/// Render agent progress to stderr so stdout stays clean for the answer.
fn print_event(event: Event) {
    match event {
        Event::AssistantText(_) => {}
        Event::ToolStart(call) => {
            let args = serde_json::to_string(&call.args).unwrap_or_default();
            let args = if args.len() > 120 {
                format!("{}…", &args[..120])
            } else {
                args
            };
            eprintln!("\x1b[2m⚙ {} {args}\x1b[0m", call.name);
        }
        Event::ToolEnd(_, result) if result.is_error => {
            let first = result.output.lines().next().unwrap_or_default();
            eprintln!("\x1b[2m  ↳ error: {first}\x1b[0m");
        }
        Event::ToolEnd(..) => {}
        Event::Compacted { kept } => {
            eprintln!("\x1b[2m⊜ history compacted to {kept} messages\x1b[0m");
        }
    }
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
    let workdir: PathBuf =
        std::env::current_dir().map_err(|e| format!("cannot resolve working directory: {e}"))?;
    let provider = config.build_provider();
    let tools = default_registry(&workdir);
    let agent = Agent::new(provider.as_ref(), &tools, system_prompt(&workdir));

    let mut session = if args.resume {
        match Session::latest(&workdir) {
            Some(session) => session.map_err(|e| format!("cannot resume session: {e}"))?,
            None => return Err("no previous session to continue".into()),
        }
    } else {
        Session::create(&workdir).map_err(|e| format!("cannot create session: {e}"))?
    };

    if let Some(prompt) = args.prompt {
        let answer = agent
            .run_turn(&mut session, &prompt, print_event)
            .map_err(|e| e.to_string())?;
        println!("{answer}");
        return Ok(());
    }

    eprintln!(
        "bridgent {} · {} · {}",
        env!("CARGO_PKG_VERSION"),
        config.model,
        workdir.display()
    );
    eprintln!("empty line or ctrl-d to exit\n");
    let stdin = std::io::stdin();
    loop {
        eprint!("\x1b[1m>\x1b[0m ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            return Ok(()); // EOF
        }
        let input = line.trim();
        if input.is_empty() {
            return Ok(());
        }
        match agent.run_turn(&mut session, input, print_event) {
            Ok(answer) => println!("{answer}\n"),
            // Provider errors don't kill the REPL; the session file is intact.
            Err(e) => eprintln!("\x1b[31m{e}\x1b[0m\n"),
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("bridgent: {message}");
            ExitCode::FAILURE
        }
    }
}
