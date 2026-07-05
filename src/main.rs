//! bridgent CLI: one-shot prompts, an interactive REPL, and session resume.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use bridgent::agent::{Agent, Event};
use bridgent::cli::{self, ProviderFlags};
use bridgent::config::Config;
use bridgent::context::system_prompt;
use bridgent::providers::Usage;
use bridgent::session::Session;
use bridgent::tools::default_registry;

const USAGE: &str = "\
bridgent — a minimal, provider-agnostic coding agent

Usage:
  bridgent [OPTIONS]              interactive session in the current directory
  bridgent [OPTIONS] <PROMPT>     run one prompt, print the answer, exit

Options:
  -c, --continue          resume the most recent session
      --resume <PATH>     resume a specific session file (see --sessions)
      --sessions          list sessions in this directory and exit
      --json              one-shot mode: emit JSONL events instead of text
      --provider <NAME>   anthropic (default) or openai
      --model <MODEL>     model id (default per provider)
      --base-url <URL>    override API base URL (local models, proxies)
  -h, --help              show this help
  -V, --version           show version

In the REPL, /help lists session commands (/new, /compact, /usage).

Environment:
  ANTHROPIC_API_KEY / OPENAI_API_KEY   provider credentials
  BRIDGENT_PROVIDER, BRIDGENT_MODEL, BRIDGENT_BASE_URL   defaults for the flags";

#[derive(Default)]
struct Args {
    prompt: Option<String>,
    resume: bool,
    resume_path: Option<PathBuf>,
    list_sessions: bool,
    json: bool,
    flags: ProviderFlags,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut args = Args::default();
    let mut words: Vec<String> = Vec::new();
    let mut iter = argv.iter();
    while let Some(arg) = iter.next() {
        if args.flags.parse(arg, &mut iter)? {
            continue;
        }
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
            "--resume" => {
                args.resume_path = Some(PathBuf::from(cli::flag_value(&mut iter, "--resume")?))
            }
            "--sessions" => args.list_sessions = true,
            "--json" => args.json = true,
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            word => words.push(word.to_string()),
        }
    }
    if !words.is_empty() {
        args.prompt = Some(words.join(" "));
    }
    Ok(Some(args))
}

/// Total tokens across every message in the session.
fn session_usage(session: &Session) -> Usage {
    let mut total = Usage::default();
    for usage in session.messages.iter().filter_map(|m| m.usage) {
        total.add(usage);
    }
    total
}

/// One JSONL line per event, for `--json` headless consumers.
fn print_json_event(event: Event) {
    let value = match event {
        Event::AssistantDelta(text) => serde_json::json!({"type": "delta", "text": text}),
        Event::AssistantText(text) => serde_json::json!({"type": "text", "text": text}),
        Event::ToolStart(call) => serde_json::json!({
            "type": "tool_start", "id": call.id, "name": call.name, "args": call.args,
        }),
        Event::ToolEnd(call, result) => serde_json::json!({
            "type": "tool_end", "id": call.id, "is_error": result.is_error,
            "output": result.output,
        }),
        Event::Compacted { kept } => serde_json::json!({"type": "compacted", "kept": kept}),
    };
    println!("{value}");
}

/// Render agent progress to stderr so stdout stays clean for the answer.
fn print_event(event: Event) {
    match event {
        Event::AssistantDelta(text) => {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
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

    let workdir: PathBuf =
        std::env::current_dir().map_err(|e| format!("cannot resolve working directory: {e}"))?;

    if args.list_sessions {
        let paths = Session::list(&workdir);
        if paths.is_empty() {
            eprintln!("no sessions in {}", workdir.display());
        }
        for path in paths {
            match Session::open(&path) {
                Ok(session) => {
                    let first = session
                        .messages
                        .iter()
                        .find(|m| m.role == bridgent::providers::Role::User)
                        .map_or("(empty)", |m| m.content.as_str());
                    let first: String = first.chars().take(60).collect();
                    println!(
                        "{}  {} msgs  {first}",
                        path.display(),
                        session.messages.len()
                    );
                }
                Err(e) => eprintln!("{}  (unreadable: {e})", path.display()),
            }
        }
        return Ok(());
    }

    let mut config = Config::from_env(&args.flags)?;
    let mut provider = config.build_provider();
    let tools = default_registry(&workdir);
    let system = system_prompt(&workdir);

    let mut session = match (&args.resume_path, args.resume) {
        (Some(path), _) => {
            Session::open(path).map_err(|e| format!("cannot resume session: {e}"))?
        }
        (None, true) => match Session::latest(&workdir) {
            Some(session) => session.map_err(|e| format!("cannot resume session: {e}"))?,
            None => return Err("no previous session to continue".into()),
        },
        (None, false) => {
            Session::create(&workdir).map_err(|e| format!("cannot create session: {e}"))?
        }
    };

    if let Some(prompt) = args.prompt {
        let agent = Agent::new(provider.as_ref(), &tools, system);
        if args.json {
            // Headless mode: JSONL events, machine-readable errors, and a
            // final done record with the session's token totals.
            match agent.run_turn(&mut session, &prompt, print_json_event) {
                Ok(answer) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "type": "done", "answer": answer,
                            "usage": session_usage(&session),
                            "session": session.path,
                        })
                    );
                    return Ok(());
                }
                Err(e) => {
                    println!(
                        "{}",
                        serde_json::json!({"type": "error", "message": e.to_string()})
                    );
                    std::process::exit(1);
                }
            }
        }
        // The answer streams through print_event; just terminate the line.
        agent
            .run_turn(&mut session, &prompt, print_event)
            .map_err(|e| e.to_string())?;
        println!();
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
        // The agent is rebuilt per turn (it's just borrows) so /model can
        // swap the provider between turns.
        if let Some(command) = input.strip_prefix('/') {
            let result = run_command(
                command,
                &mut config,
                &mut provider,
                &tools,
                &system,
                &mut session,
                &workdir,
            );
            match result {
                Ok(output) => eprintln!("{output}\n"),
                Err(e) => eprintln!("\x1b[31m{e}\x1b[0m\n"),
            }
            continue;
        }
        let agent = Agent::new(provider.as_ref(), &tools, system.clone());
        match agent.run_turn(&mut session, input, print_event) {
            Ok(_) => println!("\n"), // answer already streamed
            // Provider errors don't kill the REPL; the session file is intact.
            Err(e) => eprintln!("\x1b[31m{e}\x1b[0m\n"),
        }
    }
}

/// REPL slash commands. Everything else in the loop goes to the model.
#[allow(clippy::too_many_arguments)]
fn run_command(
    command: &str,
    config: &mut Config,
    provider: &mut Box<dyn bridgent::providers::Provider>,
    tools: &bridgent::tools::ToolRegistry,
    system: &str,
    session: &mut Session,
    workdir: &std::path::Path,
) -> Result<String, String> {
    match command
        .split_once(' ')
        .map_or((command, ""), |(c, rest)| (c, rest.trim()))
    {
        ("help", _) => Ok("/new           start a fresh session\n\
                      /compact       summarize old history to reclaim context\n\
                      /model [ID]    show or switch the model\n\
                      /usage         token totals for this session\n\
                      /help          this text"
            .into()),
        ("new", _) => {
            *session = Session::create(workdir).map_err(|e| e.to_string())?;
            Ok("started a fresh session".into())
        }
        ("compact", _) => {
            let agent = Agent::new(provider.as_ref(), tools, system.to_string());
            match agent.compact(session).map_err(|e| e.to_string())? {
                true => Ok(format!(
                    "history compacted to {} messages",
                    session.messages.len()
                )),
                false => Ok("nothing to compact yet".into()),
            }
        }
        ("model", "") => Ok(format!("current model: {}", config.model)),
        ("model", id) => {
            config.model = id.to_string();
            *provider = config.build_provider();
            Ok(format!("switched to {id}"))
        }
        ("usage", _) => {
            let total = session_usage(session);
            Ok(format!(
                "session: {} messages · {} input + {} output tokens",
                session.messages.len(),
                total.input_tokens,
                total.output_tokens
            ))
        }
        (other, _) => Err(format!("unknown command /{other} (try /help)")),
    }
}

fn main() -> ExitCode {
    cli::exit("bridgent", run())
}
