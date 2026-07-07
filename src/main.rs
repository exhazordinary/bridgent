//! bridgent CLI: one-shot prompts, an interactive REPL, and session resume.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bridgent::agent::{Agent, Event};
use bridgent::cli::{self, ProviderFlags};
use bridgent::config::Config;
use bridgent::context::system_prompt;
use bridgent::providers::{Provider, Role};
use bridgent::session::Session;
use bridgent::tools::{default_registry, ToolRegistry};

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

In the REPL, /help lists session commands (/new, /compact, /model, /usage).

Environment:
  ANTHROPIC_API_KEY / OPENAI_API_KEY   provider credentials
  ANTHROPIC_AUTH_TOKEN                 bearer token, replaces the API key
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

/// First `n` characters of `s`, with an ellipsis when truncated.
fn truncate_chars(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if out.len() < s.len() {
        out.push('…');
    }
    out
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
            eprintln!(
                "\x1b[2m⚙ {} {}\x1b[0m",
                call.name,
                truncate_chars(&args, 120)
            );
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

fn list_sessions(workdir: &Path) {
    let paths = Session::list(workdir);
    if paths.is_empty() {
        eprintln!("no sessions in {}", workdir.display());
    }
    for path in paths {
        match Session::open(&path) {
            Ok(session) => {
                let first = session
                    .messages
                    .iter()
                    .find(|m| m.role == Role::User)
                    .map_or("(empty)", |m| m.content.as_str());
                println!(
                    "{}  {} msgs  {}",
                    path.display(),
                    session.messages.len(),
                    truncate_chars(first, 60)
                );
            }
            Err(e) => eprintln!("{}  (unreadable: {e})", path.display()),
        }
    }
}

/// Run one prompt to completion, as text or as JSONL events.
fn run_one_shot(repl: &mut Repl, prompt: &str, json: bool) -> Result<(), String> {
    if json {
        // Headless mode: JSONL events on stdout, a final done record with
        // the session's token totals, machine-readable errors.
        match agent(repl.provider.as_ref(), &repl.tools, &repl.system).run_turn(
            &mut repl.session,
            prompt,
            |event| {
                println!("{}", event.to_json());
            },
        ) {
            Ok(answer) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "type": "done", "answer": answer,
                        "usage": repl.session.usage(),
                        "session": repl.session.path,
                    })
                );
                Ok(())
            }
            Err(e) => {
                println!(
                    "{}",
                    serde_json::json!({"type": "error", "message": e.to_string()})
                );
                Err(e.to_string())
            }
        }
    } else {
        // The answer streams through print_event; just terminate the line.
        agent(repl.provider.as_ref(), &repl.tools, &repl.system)
            .run_turn(&mut repl.session, prompt, print_event)
            .map_err(|e| e.to_string())?;
        println!();
        Ok(())
    }
}

/// Interactive state: everything a turn or a slash command needs.
struct Repl {
    config: Config,
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    system: String,
    session: Session,
    workdir: PathBuf,
}

/// Agents are cheap bundles of borrows, rebuilt per use so `/model` can swap
/// the provider between turns. A free function (not a `Repl` method) so the
/// borrow of these fields stays disjoint from `&mut repl.session`.
fn agent<'a>(provider: &'a dyn Provider, tools: &'a ToolRegistry, system: &str) -> Agent<'a> {
    Agent::new(provider, tools, system.to_string())
}

/// Set by the SIGINT handler; each turn clears it on start and stops at the
/// next safe checkpoint when it flips.
static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Repl {
    fn run(&mut self) -> Result<(), String> {
        // Ctrl-C interrupts the current turn instead of killing the REPL.
        ctrlc::set_handler(|| INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed))
            .map_err(|e| format!("cannot install interrupt handler: {e}"))?;
        eprintln!(
            "bridgent {} · {} · {}",
            env!("CARGO_PKG_VERSION"),
            self.config.model,
            self.workdir.display()
        );
        eprintln!("empty line or ctrl-d to exit · ctrl-c interrupts a turn\n");
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
            if let Some(command) = input.strip_prefix('/') {
                match self.command(command) {
                    Ok(output) => eprintln!("{output}\n"),
                    Err(e) => eprintln!("\x1b[31m{e}\x1b[0m\n"),
                }
                continue;
            }
            INTERRUPTED.store(false, std::sync::atomic::Ordering::Relaxed);
            let mut turn = agent(self.provider.as_ref(), &self.tools, &self.system);
            turn.interrupt = Some(&INTERRUPTED);
            match turn.run_turn(&mut self.session, input, print_event) {
                Ok(_) => println!("\n"), // answer already streamed
                Err(e) if e.message == bridgent::agent::INTERRUPTED => {
                    eprintln!("\x1b[2m⏹ turn interrupted\x1b[0m\n");
                }
                // Provider errors don't kill the REPL; the session is intact.
                Err(e) => eprintln!("\x1b[31m{e}\x1b[0m\n"),
            }
        }
    }

    /// Slash commands. Everything else in the loop goes to the model.
    fn command(&mut self, command: &str) -> Result<String, String> {
        let (name, arg) = command
            .split_once(' ')
            .map_or((command, ""), |(c, rest)| (c, rest.trim()));
        match (name, arg) {
            ("help", _) => Ok("/new           start a fresh session\n\
                          /compact       summarize old history to reclaim context\n\
                          /model [ID]    show or switch the model\n\
                          /usage         token totals for this session\n\
                          /help          this text"
                .into()),
            ("new", _) => {
                self.session = Session::create(&self.workdir).map_err(|e| e.to_string())?;
                Ok("started a fresh session".into())
            }
            ("compact", _) => {
                let agent = agent(self.provider.as_ref(), &self.tools, &self.system);
                if agent
                    .compact(&mut self.session)
                    .map_err(|e| e.to_string())?
                {
                    Ok(format!(
                        "history compacted to {} messages",
                        self.session.messages.len()
                    ))
                } else {
                    Ok("nothing to compact yet".into())
                }
            }
            ("model", "") => Ok(format!("current model: {}", self.config.model)),
            ("model", id) => {
                self.config.model = id.to_string();
                self.provider = self.config.build_provider();
                Ok(format!("switched to {id}"))
            }
            ("usage", _) => {
                let total = self.session.usage();
                let mut line = format!(
                    "session: {} messages · {} input + {} output tokens",
                    self.session.messages.len(),
                    total.input_tokens,
                    total.output_tokens
                );
                if total.cache_read_input_tokens + total.cache_creation_input_tokens > 0 {
                    line.push_str(&format!(
                        " · cache: {} read, {} written",
                        total.cache_read_input_tokens, total.cache_creation_input_tokens
                    ));
                }
                Ok(line)
            }
            (other, _) => Err(format!("unknown command /{other} (try /help)")),
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
        list_sessions(&workdir);
        return Ok(());
    }

    let config = Config::from_env(&args.flags)?;
    let session = match (&args.resume_path, args.resume) {
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
    let mut repl = Repl {
        provider: config.build_provider(),
        tools: default_registry(&workdir),
        system: system_prompt(&workdir),
        session,
        config,
        workdir,
    };

    match args.prompt {
        Some(prompt) => run_one_shot(&mut repl, &prompt, args.json),
        None => repl.run(),
    }
}

fn main() -> ExitCode {
    cli::exit("bridgent", run())
}
