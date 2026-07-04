# bridgent

A minimal, provider-agnostic coding agent harness in Rust. Single static
binary, four tools, no bloat.

> An agent is a model plus a harness. The model is rented; the harness is
> yours. This is the harness.

## Why

Built on the converging lessons of harness-engineering research and practice:

- **Four tools are enough.** `read`, `write`, `edit`, `bash`. Frontier models
  are RL-trained as coding agents; everything else (search, git, tests,
  process control) goes through bash. *(pi)*
- **A minimal system prompt beats a clever one.** Under 200 words, plus your
  project's own `AGENTS.md`. Nothing is injected that you can't see in your
  repo. *(pi)*
- **The loop just loops.** Call the model, run the tools it asks for, feed
  results back, repeat until it answers in plain text. No step limits.
- **Errors are feedback, not failures.** Tool errors go back to the model
  verbatim so it can correct course; transient provider errors retry with
  backoff. *(Anthropic, "Effective harnesses for long-running agents")*
- **State lives in files.** Sessions are append-only JSONL in
  `.bridgent/sessions/` — crash-safe, greppable, resumable. *(hermes)*
- **Core is separate from frontend.** The library (`bridgent::agent`,
  `bridgent::tools`, `bridgent::providers`, `bridgent::session`) has no CLI
  dependency; the binary is a thin client. *(opencode)*

## Install

```sh
cargo install --path .
```

## Usage

```sh
export ANTHROPIC_API_KEY=sk-ant-...

bridgent "fix the failing test in this repo"   # one-shot
bridgent                                       # interactive REPL
bridgent -c                                    # resume the latest session
bridgent --sessions                            # list sessions in this directory
```

Responses stream token by token. Inside the REPL, `/help` lists session
commands: `/new`, `/compact` (summarize old history to reclaim context —
this also happens automatically as the session grows), and `/usage` (token
totals). Any OpenAI-compatible server works, including local models:

```sh
bridgent --provider openai --base-url http://localhost:11434/v1 --model qwen3 "hi"
```

### Configuration

| Flag | Env var | Default |
|---|---|---|
| `--provider` | `BRIDGENT_PROVIDER` | `anthropic` |
| `--model` | `BRIDGENT_MODEL` | per provider |
| `--base-url` | `BRIDGENT_BASE_URL` | provider default |
| — | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | required (unless `--base-url` or a bearer token is set) |
| — | `ANTHROPIC_AUTH_TOKEN` | optional bearer token, replaces the API key |

Flags override env; env overrides defaults.

### A note on Claude subscriptions

A Claude Pro/Max subscription authenticates through OAuth tokens issued for
Claude Code; Anthropic's terms don't allow third-party harnesses to reuse
that flow, and tools that did have been blocked. bridgent therefore doesn't
impersonate Claude Code. It does accept any legitimately issued bearer token
via `ANTHROPIC_AUTH_TOKEN` (enterprise gateways, LLM proxies), so if
subscription auth ever opens up to third-party harnesses, bridgent already
speaks it. For metered API access, use `ANTHROPIC_API_KEY`.

### Project instructions

Put an `AGENTS.md` (or `CLAUDE.md`) in your working directory and its content
is appended to the system prompt. That's the whole customization model.

## The refine engine and `bridgent-arc`

Beyond the interactive agent, bridgent ships the other fundamental harness
pattern: **generate → verify → revise**. The `refine` module samples
candidates from the model, scores them with a deterministic verifier, and
feeds scores plus failure diffs back so the model evolves its best attempts
across rounds — the pattern behind the strongest ARC-AGI results
([Greenblatt's sample-and-verify](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis),
Berman's evolutionary test-time compute,
[Pang's evolutionary program synthesis](https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary)),
generalized to anything with a verifier.

`bridgent-arc` applies it to ARC-AGI tasks: the model writes Python
`transform` programs, bridgent executes them against the task's training
pairs, failures come back as exact expected-vs-got grid diffs, and the
winning program produces the test predictions.

```sh
bridgent-arc --rounds 3 --samples 5 task.json   # predictions as JSON on stdout
bridgent-arc --dir tasks/                       # evaluate a whole benchmark
```

Evaluation mode scores every task against its known outputs and prints an
accuracy report headed by the full harness configuration — because agent
benchmarks are meaningless when the harness is undisclosed
([Stop Comparing LLM Agents Without Disclosing the Harness](https://arxiv.org/pdf/2605.23950)).

## Architecture

```
src/
  tools.rs         Tool trait + registry + read/write/edit/bash
  providers.rs     Provider trait + neutral Message type + Anthropic/OpenAI clients
  streaming.rs     SSE parsing + per-provider delta accumulators
  agent.rs         the loop: complete → run tools → feed back → repeat; compaction
  refine.rs        generate–verify–revise engine (evolutionary test-time compute)
  arc.rs           ARC-AGI harness: prompts, python runner, grid verifier
  session.rs       append-only JSONL persistence, resume
  context.rs       system prompt builder (base + AGENTS.md)
  process.rs       shared subprocess runner (timeout, deadlock-safe)
  config.rs        env + flag resolution
  main.rs          bridgent CLI
  bin/bridgent-arc.rs  ARC solver CLI
```

Request building and response parsing are pure functions over JSON values, so
every provider quirk is unit-tested without a network. The agent loop is
tested against a scripted mock provider. `cargo test` runs the whole suite in
about a second.

## Security model

bridgent executes what the model asks, including arbitrary shell commands, with
your permissions and no sandbox — same trust model as every minimal coding
agent. Two consequences to be aware of:

- **Only run it in repositories you trust.** `AGENTS.md`/`CLAUDE.md` content
  goes into the system prompt, so a malicious repo can steer the agent
  (prompt injection → code execution).
- Untrusted *data* the agent reads (issues, web content piped through bash)
  can attempt the same. Review what it did — sessions in `.bridgent/sessions/`
  are the full audit log.

If you need containment, run bridgent inside a container or VM.

## Non-goals (for now)

MCP, sub-agents, built-in todo lists, plan modes. Most of these are better
served by bash and files; the rest can land later without touching the core.
Known limitation: Ctrl-C aborts the whole process, not just the current turn.

## References

- [Code as Agent Harness](https://arxiv.org/abs/2605.18747) (survey)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (Anthropic)
- [What I learned building an opinionated and minimal coding agent](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/) (pi)
- [opencode](https://github.com/sst/opencode) · [hermes-agent](https://github.com/NousResearch/hermes-agent)
