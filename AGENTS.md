# bridgent — agent instructions

- Run `cargo test` before claiming any change works; run `cargo fmt` and
  `cargo clippy --all-targets` before committing.
- Every behavior change lands with a test. Provider request/response logic is
  tested as pure functions on JSON values — never add tests that hit the
  network.
- Keep the library free of CLI concerns: `main.rs` is the only file that may
  print or read stdin.
- Keep the base system prompt in `context.rs` under 1500 characters; the test
  suite enforces this.
- Conventional commits, one logical unit per commit.
