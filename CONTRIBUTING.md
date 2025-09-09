# Contributing

[![Tests](https://github.com/graniet/llm/actions/workflows/test.yml/badge.svg)](https://github.com/graniet/llm/actions/workflows/test.yml)

Instructions to run the project in development.

> [!IMPORTANT]
>
> Requirements:
>
> - [Rust](https://www.rust-lang.org/tools/install)
> - You might need to install [`protobuf`](https://protobuf.dev/installation/), e.g. with `apt install protobuf-compiler` or `brew install protobuf`
> - Optionally, [`uv`](https://docs.astral.sh/uv/getting-started/installation/) required if you want to build the python wheels
>
> Recommended VSCode extension: [`rust-analyzer`](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## 📥 Install dev dependencies

```sh
rustup update
cargo install cargo-release cargo-deny git-cliff
```

Create a `.cargo/config.toml` file with your API keys:

```toml
[env]
MISTRAL_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"
GROQ_API_KEY = "YOUR_API_KEY"
COHERE_API_KEY = "YOUR_API_KEY"
OPENROUTER_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
ANTHROPIC_API_KEY = "YOUR_API_KEY"
```

> [!TIP]
>
> Ideally you should provide API keys for a few LLM providers if you want to run tests against these APIs.
>
> Providers with a free tier:
>
> - [Mistral.ai](https://console.mistral.ai/api-keys)
> - Google
> - Groq
> - Cohere
>
> No free tier providers:
>
> - OpenAI
> - Anthropic

## ✅ Run tests

```sh
cargo test
```

> [!NOTE]
>
> Tests for providers for which you did not provided an API key will be skipped.

## 📚 Examples

Many usage examples can be found in the [`examples/`](https://github.com/graniet/llm/tree/main/examples) folder. Run an example with:

```sh
cargo run --example mistral_example
```

## 📦 Build

Build binaries for production in `target/release/`

```sh
cargo build --release
```

> [!NOTE]
>
> Run the built CLI with:
>
> ```sh
> ./target/release/llm --help
> ```

🐍 Bundle the CLI as python wheels in `target/wheels`:

```sh
uvx maturin build
```

## 🧼 Format & lint

Automatically format the codebase using `rustfmt`:

```sh
cargo fmt
```

Lint with `clippy`:

```sh
cargo clippy --all --all-features
```

Automatically apply possible fixes:

```sh
cargo clippy --fix
```

## 📖 Docs

Build:

```sh
cargo doc --open
```

### ⛓️ Check supply chain

Check the dependency supply chain: licenses (only accept dependencies with OSI or FSF approved licenses), and vulnerabilities (CVE advisories).

```sh
cargo deny check
```

Update dependencies in `Cargo.lock`:

```sh
cargo update
```

### 🏷️ Release

Dry run:

```sh
cargo release patch
```

> Or `minor` / `major`

Create release:

```sh
cargo release patch --execute
```

This will generate the `CHANGELOG.md` and create a git tag for the new version, push this tag to trigger a GitHub actions workflow that will:

- Build and publish the rust crate to crates.io
- Build CLI binaries for every common platform architecture (linux/macos/windows x86/arm)
- Build python wheels for the CLI for common platform architecture, and publish the new version to PyPI
- Create a GitHub release using the git cliff changelog of the new version
