---
applyTo: "**/*.rs"
---

# Agent Instruction Set for a Rust Library

## Prioritize Maintainability

- Adhere to the Rust API Guidelines and their API design checklist (naming, flexibility, type safety, debug, future-proofing, etc.).
- Follow best modular architecture practices
- Stick to stable Rust, avoid overusing dependencies

## Provide a Nice & Ergonomic API for Users

- Lean on the Rust API Guidelines, including: use meaningful naming, hide struct internals to preserve invariants, expose builder patterns, avoid ambiguous boolean parameters, favor strong type safety.
- Follow Carl Kadie’s “Nine Rules for Elegant Rust Library APIs,” such as replacing boolean flags with builder patterns
- Create usage examples you're proud to share—ensure clarity, ergonomics, and readability

## Custom instructions

- Use Rust’s captured identifier shorthand introduced in 1.58: `println!("Value: {x}");` instead of `println!("Value: {}", x);`
- Do not use timeout or gtimeout when running commands
