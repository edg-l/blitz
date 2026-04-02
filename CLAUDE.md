# Blitz Compiler

## Testing

Lit tests live in `tests/lit/` and use blitztest for FileCheck-style pattern matching.
See `crates/blitztest/how-to-use.txt` for the full directive reference (CHECK, CHECK-LABEL, CHECK-NOT, CHECK-NEXT, CHECK-SAME, CHECK-COUNT-N, CHECK-DAG, EXIT, RUN).

Run lit tests: `bash tests/lit/run_tests.sh`
Run unit tests: `cargo test`
