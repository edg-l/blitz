.PHONY: build test test-rust test-lit clean

build:
	cargo build --workspace

test: test-rust test-lit

test-rust:
	cargo test --lib

test-lit: build
	./tests/lit/run_tests.sh

clean:
	cargo clean
