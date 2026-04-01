// blitztest - FileCheck-style pattern matching tool for Blitz compiler tests.
//
// Usage: blitztest <check-file> [< input]
//
// Reads CHECK directives from <check-file> (in // comments).
// Matches them against stdin.
//
// Directives:
//   // CHECK: <pattern>       - pattern must appear (in order)
//   // CHECK-NEXT: <pattern>  - pattern must appear on the very next line
//   // CHECK-NOT: <pattern>   - pattern must NOT appear before the next CHECK
//   // CHECK-LABEL: <pattern> - like CHECK, but resets the scan position
//
// Patterns are literal substrings by default. Use {{regex}} for regex.
//
// Exit code: 0 on success, 1 on check failure, 2 on usage error.

use std::io::Read;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: blitztest <check-file>");
        eprintln!("  Reads CHECK directives from <check-file>.");
        eprintln!("  Matches them against stdin.");
        exit(2);
    }

    let check_file = &args[1];

    let source = match std::fs::read_to_string(check_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("blitztest: cannot read '{}': {}", check_file, e);
            exit(2);
        }
    };

    let directives = match blitztest::directive::parse_directives(&source) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("blitztest: {}: {}", check_file, e);
            exit(2);
        }
    };

    let checks: Vec<&blitztest::directive::CheckPattern> = directives
        .iter()
        .filter_map(|d| match d {
            blitztest::directive::Directive::Check(p) => Some(p),
            _ => None,
        })
        .collect();

    if checks.is_empty() {
        eprintln!("blitztest: {}: no CHECK directives found", check_file);
        exit(2);
    }

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .unwrap_or_else(|e| {
            eprintln!("blitztest: cannot read stdin: {}", e);
            exit(2);
        });

    match blitztest::check::run_checks(&input, &checks) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("blitztest: {}: {}", check_file, e);
            exit(1);
        }
    }
}
