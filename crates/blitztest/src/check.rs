use crate::directive::{CheckKind, CheckPattern};

#[derive(Debug)]
pub struct CheckError {
    pub message: String,
    pub check_line: usize,
    pub pattern: String,
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "check line {}: {} (pattern: '{}')",
            self.check_line, self.message, self.pattern
        )
    }
}

/// Format a context snippet showing lines around a search position.
/// `pos` is the 0-based line index where the search starts.
fn format_context(lines: &[&str], pos: usize) -> String {
    let start = pos.saturating_sub(3);
    let end = (pos + 3).min(lines.len());
    let mut s = String::from("  output near search start:\n");
    for i in start..end {
        let marker = if i == pos {
            "  <-- search starts here"
        } else {
            ""
        };
        s.push_str(&format!("    {}: {}{}\n", i + 1, lines[i], marker));
    }
    s
}

/// Run CHECK directives against output text.
pub fn run_checks(output: &str, checks: &[&CheckPattern]) -> Result<(), CheckError> {
    let lines: Vec<&str> = output.lines().collect();
    let mut pos: usize = 0;
    let mut check_idx = 0;

    while check_idx < checks.len() {
        let check = checks[check_idx];

        match &check.kind {
            CheckKind::Check => {
                let found = lines[pos..]
                    .iter()
                    .position(|line| check.regex.is_match(line));
                match found {
                    Some(offset) => {
                        pos = pos + offset + 1;
                    }
                    None => {
                        let ctx = format_context(&lines, pos);
                        return Err(CheckError {
                            message: format!(
                                "CHECK: pattern not found after line {}\n{}",
                                pos + 1,
                                ctx
                            ),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
                check_idx += 1;
            }
            CheckKind::CheckNext => {
                if pos >= lines.len() {
                    return Err(CheckError {
                        message: "CHECK-NEXT: no more output lines".into(),
                        check_line: check.line_no,
                        pattern: check.raw.clone(),
                    });
                }
                if !check.regex.is_match(lines[pos]) {
                    return Err(CheckError {
                        message: format!(
                            "CHECK-NEXT: expected on next line (line {}), got: '{}'",
                            pos + 1,
                            lines[pos]
                        ),
                        check_line: check.line_no,
                        pattern: check.raw.clone(),
                    });
                }
                pos += 1;
                check_idx += 1;
            }
            CheckKind::CheckNot => {
                // Scan from pos to the position of the NEXT Check/CheckLabel, or end.
                let scan_end = find_next_positive_check_end(checks, check_idx, &lines, pos);
                for (i, line) in lines[pos..scan_end].iter().enumerate() {
                    if check.regex.is_match(line) {
                        return Err(CheckError {
                            message: format!(
                                "CHECK-NOT: pattern found but should not be present (line {}): '{}'",
                                pos + i + 1,
                                line
                            ),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
                // Don't advance pos for CHECK-NOT
                check_idx += 1;
            }
            CheckKind::CheckLabel => {
                let found = lines[pos..]
                    .iter()
                    .position(|line| check.regex.is_match(line));
                match found {
                    Some(offset) => {
                        pos = pos + offset + 1;
                    }
                    None => {
                        let ctx = format_context(&lines, pos);
                        return Err(CheckError {
                            message: format!(
                                "CHECK-LABEL: pattern not found after line {}\n{}",
                                pos + 1,
                                ctx
                            ),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
                check_idx += 1;
            }
            CheckKind::CheckCount { n } => {
                let count = lines[pos..]
                    .iter()
                    .filter(|line| check.regex.is_match(line))
                    .count();
                if count != *n {
                    return Err(CheckError {
                        message: format!(
                            "CHECK-COUNT-{}: expected {} occurrences but found {}",
                            n, n, count
                        ),
                        check_line: check.line_no,
                        pattern: check.raw.clone(),
                    });
                }
                // Don't advance pos for CHECK-COUNT
                check_idx += 1;
            }
            CheckKind::CheckDag => {
                // Collect all consecutive CHECK-DAG directives.
                let dag_start = check_idx;
                let mut dag_end = check_idx;
                while dag_end < checks.len() && matches!(checks[dag_end].kind, CheckKind::CheckDag)
                {
                    dag_end += 1;
                }
                let dag_group = &checks[dag_start..dag_end];

                // For each DAG pattern, find the first unmatched line in [pos..] that matches it.
                // No two patterns may claim the same line.
                let mut used: Vec<bool> = vec![false; lines.len()];
                let mut max_matched_line = pos;

                for dag_check in dag_group {
                    let found = lines[pos..]
                        .iter()
                        .enumerate()
                        .position(|(i, line)| !used[pos + i] && dag_check.regex.is_match(line));
                    match found {
                        Some(offset) => {
                            let abs = pos + offset;
                            used[abs] = true;
                            if abs + 1 > max_matched_line {
                                max_matched_line = abs + 1;
                            }
                        }
                        None => {
                            return Err(CheckError {
                                message: "CHECK-DAG: pattern not found in remaining output".into(),
                                check_line: dag_check.line_no,
                                pattern: dag_check.raw.clone(),
                            });
                        }
                    }
                }

                pos = max_matched_line;
                check_idx = dag_end;
            }
            CheckKind::CheckSame => {
                if pos == 0 {
                    return Err(CheckError {
                        message: "CHECK-SAME: no previous line to match against".into(),
                        check_line: check.line_no,
                        pattern: check.raw.clone(),
                    });
                }
                let prev_line = lines[pos - 1];
                if !check.regex.is_match(prev_line) {
                    return Err(CheckError {
                        message: format!(
                            "CHECK-SAME: pattern not found on line {} (same line as previous check): '{}'",
                            pos, prev_line
                        ),
                        check_line: check.line_no,
                        pattern: check.raw.clone(),
                    });
                }
                // Don't advance pos for CHECK-SAME
                check_idx += 1;
            }
        }
    }

    Ok(())
}

/// Find the end of the scan range for CHECK-NOT: the line where the next
/// Check or CheckLabel would match, or the end of the output.
fn find_next_positive_check_end(
    checks: &[&CheckPattern],
    current_idx: usize,
    lines: &[&str],
    pos: usize,
) -> usize {
    // Look for the next Check or CheckLabel after current_idx
    for check in &checks[current_idx + 1..] {
        if matches!(check.kind, CheckKind::Check | CheckKind::CheckLabel) {
            // Find where this check would match
            if let Some(offset) = lines[pos..]
                .iter()
                .position(|line| check.regex.is_match(line))
            {
                return pos + offset;
            }
        }
    }
    lines.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    fn make_check(kind: CheckKind, raw: &str, line_no: usize) -> CheckPattern {
        CheckPattern {
            kind,
            raw: raw.to_string(),
            regex: Regex::new(&regex::escape(raw)).unwrap(),
            line_no,
        }
    }

    fn as_refs(checks: &[CheckPattern]) -> Vec<&CheckPattern> {
        checks.iter().collect()
    }

    #[test]
    fn check_finds_substring() {
        let output = "line1\nline2 hello world\nline3\n";
        let checks = vec![make_check(CheckKind::Check, "hello", 1)];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_not_found() {
        let output = "line1\nline2\nline3\n";
        let checks = vec![make_check(CheckKind::Check, "missing", 1)];
        assert!(run_checks(output, &as_refs(&checks)).is_err());
    }

    #[test]
    fn check_next_passes_on_adjacent() {
        let output = "alpha\nbeta\ngamma\n";
        let checks = vec![
            make_check(CheckKind::Check, "alpha", 1),
            make_check(CheckKind::CheckNext, "beta", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_next_fails_on_non_adjacent() {
        let output = "alpha\nfiller\nbeta\n";
        let checks = vec![
            make_check(CheckKind::Check, "alpha", 1),
            make_check(CheckKind::CheckNext, "beta", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_err());
    }

    #[test]
    fn check_not_fails_when_pattern_present() {
        let output = "good\nbad\ngood\n";
        let checks = vec![make_check(CheckKind::CheckNot, "bad", 1)];
        assert!(run_checks(output, &as_refs(&checks)).is_err());
    }

    #[test]
    fn check_not_passes_when_absent() {
        let output = "good\nfine\ngreat\n";
        let checks = vec![make_check(CheckKind::CheckNot, "bad", 1)];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_label_resets_position() {
        let output = "header1\nalpha\nheader2\nbeta\n";
        let checks = vec![
            make_check(CheckKind::CheckLabel, "header2", 1),
            make_check(CheckKind::Check, "beta", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    // CHECK-COUNT tests

    #[test]
    fn check_count_passes_exact() {
        let output = "call foo\ncall bar\ncall baz\n";
        let checks = vec![make_check(CheckKind::CheckCount { n: 3 }, "call", 1)];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_count_fails_too_few() {
        let output = "call foo\ncall bar\n";
        let checks = vec![make_check(CheckKind::CheckCount { n: 3 }, "call", 1)];
        let err = run_checks(output, &as_refs(&checks)).unwrap_err();
        assert!(err.message.contains("expected 3 occurrences but found 2"));
    }

    #[test]
    fn check_count_fails_too_many() {
        let output = "call foo\ncall bar\ncall baz\ncall qux\n";
        let checks = vec![make_check(CheckKind::CheckCount { n: 3 }, "call", 1)];
        let err = run_checks(output, &as_refs(&checks)).unwrap_err();
        assert!(err.message.contains("expected 3 occurrences but found 4"));
    }

    // CHECK-DAG tests

    #[test]
    fn check_dag_passes_any_order() {
        // Patterns appear in reverse order in the output.
        let output = "line1\npattern_b\npattern_a\nline4\n";
        let checks = vec![
            make_check(CheckKind::CheckDag, "pattern_a", 1),
            make_check(CheckKind::CheckDag, "pattern_b", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_dag_fails_missing() {
        let output = "line1\npattern_a\nline3\n";
        let checks = vec![
            make_check(CheckKind::CheckDag, "pattern_a", 1),
            make_check(CheckKind::CheckDag, "pattern_missing", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_err());
    }

    // CHECK-SAME tests

    #[test]
    fn check_same_passes() {
        let output = "hello world\nother line\n";
        let checks = vec![
            make_check(CheckKind::Check, "hello", 1),
            make_check(CheckKind::CheckSame, "world", 2),
        ];
        assert!(run_checks(output, &as_refs(&checks)).is_ok());
    }

    #[test]
    fn check_same_fails() {
        let output = "hello\nworld\n";
        let checks = vec![
            make_check(CheckKind::Check, "hello", 1),
            make_check(CheckKind::CheckSame, "world", 2),
        ];
        // "hello" is on line 1, "world" is on line 2. CHECK-SAME checks line 1.
        let err = run_checks(output, &as_refs(&checks)).unwrap_err();
        assert!(err.message.contains("CHECK-SAME"));
    }
}
