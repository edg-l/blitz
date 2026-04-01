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

/// Run CHECK directives against output text.
pub fn run_checks(output: &str, checks: &[CheckPattern]) -> Result<(), CheckError> {
    let lines: Vec<&str> = output.lines().collect();
    let mut pos: usize = 0;

    for (check_idx, check) in checks.iter().enumerate() {
        match check.kind {
            CheckKind::Check => {
                let found = lines[pos..]
                    .iter()
                    .position(|line| check.regex.is_match(line));
                match found {
                    Some(offset) => {
                        pos = pos + offset + 1;
                    }
                    None => {
                        return Err(CheckError {
                            message: format!("CHECK: pattern not found after line {}", pos + 1),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
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
            }
            CheckKind::CheckNot => {
                // Scan from pos to the position of the NEXT Check/CheckLabel, or end.
                let scan_end = find_next_positive_check_end(checks, check_idx, &lines, pos);
                for line in &lines[pos..scan_end] {
                    if check.regex.is_match(line) {
                        return Err(CheckError {
                            message: "CHECK-NOT: pattern found but should not be present".into(),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
                // Don't advance pos for CHECK-NOT
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
                        return Err(CheckError {
                            message: format!(
                                "CHECK-LABEL: pattern not found after line {}",
                                pos + 1
                            ),
                            check_line: check.line_no,
                            pattern: check.raw.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Find the end of the scan range for CHECK-NOT: the line where the next
/// Check or CheckLabel would match, or the end of the output.
fn find_next_positive_check_end(
    checks: &[CheckPattern],
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

    #[test]
    fn check_finds_substring() {
        let output = "line1\nline2 hello world\nline3\n";
        let checks = vec![make_check(CheckKind::Check, "hello", 1)];
        assert!(run_checks(output, &checks).is_ok());
    }

    #[test]
    fn check_not_found() {
        let output = "line1\nline2\nline3\n";
        let checks = vec![make_check(CheckKind::Check, "missing", 1)];
        assert!(run_checks(output, &checks).is_err());
    }

    #[test]
    fn check_next_passes_on_adjacent() {
        let output = "alpha\nbeta\ngamma\n";
        let checks = vec![
            make_check(CheckKind::Check, "alpha", 1),
            make_check(CheckKind::CheckNext, "beta", 2),
        ];
        assert!(run_checks(output, &checks).is_ok());
    }

    #[test]
    fn check_next_fails_on_non_adjacent() {
        let output = "alpha\nfiller\nbeta\n";
        let checks = vec![
            make_check(CheckKind::Check, "alpha", 1),
            make_check(CheckKind::CheckNext, "beta", 2),
        ];
        assert!(run_checks(output, &checks).is_err());
    }

    #[test]
    fn check_not_fails_when_pattern_present() {
        let output = "good\nbad\ngood\n";
        let checks = vec![make_check(CheckKind::CheckNot, "bad", 1)];
        assert!(run_checks(output, &checks).is_err());
    }

    #[test]
    fn check_not_passes_when_absent() {
        let output = "good\nfine\ngreat\n";
        let checks = vec![make_check(CheckKind::CheckNot, "bad", 1)];
        assert!(run_checks(output, &checks).is_ok());
    }

    #[test]
    fn check_label_resets_position() {
        let output = "header1\nalpha\nheader2\nbeta\n";
        let checks = vec![
            make_check(CheckKind::CheckLabel, "header2", 1),
            make_check(CheckKind::Check, "beta", 2),
        ];
        assert!(run_checks(output, &checks).is_ok());
    }
}
