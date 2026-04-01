use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckKind {
    Check,
    CheckNext,
    CheckNot,
    CheckLabel,
}

#[derive(Debug)]
pub struct CheckPattern {
    pub kind: CheckKind,
    pub raw: String,
    pub regex: Regex,
    pub line_no: usize,
}

#[derive(Debug)]
pub enum Directive {
    Run { cmd: String, line_no: usize },
    Check(CheckPattern),
    Exit { code: i32, line_no: usize },
}

/// Build a regex from a pattern string with optional `{{regex}}` segments.
fn build_pattern_regex(raw: &str) -> Result<Regex, String> {
    let mut result = String::new();
    let mut rest = raw;

    loop {
        match rest.find("{{") {
            Some(start) => {
                // Literal part before {{
                let literal = &rest[..start];
                result.push_str(&regex::escape(literal));

                let after_open = &rest[start + 2..];
                match after_open.find("}}") {
                    Some(end) => {
                        let regex_part = &after_open[..end];
                        result.push_str(regex_part);
                        rest = &after_open[end + 2..];
                    }
                    None => {
                        return Err(format!("unmatched '{{{{' in pattern: {raw}"));
                    }
                }
            }
            None => {
                result.push_str(&regex::escape(rest));
                break;
            }
        }
    }

    Regex::new(&result).map_err(|e| format!("invalid regex in pattern '{raw}': {e}"))
}

/// Parse directives from a source file.
pub fn parse_directives(source: &str) -> Result<Vec<Directive>, String> {
    let mut directives = Vec::new();

    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;

        // Find // comment
        let Some(comment_start) = line.find("//") else {
            continue;
        };
        let comment = line[comment_start + 2..].trim_start();

        if let Some(rest) = comment.strip_prefix("RUN:") {
            directives.push(Directive::Run {
                cmd: rest.trim().to_string(),
                line_no,
            });
        } else if let Some(rest) = comment.strip_prefix("CHECK-NEXT:") {
            let raw = rest.trim().to_string();
            let regex = build_pattern_regex(&raw)?;
            directives.push(Directive::Check(CheckPattern {
                kind: CheckKind::CheckNext,
                raw,
                regex,
                line_no,
            }));
        } else if let Some(rest) = comment.strip_prefix("CHECK-NOT:") {
            let raw = rest.trim().to_string();
            let regex = build_pattern_regex(&raw)?;
            directives.push(Directive::Check(CheckPattern {
                kind: CheckKind::CheckNot,
                raw,
                regex,
                line_no,
            }));
        } else if let Some(rest) = comment.strip_prefix("CHECK-LABEL:") {
            let raw = rest.trim().to_string();
            let regex = build_pattern_regex(&raw)?;
            directives.push(Directive::Check(CheckPattern {
                kind: CheckKind::CheckLabel,
                raw,
                regex,
                line_no,
            }));
        } else if let Some(rest) = comment.strip_prefix("CHECK:") {
            let raw = rest.trim().to_string();
            let regex = build_pattern_regex(&raw)?;
            directives.push(Directive::Check(CheckPattern {
                kind: CheckKind::Check,
                raw,
                regex,
                line_no,
            }));
        } else if let Some(rest) = comment.strip_prefix("EXIT:") {
            let code: i32 = rest
                .trim()
                .parse()
                .map_err(|e| format!("line {line_no}: invalid EXIT code: {e}"))?;
            directives.push(Directive::Exit { code, line_no });
        }
    }

    Ok(directives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_directive_types() {
        let source = r#"
// RUN: %tinyc %s -o %t && %t
// CHECK: hello
// CHECK-NEXT: world
// CHECK-NOT: error
// CHECK-LABEL: function main
// EXIT: 42
int main() { return 42; }
"#;
        let directives = parse_directives(source).unwrap();
        assert_eq!(directives.len(), 6);

        assert!(
            matches!(&directives[0], Directive::Run { cmd, .. } if cmd == "%tinyc %s -o %t && %t")
        );
        assert!(
            matches!(&directives[1], Directive::Check(p) if p.kind == CheckKind::Check && p.raw == "hello")
        );
        assert!(
            matches!(&directives[2], Directive::Check(p) if p.kind == CheckKind::CheckNext && p.raw == "world")
        );
        assert!(
            matches!(&directives[3], Directive::Check(p) if p.kind == CheckKind::CheckNot && p.raw == "error")
        );
        assert!(
            matches!(&directives[4], Directive::Check(p) if p.kind == CheckKind::CheckLabel && p.raw == "function main")
        );
        assert!(matches!(&directives[5], Directive::Exit { code: 42, .. }));
    }

    #[test]
    fn regex_pattern_compiles_and_matches() {
        let source = r#"// CHECK: iconst({{[0-9]+}}, I32)"#;
        let directives = parse_directives(source).unwrap();
        assert_eq!(directives.len(), 1);
        if let Directive::Check(p) = &directives[0] {
            assert!(p.regex.is_match("    v0 = iconst(42, I32)"));
            assert!(!p.regex.is_match("    v0 = iconst(hello, I32)"));
        } else {
            panic!("expected Check directive");
        }
    }

    #[test]
    fn plain_pattern_is_escaped() {
        let source = r#"// CHECK: a+b"#;
        let directives = parse_directives(source).unwrap();
        if let Directive::Check(p) = &directives[0] {
            // The "+" should be escaped, so "aaab" should NOT match
            assert!(!p.regex.is_match("aaab"));
            assert!(p.regex.is_match("a+b"));
        } else {
            panic!("expected Check directive");
        }
    }
}
