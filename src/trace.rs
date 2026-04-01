//! Tracing infrastructure for Blitz compiler debugging.
//!
//! Controlled by two environment variables:
//!
//! - `BLITZ_DEBUG`: comma-separated list of categories to enable.
//!   Categories: `sched`, `liveness`, `regalloc`, `asm`, `all`.
//!
//! - `BLITZ_DEBUG_FN`: optional substring filter on function names.
//!   When set, only functions whose name contains this string produce output.
//!
//! Example: `BLITZ_DEBUG=sched,regalloc BLITZ_DEBUG_FN=swap cargo test`

use std::collections::HashSet;
use std::fmt;
use std::io;
use std::sync::OnceLock;
use std::time::Instant;

use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;

/// Global debug configuration, parsed once from env vars.
static CONFIG: OnceLock<BlitzDebugConfig> = OnceLock::new();

/// Process start time for dmesg-style timestamps.
static START: OnceLock<Instant> = OnceLock::new();

struct BlitzDebugConfig {
    categories: HashSet<&'static str>,
    fn_filter: Option<String>,
}

fn start_time() -> &'static Instant {
    START.get_or_init(Instant::now)
}

fn config() -> &'static BlitzDebugConfig {
    CONFIG.get_or_init(|| {
        let categories = match std::env::var("BLITZ_DEBUG") {
            Ok(val) => {
                let mut set = HashSet::new();
                for part in val.split(',') {
                    let part = part.trim().to_ascii_lowercase();
                    match part.as_str() {
                        "all" => {
                            set.extend(["sched", "liveness", "regalloc", "asm"]);
                        }
                        "sched" | "liveness" | "regalloc" | "asm" => {
                            set.insert(match part.as_str() {
                                "sched" => "sched",
                                "liveness" => "liveness",
                                "regalloc" => "regalloc",
                                "asm" => "asm",
                                _ => unreachable!(),
                            });
                        }
                        "" => {}
                        other => {
                            eprintln!(
                                "warning: unknown BLITZ_DEBUG category '{other}', \
                                 valid: sched, liveness, regalloc, asm, all"
                            );
                        }
                    }
                }
                set
            }
            Err(_) => HashSet::new(),
        };

        let fn_filter = std::env::var("BLITZ_DEBUG_FN")
            .ok()
            .filter(|s| !s.is_empty());

        BlitzDebugConfig {
            categories,
            fn_filter,
        }
    })
}

/// Returns true if the given debug category is enabled via `BLITZ_DEBUG`.
pub fn is_enabled(category: &str) -> bool {
    config().categories.contains(category)
}

/// Returns true if debug output should fire for the given function name.
///
/// Always returns true if `BLITZ_DEBUG_FN` is not set.
/// Otherwise returns true if `func_name` contains the filter as a substring.
pub fn fn_matches(func_name: &str) -> bool {
    match &config().fn_filter {
        None => true,
        Some(filter) => func_name.contains(filter.as_str()),
    }
}

/// Returns true if any BLITZ_DEBUG category is enabled.
pub fn any_enabled() -> bool {
    !config().categories.is_empty()
}

/// Dmesg-style timer: `[  elapsed_ms]`.
struct DmesgTimer;

impl FormatTime for DmesgTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let elapsed = start_time().elapsed();
        let ms = elapsed.as_millis();
        write!(w, "[{ms:>8}]")
    }
}

/// Install the global tracing subscriber. Safe to call multiple times (no-op after first).
///
/// When `BLITZ_DEBUG` is set, installs a subscriber that outputs:
///   `[  elapsed_ms] LEVEL target message`
///
/// When `BLITZ_DEBUG` is not set, installs a subscriber with OFF level (zero cost).
pub fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        // Ensure the start time is captured early.
        let _ = start_time();

        let level = if any_enabled() {
            tracing::Level::DEBUG
        } else {
            tracing::Level::ERROR // effectively off for our debug! calls
        };

        let subscriber = tracing_subscriber::fmt()
            .with_timer(DmesgTimer)
            .with_ansi(false)
            .with_target(true)
            .with_level(true)
            .with_max_level(level)
            .with_writer(io::stderr)
            .finish();

        // Ignore error if another subscriber was already set.
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

// ── Format helpers for dump points ──────────────────────────────────────────

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use std::collections::HashMap;

/// Format a schedule with optional barrier group annotations.
pub fn format_schedule(
    insts: &[ScheduledInst],
    vreg_group: Option<&HashMap<VReg, usize>>,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (i, inst) in insts.iter().enumerate() {
        let ops: Vec<u32> = inst.operands.iter().map(|v| v.0).collect();
        if let Some(groups) = vreg_group {
            let g = groups.get(&inst.dst).copied().unwrap_or(0);
            writeln!(
                out,
                "  [{i:>3}] v{} = {:?}({ops:?}) g={g}",
                inst.dst.0, inst.op
            )
            .unwrap();
        } else {
            writeln!(out, "  [{i:>3}] v{} = {:?}({ops:?})", inst.dst.0, inst.op).unwrap();
        }
    }
    out
}

/// Format a VReg-to-Reg mapping sorted by VReg index.
pub fn format_vreg_to_reg(map: &HashMap<VReg, crate::x86::reg::Reg>) -> String {
    use std::fmt::Write;
    let mut sorted: Vec<_> = map.iter().collect();
    sorted.sort_by_key(|(v, _)| v.0);
    let mut out = String::new();
    for (v, r) in sorted {
        writeln!(out, "  v{} -> {r:?}", v.0).unwrap();
    }
    out
}

/// Format a liveness info's live_at sets.
pub fn format_liveness(
    insts: &[ScheduledInst],
    live_at: &[HashSet<VReg>],
    live_out: &HashSet<VReg>,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (i, (inst, live)) in insts.iter().zip(live_at.iter()).enumerate() {
        let mut vregs: Vec<u32> = live.iter().map(|v| v.0).collect();
        vregs.sort();
        writeln!(
            out,
            "  [{i:>3}] v{} = {:?}  live_before={vregs:?}",
            inst.dst.0, inst.op
        )
        .unwrap();
    }
    let mut lo: Vec<u32> = live_out.iter().map(|v| v.0).collect();
    lo.sort();
    writeln!(out, "  live_out={lo:?}").unwrap();
    out
}
