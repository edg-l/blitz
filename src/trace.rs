//! Tracing infrastructure for Blitz compiler debugging.
//!
//! Controlled by two environment variables:
//!
//! - `BLITZ_DEBUG`: comma-separated list of categories to enable.
//!   Categories: `sched`, `liveness`, `regalloc`, `asm`, `licm`, `egraph`, `dce`, `alias`,
//!   `split`, `phi`, `coalesce`, `slots`, `stats`, `all`.
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

/// Every valid `BLITZ_DEBUG` category. `all` enables the lot.
const CATEGORIES: [&str; 15] = [
    "sched", "liveness", "regalloc", "asm", "licm", "egraph", "dce", "alias", "split", "phi",
    "coalesce", "slots", "stats", "paramsrc", "merges",
];

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
                        "all" => set.extend(CATEGORIES),
                        "" => {}
                        other => match CATEGORIES.iter().find(|c| **c == other) {
                            Some(&known) => {
                                set.insert(known);
                            }
                            None => eprintln!(
                                "warning: unknown BLITZ_DEBUG category '{other}', valid: {}, all",
                                CATEGORIES.join(", ")
                            ),
                        },
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
use crate::regalloc::interference::VRegSet;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;
use std::collections::BTreeMap;

/// Format a schedule with optional barrier group annotations.
pub fn format_schedule(
    insts: &[ScheduledInst],
    vreg_group: Option<&BTreeMap<VReg, usize>>,
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
pub fn format_assignment(map: &BTreeMap<VReg, crate::regalloc::Assignment>) -> String {
    use std::fmt::Write;
    let mut sorted: Vec<_> = map.iter().collect();
    sorted.sort_by_key(|(v, _)| v.0);
    let mut out = String::new();
    for (v, r) in sorted {
        writeln!(out, "  v{} -> {r:?}", v.0).unwrap();
    }
    out
}

/// Format every access to every spill slot in the final instruction stream.
///
/// A slot's traffic is the one thing the register-level dumps cannot show, and
/// three separate wrong-code bugs came down to reading it out of a disassembly
/// by hand. Each line is one slot: its frame displacement, the pass that owns it,
/// then every load and store against it in instruction order with the register
/// moved. A displacement inside the slot region that no pass owns is reported as
/// `UNOWNED` -- the frame does not reserve it.
///
/// The notes after a slot are the shapes that have been bugs:
///
/// - `NEVER STORED` -- every read of it returns whatever the frame held. The
///   machine verifier reports this per path; here it is a whole-function fact,
///   so a slot written on some other path does not hide it.
/// - `FIRST ACCESS IS A LOAD` -- weaker than the above and worth seeing: in
///   program order the value is read before it is written.
/// - `SELF-COPY` -- the only store takes the register the immediately preceding
///   load of the SAME slot wrote, so the slot is copied onto itself and nothing
///   ever puts a real value in it.
///
/// Deliberately not a flag: a slot stored from two different registers. It reads
/// like two values sharing one slot and is almost always one value re-spilled
/// after a reload put it somewhere else, which fired on a third of the slots in
/// the first program tried. The store registers are in the line; judge them
/// there.
/// The displacement a memory operand names, if it is a spill slot.
///
/// A plain `[spill_base + disp]` inside the slot region. An indexed address is
/// not a slot reference: nothing addresses a spill slot that way.
fn slot_disp(
    addr: &crate::x86::addr::Addr,
    spill_base: crate::x86::reg::Reg,
    spill_offset: i32,
    spill_slots: u32,
) -> Option<i32> {
    if addr.base != Some(spill_base) || addr.index.is_some() {
        return None;
    }
    let spill_hi = spill_offset + (spill_slots as i32) * 8;
    (addr.disp >= spill_offset && addr.disp < spill_hi).then_some(addr.disp)
}

/// How many spill stores and reloads the emitted code performs.
///
/// Counted off the instruction stream rather than off the passes that planned
/// them, so a spill three passes disagree about is still counted once, the way
/// the processor will execute it.
pub fn count_slot_traffic(
    insts: &[crate::x86::inst::MachInst],
    spill_base: crate::x86::reg::Reg,
    spill_offset: i32,
    slots: &crate::regalloc::SlotAllocator,
) -> (usize, usize) {
    let spill_slots = slots.count();
    let mut stores = 0;
    let mut loads = 0;
    for inst in insts {
        if let Some(addr) = inst.mem_store_addr()
            && slot_disp(addr, spill_base, spill_offset, spill_slots).is_some()
        {
            stores += 1;
        }
        if let Some(addr) = inst.mem_load_addr()
            && slot_disp(addr, spill_base, spill_offset, spill_slots).is_some()
        {
            loads += 1;
        }
    }
    (stores, loads)
}

pub fn format_slot_traffic(
    insts: &[crate::x86::inst::MachInst],
    spill_base: crate::x86::reg::Reg,
    spill_offset: i32,
    slots: &crate::regalloc::SlotAllocator,
) -> String {
    use std::fmt::Write;

    let spill_slots = slots.count();

    let slot_of =
        |addr: &crate::x86::addr::Addr| slot_disp(addr, spill_base, spill_offset, spill_slots);

    // disp -> accesses, in instruction order.
    let mut traffic: BTreeMap<i32, Vec<(usize, bool, Vec<crate::x86::reg::Reg>)>> = BTreeMap::new();
    for (i, inst) in insts.iter().enumerate() {
        if let Some(addr) = inst.mem_load_addr()
            && let Some(disp) = slot_of(addr)
        {
            traffic
                .entry(disp)
                .or_default()
                .push((i, false, inst.defs()));
        }
        if let Some(addr) = inst.mem_store_addr()
            && let Some(disp) = slot_of(addr)
        {
            // The address registers are a read of the frame pointer, not of the
            // value being stored. Counting them made every slot look like it was
            // stored from two registers.
            let value: Vec<crate::x86::reg::Reg> = inst
                .uses()
                .into_iter()
                .filter(|r| Some(*r) != addr.base && Some(*r) != addr.index)
                .collect();
            traffic.entry(disp).or_default().push((i, true, value));
        }
    }

    let mut out = String::new();
    if spill_slots == 0 {
        writeln!(out, "  no spill slots").unwrap();
        return out;
    }
    for (disp, accesses) in &traffic {
        let regs = |set: &[crate::x86::reg::Reg]| -> String {
            set.iter()
                .map(|r| format!("{r:?}"))
                .collect::<Vec<_>>()
                .join("/")
        };
        let body: Vec<String> = accesses
            .iter()
            .map(|(i, is_store, rs)| {
                let arrow = if *is_store { "<-" } else { "->" };
                format!("[{i}] {arrow} {}", regs(rs))
            })
            .collect();
        let stores: Vec<&(usize, bool, Vec<crate::x86::reg::Reg>)> =
            accesses.iter().filter(|(_, s, _)| *s).collect();

        let mut notes: Vec<String> = Vec::new();
        if stores.is_empty() {
            notes.push("NEVER STORED".to_string());
        } else if accesses.first().is_some_and(|(_, is_store, _)| !is_store) {
            notes.push("FIRST ACCESS IS A LOAD".to_string());
        }
        // The store's source is exactly what the load right before it produced,
        // and that load read this same slot.
        let self_copy = stores.len() == 1
            && accesses.len() == 2
            && !accesses[0].1
            && accesses[0].2 == accesses[1].2;
        if self_copy {
            notes.push("SELF-COPY".to_string());
        }
        let slot = (disp - spill_offset) / 8;
        // Which pass owns the slot, since what a suspicious access means depends
        // on it: an early-barrier slot is stored and reloaded inside one block,
        // a splitter slot spans blocks.
        let owner = slots
            .owner(slot as u32)
            .map(|o| o.as_str())
            .unwrap_or("UNOWNED");
        writeln!(
            out,
            "  slot {slot:>3} {spill_base:?}{disp:+} {owner:>13}  {}{}",
            body.join("  "),
            if notes.is_empty() {
                String::new()
            } else {
                format!("   !! {}", notes.join(", "))
            },
        )
        .unwrap();
    }
    let touched = traffic.len();
    writeln!(out, "  {spill_slots} slot(s) allocated, {touched} touched",).unwrap();
    out
}

/// Format a liveness info's live_at sets.
///
/// Every VReg carries its register class, since a value in `Flags` takes no
/// general register and a value in `XMM` competes with a disjoint budget: a
/// live set read without the classes says nothing about the pressure it makes.
pub fn format_liveness(insts: &[ScheduledInst], live_at: &[VRegSet], live_out: &VRegSet) -> String {
    use std::fmt::Write;
    let classes = crate::regalloc::build_vreg_classes_from_insts(insts);
    let name = |v: u32| match classes.get(&VReg(v)) {
        Some(RegClass::GPR) => format!("v{v}:g"),
        Some(RegClass::XMM) => format!("v{v}:x"),
        Some(RegClass::Flags) => format!("v{v}:f"),
        None => format!("v{v}:?"),
    };
    let mut out = String::new();
    for (i, (inst, live)) in insts.iter().zip(live_at.iter()).enumerate() {
        let mut vregs: Vec<u32> = live.iter().map(|v| v as u32).collect();
        vregs.sort();
        let live_before: Vec<String> = vregs.into_iter().map(name).collect();
        writeln!(
            out,
            "  [{i:>3}] {} = {:?}  live_before=[{}]",
            name(inst.dst.0),
            inst.op,
            live_before.join(", ")
        )
        .unwrap();
    }
    let lo: Vec<String> = live_out.iter().map(|v| name(v as u32)).collect();
    writeln!(out, "  live_out=[{}]", lo.join(", ")).unwrap();
    out
}

#[cfg(test)]
mod slot_traffic_tests {
    use super::format_slot_traffic;
    use crate::regalloc::{SlotAllocator, SlotOwner};
    use crate::x86::addr::Addr;
    use crate::x86::inst::{MachInst, OpSize, Operand};
    use crate::x86::reg::Reg;

    fn slot(disp: i32) -> Addr {
        Addr {
            base: Some(Reg::RSP),
            index: None,
            scale: 1,
            disp,
        }
    }

    fn load(disp: i32, dst: Reg) -> MachInst {
        MachInst::MovRM {
            size: OpSize::S64,
            dst: Operand::Reg(dst),
            addr: slot(disp),
        }
    }

    fn store(disp: i32, src: Reg) -> MachInst {
        MachInst::MovMR {
            size: OpSize::S64,
            addr: slot(disp),
            src: Operand::Reg(src),
        }
    }

    /// `n` splitter-owned slots, which is what the flags below are about.
    fn slots(n: u32) -> SlotAllocator {
        let mut slots = SlotAllocator::new();
        for _ in 0..n {
            slots.alloc(SlotOwner::Splitter);
        }
        slots
    }

    #[test]
    fn a_slot_only_ever_read_is_named() {
        let out = format_slot_traffic(&[load(0, Reg::RBX)], Reg::RSP, 0, &slots(1));
        assert!(out.contains("NEVER STORED"), "{out}");
    }

    /// The shape that cost a session: a routed block parameter's slot reloaded
    /// into a scratch register and stored straight back, so the value the
    /// predecessor was supposed to put there never arrived.
    #[test]
    fn a_slot_copied_onto_itself_is_named() {
        let out = format_slot_traffic(
            &[load(8, Reg::RBP), store(8, Reg::RBP)],
            Reg::RSP,
            0,
            &slots(2),
        );
        assert!(out.contains("SELF-COPY"), "{out}");
    }

    #[test]
    fn an_ordinary_spill_and_reload_is_not_flagged() {
        let out = format_slot_traffic(
            &[store(0, Reg::RBX), load(0, Reg::RCX), load(0, Reg::RDX)],
            Reg::RSP,
            0,
            &slots(1),
        );
        assert!(!out.contains("!!"), "{out}");
    }

    /// The store's address register is not the value being stored. Counting it
    /// made every slot in a real function look suspicious.
    #[test]
    fn the_address_register_is_not_a_stored_value() {
        let out = format_slot_traffic(&[store(0, Reg::RBX)], Reg::RSP, 0, &slots(1));
        assert!(out.contains("<- RBX"), "{out}");
        assert!(!out.contains("RSP/"), "{out}");
    }
}
