//! Pressure-driven live-range splitter for Blitz.
//!
//! Runs AFTER scheduling and effectful-op operand population, BEFORE
//! `allocate_global`. `plan_splits` scans each block for register-pressure
//! overshoots and inserts either rematerialization or spill/reload pairs to
//! bring pressure within budget.

use std::collections::{BTreeMap, BTreeSet};

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::cost::CostModel;
use crate::egraph::extract::{ClassVRegMap, ExtractionResult, extract_at_with_memo};
use crate::ir::effectful::BlockId;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::regalloc::global_liveness::GlobalLiveness;
use crate::regalloc::spill::LOOP_DEPTH_PENALTY_BASE;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::CALLER_SAVED_GPR;
use crate::x86::reg::RegClass;

// Re-export the VReg type used throughout.
use crate::egraph::extract::VReg;

/// The kind of split to apply to a victim live range.
///
/// Note: block-param splitting (Phase 6) uses a dedicated
/// `detect_blockparam_call_crossings` pass rather than routing through this
/// enum. Block-param VRegs have no explicit def instruction in the block, so
/// the pressure-victim-picker cannot determine a def site or cost for them.
/// The dedicated pass reconstructs the necessary context (param index, block
/// id, predecessor edges) independently and emits `XmmSpillLoad` insertions
/// and `slot_spilled_params` entries directly. `SlotSpillBlockParam` was
/// removed because it was never constructed; the implementation diverged from
/// the original single-dispatch design.
#[derive(Debug, Clone)]
pub(crate) enum SplitKind {
    /// Rematerialise the value at each use site by re-emitting the cheap op.
    Remat(Op),
    /// Spill to a stack slot: insert `SpillStore` after the def,
    /// `SpillLoad`/`XmmSpillLoad` before each use.
    SlotSpill,
}

/// Type alias for the map of slot-spilled block params.
///
/// Keyed by `(block_id: BlockId, param_idx: u32)`, value is spill info.
/// Passed to `lower_terminator` so predecessor blocks can emit slot stores.
pub type BlockParamSlotMap = BTreeMap<(BlockId, u32), SlotSpilledParamInfo>;

/// Information about a block param that is being slot-spilled (Phase 6).
///
/// When a block param VReg is live across a call in its block, we spill it
/// to a stack slot via the phi-copy-to-slot strategy: predecessor terminators
/// store the arg reg to the slot, uses in the block reload from the slot.
#[derive(Debug, Clone)]
pub struct SlotSpilledParamInfo {
    /// The VReg assigned to this block param (from `class_to_vreg`).
    pub vreg: VReg,
    /// The allocated spill slot number.
    pub slot: i64,
    /// The register class (GPR or XMM).
    pub reg_class: RegClass,
    /// The block index (for `truncate_segment_start`).
    pub block_idx: usize,
}

/// Output of `plan_splits`: new instructions to insert and new segments to add.
pub struct SplitPlan {
    /// For each block: instructions to insert at specific positions.
    /// Each entry is `(insert_before_inst_idx, instruction)`.
    pub per_block_insertions: Vec<Vec<(usize, ScheduledInst)>>,
    /// New `(class, vreg, start, end)` segments for `class_to_vreg`.
    pub new_segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    /// Updated VReg operands: `(block_idx, inst_idx, operand_idx, new_vreg)`.
    pub operand_rewrites: Vec<(usize, usize, usize, VReg)>,
    /// Block params that are slot-spilled (Phase 6).
    ///
    /// Key: `(block_id, param_idx)` identifying the block param in the func.
    /// Value: slot info (VReg to truncate, slot number, class, block_idx).
    pub slot_spilled_params: BTreeMap<(BlockId, u32), SlotSpilledParamInfo>,
    /// VRegs whose class_to_vreg segments must be truncated at the END to the
    /// given ProgramPoint. Used by cross-block slot spills to shorten the
    /// spilled VReg's live range so the terminator point maps to a reload VReg.
    pub segment_end_truncations: Vec<(VReg, ProgramPoint)>,
    /// Total number of spill slots allocated by the splitter. The caller must
    /// include this in the function's stack frame size calculation.
    pub slots_allocated: u32,
}

impl SplitPlan {
    /// Whether the plan would change nothing.
    ///
    /// The caller re-plans until this holds, so "nothing to do" has to mean
    /// exactly that: an empty plan applied in a loop would never terminate.
    pub fn is_empty(&self) -> bool {
        self.per_block_insertions.iter().all(|b| b.is_empty())
            && self.slot_spilled_params.is_empty()
            && self.operand_rewrites.is_empty()
            && self.new_segments.is_empty()
            && self.segment_end_truncations.is_empty()
    }
}

/// How many times the caller re-plans splits before giving up.
///
/// Each round removes at least one value from the worst pressure point in some
/// block, so the bound only has to exceed the deepest overshoot seen in
/// practice (18 GPR colors); it exists to turn a splitter that fails to
/// converge into a diagnosable error rather than a hang.
pub const MAX_SPLIT_ROUNDS: usize = 48;

/// Cost above which we prefer SlotSpill over Remat.
///
/// If `CostModel::cost(op) * loop_depth_penalty > SLOT_STORE_LOAD_COST`,
/// Remat is more expensive than slot spill so we use SlotSpill.
const SLOT_STORE_LOAD_COST: f64 = 5.0;

// ── Local liveness ────────────────────────────────────────────────────────────

/// Compute per-instruction live-before sets for a single block.
///
/// The scan is seeded from `live_out_seed` (the block's `GlobalLiveness::live_out`
/// entry) so values defined in this block that are consumed in successor blocks are
/// correctly tracked as live throughout their range in this block.
///
/// Returns a `Vec` of length `n_insts + 1` where `result[i]` is the set of
/// VRegs live immediately BEFORE instruction `i`. `result[n_insts]` is the
/// live-out set (equal to `live_out_seed` plus any locally-used pass-through
/// values that survive backward from uses inside the block).
fn compute_local_liveness(
    block_idx: usize,
    schedule: &[ScheduledInst],
    live_out_seed: &BTreeSet<VReg>,
) -> Vec<BTreeSet<VReg>> {
    let n = schedule.len();
    // result[i] = live set before instruction i.
    // result[n] = live set at exit of block (= live_out_seed for this block).
    let mut result: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n + 1];

    // Seed the backward scan from live_out: values live at the exit of the block.
    let mut live: BTreeSet<VReg> = live_out_seed.clone();
    result[n] = live.clone();

    // Walk backward.
    for i in (0..n).rev() {
        let inst = &schedule[i];

        // Standard backward liveness: live-before(I) = USE(I) ∪ (live-after(I) \ DEF(I)).
        // Remove def first (before adding uses) to handle the case where live-after
        // already contains def (possible when seeding from live_out, where values
        // defined in this block that flow to successors are included).
        live.remove(&inst.dst);

        // Add uses to live (a value used by I is live before I).
        for &use_vreg in &inst.operands {
            live.insert(use_vreg);
        }

        // Record live-before[i].
        result[i] = live.clone();
    }

    // Suppress unused variable warning for block_idx (kept for callers that need it).
    let _ = block_idx;

    result
}

// ── Pressure computation ──────────────────────────────────────────────────────

/// Compute per-instruction register pressure for a specific register class.
///
/// Returns a `Vec<u32>` of length `live_sets.len()` where each entry is the
/// count of live VRegs of the specified `class` at that position.
fn compute_pressure_for_class(
    live_sets: &[BTreeSet<VReg>],
    schedule: &[ScheduledInst],
    vreg_classes: &BTreeMap<VReg, RegClass>,
    class: RegClass,
) -> Vec<u32> {
    live_sets
        .iter()
        .enumerate()
        .map(|(i, live)| {
            let mut n = live
                .iter()
                .filter(|&v| vreg_classes.get(v).copied() == Some(class))
                .count() as u32;
            // Count the definition, not just what is live before it.
            // `interference.rs` makes every def interfere with everything live
            // before it, so the clique the allocator sees here is
            // `live_before ∪ {dst}` -- one more than the live count. Measuring
            // only the live count and testing `> budget` let a point with
            // exactly `budget` live values through while the allocator needed
            // `budget + 1` colors for it, which is the whole overshoot on most
            // of the programs that fail to allocate.
            if let Some(inst) = schedule.get(i)
                && !inst.op.has_no_result()
                && vreg_classes.get(&inst.dst).copied() == Some(class)
                && !live.contains(&inst.dst)
            {
                n += 1;
            }
            n
        })
        .collect()
}

// ── Victim scoring ────────────────────────────────────────────────────────────

/// Score a victim candidate; higher score = better spill target.
///
/// Uses the same loop-depth-penalized heuristic as `spill.rs`:
/// - Prefers values with long live ranges (more pressure relief).
/// - Penalises values inside deep loops (prefer not to spill inside loops).
fn score_victim(_vreg: VReg, live_range_length: usize, loop_depth: u32) -> u64 {
    let penalty = LOOP_DEPTH_PENALTY_BASE.saturating_pow(loop_depth).max(1);
    (live_range_length as u64).saturating_div(penalty)
}

/// Compute approximate live-range lengths for VRegs in a single block.
///
/// `live_sets[i]` is live-before of instruction i. Length is defined as the
/// number of instructions at which the VReg appears live.
fn compute_live_range_lengths(live_sets: &[BTreeSet<VReg>]) -> BTreeMap<VReg, usize> {
    let mut lengths: BTreeMap<VReg, usize> = BTreeMap::new();
    for live in live_sets {
        for &v in live {
            *lengths.entry(v).or_insert(0) += 1;
        }
    }
    lengths
}

// ── Split kind selection ──────────────────────────────────────────────────────

/// Determine whether a victim should be rematerialised or slot-spilled.
///
/// Uses the e-graph's `extract_at_with_memo` to find the cheapest op for
/// `victim_class` given the set of classes live at the use point. If
/// `extract_at` returns a free-remat op (cost <= `SLOT_STORE_LOAD_COST` after
/// loop-depth penalty), `SplitKind::Remat` is returned; otherwise `SlotSpill`.
///
/// Call-arg VRegs are NEVER rematerialised (see CLAUDE.md): they must remain
/// live at the call point.
#[allow(clippy::too_many_arguments)]
fn choose_split_kind(
    victim: VReg,
    loop_depth: u32,
    call_arg_vregs: &BTreeSet<VReg>,
    victim_class: Option<ClassId>,
    live_classes_at_use: &BTreeSet<ClassId>,
    egraph: &EGraph,
    cost_model: &CostModel,
    extraction: &ExtractionResult,
) -> SplitKind {
    // Call-arg VRegs must never be remat'd.
    if call_arg_vregs.contains(&victim) {
        return SplitKind::SlotSpill;
    }

    if let Some(class) = victim_class
        && let Some(extracted) = extract_at_with_memo(
            egraph,
            class,
            live_classes_at_use,
            cost_model,
            &extraction.choices,
        )
    {
        // Cheap is not the same as recomputable. `BlockParam`, `Param`,
        // `CallResult` and `LoadResult` are all leaves of cost ~0, so cost alone
        // waved them through -- but they compute nothing. Their value is
        // *already* in a register at one particular point, put there by a
        // predecessor's phi copy, by the caller, or by the instruction before.
        // Re-emitting the pseudo-op mints a VReg that no instruction ever
        // writes, and the allocator hands it a register nothing initialises.
        // A `BlockParam(1, 0)` re-emitted mid-block got RAX, which still held
        // the phi *source*, so a `for (i = 0; i < 5; i++)` loop tested a copy of
        // the counter that the latch never incremented and never terminated
        // (findings/seed34_loop_never_terminates.c).
        //
        // `Op::is_rematerializable` is the existing answer to this question --
        // `Iconst`, `StackAddr`, `GlobalAddr`, the three that genuinely
        // recompute from nothing -- and the splitter simply never asked it.
        let penalty = LOOP_DEPTH_PENALTY_BASE.saturating_pow(loop_depth).max(1) as f64;
        if extracted.op.is_rematerializable() && extracted.cost * penalty <= SLOT_STORE_LOAD_COST {
            return SplitKind::Remat(extracted.op);
        }
    }

    SplitKind::SlotSpill
}

/// Collect all call-arg VRegs in a schedule (used by CallResult/VoidCallBarrier).
fn collect_call_arg_vregs_set(schedule: &[ScheduledInst]) -> BTreeSet<VReg> {
    let mut set = BTreeSet::new();
    for inst in schedule {
        if matches!(inst.op, Op::CallResult(..) | Op::VoidCallBarrier) {
            for &op in &inst.operands {
                set.insert(op);
            }
        }
    }
    set
}

/// Compute call-crossing overshoot per call point in the schedule.
///
/// For each instruction that is a call (`CallResult` or `VoidCallBarrier`),
/// count GPR values that are live BEFORE the instruction. These values must
/// survive the call using callee-saved registers only. If the count exceeds
/// `callee_saved_budget`, return `(inst_idx, count, excess)` for the worst
/// call point.
///
/// `callee_saved_budget` = total GPRs − caller-saved GPRs.
fn find_call_crossing_overshoot(
    live_sets: &[BTreeSet<VReg>],
    schedule: &[ScheduledInst],
    vreg_classes: &BTreeMap<VReg, RegClass>,
    class: RegClass,
    callee_saved_budget: u32,
    call_arg_vregs: &BTreeSet<VReg>,
) -> Option<(usize, u32)> {
    let mut worst: Option<(usize, u32)> = None;
    for (inst_idx, inst) in schedule.iter().enumerate() {
        if !matches!(inst.op, Op::CallResult(..) | Op::VoidCallBarrier) {
            continue;
        }
        // Count values of `class` live before this call instruction.
        //
        // Arguments to this call are excluded when they die at it: they must
        // occupy their ABI register at the call, so they are not candidates for
        // being kept anywhere else, and counting them would demand a spill that
        // cannot help. This mirrors `exclude_call_args` in the allocator's
        // `add_clobber_interferences`.
        let live_after: Option<&BTreeSet<VReg>> = live_sets.get(inst_idx + 1);
        let live_before = &live_sets[inst_idx];
        let live_count: u32 = live_before
            .iter()
            .filter(|&&v| vreg_classes.get(&v).copied() == Some(class))
            .filter(|&&v| {
                !call_arg_vregs.contains(&v) || live_after.is_some_and(|la| la.contains(&v))
            })
            .count() as u32;
        if live_count > callee_saved_budget {
            let excess = live_count - callee_saved_budget;
            match worst {
                None => worst = Some((inst_idx, excess)),
                Some((_, worst_excess)) => {
                    if excess > worst_excess {
                        worst = Some((inst_idx, excess));
                    }
                }
            }
        }
    }
    worst
}

// ── Plan construction ─────────────────────────────────────────────────────────

/// Apply splits for a set of victims at an overshoot point in a single block.
///
/// Used by both the standard pressure path and the call-crossing path so the
/// same logic is not duplicated.
/// Whether the overshoot requires cross-block spill (call-crossing) or per-block split.
///
/// - `PerBlock`: victim's uses are in the same block; use `apply_split_planned`.
/// - `CrossBlock`: victim is live-out only; use `apply_cross_block_slot_spill`.
#[derive(Clone, Copy)]
enum SplitScope {
    PerBlock,
    CrossBlock,
}

#[allow(clippy::too_many_arguments)]
fn apply_splits_for_overshoot(
    fn_name: &str,
    block_idx: usize,
    overshoot_inst_idx: usize,
    overshoot_class: RegClass,
    overshoot_excess: u32,
    scope: SplitScope,
    live_sets: &[BTreeSet<VReg>],
    all_block_schedules: &[Vec<ScheduledInst>],
    vreg_classes: &BTreeMap<VReg, RegClass>,
    call_arg_vregs: &BTreeSet<VReg>,
    def_inst_map: &BTreeMap<VReg, usize>,
    range_lengths: &BTreeMap<VReg, usize>,
    loop_depths: &BTreeMap<VReg, u32>,
    class_to_vreg: &ClassVRegMap,
    egraph: &EGraph,
    cost_model: &CostModel,
    extraction: &ExtractionResult,
    next_vreg: &mut u32,
    new_slot_count: &mut u32,
    all_per_block_insertions: &mut [Vec<(usize, ScheduledInst)>],
    new_segments: &mut Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    operand_rewrites: &mut Vec<(usize, usize, usize, VReg)>,
    segment_end_truncations: &mut Vec<(VReg, ProgramPoint)>,
    planned_victims: &mut BTreeSet<VReg>,
) {
    let block_schedule = &all_block_schedules[block_idx];
    let live_at = &live_sets[overshoot_inst_idx];
    let overshoot_point = if overshoot_inst_idx == 0 {
        ProgramPoint::block_entry(block_idx)
    } else {
        ProgramPoint::inst_point(block_idx, overshoot_inst_idx)
    };

    // Collect candidates: VRegs of the target class that are live at the overshoot
    // point, excluding Flags-typed VRegs, spill pseudo-op defs, and already-planned victims.
    let mut candidates: Vec<VReg> = live_at
        .iter()
        .filter(|&&v| vreg_classes.get(&v).copied() == Some(overshoot_class))
        .filter(|&&v| !planned_victims.contains(&v))
        .filter(|&&v| {
            if let Some(&def_idx) = def_inst_map.get(&v) {
                let op = &block_schedule[def_idx].op;
                // Skip spill pseudo-ops (no result_type) and Flags-typed defs.
                if matches!(
                    op,
                    Op::SpillStore(_)
                        | Op::SpillLoad(_)
                        | Op::XmmSpillStore(_)
                        | Op::XmmSpillLoad(_)
                ) {
                    return false;
                }
                // Ask the e-graph for the value's type rather than
                // `op.result_type(&[])`: that asserts arity, so any def taking
                // children (every FP binary op, every ALU op) panics. The
                // e-graph type is also the authoritative one, and it covers
                // shapes whose type is not knowable from the op alone, such as
                // `Proj1` of an ALU pair.
                //
                // Queried at the overshoot point, where the candidate is live
                // by construction, so its segment is guaranteed to cover it.
                let ty = class_to_vreg
                    .vreg_to_class(v, overshoot_point)
                    .map(|cid| egraph.class(egraph.find_immutable(cid)).ty.clone());
                !matches!(ty, Some(Type::Flags))
            } else {
                // Defined in another block: a value flowing through this one.
                // Valid in both scopes -- `apply_cross_block_slot_spill` stores
                // it in its own defining block and reloads it at each use, which
                // needs no local def here.
                //
                // The standard path used to skip these, and at the peak-pressure
                // point of a typical failing function only one or two of thirteen
                // live values had a local def: everything else was a loop-carried
                // block param passing through. With nothing eligible the pass
                // returned having planned nothing, and `plan.is_empty()` reads as
                // convergence, so the overshoot went to the allocator untouched.
                true
            }
        })
        .copied()
        .collect();

    // Score and sort candidates (highest score = best victim).
    candidates.sort_by(|&a, &b| {
        let depth_a = loop_depths.get(&a).copied().unwrap_or(0);
        let depth_b = loop_depths.get(&b).copied().unwrap_or(0);
        let len_a = range_lengths.get(&a).copied().unwrap_or(0);
        let len_b = range_lengths.get(&b).copied().unwrap_or(0);
        score_victim(b, len_b, depth_b).cmp(&score_victim(a, len_a, depth_a))
    });

    // An overshoot with nothing to split is a stall: the pass reports the
    // overshoot, plans nothing, and `plan.is_empty()` reads as convergence, so
    // the point reaches the allocator untouched and it fails to color. Name every
    // value live here and why it was rejected -- the answer is always in this
    // list.
    if candidates.is_empty() {
        if crate::trace::is_enabled("split") && crate::trace::fn_matches(fn_name) {
            let rejected: Vec<String> = live_at
                .iter()
                .filter(|&&v| vreg_classes.get(&v).copied() == Some(overshoot_class))
                .map(|&v| {
                    let why = if planned_victims.contains(&v) {
                        "already a victim".to_string()
                    } else {
                        match def_inst_map.get(&v) {
                            Some(&def_idx) => format!("def {:?}", block_schedule[def_idx].op),
                            None => "no local def".to_string(),
                        }
                    };
                    format!("{v:?} ({why})")
                })
                .collect();
            tracing::debug!(
                target: "blitz::split",
                "[{fn_name}] block {block_idx} at [{overshoot_inst_idx}]: STALLED, \
                 excess={overshoot_excess} with no eligible victim among {}",
                rejected.join(", "),
            );
        }
        return;
    }

    // Pick enough victims to eliminate the entire overshoot at this point.
    let n_victims = (overshoot_excess as usize).min(candidates.len());
    let victims = &candidates[..n_victims];
    if crate::trace::is_enabled("split") && crate::trace::fn_matches(fn_name) {
        tracing::debug!(
            target: "blitz::split",
            "[{fn_name}] block {block_idx} at [{overshoot_inst_idx}]: excess={overshoot_excess}, \
             victims={victims:?} of {} candidates",
            candidates.len(),
        );
    }

    let use_point = overshoot_point;
    let live_classes_at_use: BTreeSet<ClassId> = live_at
        .iter()
        .filter_map(|&v| class_to_vreg.vreg_to_class(v, use_point))
        .collect();

    for &victim in victims {
        planned_victims.insert(victim);
        // The scope is per victim, not per call site. A per-block split rewrites
        // the victim's uses in this block, so it needs both a def and a use here;
        // with either one missing there is nothing to split at locally, whichever
        // path found the victim, and the cross-block spill is the only shape that
        // helps -- it stores after the def wherever that is and reloads at every
        // use block.
        //
        // A value defined here and consumed only downstream is the case the def
        // test alone misses: it occupies a register across the whole block, is the
        // best victim at the peak, and `apply_split_planned` finds no use to
        // rewrite and plans nothing. An empty plan reads as convergence, so the
        // overshoot reached the allocator untouched.
        let used_in_block = block_schedule
            .iter()
            .any(|inst| inst.operands.contains(&victim));
        let victim_scope = if def_inst_map.contains_key(&victim) && used_in_block {
            scope
        } else {
            SplitScope::CrossBlock
        };
        match victim_scope {
            SplitScope::PerBlock => {
                let loop_depth = loop_depths.get(&victim).copied().unwrap_or(0);
                let victim_class = class_to_vreg.vreg_to_class(victim, use_point);
                let kind = choose_split_kind(
                    victim,
                    loop_depth,
                    call_arg_vregs,
                    victim_class,
                    &live_classes_at_use,
                    egraph,
                    cost_model,
                    extraction,
                );
                apply_split_planned(
                    block_idx,
                    victim,
                    kind,
                    block_schedule,
                    overshoot_class,
                    next_vreg,
                    new_slot_count,
                    &mut all_per_block_insertions[block_idx],
                    new_segments,
                    operand_rewrites,
                    class_to_vreg,
                );
            }
            SplitScope::CrossBlock => {
                apply_cross_block_slot_spill(
                    victim,
                    overshoot_class,
                    all_block_schedules,
                    next_vreg,
                    new_slot_count,
                    all_per_block_insertions,
                    new_segments,
                    operand_rewrites,
                    segment_end_truncations,
                    class_to_vreg,
                );
            }
        }
    }
}

/// Build a `SplitPlan` for the given function-wide schedules.
///
/// Scans each block for register-pressure overshoots. For each overshoot,
/// picks a victim and plans either a remat or slot-spill split.
///
/// Also detects block param VRegs live across calls and plans
/// `SlotSpillBlockParam` splits for them (Phase 6).
///
/// `loop_depths` maps VReg -> loop nesting depth (from `compute_loop_depths`).
/// `first_slot` is the first spill-slot index the splitter may allocate; must
/// be set to the number of slots already allocated by `insert_early_barrier_spills`
/// so that splitter-allocated slots do not alias early-barrier slots.
#[allow(clippy::too_many_arguments)]
pub fn plan_splits(
    block_schedules: &[Vec<ScheduledInst>],
    class_to_vreg: &ClassVRegMap,
    extraction: &ExtractionResult,
    egraph: &EGraph,
    cost_model: &CostModel,
    global_liveness: &GlobalLiveness,
    gpr_budget: u32,
    xmm_budget: u32,
    mut next_vreg: u32,
    first_slot: u32,
    loop_depths: &BTreeMap<VReg, u32>,
    func: &Function,
    already_slot_spilled: &BlockParamSlotMap,
) -> SplitPlan {
    let trace = crate::trace::is_enabled("split") && crate::trace::fn_matches(&func.name);
    let n_blocks = block_schedules.len();
    let mut per_block_insertions: Vec<Vec<(usize, ScheduledInst)>> = vec![Vec::new(); n_blocks];
    let mut new_segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)> = Vec::new();
    let mut new_slot_count: u32 = first_slot;
    let mut operand_rewrites: Vec<(usize, usize, usize, VReg)> = Vec::new();
    let mut slot_spilled_params: BTreeMap<(BlockId, u32), SlotSpilledParamInfo> = BTreeMap::new();
    let mut segment_end_truncations: Vec<(VReg, ProgramPoint)> = Vec::new();
    // End-of-block SpillLoad VRegs inserted by cross-block spills.
    // The lowering pass forces these into the trailing barrier group so they
    // execute after all calls in the block (not before them).
    // Deduplication set: VRegs already planned for spill by either path.
    // Prevents the same VReg from being spilled twice when standard pressure
    // and call-crossing pressure both select it as a victim.
    let mut planned_victims: BTreeSet<VReg> = BTreeSet::new();

    // Callee-saved budget: GPR values that can stay live across a call.
    // Values exceeding this budget at any call site cause chromatic overshoot
    // even when raw pressure is below gpr_budget.
    let callee_saved_budget = gpr_budget.saturating_sub(CALLER_SAVED_GPR.len() as u32);

    // Function-wide VReg -> RegClass. Must match what the allocator uses, or
    // the splitter measures a different graph than the one being colored.
    let function_vreg_classes =
        crate::regalloc::build_vreg_classes_from_all_blocks(block_schedules);

    // Block-param slot routing (Phase 6).
    //
    // Runs before the pressure loop below and claims what it handles in
    // `planned_victims`: a block parameter is written by its predecessors' phi
    // copies, so relieving it means changing every predecessor -- which the
    // generic pressure paths cannot express, so whatever this plans has to win.
    detect_blockparam_slot_routing(
        func,
        egraph,
        block_schedules,
        class_to_vreg,
        global_liveness,
        gpr_budget,
        xmm_budget,
        &mut new_slot_count,
        &mut next_vreg,
        &mut per_block_insertions,
        &mut new_segments,
        &mut operand_rewrites,
        &mut slot_spilled_params,
        &mut planned_victims,
        already_slot_spilled,
    );

    for block_idx in 0..n_blocks {
        let schedule = &block_schedules[block_idx];
        if schedule.is_empty() {
            continue;
        }

        // Seed from global live_out so values used in successor blocks are
        // correctly tracked as live throughout their range in this block.
        let live_out_seed = if block_idx < global_liveness.live_out.len() {
            &global_liveness.live_out[block_idx]
        } else {
            continue;
        };
        // Also ensure blocks array bounds are valid.
        if block_idx >= global_liveness.live_in.len() {
            continue;
        }

        // VReg -> RegClass, built from EVERY block rather than just this one.
        //
        // A value that is live *through* this block appears in no instruction
        // here, so a per-block map has no entry for it and every pressure count
        // silently skips it -- while the allocator, which uses a function-wide
        // map, sees it interfering with the call-clobber phantoms. That
        // disagreement is what let an XMM value live across a call reach the
        // allocator unsplit.
        let vreg_classes = &function_vreg_classes;

        // Compute per-instruction live-before sets.
        let live_sets = compute_local_liveness(block_idx, schedule, live_out_seed);

        // Compute pressure per class.
        let gpr_pressure =
            compute_pressure_for_class(&live_sets, schedule, vreg_classes, RegClass::GPR);
        let xmm_pressure =
            compute_pressure_for_class(&live_sets, schedule, vreg_classes, RegClass::XMM);

        // Build def-site map and call-arg set once per block (shared by both paths).
        let def_inst_map: BTreeMap<VReg, usize> = schedule
            .iter()
            .enumerate()
            .map(|(idx, inst)| (inst.dst, idx))
            .collect();
        let call_arg_vregs = collect_call_arg_vregs_set(schedule);
        let range_lengths = compute_live_range_lengths(&live_sets);

        // ── Standard pressure path ────────────────────────────────────────────
        //
        // Find the first overshoot point against the full register budget.
        let gpr_overshoot: Option<(usize, u32)> = gpr_pressure
            .iter()
            .enumerate()
            .find(|(_, p)| **p > gpr_budget)
            .map(|(i, p)| (i, *p));
        let xmm_overshoot: Option<(usize, u32)> = xmm_pressure
            .iter()
            .enumerate()
            .find(|(_, p)| **p > xmm_budget)
            .map(|(i, p)| (i, *p));

        // Pick the most urgent overshoot (by class and amount). Prefer XMM.
        let standard_overshoot = match (gpr_overshoot, xmm_overshoot) {
            (None, None) => None,
            (Some(g), None) => Some((g.0, RegClass::GPR, g.1.saturating_sub(gpr_budget))),
            (None, Some(x)) => Some((x.0, RegClass::XMM, x.1.saturating_sub(xmm_budget))),
            (Some(g), Some(x)) => {
                let xmm_excess = x.1.saturating_sub(xmm_budget);
                let gpr_excess = g.1.saturating_sub(gpr_budget);
                if xmm_excess >= gpr_excess {
                    Some((x.0, RegClass::XMM, xmm_excess))
                } else {
                    Some((g.0, RegClass::GPR, gpr_excess))
                }
            }
        };

        if trace {
            tracing::debug!(
                target: "blitz::split",
                "[{}] block {block_idx}: gpr_budget={gpr_budget} xmm_budget={xmm_budget} \
                 callee_saved_budget={callee_saved_budget}\n{}",
                func.name,
                format_pressure(schedule, &gpr_pressure, &xmm_pressure, &live_sets, vreg_classes),
            );
        }

        if let Some((inst_idx, class, excess)) = standard_overshoot {
            if trace {
                tracing::debug!(
                    target: "blitz::split",
                    "[{}] block {block_idx}: standard overshoot at [{inst_idx}] class={class:?} excess={excess}",
                    func.name,
                );
            }
            apply_splits_for_overshoot(
                &func.name,
                block_idx,
                inst_idx,
                class,
                excess,
                SplitScope::PerBlock,
                &live_sets,
                block_schedules,
                vreg_classes,
                &call_arg_vregs,
                &def_inst_map,
                &range_lengths,
                loop_depths,
                class_to_vreg,
                egraph,
                cost_model,
                extraction,
                &mut next_vreg,
                &mut new_slot_count,
                &mut per_block_insertions,
                &mut new_segments,
                &mut operand_rewrites,
                &mut segment_end_truncations,
                &mut planned_victims,
            );
        }

        // ── Call-crossing pressure path ───────────────────────────────────────
        //
        // Values live at a call site must use callee-saved registers to survive.
        // If more than `callee_saved_budget` GPR values are live at a call,
        // the allocator will fail even if raw pressure is below `gpr_budget`.
        // Pick victims from the GPR values live at the worst call-crossing point.
        // Victims that have no local uses (defined here, consumed in successors)
        // require cross-block spilling: SpillStore in this block, SpillLoad in
        // each use block.
        let call_crossing = find_call_crossing_overshoot(
            &live_sets,
            schedule,
            vreg_classes,
            RegClass::GPR,
            callee_saved_budget,
            &call_arg_vregs,
        );
        if let Some((call_inst_idx, excess)) = call_crossing {
            if trace {
                tracing::debug!(
                    target: "blitz::split",
                    "[{}] block {block_idx}: GPR call-crossing overshoot at [{call_inst_idx}] excess={excess}",
                    func.name,
                );
            }
            apply_splits_for_overshoot(
                &func.name,
                block_idx,
                call_inst_idx,
                RegClass::GPR,
                excess,
                SplitScope::CrossBlock,
                &live_sets,
                block_schedules,
                vreg_classes,
                &call_arg_vregs,
                &def_inst_map,
                &range_lengths,
                loop_depths,
                class_to_vreg,
                egraph,
                cost_model,
                extraction,
                &mut next_vreg,
                &mut new_slot_count,
                &mut per_block_insertions,
                &mut new_segments,
                &mut operand_rewrites,
                &mut segment_end_truncations,
                &mut planned_victims,
            );
        }

        // ── XMM call-crossing pressure path ──────────────────────────────────
        //
        // Every XMM register is caller-saved, so the callee-saved budget for
        // this class is zero: any XMM value still live after a call must go
        // through a slot. The allocator models exactly this with clobber
        // phantoms pre-colored to all 16 XMM registers at each call point
        // (`add_clobber_interferences`), which makes such a value uncolorable
        // and reports a chromatic overshoot.
        //
        // Block params are excluded via `planned_victims`, which the block-param
        // strategy above has already populated.
        let xmm_call_crossing = find_call_crossing_overshoot(
            &live_sets,
            schedule,
            vreg_classes,
            RegClass::XMM,
            0, // no callee-saved XMM registers exist
            &call_arg_vregs,
        );
        if let Some((call_inst_idx, excess)) = xmm_call_crossing {
            if trace {
                tracing::debug!(
                    target: "blitz::split",
                    "[{}] block {block_idx}: XMM call-crossing overshoot at [{call_inst_idx}] excess={excess}",
                    func.name,
                );
            }
            apply_splits_for_overshoot(
                &func.name,
                block_idx,
                call_inst_idx,
                RegClass::XMM,
                excess,
                SplitScope::CrossBlock,
                &live_sets,
                block_schedules,
                vreg_classes,
                &call_arg_vregs,
                &def_inst_map,
                &range_lengths,
                loop_depths,
                class_to_vreg,
                egraph,
                cost_model,
                extraction,
                &mut next_vreg,
                &mut new_slot_count,
                &mut per_block_insertions,
                &mut new_segments,
                &mut operand_rewrites,
                &mut segment_end_truncations,
                &mut planned_victims,
            );
        }
    }

    let plan = SplitPlan {
        per_block_insertions,
        new_segments,
        operand_rewrites,
        slot_spilled_params,
        segment_end_truncations,
        slots_allocated: new_slot_count,
    };
    if trace {
        tracing::debug!(
            target: "blitz::split",
            "[{}] plan:\n{}", func.name, format_plan(&plan),
        );
    }
    plan
}

// ── Trace formatting (`BLITZ_DEBUG=split`) ───────────────────────────────────

/// Format a block's per-instruction pressure next to the live sets that produced
/// it, so a disagreement with the allocator's chromatic number can be read off
/// directly instead of re-derived.
fn format_pressure(
    schedule: &[ScheduledInst],
    gpr_pressure: &[u32],
    xmm_pressure: &[u32],
    live_sets: &[BTreeSet<VReg>],
    vreg_classes: &BTreeMap<VReg, RegClass>,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for i in 0..=schedule.len() {
        let (gpr, xmm) = (
            gpr_pressure.get(i).copied().unwrap_or(0),
            xmm_pressure.get(i).copied().unwrap_or(0),
        );
        let live = live_sets.get(i).map(|s| {
            let mut v: Vec<String> = s
                .iter()
                .map(|r| match vreg_classes.get(r) {
                    Some(RegClass::XMM) => format!("v{}x", r.0),
                    _ => format!("v{}", r.0),
                })
                .collect();
            v.sort();
            v.join(",")
        });
        match schedule.get(i) {
            Some(inst) => {
                let ops: Vec<u32> = inst.operands.iter().map(|v| v.0).collect();
                writeln!(
                    out,
                    "  [{i:>3}] gpr={gpr:>2} xmm={xmm:>2} | v{} = {:?}({ops:?}) live_before={{{}}}",
                    inst.dst.0,
                    inst.op,
                    live.unwrap_or_default(),
                )
            }
            None => writeln!(
                out,
                "  [exit] gpr={gpr:>2} xmm={xmm:>2} | live_out={{{}}}",
                live.unwrap_or_default(),
            ),
        }
        .unwrap();
    }
    out
}

/// Format the split plan: what got inserted where, and how operands moved.
fn format_plan(plan: &SplitPlan) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (bi, insertions) in plan.per_block_insertions.iter().enumerate() {
        for (at, inst) in insertions {
            let ops: Vec<u32> = inst.operands.iter().map(|v| v.0).collect();
            writeln!(
                out,
                "  insert b{bi} before [{at}]: v{} = {:?}({ops:?})",
                inst.dst.0, inst.op
            )
            .unwrap();
        }
    }
    for (bi, ii, oi, new_vreg) in &plan.operand_rewrites {
        writeln!(out, "  rewrite b{bi}[{ii}].op{oi} -> v{}", new_vreg.0).unwrap();
    }
    for (cid, vreg, start, end) in &plan.new_segments {
        writeln!(
            out,
            "  segment c{} -> v{} [{start:?}..{end:?}]",
            cid.0, vreg.0
        )
        .unwrap();
    }
    for (vreg, at) in &plan.segment_end_truncations {
        writeln!(out, "  truncate v{} end at {at:?}", vreg.0).unwrap();
    }
    for ((bid, pidx), info) in &plan.slot_spilled_params {
        writeln!(
            out,
            "  slot-spilled param b{bid}#{pidx}: v{} slot={} {:?}",
            info.vreg.0, info.slot, info.reg_class
        )
        .unwrap();
    }
    writeln!(out, "  slots_allocated={}", plan.slots_allocated).unwrap();
    out
}

// ── Block-param slot routing (Phase 6) ──────────────────────────────────────

/// The VReg and e-class naming block `bid`'s parameter `pidx` at `entry_point`.
///
/// `None` when no class covers the entry point, which is a parameter that passes
/// a dominating definition straight through and has no `BlockParam` of its own.
fn find_block_param_vreg(
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    bid: BlockId,
    pidx: u32,
    entry_point: ProgramPoint,
) -> Option<(VReg, ClassId)> {
    (0..egraph.classes.len() as u32)
        .map(ClassId)
        .filter(|&cid| egraph.unionfind.find_immutable(cid) == cid)
        .find_map(|cid| {
            let matches =
                egraph.class(cid).nodes.iter().any(
                    |node| matches!(node.op, Op::BlockParam(b, p, _) if b == bid && p == pidx),
                );
            if !matches {
                return None;
            }
            Some((class_to_vreg.lookup(cid, entry_point)?, cid))
        })
}

/// Route block parameters through stack slots when they cannot all hold
/// registers.
///
/// Two reasons a parameter must leave the register file, and neither is
/// something the pressure loop below can express -- a parameter is written by
/// its predecessors' phi copies, so relieving it means changing every
/// predecessor, not inserting a reload here:
///
/// - **It is live in a block containing a call, and it is an XMM value.** Every
///   XMM register is caller-saved, so the parameter interferes with the call's
///   clobber set and no colouring exists.
/// - **Its block has more parameters of one class than there are registers for
///   that class.** `add_block_param_interferences` makes every parameter of a
///   block interfere with every other of its class, because the phi copies write
///   them simultaneously. That clique is real and its width is fixed by the
///   block, so a block with 16 GPR parameters against 14 registers is
///   uncolourable no matter what else is spilled. A generated loop carrying 28
///   values produces exactly that.
///
/// For each parameter routed:
/// 1. Allocate a spill slot.
/// 2. Insert a reload before each use of the parameter VReg in EVERY block where
///    it appears as a schedule operand (covers both the block defining the
///    parameter and any block where it is a live-in).
/// 3. Record it in `slot_spilled_params`, which is what makes each predecessor
///    store to the slot instead of copying to a register.
/// 4. `apply_plan_to` truncates the parameter's segment to start after
///    `BLOCK_ENTRY(param_block_idx)` so no register is allocated to it.
#[allow(clippy::too_many_arguments)]
fn detect_blockparam_slot_routing(
    func: &Function,
    egraph: &EGraph,
    block_schedules: &[Vec<ScheduledInst>],
    class_to_vreg: &ClassVRegMap,
    global_liveness: &GlobalLiveness,
    gpr_budget: u32,
    xmm_budget: u32,
    new_slot_count: &mut u32,
    next_vreg: &mut u32,
    per_block_insertions: &mut [Vec<(usize, ScheduledInst)>],
    new_segments: &mut Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    operand_rewrites: &mut Vec<(usize, usize, usize, VReg)>,
    slot_spilled_params: &mut BTreeMap<(BlockId, u32), SlotSpilledParamInfo>,
    planned_victims: &mut BTreeSet<VReg>,
    already_slot_spilled: &BlockParamSlotMap,
) {
    let n_blocks = block_schedules.len();

    // Build a set of blocks that contain calls.
    let call_blocks: BTreeSet<usize> = (0..n_blocks)
        .filter(|&bi| {
            block_schedules[bi]
                .iter()
                .any(|i| matches!(i.op, Op::CallResult(..) | Op::VoidCallBarrier))
        })
        .collect();

    for block_idx in 0..n_blocks.min(func.blocks.len()) {
        let block = &func.blocks[block_idx];
        if block.param_types.is_empty() {
            continue;
        }

        let entry_point = ProgramPoint::block_entry(block_idx);

        // Every parameter of this block that still holds a register, with the
        // class and VReg naming it. Collected first because the over-budget
        // decision is about the set, not about one parameter.
        let params: Vec<(u32, RegClass, VReg, ClassId)> = (0..block.param_types.len() as u32)
            .filter(|&pidx| {
                // Already routed through a slot by an earlier round. Spilling it
                // again would allocate a second slot, and only the newer entry
                // survives in `slot_spilled_params` -- so predecessors would
                // store to one slot while the earlier round's reloads read the
                // other.
                !already_slot_spilled.contains_key(&(block.id, pidx))
            })
            .filter_map(|pidx| {
                let (vreg, cid) =
                    find_block_param_vreg(egraph, class_to_vreg, block.id, pidx, entry_point)?;
                let class = if block.param_types[pidx as usize].is_float() {
                    RegClass::XMM
                } else {
                    RegClass::GPR
                };
                Some((pidx, class, vreg, cid))
            })
            .collect();

        // How many parameters of each class have to go, so that what is left of
        // the clique fits the register file. The budget is the only dial: the
        // clique's width is decided by the block, and every parameter in it is
        // written at the same point.
        let mut over_budget: BTreeMap<RegClass, usize> = BTreeMap::new();
        for (class, budget) in [(RegClass::GPR, gpr_budget), (RegClass::XMM, xmm_budget)] {
            let n = params.iter().filter(|&&(_, c, _, _)| c == class).count();
            over_budget.insert(class, n.saturating_sub(budget as usize));
        }

        // Cheapest first: a parameter read fewer times in this block costs fewer
        // reloads to route. Ties by index, so the choice is deterministic.
        let mut by_cost: Vec<&(u32, RegClass, VReg, ClassId)> = params.iter().collect();
        by_cost.sort_by_key(|&&(pidx, _, vreg, _)| {
            let uses: usize = block_schedules
                .iter()
                .flatten()
                .map(|inst| inst.operands.iter().filter(|&&v| v == vreg).count())
                .sum();
            (uses, pidx)
        });
        let mut chosen_for_pressure: BTreeSet<u32> = BTreeSet::new();
        for &&(pidx, class, _, _) in &by_cost {
            let remaining = over_budget.get_mut(&class).expect("both classes seeded");
            if *remaining == 0 {
                continue;
            }
            *remaining -= 1;
            chosen_for_pressure.insert(pidx);
        }

        for &(pidx, reg_class, param_vreg, param_class) in &params {
            // An XMM parameter live in a block that contains a call has no
            // colouring at all: every XMM register is caller-saved, so it
            // interferes with the clobber set wherever it is live.
            let crosses_call = reg_class == RegClass::XMM
                && !call_blocks.is_empty()
                && call_blocks.iter().any(|&call_bi| {
                    call_bi < global_liveness.live_in.len()
                        && global_liveness.live_in[call_bi].contains(&param_vreg)
                });

            if !crosses_call && !chosen_for_pressure.contains(&pidx) {
                continue;
            }

            // Allocate a spill slot for this param.
            let slot = *new_slot_count as i64;
            *new_slot_count += 1;

            // Insert a reload before each use of param_vreg in EVERY block
            // where it appears as a schedule operand.
            for other_block_idx in 0..n_blocks {
                let schedule = &block_schedules[other_block_idx];
                let use_positions: Vec<(usize, usize)> = schedule
                    .iter()
                    .enumerate()
                    .flat_map(|(inst_idx, inst)| {
                        inst.operands
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == param_vreg)
                            .map(move |(op_idx, _)| (inst_idx, op_idx))
                    })
                    .collect();

                if use_positions.is_empty() {
                    continue;
                }

                for (inst_idx, op_idx) in &use_positions {
                    let reload_vreg = VReg(*next_vreg);
                    *next_vreg += 1;

                    let load_inst = ScheduledInst {
                        op: match reg_class {
                            RegClass::XMM => Op::XmmSpillLoad(slot),
                            RegClass::GPR => Op::SpillLoad(slot),
                        },
                        dst: reload_vreg,
                        operands: vec![],
                    };
                    per_block_insertions[other_block_idx].push((*inst_idx, load_inst));
                    operand_rewrites.push((other_block_idx, *inst_idx, *op_idx, reload_vreg));

                    // Register the reload VReg's segment.
                    let point = ProgramPoint::inst_point(other_block_idx, inst_idx + 1);
                    new_segments.push((param_class, reload_vreg, point, point));
                }
            }

            // Record this param so each predecessor stores to the slot instead
            // of copying to a register (a back edge whose argument is the
            // parameter itself needs no store at all: the slot already holds it).
            slot_spilled_params.insert(
                (block.id, pidx),
                SlotSpilledParamInfo {
                    vreg: param_vreg,
                    slot,
                    reg_class,
                    block_idx,
                },
            );

            // This param is now slot-routed. Claim it so the pressure paths do
            // not plan a second, conflicting spill for the same value.
            planned_victims.insert(param_vreg);
        }
    }
}

/// Slot-spill a victim that is live across a call and whose value is consumed
/// both via schedule-operand uses in successor blocks AND via the block's
/// terminator (Jump/Branch args → block params).
///
/// # Strategy
///
/// 1. Insert `SpillStore` immediately after the def in the def block.
/// 2. Insert `SpillLoad` at the END of the def block (position `n_insts`)
///    so that `class_to_vreg.lookup(class, block_exit)` maps to the reload VReg
///    rather than the original VReg. This causes `compute_phi_uses` (recomputed
///    after the plan is applied) to include the reload VReg instead of the
///    original, breaking the live-out chain that was keeping the original VReg
///    alive across all calls.
/// 3. Record a `segment_end_truncation` to shorten the original VReg's segment
///    to end at the SpillStore. `apply_plan_to` calls `truncate_segment_end`
///    to commit this to `class_to_vreg`.
/// 4. Insert `SpillLoad` before each schedule-operand use in any block (to
///    handle direct uses that don't go through the block-param path).
#[allow(clippy::too_many_arguments)]
fn apply_cross_block_slot_spill(
    victim: VReg,
    reg_class: RegClass,
    all_block_schedules: &[Vec<ScheduledInst>],
    next_vreg: &mut u32,
    new_slot_count: &mut u32,
    per_block_insertions: &mut [Vec<(usize, ScheduledInst)>],
    new_segments: &mut Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    operand_rewrites: &mut Vec<(usize, usize, usize, VReg)>,
    segment_end_truncations: &mut Vec<(VReg, ProgramPoint)>,
    class_to_vreg: &ClassVRegMap,
) {
    // Allocate a spill slot.
    let slot = *new_slot_count as i64;
    *new_slot_count += 1;

    let load_op = if reg_class == RegClass::XMM {
        Op::XmmSpillLoad(slot)
    } else {
        Op::SpillLoad(slot)
    };
    let store_op = if reg_class == RegClass::XMM {
        Op::XmmSpillStore(slot)
    } else {
        Op::SpillStore(slot)
    };

    // Find the def block and insert the SpillStore after the def.
    let mut victim_class: Option<ClassId> = None;
    let mut def_block_idx: Option<usize> = None;
    for (bi, block_sched) in all_block_schedules.iter().enumerate() {
        if let Some(def_pos) = block_sched.iter().position(|i| i.dst == victim) {
            // Step 1: SpillStore after def.
            let store_vreg = VReg(*next_vreg);
            *next_vreg += 1;
            let store_inst = ScheduledInst {
                op: store_op.clone(),
                dst: store_vreg,
                operands: vec![victim],
            };
            per_block_insertions[bi].push((def_pos + 1, store_inst));

            // Look up the victim's ClassId for segment registration.
            let entry_point = ProgramPoint::block_entry(bi);
            victim_class = class_to_vreg.vreg_to_class(victim, entry_point);

            if victim_class.is_some() {
                // The store's dummy dst gets no segment: it names a pseudo-op
                // rather than a value, holds no register, and its point is
                // already covered by the victim's segment, truncated to end
                // exactly here.
                let store_point = ProgramPoint::inst_point(bi, def_pos + 1);

                // Step 2: truncate the original VReg's segment to end at
                // store_point, so nothing downstream resolves the class to a
                // register the store has already given up.
                //
                // The terminator's own use of the victim needs no special
                // handling: it is an operand of `Op::TerminatorArgs`, so step 3
                // reloads before it like any other use. This used to insert a
                // second reload at the very end of the block purely so that a
                // re-resolved `class_to_vreg` lookup at the block exit would
                // find something -- a value live from there to the terminator,
                // costing a register in the block whose pressure this call is
                // trying to relieve.
                segment_end_truncations.push((victim, store_point));
            }

            def_block_idx = Some(bi);
            break; // A VReg has exactly one def.
        }
    }

    // Step 4: Find all schedule-operand uses in ALL blocks and insert SpillLoad
    // before each (excluding the def block's end-of-block position, handled above).
    for (bi, block_sched) in all_block_schedules.iter().enumerate() {
        let use_positions: Vec<(usize, usize)> = block_sched
            .iter()
            .enumerate()
            .flat_map(|(inst_idx, inst)| {
                inst.operands
                    .iter()
                    .enumerate()
                    .filter(|(_, op)| **op == victim)
                    .map(move |(op_idx, _)| (inst_idx, op_idx))
            })
            .collect();

        if use_positions.is_empty() {
            continue;
        }

        for (inst_idx, op_idx) in use_positions {
            let reload_vreg = VReg(*next_vreg);
            *next_vreg += 1;

            let use_load_inst = ScheduledInst {
                op: load_op.clone(),
                dst: reload_vreg,
                operands: vec![],
            };
            per_block_insertions[bi].push((inst_idx, use_load_inst));
            operand_rewrites.push((bi, inst_idx, op_idx, reload_vreg));

            if let Some(class) = victim_class {
                let point = ProgramPoint::inst_point(bi, inst_idx + 1);
                new_segments.push((class, reload_vreg, point, point));
            }
        }

        let _ = def_block_idx;
    }
}

/// Plan the insertions and rewrites for a single split action.
#[allow(clippy::too_many_arguments)]
fn apply_split_planned(
    block_idx: usize,
    victim: VReg,
    kind: SplitKind,
    schedule: &[ScheduledInst],
    reg_class: RegClass,
    next_vreg: &mut u32,
    new_slot_count: &mut u32,
    insertions: &mut Vec<(usize, ScheduledInst)>,
    new_segments: &mut Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    operand_rewrites: &mut Vec<(usize, usize, usize, VReg)>,
    class_to_vreg: &ClassVRegMap,
) {
    // Find the def position (the first instruction with dst == victim).
    let def_pos = schedule.iter().position(|i| i.dst == victim);

    // Collect all use positions (instructions whose operands include victim).
    let use_positions: Vec<(usize, usize)> = schedule
        .iter()
        .enumerate()
        .flat_map(|(inst_idx, inst)| {
            inst.operands
                .iter()
                .enumerate()
                .filter(move |(_, op)| **op == victim)
                .map(move |(op_idx, _)| (inst_idx, op_idx))
        })
        .collect();

    if use_positions.is_empty() {
        // No uses in this block; nothing to split.
        return;
    }

    // Find the ClassId for this victim so we can register new segments.
    // Use block_entry(block_idx) as the lookup point.
    let entry_point = ProgramPoint::block_entry(block_idx);
    let victim_class = class_to_vreg.vreg_to_class(victim, entry_point);

    match kind {
        SplitKind::Remat(op) => {
            // For each use, insert a fresh VReg definition immediately before the use.
            for (inst_idx, op_idx) in use_positions {
                let fresh = VReg(*next_vreg);
                *next_vreg += 1;

                let remat_inst = ScheduledInst {
                    op: op.clone(),
                    dst: fresh,
                    operands: vec![],
                };
                insertions.push((inst_idx, remat_inst));
                operand_rewrites.push((block_idx, inst_idx, op_idx, fresh));

                // Register the new VReg's segment: from the insertion point to the use.
                if let Some(class) = victim_class {
                    let point = ProgramPoint::inst_point(block_idx, inst_idx + 1);
                    new_segments.push((class, fresh, point, point));
                }
            }
        }
        SplitKind::SlotSpill => {
            let slot = *new_slot_count as i64;
            *new_slot_count += 1;

            // Insert SpillStore after the def (or at beginning of block if no def here).
            let store_inst_op = if reg_class == RegClass::XMM {
                Op::XmmSpillStore(slot)
            } else {
                Op::SpillStore(slot)
            };

            let store_pos = def_pos.map(|p| p + 1).unwrap_or(0);
            let store_inst = ScheduledInst {
                op: store_inst_op,
                dst: VReg(*next_vreg), // dummy dst for the store pseudo-op
                operands: vec![victim],
            };
            let store_dummy_vreg = VReg(*next_vreg);
            *next_vreg += 1;
            insertions.push((store_pos, store_inst));

            // For each use, insert a SpillLoad before it.
            for (inst_idx, op_idx) in &use_positions {
                let reload_vreg = VReg(*next_vreg);
                *next_vreg += 1;

                let load_op = if reg_class == RegClass::XMM {
                    Op::XmmSpillLoad(slot)
                } else {
                    Op::SpillLoad(slot)
                };
                let load_inst = ScheduledInst {
                    op: load_op,
                    dst: reload_vreg,
                    operands: vec![],
                };
                insertions.push((*inst_idx, load_inst));
                operand_rewrites.push((block_idx, *inst_idx, *op_idx, reload_vreg));

                // Register the reload VReg's segment.
                if let Some(class) = victim_class {
                    let point = ProgramPoint::inst_point(block_idx, inst_idx + 1);
                    new_segments.push((class, reload_vreg, point, point));
                }
            }

            // No segment is registered for the store's dummy dst. It names a
            // pseudo-op, not a value: it gets no register, so a lookup landing
            // on it reports "no register" for a class that is perfectly live.
            // The victim's own segment already covers the store point.
            let _ = store_dummy_vreg;
        }
    }
}

// ── Plan application ──────────────────────────────────────────────────────────

/// Commit a `SplitPlan` to the live data structures.
///
/// Inserts new instructions into `block_schedules`, registers new segments in
/// `class_to_vreg`, advances `next_vreg`, and bumps `split_generation` by 1.
///
/// CRITICAL ORDERING: this must be called BEFORE `collect_block_param_vregs_per_block`.
#[allow(clippy::ptr_arg)] // block_schedules needs Vec::insert which isn't on slices
pub fn apply_plan_to(
    block_schedules: &mut Vec<Vec<ScheduledInst>>,
    class_to_vreg: &mut ClassVRegMap,
    next_vreg: &mut u32,
    plan: SplitPlan,
) -> AppliedSplits {
    // Apply operand rewrites first (before inserting instructions shifts indices).
    // We collect rewrites per block and apply them before insertions.
    let mut per_block_rewrites: BTreeMap<usize, Vec<(usize, usize, VReg)>> = BTreeMap::new();
    for (block_idx, inst_idx, op_idx, new_vreg) in plan.operand_rewrites {
        per_block_rewrites
            .entry(block_idx)
            .or_default()
            .push((inst_idx, op_idx, new_vreg));
    }

    for (block_idx, rewrites) in &per_block_rewrites {
        let schedule = &mut block_schedules[*block_idx];
        for &(inst_idx, op_idx, new_vreg) in rewrites {
            if let Some(inst) = schedule.get_mut(inst_idx)
                && let Some(operand) = inst.operands.get_mut(op_idx)
            {
                *operand = new_vreg;
            }
        }
    }

    // Apply insertions per block. Each insertion is `(insert_before, inst)`.
    // Insert in reverse order (high indices first) so earlier indices stay valid.
    for (block_idx, mut insertions) in plan.per_block_insertions.into_iter().enumerate() {
        if insertions.is_empty() {
            continue;
        }
        // Sort by insert position descending so we insert from back to front.
        // At the same position, SpillLoad/XmmSpillLoad must be inserted BEFORE
        // SpillStore/XmmSpillStore so that after insertion the store precedes the
        // load in the final schedule (we insert in reverse, so SpillLoad is
        // inserted last at a given position and ends up first there).
        insertions.sort_by(|a, b| {
            b.0.cmp(&a.0).then_with(|| {
                let is_load = |inst: &ScheduledInst| {
                    matches!(inst.op, Op::SpillLoad(_) | Op::XmmSpillLoad(_))
                };
                is_load(&b.1).cmp(&is_load(&a.1))
            })
        });
        let schedule = &mut block_schedules[block_idx];
        for (pos, inst) in insertions {
            let insert_at = pos.min(schedule.len());
            schedule.insert(insert_at, inst);
        }
    }

    // Register new segments in class_to_vreg, in FINAL schedule coordinates.
    //
    // Every index the plan recorded was measured against the pre-insertion
    // schedule, and goes stale the moment an earlier insertion shifts the
    // block: a use planned at index 5 with three insertions ahead of it now
    // sits at 8. `lookup` matches points exactly, so a stale point misses
    // silently -- and lowering then falls back to the register the value
    // occupied *before* it was spilled, making a load read a register the
    // reload never wrote.
    //
    // The final schedule is the authority: a VReg's segment runs from its def
    // to its last reference. Segments the plan marked as reaching block exit
    // keep that end, since `compute_phi_uses` resolves terminator args there.
    let refs = VRegRefs::of(block_schedules);
    let mut committed_segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)> =
        Vec::with_capacity(plan.new_segments.len());
    for (class, vreg, start, end) in plan.new_segments {
        let (start, end) = match refs.range_of(vreg) {
            Some((s, _)) if end.inst == u32::MAX => (s, ProgramPoint::block_exit(refs.block(vreg))),
            Some(range) => range,
            None => (start, end),
        };
        class_to_vreg.insert_segment(class, vreg, start, end);
        committed_segments.push((class, vreg, start, end));
    }

    // Phase 6: Truncate block-param segments so they start AFTER block entry.
    // This causes `collect_block_param_vregs_per_block` (which looks up at
    // BLOCK_ENTRY) to NOT find the param VReg, so no register is allocated to
    // it. Predecessors store to the slot; uses in the block reload from it.
    for info in plan.slot_spilled_params.values() {
        // Only XMM params are slot-spilled (detect_blockparam_call_crossings
        // only handles float params because only XMM regs are caller-saved).
        debug_assert_eq!(
            info.reg_class,
            RegClass::XMM,
            "slot-spilled block params must be XMM"
        );
        // Truncate to inst_point(block_idx, 1) which is strictly after
        // BLOCK_ENTRY(block_idx) (inst=0), so lookup at block_entry returns None.
        let new_start = ProgramPoint::inst_point(info.block_idx, 1);
        class_to_vreg.truncate_segment_start(info.vreg, new_start);
    }

    // Cross-block spill: truncate original VReg segments to end at the SpillStore.
    // After truncation, class_to_vreg.lookup(class, block_exit) returns the
    // end-of-block reload VReg (whose segment was registered above), enabling
    // compute_phi_uses to route the terminator through the reload VReg instead.
    //
    // The new end is recomputed from the final schedule for the same reason the
    // segments above are: the spilled VReg's last reference is its SpillStore,
    // and insertions moved it.
    let mut committed_truncations: Vec<(VReg, ProgramPoint)> =
        Vec::with_capacity(plan.segment_end_truncations.len());
    for (vreg, planned_end) in plan.segment_end_truncations {
        let new_end = refs.range_of(vreg).map(|(_, e)| e).unwrap_or(planned_end);
        class_to_vreg.truncate_segment_end(vreg, new_end);
        committed_truncations.push((vreg, new_end));
    }

    // Advance next_vreg to the highest freshly-allocated VReg + 1.
    // (The plan used *next_vreg during planning and incremented it,
    // so the caller's next_vreg is already ahead; we just sync it.)
    // Find the max VReg in all schedules to be safe.
    let max_vreg_in_schedules = block_schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .unwrap_or(0);
    *next_vreg = (*next_vreg).max(max_vreg_in_schedules + 1);

    // Bump split_generation to mark that splitter output has been committed.
    class_to_vreg.split_generation += 1;

    AppliedSplits {
        segments: committed_segments,
        truncations: committed_truncations,
    }
}

/// The segment edits `apply_plan_to` committed, in final schedule coordinates.
///
/// Phase 7 lowering resolves effectful-op operands through per-block snapshots
/// of `class_to_vreg` taken during linearization, long before the splitter runs.
/// Replaying these onto each snapshot is what keeps a spilled address resolving
/// to its reload instead of its pre-spill register.
pub struct AppliedSplits {
    pub segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    pub truncations: Vec<(VReg, ProgramPoint)>,
}

impl AppliedSplits {
    /// Replay the committed edits onto a snapshot taken before the split.
    ///
    /// Truncations run first: an untruncated original segment would otherwise
    /// overlap the reload segment that replaces its tail.
    pub fn replay_onto(&self, snapshot: &mut ClassVRegMap) {
        for &(vreg, new_end) in &self.truncations {
            snapshot.truncate_segment_end(vreg, new_end);
        }
        for &(class, vreg, start, end) in &self.segments {
            snapshot.insert_segment(class, vreg, start, end);
        }
    }
}

/// Where each VReg is defined and last referenced in the final schedules.
struct VRegRefs {
    /// vreg -> (block index, def index, last reference index).
    entries: BTreeMap<VReg, (usize, usize, usize)>,
}

impl VRegRefs {
    fn of(block_schedules: &[Vec<ScheduledInst>]) -> Self {
        let mut entries: BTreeMap<VReg, (usize, usize, usize)> = BTreeMap::new();
        for (bi, schedule) in block_schedules.iter().enumerate() {
            for (idx, inst) in schedule.iter().enumerate() {
                entries.entry(inst.dst).or_insert((bi, idx, idx));
            }
            // A reference only extends the range when it sits in the defining
            // block: segments are per-block, and a cross-block use is served by
            // its own reload segment.
            for (idx, inst) in schedule.iter().enumerate() {
                for op in &inst.operands {
                    if let Some(entry) = entries.get_mut(op)
                        && entry.0 == bi
                    {
                        entry.2 = entry.2.max(idx);
                    }
                }
            }
        }
        VRegRefs { entries }
    }

    fn block(&self, vreg: VReg) -> usize {
        self.entries.get(&vreg).map(|e| e.0).unwrap_or(0)
    }

    /// The `[def .. last reference]` program-point range for `vreg`, or `None`
    /// if it is defined by no instruction in any schedule.
    ///
    /// Index 0 is clamped to 1: `inst = 0` is the `BLOCK_ENTRY` sentinel, so a
    /// reload landing at the top of a block shares a point with the instruction
    /// that consumes it, which is the point lowering asks about anyway.
    fn range_of(&self, vreg: VReg) -> Option<(ProgramPoint, ProgramPoint)> {
        let &(bi, def, last) = self.entries.get(&vreg)?;
        Some((
            ProgramPoint::inst_point(bi, def.max(1)),
            ProgramPoint::inst_point(bi, last.max(1)),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::EGraph;
    use crate::egraph::cost::{CostModel, OptGoal};
    use crate::egraph::extract::ExtractionResult;
    use crate::ir::op::Op;
    use crate::ir::types::Type;
    use crate::regalloc::global_liveness::GlobalLiveness;
    use crate::schedule::scheduler::ScheduledInst;
    use std::collections::{BTreeMap, BTreeSet};

    fn fconst_inst(dst: u32, val: f64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Fconst(val.to_bits(), Type::F64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn addsd_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Addsd,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn call_result_inst(dst: u32, args: Vec<u32>) -> ScheduledInst {
        ScheduledInst {
            op: Op::CallResult(0, Type::F64),
            dst: VReg(dst),
            operands: args.into_iter().map(VReg).collect(),
        }
    }

    fn make_empty_extraction() -> ExtractionResult {
        ExtractionResult {
            choices: BTreeMap::new(),
        }
    }

    fn make_global_liveness(n_blocks: usize) -> GlobalLiveness {
        GlobalLiveness {
            live_in: vec![BTreeSet::new(); n_blocks],
            live_out: vec![BTreeSet::new(); n_blocks],
        }
    }

    fn make_simple_class_map() -> ClassVRegMap {
        ClassVRegMap::new()
    }

    fn make_empty_func() -> crate::ir::function::Function {
        crate::ir::function::Function::new("test", vec![], vec![])
    }

    // Test 1: No overshoot means no splits.
    #[test]
    fn no_overshoot_no_splits() {
        // Two XMM values live simultaneously, but budget = 16; no split needed.
        let schedule = vec![
            fconst_inst(0, 1.0), // v0 = 1.0 (XMM)
            fconst_inst(1, 2.0), // v1 = 2.0 (XMM)
            addsd_inst(2, 0, 1), // v2 = v0 + v1
        ];
        let block_schedules = vec![schedule];
        let class_to_vreg = make_simple_class_map();
        let extraction = make_empty_extraction();
        let egraph = EGraph::new();
        let cost_model = CostModel::new(OptGoal::Balanced);
        let global_liveness = make_global_liveness(1);
        let loop_depths: BTreeMap<VReg, u32> = BTreeMap::new();

        let func = make_empty_func();
        let plan = plan_splits(
            &block_schedules,
            &class_to_vreg,
            &extraction,
            &egraph,
            &cost_model,
            &global_liveness,
            15, // gpr_budget
            16, // xmm_budget
            100,
            0, // first_slot
            &loop_depths,
            &func,
            &BlockParamSlotMap::new(),
        );

        assert!(
            plan.per_block_insertions[0].is_empty(),
            "no splits expected when under budget"
        );
        assert!(plan.new_segments.is_empty());
    }

    // Test 2: XMM pressure overshoot across a call triggers a split.
    #[test]
    fn pressure_overshoot_detected_on_xmm_across_call() {
        // Build a schedule with 17 live XMM values at the call site.
        // fconst v0..v16, call_result v17(args=[]), use v0..v16
        // With budget=16, we should detect overshoot.
        let mut schedule = Vec::new();
        for i in 0..17u32 {
            schedule.push(fconst_inst(i, i as f64));
        }
        // A call (CallResult) with no args - all 17 XMM values are live across it.
        schedule.push(call_result_inst(17, vec![]));
        // Use all XMM values after the call.
        for i in 0..17u32 {
            schedule.push(addsd_inst(18 + i, i, 17));
        }

        let block_schedules = vec![schedule];
        let class_to_vreg = make_simple_class_map();
        let extraction = make_empty_extraction();
        let egraph = EGraph::new();
        let cost_model = CostModel::new(OptGoal::Balanced);
        let global_liveness = make_global_liveness(1);
        let loop_depths: BTreeMap<VReg, u32> = BTreeMap::new();

        let func = make_empty_func();
        let plan = plan_splits(
            &block_schedules,
            &class_to_vreg,
            &extraction,
            &egraph,
            &cost_model,
            &global_liveness,
            15,
            16, // xmm_budget; 17 live XMMs > 16
            200,
            0, // first_slot
            &loop_depths,
            &func,
            &BlockParamSlotMap::new(),
        );

        // Should have planned at least one split (insertions or rewrites).
        let total_insertions: usize = plan.per_block_insertions.iter().map(|v| v.len()).sum();
        assert!(
            total_insertions > 0 || !plan.operand_rewrites.is_empty(),
            "expected a split to be planned for XMM overshoot"
        );
    }

    // Test 3: Fconst victim -> Remat chosen via e-graph extraction.
    //
    // Build an EGraph with a single Fconst class. extract_at_with_memo returns
    // Fconst (cheap, cost 1), so choose_split_kind should pick Remat.
    #[test]
    fn remat_chosen_for_fconst_victim() {
        use crate::egraph::enode::ENode;

        let mut egraph = EGraph::new();
        let fconst_cid = egraph.add(ENode {
            op: Op::Fconst(2.5f64.to_bits(), Type::F64),
            children: smallvec::smallvec![],
        });
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        let kind = choose_split_kind(
            VReg(0),
            0, // loop_depth = 0
            &BTreeSet::new(),
            Some(fconst_cid),
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::Remat(Op::Fconst(..))),
            "expected Remat for Fconst, got {kind:?}"
        );
    }

    // Test 4: Non-cheap victim -> SlotSpill chosen.
    //
    // Pass victim_class=None (no class in the map) so extract_at returns None.
    // The fallback is always SlotSpill.
    #[test]
    fn slot_chosen_for_noncheap_victim() {
        let egraph = EGraph::new();
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        // No victim_class → extract_at_with_memo returns None → SlotSpill.
        let kind = choose_split_kind(
            VReg(0),
            0,
            &BTreeSet::new(),
            None, // no class: forces SlotSpill
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::SlotSpill),
            "expected SlotSpill when victim_class is None, got {kind:?}"
        );
    }

    // Test 5: Call-arg VReg is never remat'd, even if its op is free.
    //
    // The call-arg check is first so the egraph is never consulted.
    #[test]
    fn call_arg_never_remat_even_if_free() {
        use crate::egraph::enode::ENode;

        let mut egraph = EGraph::new();
        let fconst_cid = egraph.add(ENode {
            op: Op::Fconst(1.0f64.to_bits(), Type::F64),
            children: smallvec::smallvec![],
        });
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        let victim = VReg(0);
        let mut call_args = BTreeSet::new();
        call_args.insert(victim);
        let kind = choose_split_kind(
            victim,
            0,
            &call_args,
            Some(fconst_cid),
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::SlotSpill),
            "call-arg VReg must use SlotSpill, not Remat; got {kind:?}"
        );
    }

    // ── Phase 6 block-param unit tests ───────────────────────────────────────

    /// Build a minimal two-block Function and EGraph for block-param crossing tests.
    ///
    /// Block0: no params, terminates with a Jump to block1 passing arg_cid.
    /// Block1: one F64 param (param_cid = BlockParam(block1.id, 0, F64)).
    /// Block2: no params, has the call instruction using param_vreg.
    ///
    /// Returns (func, egraph, param_class, param_vreg, call_block_idx=1)
    /// where block1 is the param block and block1's schedule has the call.
    fn make_blockparam_call_setup() -> (
        crate::ir::function::Function,
        crate::egraph::EGraph,
        crate::ir::op::ClassId,
        VReg,
    ) {
        use crate::egraph::enode::ENode;
        use crate::ir::effectful::EffectfulOp;
        use crate::ir::function::BasicBlock;
        use crate::ir::op::ClassId;

        let mut func = crate::ir::function::Function::new("test_bp", vec![], vec![]);
        let b0_id = func.fresh_block_id(); // block0: no params
        let b1_id = func.fresh_block_id(); // block1: one F64 param

        let mut block0 = BasicBlock::new(b0_id, vec![]);
        block0.ops.push(EffectfulOp::Jump {
            target: b1_id,
            args: vec![ClassId(0)], // arg = ClassId(0) (placeholder)
        });
        let mut block1 = BasicBlock::new(b1_id, vec![Type::F64]);
        block1.ops.push(EffectfulOp::Ret { val: None });
        func.blocks.push(block0);
        func.blocks.push(block1);

        // Build EGraph: ClassId(0) = BlockParam(b1_id, 0, F64).
        let mut egraph = crate::egraph::EGraph::new();
        let param_cid = egraph.add(ENode {
            op: Op::BlockParam(b1_id, 0, Type::F64),
            children: smallvec::smallvec![],
        });

        // Build ClassVRegMap: param_cid -> param_vreg with full range.
        let param_vreg = VReg(0);

        (func, egraph, param_cid, param_vreg)
    }

    // Test 7: blockparam_split_truncates_segment_at_entry
    //
    // After detect_blockparam_call_crossings + apply_plan_to, the param VReg's
    // segment should start at inst_point(block_idx, 1) (strictly after
    // block_entry(block_idx)) so the allocator doesn't assign a register to it.
    #[test]
    fn blockparam_split_truncates_segment_at_entry() {
        use crate::egraph::extract::ClassVRegMap;

        let (func, egraph, param_cid, param_vreg) = make_blockparam_call_setup();
        let block1_idx = 1usize;
        let param_block_idx = block1_idx;

        let mut class_to_vreg = ClassVRegMap::new();
        class_to_vreg.insert_full_range(param_cid, param_vreg);

        // Block0 schedule: empty. Block1 schedule: one use of param_vreg via a call.
        let call_inst = call_result_inst(1, vec![0]); // param_vreg=VReg(0) as arg
        let block_schedules = vec![vec![], vec![call_inst]];

        // Global liveness: param_vreg is live_in to block1 (where the call is).
        let mut global_liveness = make_global_liveness(2);
        global_liveness.live_in[block1_idx].insert(param_vreg);

        let mut new_slot_count = 0u32;
        let mut next_vreg = 2u32;
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut slot_spilled_params = BTreeMap::new();

        detect_blockparam_call_crossings(
            &func,
            &egraph,
            &block_schedules,
            &class_to_vreg,
            &global_liveness,
            &mut new_slot_count,
            &mut next_vreg,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut slot_spilled_params,
            &mut BTreeSet::new(),
            &BlockParamSlotMap::new(),
        );

        assert_eq!(new_slot_count, 1, "one slot should be allocated");
        assert!(
            slot_spilled_params.contains_key(&(func.blocks[block1_idx].id, 0)),
            "param should be recorded in slot_spilled_params"
        );

        // Apply the plan (which truncates the segment).
        let plan = SplitPlan {
            per_block_insertions: per_block_insertions.clone(),
            new_segments,
            operand_rewrites,
            slot_spilled_params: slot_spilled_params.clone(),
            segment_end_truncations: Vec::new(),
            slots_allocated: 0,
        };
        let mut block_schedules_mut = vec![vec![], vec![call_result_inst(1, vec![0])]];
        let mut next_vreg2 = next_vreg;
        let mut class_to_vreg2 = class_to_vreg.clone();
        apply_plan_to(
            &mut block_schedules_mut,
            &mut class_to_vreg2,
            &mut next_vreg2,
            plan,
        );

        // After truncation, lookup at block_entry(block1_idx) should return None.
        let entry_point = ProgramPoint::block_entry(param_block_idx);
        let found = class_to_vreg2.lookup(param_cid, entry_point);
        assert!(
            found.is_none(),
            "param VReg segment should not cover block_entry after truncation; got {found:?}"
        );
    }

    // Test 8: blockparam_split_emits_slot_store_in_predecessor
    //
    // slot_spilled_params should have an entry for the param so lower_terminator
    // can emit slot stores in predecessor phi copies.
    #[test]
    fn blockparam_split_emits_slot_store_in_predecessor() {
        use crate::egraph::extract::ClassVRegMap;

        let (func, egraph, param_cid, param_vreg) = make_blockparam_call_setup();
        let block1_idx = 1usize;

        let mut class_to_vreg = ClassVRegMap::new();
        class_to_vreg.insert_full_range(param_cid, param_vreg);

        let call_inst = call_result_inst(1, vec![0]);
        let block_schedules = vec![vec![], vec![call_inst]];

        let mut global_liveness = make_global_liveness(2);
        global_liveness.live_in[block1_idx].insert(param_vreg);

        let mut new_slot_count = 0u32;
        let mut next_vreg = 2u32;
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut slot_spilled_params = BTreeMap::new();

        detect_blockparam_call_crossings(
            &func,
            &egraph,
            &block_schedules,
            &class_to_vreg,
            &global_liveness,
            &mut new_slot_count,
            &mut next_vreg,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut slot_spilled_params,
            &mut BTreeSet::new(),
            &BlockParamSlotMap::new(),
        );

        let b1_id = func.blocks[block1_idx].id;
        assert!(
            slot_spilled_params.contains_key(&(b1_id, 0)),
            "slot_spilled_params must have entry for param"
        );
        let info = &slot_spilled_params[&(b1_id, 0)];
        assert_eq!(info.vreg, param_vreg, "info.vreg must match param_vreg");
        assert_eq!(info.slot, 0, "first slot should be 0");
        assert_eq!(
            info.reg_class,
            RegClass::XMM,
            "F64 param must use XMM class"
        );
    }

    // Test 9: blockparam_split_emits_slot_load_at_use
    //
    // For each use of param_vreg in the schedule, an XmmSpillLoad instruction
    // must be inserted before it, and the operand rewritten.
    #[test]
    fn blockparam_split_emits_slot_load_at_use() {
        use crate::egraph::extract::ClassVRegMap;

        let (func, egraph, param_cid, param_vreg) = make_blockparam_call_setup();
        let block1_idx = 1usize;

        let mut class_to_vreg = ClassVRegMap::new();
        class_to_vreg.insert_full_range(param_cid, param_vreg);

        // Block1 has two uses: the call arg AND an addsd consumer.
        let call_inst = call_result_inst(1, vec![0]); // arg = param_vreg
        let use2 = addsd_inst(2, 0, 1); // another use of param_vreg
        let block_schedules = vec![vec![], vec![call_inst, use2]];

        let mut global_liveness = make_global_liveness(2);
        global_liveness.live_in[block1_idx].insert(param_vreg);

        let mut new_slot_count = 0u32;
        let mut next_vreg = 3u32;
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut slot_spilled_params = BTreeMap::new();

        detect_blockparam_call_crossings(
            &func,
            &egraph,
            &block_schedules,
            &class_to_vreg,
            &global_liveness,
            &mut new_slot_count,
            &mut next_vreg,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut slot_spilled_params,
            &mut BTreeSet::new(),
            &BlockParamSlotMap::new(),
        );

        // Two uses → two XmmSpillLoad insertions in block1.
        let insertions_in_block1 = &per_block_insertions[block1_idx];
        assert_eq!(
            insertions_in_block1.len(),
            2,
            "expected two XmmSpillLoad insertions for two uses"
        );
        for (_, inst) in insertions_in_block1 {
            assert!(
                matches!(inst.op, Op::XmmSpillLoad(0)),
                "inserted inst must be XmmSpillLoad(0), got {:?}",
                inst.op
            );
        }
        // Two operand rewrites.
        assert_eq!(operand_rewrites.len(), 2, "expected two operand rewrites");
    }

    // Test 10: blockparam_split_two_predecessors_both_store
    //
    // When a block param is slot-spilled, ALL predecessors should see a
    // slot_spilled_params entry for the param. The entry has a single slot
    // number; both predecessors will emit stores to the same slot.
    #[test]
    fn blockparam_split_two_predecessors_both_store() {
        use crate::egraph::enode::ENode;
        use crate::egraph::extract::ClassVRegMap;
        use crate::ir::effectful::EffectfulOp;
        use crate::ir::function::BasicBlock;
        use crate::ir::op::ClassId;

        // Build a 3-block function: block0 and block1 both jump to block2 (merge).
        // Block2 has one F64 param; block1 contains the call.
        let mut func = crate::ir::function::Function::new("test_multi_pred", vec![], vec![]);
        let b0_id = func.fresh_block_id();
        let b1_id = func.fresh_block_id();
        let b2_id = func.fresh_block_id();

        let mut block0 = BasicBlock::new(b0_id, vec![]);
        block0.ops.push(EffectfulOp::Jump {
            target: b2_id,
            args: vec![ClassId(0)],
        });
        let mut block1 = BasicBlock::new(b1_id, vec![]);
        block1.ops.push(EffectfulOp::Jump {
            target: b2_id,
            args: vec![ClassId(0)],
        });
        let mut block2 = BasicBlock::new(b2_id, vec![Type::F64]);
        block2.ops.push(EffectfulOp::Ret { val: None });
        func.blocks.push(block0);
        func.blocks.push(block1);
        func.blocks.push(block2);

        let mut egraph = crate::egraph::EGraph::new();
        let param_cid = egraph.add(ENode {
            op: Op::BlockParam(b2_id, 0, Type::F64),
            children: smallvec::smallvec![],
        });

        let param_vreg = VReg(0);
        let mut class_to_vreg = ClassVRegMap::new();
        class_to_vreg.insert_full_range(param_cid, param_vreg);

        // Block0: empty, block1: has a call using param_vreg, block2: empty.
        let call_inst = call_result_inst(1, vec![0]);
        let block_schedules = vec![vec![], vec![call_inst], vec![]];

        // param_vreg is live_in to block1 (which has the call).
        let mut global_liveness = make_global_liveness(3);
        global_liveness.live_in[1].insert(param_vreg);

        let mut new_slot_count = 0u32;
        let mut next_vreg = 2u32;
        let mut per_block_insertions = vec![vec![], vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut slot_spilled_params = BTreeMap::new();

        detect_blockparam_call_crossings(
            &func,
            &egraph,
            &block_schedules,
            &class_to_vreg,
            &global_liveness,
            &mut new_slot_count,
            &mut next_vreg,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut slot_spilled_params,
            &mut BTreeSet::new(),
            &BlockParamSlotMap::new(),
        );

        // One slot allocated for the F64 param.
        assert_eq!(new_slot_count, 1);
        // One entry in slot_spilled_params for (b2_id, 0).
        assert!(
            slot_spilled_params.contains_key(&(b2_id, 0)),
            "slot_spilled_params must have entry for block2's param"
        );
        // Both predecessors can use this entry to emit slot stores.
        // The entry has a single slot number.
        let info = &slot_spilled_params[&(b2_id, 0)];
        assert_eq!(info.slot, 0);
        // The slot number is the same for both predecessors (they both store to slot 0).
        assert_eq!(
            new_slot_count, 1,
            "only one slot needed regardless of predecessor count"
        );
    }

    // ── Ported legacy tests (Task 7.4-pre) ───────────────────────────────────
    //
    // Each test exercises behavior of the pressure-driven split pass.

    fn iconst_inst_gpr(dst: u32, val: i64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(val, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn proj0_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    fn x86add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn void_call_barrier_inst(dst: u32, args: Vec<u32>) -> ScheduledInst {
        ScheduledInst {
            op: Op::VoidCallBarrier,
            dst: VReg(dst),
            operands: args.into_iter().map(VReg).collect(),
        }
    }

    // Ported from `remat_drops_original_def` and `stackaddr_rematerialized_like_iconst`:
    // Remat path inserts a fresh copy before each use and rewrites operands.
    // The original def instruction is left in place (the allocator elides it as dead).
    #[test]
    fn remat_inserts_fresh_copy_per_use() {
        use crate::regalloc::spill::{is_rematerializable, is_spill_load, is_spill_store};

        // Schedule (single block):
        //   v0 = iconst 99  (will be remat-spilled)
        //   v1 = proj0(v0)
        //   v2 = proj0(v0)
        let schedule = vec![iconst_inst_gpr(0, 99), proj0_inst(1, 0), proj0_inst(2, 0)];
        let block_idx = 0usize;

        // Confirm is_rematerializable returns true for Iconst.
        assert!(
            is_rematerializable(&schedule[0]),
            "Iconst must be rematerializable"
        );

        let class_to_vreg = make_simple_class_map();
        let mut insertions: Vec<(usize, ScheduledInst)> = vec![];
        let mut new_segments: Vec<(_, VReg, _, _)> = vec![];
        let mut operand_rewrites: Vec<(usize, usize, usize, VReg)> = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_split_planned(
            block_idx,
            VReg(0),
            SplitKind::Remat(Op::Iconst(99, Type::I64)),
            &schedule,
            RegClass::GPR,
            &mut next_vreg,
            &mut new_slot_count,
            &mut insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &class_to_vreg,
        );

        // No slot allocated for remat.
        assert_eq!(new_slot_count, 0, "remat must not allocate a spill slot");

        // Two uses -> two fresh-copy insertions.
        let remat_copies: Vec<_> = insertions
            .iter()
            .filter(|(_, i)| matches!(i.op, Op::Iconst(99, _)))
            .collect();
        assert_eq!(
            remat_copies.len(),
            2,
            "expected two fresh Iconst copies (one per use), got {}",
            remat_copies.len()
        );

        // No SpillStore / SpillLoad.
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_store(i)),
            "remat must not emit a SpillStore"
        );
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_load(i)),
            "remat must not emit a SpillLoad"
        );

        // Two operand rewrites: each use of v0 gets a fresh VReg.
        assert_eq!(
            operand_rewrites.len(),
            2,
            "expected two operand rewrites for two uses"
        );
    }

    // Ported from `stackaddr_rematerialized_like_iconst`:
    // StackAddr is remat-eligible; choose_split_kind returns Remat for it.
    #[test]
    fn stackaddr_is_remat_eligible_in_split_pass() {
        use crate::egraph::enode::ENode;
        use crate::regalloc::spill::is_rematerializable;

        let stack_addr_inst = ScheduledInst {
            op: Op::StackAddr(5),
            dst: VReg(0),
            operands: vec![],
        };
        assert!(
            is_rematerializable(&stack_addr_inst),
            "StackAddr must be classified as rematerializable"
        );

        let mut egraph = EGraph::new();
        let cid = egraph.add(ENode {
            op: Op::StackAddr(5),
            children: smallvec::smallvec![],
        });
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        let kind = choose_split_kind(
            VReg(0),
            0,
            &BTreeSet::new(),
            Some(cid),
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::Remat(Op::StackAddr(5))),
            "choose_split_kind must select Remat for StackAddr, got {kind:?}"
        );
    }

    // Ported from `call_arg_never_rematerialized` and `call_arg_iconst_forces_slot_not_remat`:
    // A VReg that is a call argument must always use SlotSpill, even if its op is free.
    #[test]
    fn call_arg_always_slot_spill_even_if_free() {
        use crate::egraph::enode::ENode;

        let mut egraph = EGraph::new();
        let iconst_cid = egraph.add(ENode {
            op: Op::Iconst(42, Type::I64),
            children: smallvec::smallvec![],
        });
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        let victim = VReg(0);
        let mut call_args = BTreeSet::new();
        call_args.insert(victim);

        // Iconst is normally rematerializable, but call-arg check wins.
        let kind = choose_split_kind(
            victim,
            0,
            &call_args,
            Some(iconst_cid),
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::SlotSpill),
            "call-arg Iconst must produce SlotSpill (not Remat); got {kind:?}"
        );

        // Also verify the SlotSpill path actually emits SpillStore + SpillLoad.
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        let schedule = vec![
            iconst_inst_gpr(0, 42),
            void_call_barrier_inst(1, vec![0]), // v0 is a call arg
            proj0_inst(2, 0),
        ];
        let class_to_vreg = make_simple_class_map();
        let mut insertions: Vec<(usize, ScheduledInst)> = vec![];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_split_planned(
            0,
            VReg(0),
            SplitKind::SlotSpill,
            &schedule,
            RegClass::GPR,
            &mut next_vreg,
            &mut new_slot_count,
            &mut insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &class_to_vreg,
        );

        // A SpillStore must appear.
        assert!(
            insertions.iter().any(|(_, i)| is_spill_store(i)),
            "SlotSpill must emit a SpillStore for call-arg Iconst"
        );
        // A SpillLoad must appear.
        assert!(
            insertions.iter().any(|(_, i)| is_spill_load(i)),
            "SlotSpill must emit a SpillLoad for call-arg Iconst use"
        );
        // No remat Iconst copies after the call.
        let call_pos = schedule
            .iter()
            .position(|i| matches!(i.op, Op::VoidCallBarrier))
            .unwrap_or(usize::MAX);
        let remat_after_call = insertions
            .iter()
            .filter(|(pos, _)| *pos > call_pos)
            .any(|(_, i)| matches!(i.op, Op::Iconst(42, _)));
        assert!(
            !remat_after_call,
            "no remat Iconst copy may appear after the call for a call-arg VReg"
        );
    }

    // Ported from `xmm_high_pressure_uses_xmm_spill_ops`:
    // XMM VReg slot-spill emits XmmSpillStore / XmmSpillLoad, not the GPR variants.
    #[test]
    fn xmm_victim_uses_xmm_spill_ops() {
        use crate::regalloc::spill::{
            is_spill_load, is_spill_store, is_xmm_spill_load, is_xmm_spill_store,
        };

        // v0 = fconst (XMM def), v1 = addsd(v0, v0) (XMM use)
        let schedule = vec![
            fconst_inst(0, 1.0),
            addsd_inst(1, 0, 0), // two uses of v0
        ];
        let class_to_vreg = make_simple_class_map();
        let mut insertions: Vec<(usize, ScheduledInst)> = vec![];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_split_planned(
            0,
            VReg(0),
            SplitKind::SlotSpill,
            &schedule,
            RegClass::XMM, // XMM class
            &mut next_vreg,
            &mut new_slot_count,
            &mut insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &class_to_vreg,
        );

        // Must use XmmSpillStore (not GPR SpillStore).
        assert!(
            insertions.iter().any(|(_, i)| is_xmm_spill_store(i)),
            "XMM slot-spill must emit XmmSpillStore"
        );
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_store(i)),
            "GPR SpillStore must NOT appear for XMM victim"
        );

        // Must use XmmSpillLoad (not GPR SpillLoad).
        assert!(
            insertions.iter().any(|(_, i)| is_xmm_spill_load(i)),
            "XMM slot-spill must emit XmmSpillLoad"
        );
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_load(i)),
            "GPR SpillLoad must NOT appear for XMM victim"
        );
    }

    // Ported from `spill_store_exactly_once_across_many_blocks`:
    // For a cross-block slot-spill, SpillStore appears exactly once (in the def
    // block) and SpillLoad appears at least once in each use-containing block.
    #[test]
    fn cross_block_slot_spill_store_once_load_per_use_block() {
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        // Block 0: v0 = x86add(v1, v2)  [non-remat, cross-block def]
        // Blocks 1..4: each uses v0 once.
        let sched0 = vec![
            iconst_inst_gpr(1, 0),
            iconst_inst_gpr(2, 0),
            x86add_inst(0, 1, 2),
        ];
        let sched1 = vec![proj0_inst(3, 0)];
        let sched2 = vec![proj0_inst(4, 0)];
        let sched3 = vec![proj0_inst(5, 0)];
        let sched4 = vec![proj0_inst(6, 0)];

        let mut all_block_schedules = vec![sched0, sched1, sched2, sched3, sched4];
        let class_to_vreg = make_simple_class_map();
        let mut per_block_insertions = vec![vec![], vec![], vec![], vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut segment_end_truncations = vec![];
        let mut next_vreg = 20u32;
        let mut new_slot_count = 0u32;

        apply_cross_block_slot_spill(
            VReg(0),
            RegClass::GPR,
            &all_block_schedules,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut segment_end_truncations,
            &class_to_vreg,
        );

        // Apply insertions so we can check the schedules.
        for (bi, mut insertions) in per_block_insertions.into_iter().enumerate() {
            insertions.sort_by(|a, b| b.0.cmp(&a.0));
            for (pos, inst) in insertions {
                let insert_at = pos.min(all_block_schedules[bi].len());
                all_block_schedules[bi].insert(insert_at, inst);
            }
        }

        // SpillStore must appear exactly once, in block 0.
        let total_stores: usize = all_block_schedules
            .iter()
            .flat_map(|s| s.iter())
            .filter(|i| is_spill_store(i))
            .count();
        assert_eq!(
            total_stores, 1,
            "SpillStore must appear exactly once across all blocks, got {total_stores}"
        );
        let stores_b0: usize = all_block_schedules[0]
            .iter()
            .filter(|i| is_spill_store(i))
            .count();
        assert_eq!(
            stores_b0, 1,
            "the single SpillStore must be in block 0, got {stores_b0}"
        );

        // SpillLoad must appear in each use block (1-4).
        for b in 1..5 {
            let loads = all_block_schedules[b]
                .iter()
                .filter(|i| is_spill_load(i))
                .count();
            assert!(
                loads >= 1,
                "block {b} must have at least one SpillLoad, got {loads}"
            );
        }
    }

    // Ported from `coalesced_vreg_spill_finds_canonical_def`:
    // The new split pass works directly on VRegs (no alias indirection needed).
    // SpillStore appears in the def block; SpillLoad in the use block.
    #[test]
    fn slot_spill_store_in_def_block_load_in_use_block() {
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        // Block 0: v0 = x86add(v1, v2)  [non-remat def]
        // Block 1: v3 = proj0(v0)        [use]
        let sched0 = vec![
            iconst_inst_gpr(1, 0),
            iconst_inst_gpr(2, 0),
            x86add_inst(0, 1, 2),
        ];
        let sched1 = vec![proj0_inst(3, 0)];

        let mut all_block_schedules = vec![sched0, sched1];
        let class_to_vreg = make_simple_class_map();
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut segment_end_truncations = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_cross_block_slot_spill(
            VReg(0),
            RegClass::GPR,
            &all_block_schedules,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut segment_end_truncations,
            &class_to_vreg,
        );

        // Apply insertions.
        for (bi, mut insertions) in per_block_insertions.into_iter().enumerate() {
            insertions.sort_by(|a, b| b.0.cmp(&a.0));
            for (pos, inst) in insertions {
                let insert_at = pos.min(all_block_schedules[bi].len());
                all_block_schedules[bi].insert(insert_at, inst);
            }
        }

        // SpillStore must be in block 0.
        assert!(
            all_block_schedules[0].iter().any(is_spill_store),
            "SpillStore must appear in block 0 (def block)"
        );
        assert!(
            !all_block_schedules[1].iter().any(is_spill_store),
            "SpillStore must NOT appear in block 1 (use block)"
        );

        // SpillLoad must be in block 1.
        assert!(
            all_block_schedules[1].iter().any(is_spill_load),
            "SpillLoad must appear in block 1 (use block)"
        );

        // SpillStore in block 0 must come AFTER v0's def.
        let b0 = &all_block_schedules[0];
        let def_pos = b0.iter().position(|i| i.dst == VReg(0)).unwrap();
        let store_pos = b0.iter().position(is_spill_store).unwrap();
        assert!(
            store_pos > def_pos,
            "SpillStore (pos {store_pos}) must come after v0's def (pos {def_pos})"
        );
    }

    // Ported from `cross_block_non_remat_spills` (regalloc/split.rs):
    // Cross-block non-remat value gets a spill slot; SpillStore at def-block exit,
    // SpillLoad at use-block entry (via apply_cross_block_slot_spill).
    #[test]
    fn legacy_cross_block_non_remat_spills() {
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        let mut all_block_schedules = vec![
            vec![proj0_inst(0, 99)], // block 0: v0 = proj0(v99) -- non-remat
            vec![proj0_inst(1, 0)],  // block 1: v1 = use(v0)
        ];
        let class_to_vreg = make_simple_class_map();
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut segment_end_truncations = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_cross_block_slot_spill(
            VReg(0),
            RegClass::GPR,
            &all_block_schedules,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut segment_end_truncations,
            &class_to_vreg,
        );

        assert_eq!(new_slot_count, 1, "one slot must be allocated");

        // Apply insertions.
        for (bi, mut insertions) in per_block_insertions.into_iter().enumerate() {
            insertions.sort_by(|a, b| b.0.cmp(&a.0));
            for (pos, inst) in insertions {
                let insert_at = pos.min(all_block_schedules[bi].len());
                all_block_schedules[bi].insert(insert_at, inst);
            }
        }

        // SpillStore in block 0.
        assert!(
            all_block_schedules[0].iter().any(is_spill_store),
            "SpillStore must appear in block 0 (def block)"
        );
        // SpillLoad in block 1.
        assert!(
            all_block_schedules[1].iter().any(is_spill_load),
            "SpillLoad must appear in block 1 (use block)"
        );
        // Operand of block 1's use instruction rewritten to reload VReg.
        assert!(
            !operand_rewrites.is_empty(),
            "operand rewrites must be present for the use in block 1"
        );
    }

    // Ported from `cross_block_remat_no_spill` (regalloc/split.rs):
    // Rematerializable (Iconst) cross-block value: no SpillStore/SpillLoad;
    // instead choose_split_kind returns Remat, and apply_split_planned inserts
    // fresh copies. The class_to_vreg lookup determines Remat via e-graph.
    #[test]
    fn legacy_cross_block_remat_no_spill() {
        use crate::egraph::enode::ENode;
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        // v0 = Iconst 42 in block 0; block 1 uses v0.
        let schedule_b0 = vec![iconst_inst_gpr(0, 42)];
        let schedule_b1 = vec![proj0_inst(1, 0)];

        // Build an EGraph with a classid for the Iconst.
        let mut egraph = EGraph::new();
        let iconst_cid = egraph.add(ENode {
            op: Op::Iconst(42, Type::I64),
            children: smallvec::smallvec![],
        });
        let cost_model = CostModel::new(OptGoal::Balanced);
        let extraction = make_empty_extraction();

        // choose_split_kind for an Iconst (not a call arg) must return Remat.
        let kind = choose_split_kind(
            VReg(0),
            0,
            &BTreeSet::new(),
            Some(iconst_cid),
            &BTreeSet::new(),
            &egraph,
            &cost_model,
            &extraction,
        );
        assert!(
            matches!(kind, SplitKind::Remat(Op::Iconst(42, _))),
            "Iconst cross-block value must use Remat, got {kind:?}"
        );

        // apply_split_planned in block 1 must emit a fresh Iconst copy (not a SpillLoad).
        let class_to_vreg = make_simple_class_map();
        let mut insertions: Vec<(usize, ScheduledInst)> = vec![];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_split_planned(
            1, // block 1 has the use
            VReg(0),
            kind,
            &schedule_b1,
            RegClass::GPR,
            &mut next_vreg,
            &mut new_slot_count,
            &mut insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &class_to_vreg,
        );

        // No SpillLoad; a fresh Iconst is emitted instead.
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_load(i)),
            "cross-block remat must not emit SpillLoad"
        );
        assert!(
            !insertions.iter().any(|(_, i)| is_spill_store(i)),
            "cross-block remat must not emit SpillStore"
        );
        // A fresh Iconst copy must be present.
        let fresh_iconst = insertions
            .iter()
            .any(|(_, i)| matches!(i.op, Op::Iconst(42, _)));
        assert!(
            fresh_iconst,
            "cross-block remat must emit a fresh Iconst in the use block"
        );
        let _ = (schedule_b0,);
    }

    // Ported from `pass_through_value_skipped` (regalloc/split.rs):
    // A VReg that passes through a block (no use, no def there) generates no
    // insertions in that block from the splitter. apply_cross_block_slot_spill
    // only inserts SpillLoad in blocks that actually USE the victim.
    #[test]
    fn legacy_pass_through_block_gets_no_insertions() {
        use crate::regalloc::spill::{is_spill_load, is_spill_store};

        // Block 0: def v0
        // Block 1: pass-through (no use of v0)
        // Block 2: use v0
        let all_block_schedules = vec![
            vec![proj0_inst(0, 99)],     // block 0: def v0
            vec![iconst_inst_gpr(1, 2)], // block 1: no use of v0
            vec![proj0_inst(2, 0)],      // block 2: use v0
        ];
        let class_to_vreg = make_simple_class_map();
        let mut per_block_insertions = vec![vec![], vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut segment_end_truncations = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_cross_block_slot_spill(
            VReg(0),
            RegClass::GPR,
            &all_block_schedules,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut segment_end_truncations,
            &class_to_vreg,
        );

        // Block 1 (pass-through): no SpillLoad inserted.
        let b1_has_load = per_block_insertions[1]
            .iter()
            .any(|(_, i)| is_spill_load(i));
        assert!(
            !b1_has_load,
            "pass-through block 1 must not have a SpillLoad inserted"
        );
        // Block 1: no SpillStore either (def is in block 0).
        let b1_has_store = per_block_insertions[1]
            .iter()
            .any(|(_, i)| is_spill_store(i));
        assert!(
            !b1_has_store,
            "pass-through block 1 must not have a SpillStore inserted"
        );

        // Block 2 (use): SpillLoad must be present.
        let b2_has_load = per_block_insertions[2]
            .iter()
            .any(|(_, i)| is_spill_load(i));
        assert!(
            b2_has_load,
            "block 2 (use block) must have a SpillLoad inserted"
        );
    }

    // Ported from `block_params_not_reloaded` (regalloc/split.rs):
    // Block params are handled by phi elimination, not the split pass.
    // detect_blockparam_call_crossings does insert loads for block params
    // that cross calls (Phase 6 behavior), but does not treat them as ordinary
    // cross-block victims. Here we verify that a GPR block param that does NOT
    // cross a call generates no insertions from apply_cross_block_slot_spill
    // (since GPR block params are never XMM, the Phase 6 function ignores them).
    #[test]
    fn legacy_gpr_block_param_not_spilled_by_split_pass() {
        // Block 1 receives v5 as a block parameter (GPR type).
        // Block 1 uses v5 but there is no call in block 1.
        // apply_cross_block_slot_spill is NOT called for block params
        // (they are phi-eliminated); this test just verifies that even if
        // erroneously called, the pass treats v5 like any other VReg:
        // it finds v5's def in block 0 (iconst) and inserts SpillLoad in block 1.
        // The point: the split pass makes no special exception for block params
        // as block params — that's handled at a higher level (phi elim).
        //
        // This test verifies the operational invariant: the split pass is purely
        // mechanical about VReg def/use locations; it does not check "is this a
        // block param?" That responsibility belongs to the allocator pipeline.
        let all_block_schedules = vec![
            vec![iconst_inst_gpr(5, 1)], // block 0: v5 = iconst (simulating param source)
            vec![proj0_inst(1, 5)],      // block 1: uses v5 (simulating param block)
        ];
        let class_to_vreg = make_simple_class_map();
        let mut per_block_insertions = vec![vec![], vec![]];
        let mut new_segments = vec![];
        let mut operand_rewrites = vec![];
        let mut segment_end_truncations = vec![];
        let mut next_vreg = 10u32;
        let mut new_slot_count = 0u32;

        apply_cross_block_slot_spill(
            VReg(5),
            RegClass::GPR,
            &all_block_schedules,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions,
            &mut new_segments,
            &mut operand_rewrites,
            &mut segment_end_truncations,
            &class_to_vreg,
        );

        assert_eq!(
            new_slot_count, 1,
            "slot must be allocated for cross-block GPR"
        );
        assert!(
            !per_block_insertions[1].is_empty() || !operand_rewrites.is_empty(),
            "use block must have insertions or rewrites for the cross-block VReg"
        );
    }

    // Test 6: apply_plan_to updates class_to_vreg segments and bumps split_generation.
    #[test]
    fn split_updates_class_to_vreg_segments() {
        use crate::egraph::extract::ClassVRegMap;
        use crate::ir::op::ClassId;

        let mut map = ClassVRegMap::new();
        // Register a dummy class -> vreg mapping.
        map.insert_full_range(ClassId(0), VReg(0));
        assert_eq!(map.split_generation, 0);

        // Build a trivial plan with one new segment.
        let plan = SplitPlan {
            per_block_insertions: vec![Vec::new()],
            new_segments: vec![(
                ClassId(0),
                VReg(10),
                ProgramPoint::inst_point(0, 1),
                ProgramPoint::inst_point(0, 1),
            )],
            operand_rewrites: vec![],
            slot_spilled_params: BTreeMap::new(),
            segment_end_truncations: Vec::new(),
            slots_allocated: 0,
        };

        let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![vec![]];
        let mut next_vreg: u32 = 1;
        apply_plan_to(&mut block_schedules, &mut map, &mut next_vreg, plan);

        assert_eq!(map.split_generation, 1, "split_generation must be bumped");
    }
}
