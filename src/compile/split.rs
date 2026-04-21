//! Pressure-driven live-range splitter for Blitz.
//!
//! Runs AFTER scheduling and effectful-op operand population, BEFORE
//! `allocate_global`. When `BLITZ_SPLIT=1` is set, `plan_splits` scans each
//! block for register-pressure overshoots and inserts either rematerialization
//! or spill/reload pairs to bring pressure within budget.
//!
//! Gating: the pass is disabled by default. Set `BLITZ_SPLIT=1` in the
//! environment to enable it. This preserves the baseline pipeline for all
//! tests that don't exercise the splitter.

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
}

/// Cost above which we prefer SlotSpill over Remat.
///
/// If `CostModel::cost(op) * loop_depth_penalty > SLOT_STORE_LOAD_COST`,
/// Remat is more expensive than slot spill so we use SlotSpill.
const SLOT_STORE_LOAD_COST: f64 = 5.0;

// ── Local liveness ────────────────────────────────────────────────────────────

/// Compute per-instruction live-before sets for a single block.
///
/// The scan is seeded from `live_in_seed` (the block's `GlobalLiveness::live_in`
/// entry) so pass-through VRegs are counted even if they have no local def/use.
///
/// Returns a `Vec` of length `n_insts + 1` where `result[i]` is the set of
/// VRegs live immediately BEFORE instruction `i`. `result[n_insts]` is the
/// live-before set for the (virtual) instruction after the last one, which
/// equals the live-out set as seen by the splitter (not the full global
/// live-out, since we seed only from `live_in` of this block).
fn compute_local_liveness(
    block_idx: usize,
    schedule: &[ScheduledInst],
    live_in_seed: &BTreeSet<VReg>,
) -> Vec<BTreeSet<VReg>> {
    let n = schedule.len();
    // result[i] = live set before instruction i.
    // result[n] = live set after the last instruction (exit snapshot).
    let mut result: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n + 1];

    // Seed the backward scan: start with live_in_seed as the "live after last inst".
    let mut live: BTreeSet<VReg> = live_in_seed.clone();

    // Walk backward.
    for i in (0..n).rev() {
        let inst = &schedule[i];

        // Live-before = (live ∪ uses) - {def} (if def not in uses).
        // Add uses before removing def, because the def is not live before this inst.
        for &use_vreg in &inst.operands {
            live.insert(use_vreg);
        }
        // Record live-before[i] (includes uses, excludes def if def != use).
        result[i] = live.clone();

        // Now remove the def (def is live after this point only when used later).
        live.remove(&inst.dst);
    }

    // result[n] corresponds to "after last instruction" = our seed (live_in_seed
    // propagated backward), but since we walked backward starting from live_in_seed
    // we've already set result[0] correctly as the live-before of the first inst.
    // Set result[n] to be the live_in_seed so the caller has a consistent "exit" view.
    result[n] = live_in_seed.clone();

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
    vreg_classes: &BTreeMap<VReg, RegClass>,
    class: RegClass,
) -> Vec<u32> {
    live_sets
        .iter()
        .map(|live| {
            live.iter()
                .filter(|&v| vreg_classes.get(v).copied() == Some(class))
                .count() as u32
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
        let penalty = LOOP_DEPTH_PENALTY_BASE.saturating_pow(loop_depth).max(1) as f64;
        if extracted.cost * penalty <= SLOT_STORE_LOAD_COST {
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

// ── Plan construction ─────────────────────────────────────────────────────────

/// Build a `SplitPlan` for the given function-wide schedules.
///
/// Scans each block for register-pressure overshoots. For each overshoot,
/// picks a victim and plans either a remat or slot-spill split.
///
/// Also detects block param VRegs live across calls and plans
/// `SlotSpillBlockParam` splits for them (Phase 6).
///
/// `loop_depths` maps VReg -> loop nesting depth (from `compute_loop_depths`).
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
    loop_depths: &BTreeMap<VReg, u32>,
    func: &Function,
) -> SplitPlan {
    let n_blocks = block_schedules.len();
    let mut per_block_insertions: Vec<Vec<(usize, ScheduledInst)>> = vec![Vec::new(); n_blocks];
    let mut new_segments: Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)> = Vec::new();
    let mut new_slot_count: u32 = 0;
    let mut operand_rewrites: Vec<(usize, usize, usize, VReg)> = Vec::new();
    let mut slot_spilled_params: BTreeMap<(BlockId, u32), SlotSpilledParamInfo> = BTreeMap::new();

    for block_idx in 0..n_blocks {
        let schedule = &block_schedules[block_idx];
        if schedule.is_empty() {
            continue;
        }

        // Seed from global live_in so pass-through VRegs are counted.
        let live_in_seed = if block_idx < global_liveness.live_in.len() {
            &global_liveness.live_in[block_idx]
        } else {
            continue;
        };

        // Build VReg -> RegClass map for this block.
        let vreg_classes = crate::regalloc::build_vreg_classes_from_insts(schedule);

        // Compute per-instruction live-before sets.
        let live_sets = compute_local_liveness(block_idx, schedule, live_in_seed);

        // Compute pressure per class.
        let gpr_pressure = compute_pressure_for_class(&live_sets, &vreg_classes, RegClass::GPR);
        let xmm_pressure = compute_pressure_for_class(&live_sets, &vreg_classes, RegClass::XMM);

        // Find the first overshoot point.
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
        let (overshoot_inst_idx, overshoot_class) = match (gpr_overshoot, xmm_overshoot) {
            (None, None) => continue,
            (Some(g), None) => (g.0, RegClass::GPR),
            (None, Some(x)) => (x.0, RegClass::XMM),
            (Some(g), Some(x)) => {
                let xmm_excess = x.1.saturating_sub(xmm_budget);
                let gpr_excess = g.1.saturating_sub(gpr_budget);
                if xmm_excess >= gpr_excess {
                    (x.0, RegClass::XMM)
                } else {
                    (g.0, RegClass::GPR)
                }
            }
        };

        // Gather victims: VRegs of the target class live at the overshoot point.
        let live_at = &live_sets[overshoot_inst_idx];
        let range_lengths = compute_live_range_lengths(&live_sets);

        // Build def-site map for this block.
        let def_inst_map: BTreeMap<VReg, &ScheduledInst> = schedule
            .iter()
            .rev()
            .map(|inst| (inst.dst, inst))
            .collect::<BTreeMap<_, _>>();

        // Collect call-arg VRegs (must not remat).
        let call_arg_vregs = collect_call_arg_vregs_set(schedule);

        // Collect candidates: VRegs of the right class that are live at overshoot,
        // excluding Flags-typed VRegs.
        let mut candidates: Vec<VReg> = live_at
            .iter()
            .filter(|&&v| vreg_classes.get(&v).copied() == Some(overshoot_class))
            .filter(|&&v| {
                // Skip Flags-typed VRegs (cannot be split).
                if let Some(inst) = def_inst_map.get(&v) {
                    // Check if the result type is Flags.
                    let result_ty = inst.op.result_type(&[]);
                    !matches!(result_ty, Type::Flags)
                } else {
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

        let victim = match candidates.first() {
            Some(&v) => v,
            None => continue,
        };

        let loop_depth = loop_depths.get(&victim).copied().unwrap_or(0);

        // Look up the victim's ClassId and compute the set of live classes at
        // the overshoot point. These are passed to `extract_at_with_memo` so
        // the e-graph can pick the cheapest op whose children are already live
        // (cost = 0) rather than requiring re-extraction from scratch.
        let use_point = ProgramPoint::inst_point(block_idx, overshoot_inst_idx);
        let victim_class = class_to_vreg.vreg_to_class(victim, use_point);
        let live_classes_at_use: BTreeSet<ClassId> = live_at
            .iter()
            .filter_map(|&v| class_to_vreg.vreg_to_class(v, use_point))
            .collect();

        let kind = choose_split_kind(
            victim,
            loop_depth,
            &call_arg_vregs,
            victim_class,
            &live_classes_at_use,
            egraph,
            cost_model,
            extraction,
        );

        // Apply the split: plan insertions and operand rewrites.
        apply_split_planned(
            block_idx,
            victim,
            kind,
            schedule,
            overshoot_class,
            &mut next_vreg,
            &mut new_slot_count,
            &mut per_block_insertions[block_idx],
            &mut new_segments,
            &mut operand_rewrites,
            class_to_vreg,
        );
    }

    // Phase 6: Block-param split strategy.
    //
    // Scan each block for block param VRegs that are live across a call in the
    // same block. Since all XMM registers are caller-saved, a block param that
    // is live-in AND there is a call in the block will collide with the call's
    // clobber set, forcing the allocator into an infeasible coloring. We fix
    // this by routing the block param's phi copy to a slot in predecessors and
    // inserting SpillLoad at each use in the block.
    detect_blockparam_call_crossings(
        func,
        egraph,
        block_schedules,
        class_to_vreg,
        global_liveness,
        &mut new_slot_count,
        &mut next_vreg,
        &mut per_block_insertions,
        &mut new_segments,
        &mut operand_rewrites,
        &mut slot_spilled_params,
    );

    SplitPlan {
        per_block_insertions,
        new_segments,
        operand_rewrites,
        slot_spilled_params,
    }
}

// ── Block-param call-crossing detection (Phase 6) ───────────────────────────

/// Detect XMM block param VRegs that are live in some block containing a call.
///
/// For each block B with XMM params, check if the param VReg is live-in to
/// any block that has a call instruction. If so, the param will conflict with
/// the call's clobber of all XMM regs, causing a chromatic overshoot in the
/// allocator. We fix this by routing the param through a slot.
///
/// For each such param:
/// 1. Allocate a spill slot.
/// 2. Insert `XmmSpillLoad` before each use of the param VReg in EVERY block
///    where the param is live AND has uses (covers both the block defining the
///    param and any blocks where it appears as a live-in).
/// 3. Record in `slot_spilled_params` so `lower_terminator` emits a slot
///    store in each predecessor's phi copies (for forward edges only —
///    the back-edge skips the store since the slot is already populated and
///    the param's arg VReg has no register on the back-edge).
/// 4. `apply_plan_to` truncates the param's segment to start after
///    `BLOCK_ENTRY(param_block_idx)` so no register is allocated to it.
#[allow(clippy::too_many_arguments)]
fn detect_blockparam_call_crossings(
    func: &Function,
    egraph: &EGraph,
    block_schedules: &[Vec<ScheduledInst>],
    class_to_vreg: &ClassVRegMap,
    global_liveness: &GlobalLiveness,
    new_slot_count: &mut u32,
    next_vreg: &mut u32,
    per_block_insertions: &mut [Vec<(usize, ScheduledInst)>],
    new_segments: &mut Vec<(ClassId, VReg, ProgramPoint, ProgramPoint)>,
    operand_rewrites: &mut Vec<(usize, usize, usize, VReg)>,
    slot_spilled_params: &mut BTreeMap<(BlockId, u32), SlotSpilledParamInfo>,
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

    if call_blocks.is_empty() {
        return;
    }

    // For each block with XMM params, check if the param VReg is live in
    // a block that has a call.
    for block_idx in 0..n_blocks.min(func.blocks.len()) {
        let block = &func.blocks[block_idx];
        if block.param_types.is_empty() {
            continue;
        }

        let entry_point = ProgramPoint::block_entry(block_idx);

        for pidx in 0..block.param_types.len() as u32 {
            // Only handle XMM (float) params.
            if !block.param_types[pidx as usize].is_float() {
                continue;
            }

            // Find the ClassId and VReg for this BlockParam.
            let mut param_vreg_and_class: Option<(VReg, ClassId)> = None;
            'outer: for i in 0..egraph.classes.len() as u32 {
                let cid = ClassId(i);
                let canon = egraph.unionfind.find_immutable(cid);
                if canon != cid {
                    continue;
                }
                let class = egraph.class(cid);
                for node in &class.nodes {
                    if let Op::BlockParam(bid, pidx2, _) = &node.op
                        && *bid == block.id
                        && *pidx2 == pidx
                        && let Some(vreg) = class_to_vreg.lookup(cid, entry_point)
                    {
                        param_vreg_and_class = Some((vreg, cid));
                        break 'outer;
                    }
                }
            }

            let (param_vreg, param_class) = match param_vreg_and_class {
                Some(p) => p,
                None => continue,
            };

            // Check if this param VReg is live-in to any call block.
            let is_live_in_call_block = call_blocks.iter().any(|&call_bi| {
                call_bi < global_liveness.live_in.len()
                    && global_liveness.live_in[call_bi].contains(&param_vreg)
            });

            if !is_live_in_call_block {
                continue;
            }

            // Allocate a spill slot for this param.
            let slot = *new_slot_count as i64;
            *new_slot_count += 1;

            // Insert XmmSpillLoad before each use of param_vreg in EVERY block
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
                        op: Op::XmmSpillLoad(slot),
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

            // Record this param for slot-spill so lower_terminator can emit
            // slot stores in phi copies for forward edges (back-edges with no
            // register for the arg will be skipped automatically in build_phi_copies).
            slot_spilled_params.insert(
                (block.id, pidx),
                SlotSpilledParamInfo {
                    vreg: param_vreg,
                    slot,
                    reg_class: RegClass::XMM,
                    block_idx,
                },
            );
        }
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

            // Register the store dummy VReg's segment (just for bookkeeping).
            if let Some(class) = victim_class
                && let Some(dp) = def_pos
            {
                let point = ProgramPoint::inst_point(block_idx, dp + 1);
                new_segments.push((class, store_dummy_vreg, point, point));
            }
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
) {
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
        insertions.sort_by(|a, b| b.0.cmp(&a.0));
        let schedule = &mut block_schedules[block_idx];
        for (pos, inst) in insertions {
            let insert_at = pos.min(schedule.len());
            schedule.insert(insert_at, inst);
        }
    }

    // Register new segments in class_to_vreg.
    for (class, vreg, start, end) in plan.new_segments {
        class_to_vreg.insert_segment(class, vreg, start, end);
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
            &loop_depths,
            &func,
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
            &loop_depths,
            &func,
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
            op: Op::Fconst(3.14f64.to_bits(), Type::F64),
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
        use crate::ir::op::ClassId;

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
        };

        let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![vec![]];
        let mut next_vreg: u32 = 1;
        apply_plan_to(&mut block_schedules, &mut map, &mut next_vreg, plan);

        assert_eq!(map.split_generation, 1, "split_generation must be bumped");
    }
}
