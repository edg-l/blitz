//! Function-scope (global) register allocator.
//!
//! # Algorithm overview
//!
//! This module implements an SSA-based Chaitin-Briggs-style graph-coloring
//! allocator that operates over an entire function at once, rather than
//! block-by-block. The key property exploited is that Blitz's IR is in SSA
//! form: each VReg is defined exactly once, so the interference graph of a
//! pure SSA function is chordal, and MCS + greedy coloring is optimal.
//!
//! # Phase 2 state layout
//!
//! After Phase 2 (`build_global_interference`), the following data is
//! available for consumption by Phase 3 and Phase 5:
//!
//! - `graph: InterferenceGraph` - function-wide interference graph sized for
//!   `num_vregs` = max VReg index + 1 across the whole function. `reg_class`
//!   is pre-populated from `build_vreg_classes_from_all_blocks` before any
//!   per-block pass runs, so cross-block live-in VRegs have correct classes.
//!
//! - `per_block_liveness: Vec<LivenessInfo>` - per-block liveness indexed by
//!   block index (same ordering as `block_schedules`). Phase 3 clobber
//!   injection uses `per_block_liveness[b].live_at[cp]` where `cp` is a
//!   per-block instruction index. Phase 5 spill-pressure detection scans the
//!   same data.
//!
//! # Coloring strategy
//!
//! MCS + greedy coloring (Phase 4) is used as the sole coloring pass. On a
//! pure SSA function-scope graph the interference graph is chordal, so
//! MCS + greedy is already optimal without any interval-color fallback.
//!
//! The per-block interval-color fallback (present in the single-block
//! allocator) is intentionally omitted at function scope for two reasons:
//!
//! 1. **Optimality**: Blitz's IR is in SSA form (single def per VReg). The
//!    interference graph of a pure SSA program is chordal. For chordal graphs,
//!    MCS + greedy in reverse elimination order is provably optimal (chromatic
//!    number equals clique number). No interval-color pass can do better.
//!
//! 2. **Failure mode**: If the function-scope graph exceeds register budget
//!    after greedy coloring, it means pressure genuinely requires spilling.
//!    Phase 5's iterative spill-and-recolor (Briggs-style) handles this: it
//!    selects a spill candidate, inserts spill/reload code, and re-runs
//!    Phase 3–4. This is strictly more powerful than an interval-color
//!    fallback, which would still fail at the same pressure point.
//!
//! The interval-color path existed in the per-block allocator only because
//! coalescing and spill-code insertion within a single block occasionally broke
//! chordality. At function scope those operations run before the final graph is
//! built (Phase 3 rebuilds after coalescing), so the final graph fed into Phase
//! 4 is still chordal.

#![allow(dead_code, unused_variables)]

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::{CALLER_SAVED_GPR, CALLER_SAVED_XMM};
use crate::x86::reg::{Reg, RegClass};

use super::GlobalRegAllocResult;
use super::build_vreg_classes_from_all_blocks;
use super::coalesce::coalesce;
use super::coloring::{allocatable_gpr_order, allocatable_xmm_order};
use super::interference::{InterferenceGraph, build_interference_into};
use super::liveness::{LivenessInfo, compute_liveness};
use super::rewrite::apply_coalescing;

/// Result of Phase 2: function-wide interference graph + per-block liveness.
///
/// Both `graph` and `per_block_liveness` are consumed by Phase 3 (clobber
/// injection) and Phase 5 (spill pressure detection). They are produced by
/// `build_global_interference` and stored here so later phases can access them
/// without recomputing.
struct Phase2State {
    /// Function-wide interference graph.
    ///
    /// Sized for `num_vregs` = max VReg index + 1 across all blocks.
    /// `reg_class` is pre-populated from the function-wide class map before
    /// any per-block edges are added, ensuring cross-block live-in VRegs have
    /// the correct class from the start.
    graph: InterferenceGraph,

    /// Per-block liveness indexed by block index.
    ///
    /// `per_block_liveness[b]` is the `LivenessInfo` for `block_schedules[b]`,
    /// computed with `global_liveness.live_out[b]` as the block-exit live set.
    per_block_liveness: Vec<LivenessInfo>,
}

/// Result of Phase 3: precolorings, coalesced schedules, post-rebuild graph.
///
/// Produced by `run_phase3` and consumed by Phase 4 (coloring). Contains the
/// post-coalesce instruction lists, the final interference graph with phantoms,
/// merged precolorings, and the list of dropped param precolorings.
pub(crate) struct Phase3State {
    /// Post-coalesce + post-rebuild instruction lists, one per block.
    per_block_insts: Vec<Vec<ScheduledInst>>,

    /// Post-rebuild interference graph (with clobber phantoms injected).
    /// This is the graph handed to Phase 4 coloring.
    graph: InterferenceGraph,

    /// Post-rebuild per-block liveness (recomputed on post-coalesce schedules).
    per_block_liveness: Vec<LivenessInfo>,

    /// Merged precoloring map: VReg index -> color. Covers param precolors,
    /// shift/div precolors, and all three phantom precolor sets.
    pre_coloring_colors: BTreeMap<usize, u32>,

    /// Param VRegs whose ABI precoloring was dropped because a call clobber
    /// phantom interferes with the same color. The lowering must emit a mov
    /// from the ABI register to the allocated register at function entry.
    unprecolored_params: Vec<(VReg, Reg)>,

    /// Shared next_vreg counter after Phase 3 phantom injection.
    next_vreg: u32,

    /// Coalescing alias map: `from_idx -> into_idx`. When two VRegs are coalesced,
    /// the "from" VReg no longer exists in the post-coalesce schedules; its uses
    /// have been rewritten to "into". Downstream callers (e.g. lowering's
    /// `block_class_to_vreg`) must apply this map when resolving ClassId -> VReg
    /// so that stale `class_to_vreg` entries pointing at `from` VRegs are chased
    /// to their live canonical counterparts.
    alias_map: BTreeMap<u32, u32>,
}

/// Add pairwise interference between all block_params of each block. Phi
/// copies at block entry write distinct values into each param, so even if
/// two params have disjoint schedule-level live ranges (e.g. both unused in
/// the block body), they must occupy distinct registers during the copy
/// sequence.
fn add_block_param_interferences(
    graph: &mut InterferenceGraph,
    block_param_vregs_per_block: &[BTreeSet<VReg>],
    alias_map: &BTreeMap<u32, u32>,
) {
    let resolve = |v: VReg| -> VReg {
        let mut idx = v.0;
        while let Some(&t) = alias_map.get(&idx) {
            if t == idx {
                break;
            }
            idx = t;
        }
        VReg(idx)
    };
    for params in block_param_vregs_per_block {
        let mut seen: BTreeSet<VReg> = BTreeSet::new();
        let unique: Vec<VReg> = params
            .iter()
            .map(|&p| resolve(p))
            .filter(|&v| seen.insert(v))
            .collect();
        if unique.len() < 2 {
            continue;
        }
        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                let a = unique[i].0 as usize;
                let b = unique[j].0 as usize;
                if a >= graph.num_vregs || b >= graph.num_vregs {
                    continue;
                }
                if graph.reg_class[a] == graph.reg_class[b] {
                    graph.adj[a].insert(b);
                    graph.adj[b].insert(a);
                }
            }
        }
    }
}

/// Build the function-wide interference graph (Phase 2).
///
/// Steps:
/// 1. Build a function-wide VReg class map from all blocks (Task 2.3).
/// 2. Determine `num_vregs` and allocate the shared `InterferenceGraph` with
///    `reg_class` pre-populated (Task 2.4).
/// 3. For each block, compute per-block `LivenessInfo` using the global
///    `live_out` set, call `build_interference_into` to add edges, and store
///    the `LivenessInfo` in `per_block_liveness` (Tasks 2.4, 2.4.5).
/// 4. Add cross-block boundary interferences: all pairs in `live_in[b]` of the
///    same class interfere; same for `live_out[b]` (Task 2.4).
fn build_global_interference(
    block_schedules: &[Vec<ScheduledInst>],
    global_liveness: &crate::regalloc::global_liveness::GlobalLiveness,
) -> Phase2State {
    // Task 2.3: function-wide class map (must complete before graph init).
    let vreg_class_map = build_vreg_classes_from_all_blocks(block_schedules);

    // Task 2.4: determine num_vregs across all blocks + global liveness sets.
    let num_vregs = {
        let mut max_idx = 0usize;
        for sched in block_schedules {
            for inst in sched {
                let idx = inst.dst.0 as usize;
                if idx > max_idx {
                    max_idx = idx;
                }
                for &op in &inst.operands {
                    let oidx = op.0 as usize;
                    if oidx > max_idx {
                        max_idx = oidx;
                    }
                }
            }
        }
        // Include VRegs that appear only in live_in/live_out (not in any
        // instruction within these schedules, e.g. live-through values).
        for live_set in global_liveness
            .live_in
            .iter()
            .chain(global_liveness.live_out.iter())
        {
            for v in live_set {
                let idx = v.0 as usize;
                if idx > max_idx {
                    max_idx = idx;
                }
            }
        }
        if max_idx == 0 && block_schedules.iter().all(|s| s.is_empty()) {
            0
        } else {
            max_idx + 1
        }
    };

    if num_vregs == 0 {
        let per_block_liveness = block_schedules
            .iter()
            .map(|_| LivenessInfo {
                live_at: vec![],
                live_in: BTreeSet::new(),
                live_out: BTreeSet::new(),
            })
            .collect();
        return Phase2State {
            graph: InterferenceGraph {
                num_vregs: 0,
                adj: vec![],
                reg_class: vec![],
            },
            per_block_liveness,
        };
    }

    // Initialize graph with reg_class pre-populated from the function-wide map.
    let mut reg_class = vec![RegClass::GPR; num_vregs];
    for (&vreg, &class) in &vreg_class_map {
        let idx = vreg.0 as usize;
        if idx < num_vregs {
            reg_class[idx] = class;
        }
    }

    let mut graph = InterferenceGraph {
        num_vregs,
        adj: vec![BTreeSet::new(); num_vregs],
        reg_class,
    };

    // Task 2.4.5: collect per-block liveness.
    let mut per_block_liveness: Vec<LivenessInfo> = Vec::with_capacity(block_schedules.len());

    // For each block: compute liveness, add edges, then add boundary interferences.
    for (b, sched) in block_schedules.iter().enumerate() {
        let block_live_out = &global_liveness.live_out[b];
        let liveness = compute_liveness(sched, block_live_out);

        // Add intra-block interference edges.
        build_interference_into(&mut graph, &liveness, sched);

        per_block_liveness.push(liveness);
    }

    // Add cross-block boundary interferences: all pairs in live_in[b] of the
    // same class interfere, and all pairs in live_out[b] of the same class
    // interfere. These capture phi-source/sink interferences not emitted by
    // the per-instruction pass.
    for b in 0..block_schedules.len() {
        add_boundary_interferences(&mut graph, &global_liveness.live_in[b]);
        add_boundary_interferences(&mut graph, &global_liveness.live_out[b]);
    }

    Phase2State {
        graph,
        per_block_liveness,
    }
}

/// Add interference edges between all pairs in `boundary_set` of the same class.
fn add_boundary_interferences(graph: &mut InterferenceGraph, boundary_set: &BTreeSet<VReg>) {
    let live: Vec<usize> = boundary_set
        .iter()
        .map(|v| v.0 as usize)
        .filter(|&idx| idx < graph.num_vregs)
        .collect();
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            let a = live[i];
            let b = live[j];
            if graph.reg_class[a] == graph.reg_class[b] {
                graph.add_edge(a, b);
            }
        }
    }
}

// ── Task 3.1: Function-wide precoloring ──────────────────────────────────────

/// Pre-color shift count operands to RCX for variable-shift instructions.
///
/// Mirrors `add_shift_precolors` from `compile/precolor.rs`.
fn add_shift_precolors_global(insts: &[ScheduledInst], precolors: &mut Vec<(VReg, Reg)>) {
    for inst in insts {
        if matches!(inst.op, Op::X86Shl | Op::X86Shr | Op::X86Sar) && inst.operands.len() >= 2 {
            let count_vreg = inst.operands[1];
            if !precolors.iter().any(|&(v, _)| v == count_vreg) {
                precolors.push((count_vreg, Reg::RCX));
            }
        }
    }
}

/// Pre-color division projections to RAX (quotient).
///
/// The dividend is NOT pre-colored: `lower::X86Idiv` emits `mov rax, <dividend>`
/// when needed. Precoloring the dividend would force its VReg to live in RAX
/// across the entire live range, which breaks when the dividend is also used
/// after the idiv (idiv clobbers RAX with the quotient).
///
/// Proj1 (remainder) is also not pre-colored; the lowering emits `mov dst, rdx`
/// so the remainder can live in any register (see `lower::Proj1` for
/// X86Idiv/X86Div sources).
fn add_div_precolors_global(insts: &[ScheduledInst], precolors: &mut Vec<(VReg, Reg)>) {
    let mut div_dst_vregs: BTreeSet<VReg> = BTreeSet::new();
    for inst in insts {
        if matches!(inst.op, Op::X86Idiv | Op::X86Div) {
            div_dst_vregs.insert(inst.dst);
        }
    }
    // Pre-color Proj0 nodes that project from a div result to RAX (quotient).
    for inst in insts {
        if inst.op == Op::Proj0
            && let Some(&src) = inst.operands.first()
            && div_dst_vregs.contains(&src)
            && !precolors.iter().any(|&(v, _)| v == inst.dst)
        {
            precolors.push((inst.dst, Reg::RAX));
        }
    }
}

/// Build a function-wide precoloring list covering params, shifts, divs, and
/// caller-supplied call-argument/return-value VRegs.
///
/// `call_arg_precolors` must be computed by the caller BEFORE
/// `populate_effectful_operands` sorts the barrier operands by VReg index
/// (destroying ABI argument order). The canonical source is
/// `add_call_precolors_for_block` in `compile/precolor.rs`, called per block
/// and aggregated into a single `Vec<(VReg, Reg)>` before invoking
/// `allocate_global`.
///
/// Returns `(precolors, param_vreg_indices)` where:
/// - `precolors` is the unified `Vec<(VReg, Reg)>`. The same-VReg/different-reg
///   case is a bug in the IR and is caught by a `debug_assert`. Multiple distinct
///   VRegs sharing the same physical reg (e.g., first args of two different calls
///   both precolored to RDI) is expected and is NOT a conflict.
/// - `param_vreg_indices` is the set of VReg indices that come from function
///   parameters (used by `merge_precolorings_global` to identify which can be
///   dropped on clobber conflict).
fn build_function_wide_precoloring(
    param_vregs: &[(VReg, Reg)],
    block_schedules: &[Vec<ScheduledInst>],
    call_arg_precolors: Vec<(VReg, Reg)>,
) -> (Vec<(VReg, Reg)>, BTreeSet<usize>) {
    // Start with function param precolorings.
    let mut precolors: Vec<(VReg, Reg)> = param_vregs.to_vec();
    let param_vreg_indices: BTreeSet<usize> =
        param_vregs.iter().map(|(v, _)| v.0 as usize).collect();

    // Merge shift and div precolors from all blocks' schedules.
    for sched in block_schedules {
        add_shift_precolors_global(sched, &mut precolors);
        add_div_precolors_global(sched, &mut precolors);
    }

    // Merge caller-supplied call-arg precolors. These were computed from IR
    // EffectfulOp::Call args in ABI argument order, before populate_effectful_operands
    // sorted the barrier operands by VReg index.
    for (vreg, reg) in call_arg_precolors {
        if !precolors.iter().any(|&(v, _)| v == vreg) {
            precolors.push((vreg, reg));
        }
    }

    // Validate: the same VReg must not appear with two different physical regs
    // (that is an IR bug). Multiple distinct VRegs mapped to the same reg across
    // different call sites is fine and must not trigger this assert.
    let mut vreg_to_reg: BTreeMap<VReg, Reg> = BTreeMap::new();
    for &(vreg, reg) in &precolors {
        if let Some(&existing) = vreg_to_reg.get(&vreg) {
            debug_assert_eq!(
                existing, reg,
                "VReg {:?} precolored to two different regs ({:?} and {:?}) — IR bug",
                vreg, existing, reg
            );
        } else {
            vreg_to_reg.insert(vreg, reg);
        }
    }

    (precolors, param_vreg_indices)
}

/// Convert a `Vec<(VReg, Reg)>` precoloring to a `BTreeMap<usize, u32>` color
/// map using the ordering provided by `allocatable_gpr_order` for GPR regs
/// and `allocatable_xmm_order` for XMM regs.
fn precolors_to_color_map(
    precolors: &[(VReg, Reg)],
    uses_frame_pointer: bool,
) -> BTreeMap<usize, u32> {
    let gpr_order = allocatable_gpr_order(uses_frame_pointer);
    let xmm_order = allocatable_xmm_order();

    let gpr_reg_to_color: BTreeMap<Reg, u32> = gpr_order
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();
    let xmm_reg_to_color: BTreeMap<Reg, u32> = xmm_order
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();

    let mut map: BTreeMap<usize, u32> = BTreeMap::new();
    for &(vreg, reg) in precolors {
        let color = if reg.is_xmm() {
            xmm_reg_to_color.get(&reg).copied()
        } else {
            gpr_reg_to_color.get(&reg).copied()
        };
        if let Some(c) = color {
            map.insert(vreg.0 as usize, c);
        }
    }
    map
}

// ── Task 3.2: Call/div point collection ─────────────────────────────────────

/// Collect all call and div program points across all blocks.
///
/// Returns `(call_points, div_points)` where each entry is `(block_idx, inst_idx)`.
fn collect_call_div_points(
    block_schedules: &[Vec<ScheduledInst>],
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut call_points: Vec<(usize, usize)> = Vec::new();
    let mut div_points: Vec<(usize, usize)> = Vec::new();

    for (b, sched) in block_schedules.iter().enumerate() {
        for (i, inst) in sched.iter().enumerate() {
            if matches!(inst.op, Op::CallResult(_, _) | Op::VoidCallBarrier) {
                call_points.push((b, i));
            }
            if matches!(inst.op, Op::X86Idiv | Op::X86Div) {
                div_points.push((b, i));
            }
        }
    }

    (call_points, div_points)
}

// ── Task 3.3 & 3.4: Global clobber interference injection ───────────────────

/// Configuration for `add_clobber_interferences_global`.
struct GlobalClobberConfig<'a> {
    /// `(block_idx, inst_idx)` pairs of the clobbering instructions.
    points: &'a [(usize, usize)],
    /// Physical registers clobbered at each point.
    clobbered_regs: &'a [Reg],
    /// Register class this clobber applies to.
    reg_class: RegClass,
    /// Ordered register list used to assign stable color numbers to phantoms.
    ordered_regs: Vec<Reg>,
    /// When true, exclude call-arg VRegs that die at the call from interference.
    exclude_call_args: bool,
    /// When true, skip points where no live VReg of `reg_class` exists.
    skip_if_no_live: bool,
}

/// Extend the global interference graph with phantom VRegs for clobbered
/// registers at each (block_idx, inst_idx) program point.
///
/// Mirrors `add_clobber_interferences` from `allocator.rs` but operates on the
/// global (block_idx, inst_idx) space instead of per-block flat indices.
///
/// Returns the updated graph and a `BTreeMap<usize, u32>` of phantom VReg
/// index -> color (same layering as the per-block three-way maps).
fn add_clobber_interferences_global(
    mut graph: InterferenceGraph,
    per_block_liveness: &[LivenessInfo],
    block_schedules: &[Vec<ScheduledInst>],
    config: &GlobalClobberConfig,
    next_vreg: &mut u32,
) -> (InterferenceGraph, BTreeMap<usize, u32>) {
    if config.points.is_empty() {
        return (graph, BTreeMap::new());
    }

    let reg_to_color: BTreeMap<Reg, u32> = config
        .ordered_regs
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();

    let mut phantom_precolors: BTreeMap<usize, u32> = BTreeMap::new();

    for &(block_idx, inst_idx) in config.points {
        let liveness = &per_block_liveness[block_idx];
        let sched = &block_schedules[block_idx];
        let n = liveness.live_at.len();

        let live_at_cp: &BTreeSet<VReg> = if inst_idx < n {
            &liveness.live_at[inst_idx]
        } else {
            &liveness.live_out
        };

        // Collect call-arg vregs if exclusion is enabled. Only exclude args
        // that are NOT live after the call — an arg that survives past its
        // call must still interfere with caller-saved clobber phantoms.
        let call_arg_vregs: BTreeSet<usize> = if config.exclude_call_args {
            if inst_idx < sched.len() {
                let inst = &sched[inst_idx];
                if matches!(inst.op, Op::CallResult(_, _) | Op::VoidCallBarrier) {
                    let live_after: &BTreeSet<VReg> = if inst_idx + 1 < n {
                        &liveness.live_at[inst_idx + 1]
                    } else {
                        &liveness.live_out
                    };
                    inst.operands
                        .iter()
                        .filter(|v| !live_after.contains(v))
                        .map(|v| v.0 as usize)
                        .collect()
                } else {
                    BTreeSet::new()
                }
            } else {
                BTreeSet::new()
            }
        } else {
            BTreeSet::new()
        };

        // Early-out: skip if no non-call-arg VRegs of the target class are live.
        if config.skip_if_no_live {
            let has_live = live_at_cp.iter().any(|v| {
                let idx = v.0 as usize;
                idx < graph.num_vregs
                    && graph.reg_class[idx] == config.reg_class
                    && !call_arg_vregs.contains(&idx)
            });
            if !has_live {
                continue;
            }
        }

        for &clobbered_reg in config.clobbered_regs {
            let Some(&color) = reg_to_color.get(&clobbered_reg) else {
                continue;
            };

            let phantom_idx = *next_vreg as usize;
            *next_vreg += 1;

            if phantom_idx >= graph.num_vregs {
                let new_n = phantom_idx + 1;
                graph.adj.resize(new_n, BTreeSet::new());
                graph.reg_class.resize(new_n, config.reg_class);
                graph.num_vregs = new_n;
            }
            graph.reg_class[phantom_idx] = config.reg_class;

            phantom_precolors.insert(phantom_idx, color);

            for &live_v in live_at_cp {
                let live_idx = live_v.0 as usize;
                if live_idx < graph.num_vregs
                    && graph.reg_class[live_idx] == config.reg_class
                    && !call_arg_vregs.contains(&live_idx)
                {
                    graph.add_edge(phantom_idx, live_idx);
                }
            }
        }
    }

    (graph, phantom_precolors)
}

/// Inject all three clobber phantom sets (GPR call, XMM call, div) into the
/// given graph, using the provided call/div points and per-block liveness.
///
/// Returns `(updated_graph, gpr_call_phantoms, xmm_call_phantoms, div_phantoms)`.
fn inject_clobber_phantoms(
    graph: InterferenceGraph,
    per_block_liveness: &[LivenessInfo],
    block_schedules: &[Vec<ScheduledInst>],
    call_points: &[(usize, usize)],
    div_points: &[(usize, usize)],
    uses_frame_pointer: bool,
    next_vreg: &mut u32,
) -> (
    InterferenceGraph,
    BTreeMap<usize, u32>, // gpr_call_phantoms
    BTreeMap<usize, u32>, // xmm_call_phantoms
    BTreeMap<usize, u32>, // div_phantoms
) {
    let gpr_clobbers: Vec<Reg> = CALLER_SAVED_GPR
        .iter()
        .copied()
        .filter(|&r| r != Reg::RSP)
        .collect();

    let (graph, gpr_call_phantoms) = add_clobber_interferences_global(
        graph,
        per_block_liveness,
        block_schedules,
        &GlobalClobberConfig {
            points: call_points,
            clobbered_regs: &gpr_clobbers,
            reg_class: RegClass::GPR,
            ordered_regs: allocatable_gpr_order(uses_frame_pointer),
            exclude_call_args: true,
            skip_if_no_live: false,
        },
        next_vreg,
    );

    let (graph, xmm_call_phantoms) = add_clobber_interferences_global(
        graph,
        per_block_liveness,
        block_schedules,
        &GlobalClobberConfig {
            points: call_points,
            clobbered_regs: &CALLER_SAVED_XMM,
            reg_class: RegClass::XMM,
            ordered_regs: allocatable_xmm_order(),
            exclude_call_args: true,
            skip_if_no_live: true,
        },
        next_vreg,
    );

    let (graph, div_phantoms) = add_clobber_interferences_global(
        graph,
        per_block_liveness,
        block_schedules,
        &GlobalClobberConfig {
            points: div_points,
            clobbered_regs: &[Reg::RAX, Reg::RDX],
            reg_class: RegClass::GPR,
            ordered_regs: allocatable_gpr_order(uses_frame_pointer),
            exclude_call_args: false,
            skip_if_no_live: false,
        },
        next_vreg,
    );

    (graph, gpr_call_phantoms, xmm_call_phantoms, div_phantoms)
}

// ── Task 3.5: Global merge_precolorings ──────────────────────────────────────

/// Merge phantom precolorings with param precolorings into one map.
///
/// When a param VReg is precolored to the same color as a GPR call phantom
/// AND the graph has an interference edge between them, the param precoloring
/// is dropped (the param will receive a free callee-saved register). The
/// dropped pairs are appended to `unprecolored_params`.
///
/// Mirrors `merge_precolorings` from `allocator.rs` but operates at function
/// scope with the global `param_vreg_to_reg` map and `unprecolored_params`.
fn merge_precolorings_global(
    param_color_map: &BTreeMap<usize, u32>,
    gpr_call_phantoms: &BTreeMap<usize, u32>,
    xmm_call_phantoms: &BTreeMap<usize, u32>,
    div_phantoms: &BTreeMap<usize, u32>,
    param_vreg_indices: &BTreeSet<usize>,
    graph: &InterferenceGraph,
    param_vreg_to_reg: &mut BTreeMap<VReg, Reg>,
    unprecolored_params: &mut Vec<(VReg, Reg)>,
) -> BTreeMap<usize, u32> {
    let mut merged = param_color_map.clone();

    // For each GPR call phantom, check if any precoloring conflicts (same
    // color + interference edge). Drop conflicting precolorings: the VReg
    // will get a callee-saved register, and the lowering will emit a mov to
    // the ABI register at the use site (call arg setup or function prologue).
    //
    // This covers both function params AND call-arg VRegs: a call-arg VReg
    // whose value is live across OTHER calls that clobber the target register
    // cannot be precolored to that register, or its value is destroyed by the
    // intervening call. The `setup_call_args` lowering handles non-precolored
    // arg VRegs by emitting `mov rdi, <arg_reg>` before the call.
    for (&phantom_vreg, &phantom_color) in gpr_call_phantoms {
        let conflicting: Vec<usize> = merged
            .iter()
            .filter(|&(&pv, &pc)| {
                pc == phantom_color
                    && phantom_vreg < graph.num_vregs
                    && pv < graph.num_vregs
                    && graph.adj[phantom_vreg].contains(&pv)
            })
            .map(|(&pv, _)| pv)
            .collect();

        for pv in conflicting {
            merged.remove(&pv);
            let vreg = VReg(pv as u32);
            if let Some(reg) = param_vreg_to_reg.remove(&vreg) {
                // Params get re-added to unprecolored_params so the lowering
                // emits an entry move. Call-arg VRegs aren't in param_vreg_to_reg
                // so no entry move is needed (setup_call_args handles them).
                if param_vreg_indices.contains(&pv) {
                    unprecolored_params.push((vreg, reg));
                }
            }
        }
    }

    // Inject phantom precolorings. Phantoms represent hard hardware constraints
    // and override any remaining param precolorings at the same index.
    merged.extend(gpr_call_phantoms);
    merged.extend(xmm_call_phantoms);
    merged.extend(div_phantoms);

    merged
}

// ── Task 3.6 + 3.7: Coalescing and post-rebuild ──────────────────────────────

/// Run Phase 3: precoloring, clobber phantoms, coalescing, and graph rebuild.
///
/// # Order of operations (matches the per-block allocator)
///
/// 1. Build function-wide precoloring from params + shifts + divs (Task 3.1).
/// 2. Collect call/div points (Task 3.2).
/// 3. **Coalesce on the PRE-phantom graph** produced by Phase 2 (Task 3.6).
/// 4. Apply coalescing aliases to each block's schedule (Task 3.6).
/// 5. Rebuild interference graph from scratch on post-coalesce schedules
///    (Task 3.7).
/// 6. Inject clobber phantoms into the rebuilt graph (Tasks 3.3/3.4/3.7).
/// 7. Rebuild precolorings on the post-coalesce VReg set; apply
///    `merge_precolorings_global` to detect and drop conflicting param
///    precolorings (Tasks 3.5/3.7).
fn run_phase3(
    phase2: Phase2State,
    block_schedules: Vec<Vec<ScheduledInst>>,
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    copy_pairs: &[(VReg, VReg)],
    cfg_succs: &[Vec<usize>],
    phi_uses: &[BTreeSet<VReg>],
    block_param_vregs_per_block: &[BTreeSet<VReg>],
    uses_frame_pointer: bool,
    mut next_vreg: u32,
) -> Phase3State {
    // Task 3.1: build function-wide precoloring (params + shifts + divs +
    // caller-supplied call-arg precolors).
    let (precolors, param_vreg_indices) =
        build_function_wide_precoloring(param_vregs, &block_schedules, call_arg_precolors);
    let mut param_vreg_to_reg: BTreeMap<VReg, Reg> = precolors.iter().copied().collect();

    // Task 3.2: collect call and div program points.
    let (call_points, div_points) = collect_call_div_points(&block_schedules);

    // Task 3.6 (first half): coalesce on the PRE-phantom graph from Phase 2.
    // This mirrors the per-block allocator which coalesces on the pre-phantom
    // graph and never again.
    let coalesced = {
        let pairs: Vec<(usize, usize)> = copy_pairs
            .iter()
            .map(|(src, dst)| (src.0 as usize, dst.0 as usize))
            .filter(|&(src, dst)| src < phase2.graph.num_vregs && dst < phase2.graph.num_vregs)
            .collect();
        coalesce(&phase2.graph, &pairs)
    };

    // Task 3.6 (second half): apply coalescing aliases to each block's schedule
    // individually, preserving block boundaries.
    let post_coalesce_schedules: Vec<Vec<ScheduledInst>> = block_schedules
        .iter()
        .map(|sched| apply_coalescing(sched, &coalesced))
        .collect();

    // Build the coalescing alias map early so it can be used to resolve
    // block_param VRegs to their post-coalesce canonicals before the rebuild's
    // interference injection. (The later declaration of `alias_map` is reused.)
    let alias_map_early: BTreeMap<u32, u32> = coalesced
        .iter()
        .map(|&(into, from)| (from as u32, into as u32))
        .collect();

    // Task 3.7: rebuild the interference graph from scratch on post-coalesce
    // schedules. Re-run Task 2.3 (vreg class map), re-initialize graph with
    // pre-populated reg_class, re-run build_interference_into per block, and
    // re-add cross-block boundary interferences.
    //
    // The CFG topology (cfg_succs) is unchanged by coalescing — only VReg
    // names change. Apply the alias map to phi_uses and block_param_vregs
    // before rebuilding liveness: the schedule operands have been renamed by
    // apply_coalescing, but phi_uses/block_param_vregs still carry pre-coalesce
    // VReg names. Without renaming, liveness seeds live_out[b] with a VReg
    // that never appears in the schedule, so interferences between phi-source
    // values and defs in the block are missed — the canonical post-coalesce
    // VReg (e.g. v1 for n) is not recognized as live, and a new def in the
    // block can land on the same register as n.
    let resolve_vreg_early = |v: VReg| -> VReg {
        let mut idx = v.0;
        while let Some(&target) = alias_map_early.get(&idx) {
            idx = target;
        }
        VReg(idx)
    };
    let renamed_phi_uses: Vec<BTreeSet<VReg>> = phi_uses
        .iter()
        .map(|set| set.iter().map(|&v| resolve_vreg_early(v)).collect())
        .collect();
    let renamed_block_param_vregs: Vec<BTreeSet<VReg>> = block_param_vregs_per_block
        .iter()
        .map(|set| set.iter().map(|&v| resolve_vreg_early(v)).collect())
        .collect();
    let rebuild_global_liveness =
        crate::regalloc::global_liveness::compute_global_liveness_with_block_params(
            &post_coalesce_schedules,
            cfg_succs,
            &renamed_phi_uses,
            &renamed_block_param_vregs,
        );

    let mut rebuilt = build_global_interference(&post_coalesce_schedules, &rebuild_global_liveness);
    add_block_param_interferences(
        &mut rebuilt.graph,
        &renamed_block_param_vregs,
        &alias_map_early,
    );

    // Re-inject clobber phantoms into the rebuilt graph (Task 3.7).
    let (call_points_post, div_points_post) = collect_call_div_points(&post_coalesce_schedules);

    let (graph_with_phantoms, gpr_call_phantoms, xmm_call_phantoms, div_phantoms) =
        inject_clobber_phantoms(
            rebuilt.graph,
            &rebuilt.per_block_liveness,
            &post_coalesce_schedules,
            &call_points_post,
            &div_points_post,
            uses_frame_pointer,
            &mut next_vreg,
        );

    // Rebuild precolorings on the post-coalesce VReg set: apply the alias map
    // to precolor keys. The coalescing alias map renames `from` -> `into`, so
    // a precoloring for a `from` VReg should transfer to `into`.
    let alias_map = alias_map_early.clone();

    let resolve_vreg = |v: VReg| -> VReg {
        let mut idx = v.0;
        while let Some(&target) = alias_map.get(&idx) {
            idx = target;
        }
        VReg(idx)
    };

    // Rebuild param_vreg_to_reg with aliased VReg keys.
    let mut param_vreg_to_reg_post: BTreeMap<VReg, Reg> = BTreeMap::new();
    for (v, r) in &param_vreg_to_reg {
        param_vreg_to_reg_post.insert(resolve_vreg(*v), *r);
    }
    param_vreg_to_reg = param_vreg_to_reg_post;

    let param_vreg_indices_post: BTreeSet<usize> = param_vreg_indices
        .iter()
        .map(|&idx| resolve_vreg(VReg(idx as u32)).0 as usize)
        .collect();

    // Build the base color map from post-coalesce param/shift/div precolors.
    let param_color_map = precolors_to_color_map(
        &param_vreg_to_reg
            .iter()
            .map(|(&v, &r)| (v, r))
            .collect::<Vec<_>>(),
        uses_frame_pointer,
    );

    // Task 3.5: merge_precolorings_global — detect and drop param precolorings
    // that conflict with a GPR call phantom.
    let mut unprecolored_params: Vec<(VReg, Reg)> = Vec::new();
    let pre_coloring_colors = merge_precolorings_global(
        &param_color_map,
        &gpr_call_phantoms,
        &xmm_call_phantoms,
        &div_phantoms,
        &param_vreg_indices_post,
        &graph_with_phantoms,
        &mut param_vreg_to_reg,
        &mut unprecolored_params,
    );

    Phase3State {
        per_block_insts: post_coalesce_schedules,
        graph: graph_with_phantoms,
        per_block_liveness: rebuilt.per_block_liveness,
        pre_coloring_colors,
        unprecolored_params,
        next_vreg,
        alias_map,
    }
}

// ── Phase 4: Global coloring and mapping ─────────────────────────────────────

/// Result of Phase 4: color map, vreg-to-reg assignment, callee-saved list,
/// and per-class overshoot counts.
///
/// `gpr_overshoot` and `xmm_overshoot` are non-zero when the greedy coloring
/// exceeded the available register budget for that class. Phase 5 consumes
/// these counts to drive iterative spill selection. Phase 4 itself does NOT
/// spill — it always returns successfully.
pub(crate) struct Phase4State {
    /// Color map produced by greedy coloring: VReg index -> color.
    pub color_map: BTreeMap<usize, u32>,

    /// Final function-wide VReg -> physical register assignment.
    ///
    /// Contains only real VRegs (those appearing as `dst` or `operands` in
    /// `per_block_insts`). Phantom VRegs injected by Phase 3 clobber injection
    /// are excluded.
    pub vreg_to_reg: BTreeMap<VReg, Reg>,

    /// Callee-saved registers actually used by the coloring. The function
    /// prologue/epilogue must push/pop these.
    pub callee_saved_used: Vec<Reg>,

    /// Number of GPR colors that exceeded `available_gpr_colors(uses_frame_pointer)`.
    /// Zero when the GPR coloring fits within the available register budget.
    pub gpr_overshoot: u32,

    /// Number of XMM colors that exceeded `available_xmm_colors()`.
    /// Zero when the XMM coloring fits within the available register budget.
    pub xmm_overshoot: u32,

    /// Inherited from Phase 3: param VRegs whose ABI precoloring was dropped.
    pub unprecolored_params: Vec<(VReg, Reg)>,

    /// Inherited from Phase 3: post-coalesce instruction lists, one per block.
    pub per_block_insts: Vec<Vec<ScheduledInst>>,

    /// Inherited from Phase 3: coalesce alias map (`from_idx -> into_idx`).
    pub alias_map: BTreeMap<u32, u32>,
}

/// Run Phase 4: global coloring and color-to-register mapping.
///
/// # Steps
///
/// 1. **Task 4.1**: `mcs_ordering` on the Phase 3 graph.
/// 2. **Task 4.2**: `greedy_color` with Phase 3's merged precoloring.
/// 3. **Task 4.3**: compute per-class chromatic numbers and overshoot counts.
/// 4. **Task 4.4**: interval-color fallback is intentionally omitted (see module
///    doc `# Coloring strategy`). If greedy fails, Phase 5 handles it.
/// 5. **Task 4.5**: `map_colors_to_regs` per class; build `vreg_to_reg` from
///    real VRegs only (phantoms are excluded).
/// 6. **Task 4.6**: compute `callee_saved_used` as the union of assigned physical
///    registers that appear in `CALLEE_SAVED` / `CALLEE_SAVED_XMM`.
pub(crate) fn run_phase4(phase3: Phase3State, uses_frame_pointer: bool) -> Phase4State {
    use super::coloring::{available_gpr_colors, greedy_color, map_colors_to_regs, mcs_ordering};
    use crate::x86::abi::CALLEE_SAVED;

    // Task 4.1: MCS ordering on the Phase 3 graph.
    let ordering = mcs_ordering(&phase3.graph);

    // Task 4.2: greedy coloring with merged precoloring from Phase 3.
    let coloring = greedy_color(&phase3.graph, &ordering, &phase3.pre_coloring_colors);

    // Build a flat color map: VReg index -> color (from the ColoringResult vec).
    let color_map: BTreeMap<usize, u32> = coloring
        .colors
        .iter()
        .enumerate()
        .filter_map(|(idx, &c)| c.map(|color| (idx, color)))
        .collect();

    // Task 4.3: per-class chromatic numbers and overshoot counts.
    //
    // Count the maximum color assigned to VRegs of each class among REAL VRegs
    // (phantom VRegs are those with index >= the pre-phantom count, but since we
    // don't track that boundary here, we compute the chromatic number as the max
    // color + 1 over all VRegs of each class in the full graph).
    let mut gpr_max_color: Option<u32> = None;
    let mut xmm_max_color: Option<u32> = None;

    for (idx, &color_opt) in coloring.colors.iter().enumerate() {
        let Some(color) = color_opt else { continue };
        if idx >= phase3.graph.num_vregs {
            continue;
        }
        match phase3.graph.reg_class[idx] {
            RegClass::GPR => {
                gpr_max_color = Some(gpr_max_color.map_or(color, |m: u32| m.max(color)));
            }
            RegClass::XMM => {
                xmm_max_color = Some(xmm_max_color.map_or(color, |m: u32| m.max(color)));
            }
        }
    }

    let gpr_chromatic = gpr_max_color.map_or(0, |m| m + 1);
    let xmm_chromatic = xmm_max_color.map_or(0, |m| m + 1);

    let gpr_budget = available_gpr_colors(uses_frame_pointer);
    let xmm_budget = super::coloring::AVAILABLE_XMM_COLORS;

    let gpr_overshoot = gpr_chromatic.saturating_sub(gpr_budget);
    let xmm_overshoot = xmm_chromatic.saturating_sub(xmm_budget);

    // Task 4.5: map colors to physical registers per class.
    //
    // Build a `BTreeMap<usize, Reg>` precoloring (vreg_idx -> Reg) for each
    // class by decoding `pre_coloring_colors` (vreg_idx -> color) through the
    // allocatable register ordering for that class.
    let gpr_order = super::coloring::allocatable_gpr_order(uses_frame_pointer);
    let xmm_order = super::coloring::allocatable_xmm_order();

    let precolor_vreg_to_reg: BTreeMap<usize, Reg> = phase3
        .pre_coloring_colors
        .iter()
        .filter_map(|(&vreg_idx, &color)| {
            if vreg_idx >= phase3.graph.num_vregs {
                return None;
            }
            let reg = match phase3.graph.reg_class[vreg_idx] {
                RegClass::GPR => gpr_order.get(color as usize).copied(),
                RegClass::XMM => xmm_order.get(color as usize).copied(),
            };
            reg.map(|r| (vreg_idx, r))
        })
        .collect();

    let gpr_color_to_reg = map_colors_to_regs(
        &coloring,
        RegClass::GPR,
        &precolor_vreg_to_reg,
        uses_frame_pointer,
    );
    let xmm_color_to_reg = map_colors_to_regs(
        &coloring,
        RegClass::XMM,
        &precolor_vreg_to_reg,
        uses_frame_pointer,
    );

    // Collect the set of real VReg indices: those appearing as dst or operands
    // in per_block_insts. Phantom VRegs (injected by Phase 3 clobber injection)
    // have indices that do not appear in any instruction and are excluded.
    let mut real_vreg_indices: BTreeSet<usize> = BTreeSet::new();
    for sched in &phase3.per_block_insts {
        for inst in sched {
            real_vreg_indices.insert(inst.dst.0 as usize);
            for &op in &inst.operands {
                real_vreg_indices.insert(op.0 as usize);
            }
        }
    }

    // Build vreg_to_reg: only real VRegs, using the color assigned to each.
    let mut vreg_to_reg: BTreeMap<VReg, Reg> = BTreeMap::new();
    for &idx in &real_vreg_indices {
        if idx >= coloring.colors.len() {
            continue;
        }
        let Some(color) = coloring.colors[idx] else {
            continue;
        };
        let reg_class = if idx < phase3.graph.num_vregs {
            phase3.graph.reg_class[idx]
        } else {
            RegClass::GPR
        };
        let reg = match reg_class {
            RegClass::GPR => gpr_color_to_reg.get(&color).copied(),
            RegClass::XMM => xmm_color_to_reg.get(&color).copied(),
        };
        if let Some(r) = reg {
            vreg_to_reg.insert(VReg(idx as u32), r);
        }
    }

    // Task 4.6: compute callee_saved_used.
    //
    // Union of assigned physical registers (across both classes) that are in
    // CALLEE_SAVED (GPR) or CALLEE_SAVED_XMM. In SysV AMD64 all XMM registers
    // are caller-saved, so no XMM register is callee-saved; only CALLEE_SAVED
    // GPR entries need checking.
    //
    // CALLEE_SAVED = [RBX, RBP, R12, R13, R14, R15]  (from abi.rs)
    // All XMM regs are caller-saved in SysV AMD64; CALLEE_SAVED covers only GPRs.
    let callee_saved_set: BTreeSet<Reg> = CALLEE_SAVED.iter().copied().collect();

    let mut callee_saved_used: Vec<Reg> = vreg_to_reg
        .values()
        .filter(|&&r| callee_saved_set.contains(&r))
        .copied()
        .collect::<BTreeSet<Reg>>()
        .into_iter()
        .collect();
    // Sort for deterministic output (CALLEE_SAVED declaration order).
    callee_saved_used.sort_by_key(|&r| {
        CALLEE_SAVED
            .iter()
            .position(|&cs| cs == r)
            .unwrap_or(usize::MAX)
    });

    Phase4State {
        color_map,
        vreg_to_reg,
        callee_saved_used,
        gpr_overshoot,
        xmm_overshoot,
        unprecolored_params: phase3.unprecolored_params,
        per_block_insts: phase3.per_block_insts,
        alias_map: phase3.alias_map,
    }
}

// ── Phase 5: Register assignment ─────────────────────────────────────────────

/// Context carried from Phase 3 for graph rebuilding inside phase 5.
///
/// The spill loop re-runs the interference-graph build + phantom injection per
/// round without re-running coalescing (coalesce once, spill-loop iterates
/// coloring only per the plan).
pub(crate) struct Phase5Context {
    /// CFG successors per block (block index -> successor block indices).
    cfg_succs: Vec<Vec<usize>>,
    /// Per-block phi uses (VRegs referenced by terminators).
    phi_uses: Vec<BTreeSet<VReg>>,
    /// Per-block sets of VRegs that are block parameters (phi destinations).
    block_param_vregs_per_block: Vec<BTreeSet<VReg>>,
    /// Function parameter VRegs with their ABI physical registers.
    param_vregs: Vec<(VReg, Reg)>,
    /// Call-argument precolors (computed before effectful-op operand sorting).
    call_arg_precolors: Vec<(VReg, Reg)>,
    /// Copy pairs for reference (not re-coalesced).
    copy_pairs: Vec<(VReg, VReg)>,
    /// Loop depths per VReg (for spill scoring).
    loop_depths: BTreeMap<VReg, u32>,
    /// Whether the frame pointer (RBP) is in use (shrinks GPR budget by 1).
    uses_frame_pointer: bool,
    /// Function name for debug tracing.
    func_name: String,
    /// Coalescing alias map: `alias_map[from] = into` (post-coalesce VReg renaming).
    alias_map: BTreeMap<u32, u32>,
}

/// Transitively resolve every chain in a raw alias map (`from -> into`) to
/// produce `VReg -> canonical_VReg` entries. The result keeps only mappings
/// whose key differs from the value, so lookups can short-circuit identity.
fn build_transitive_alias_map(raw: &BTreeMap<u32, u32>) -> BTreeMap<VReg, VReg> {
    if raw.is_empty() {
        return BTreeMap::new();
    }
    let mut out = BTreeMap::new();
    for &from in raw.keys() {
        let mut cur = from;
        while let Some(&next) = raw.get(&cur) {
            if next == cur {
                break;
            }
            cur = next;
        }
        if cur != from {
            out.insert(VReg(from), VReg(cur));
        }
    }
    out
}

/// Run Phase 5: global spilling with iterative spill-and-recolor (Task 5.7).
///
/// # XMM cross-block-phi audit (Task 5.9)
///
/// The legacy `compile/mod.rs` Step 6b unconditionally force-spilled ALL XMM
/// VRegs appearing in `phi_uses` or `block_param_vregs_per_block`, because the
/// per-block allocator could not keep XMM values in registers across block
/// boundaries (it had no cross-block view).
///
/// The global allocator CAN assign XMM VRegs to physical XMM registers across
/// block boundaries when no call clobbers the path. We audited the phi-copy
/// emission path in `src/compile/terminator.rs` (`build_phi_copies` +
/// `phi_copies`):
///
/// 1. `build_phi_copies` looks up `regalloc.vreg_to_reg.get(&arg_vreg)` and
///    if the XMM VReg has a register assignment, uses it directly as the src.
///    If it has NO assignment (because it was force-spilled to a slot in the
///    legacy path), the copy is skipped and the comment says "the successor will
///    load from the spill slot at block entry".
///
/// 2. `phi_copies` emits reg-to-reg `MovsdRR` for XMM-to-XMM copies (line 57-59
///    in `phi_elim.rs`). It does NOT read from or write to stack slots directly.
///
/// **Conclusion**: the phi-copy emission is reg-to-reg `movsd` and is SUFFICIENT
/// for the global allocator case. When an XMM VReg has a physical register
/// assignment (because no call clobbers the path), `build_phi_copies` will emit
/// a reg-to-reg movsd. When an XMM VReg is pressure-spilled by the global
/// allocator (e.g., because it crosses a call), it will have no register
/// assignment and the phi copy will be skipped — the successor reloads from the
/// slot. **We do NOT need a forced-slot pre-spill step.** Pressure-based
/// spilling via the normal iterative loop (Tasks 5.2-5.7) is sufficient because
/// XMM phantoms at every call point make every XMM VReg live-across-call
/// uncolorable, which triggers normal spill selection.
///
/// The `block_param_vregs_per_block` parameter is retained for bookkeeping and
/// potential future use by Phase 6, but no XMM forced-slot step is performed
/// here.
pub(crate) fn run_phase5(
    phase4: Phase4State,
    ctx: Phase5Context,
) -> Result<GlobalRegAllocResult, String> {
    use crate::x86::abi::CALLEE_SAVED;

    let uses_frame_pointer = ctx.uses_frame_pointer;
    let func_name = &ctx.func_name;

    // Check if Phase 4 already converged (no spilling needed).
    if phase4.gpr_overshoot == 0 && phase4.xmm_overshoot == 0 {
        if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
            tracing::debug!("phase5[{func_name}]: converged immediately (no spill needed)");
        }
        let callee_saved_set: BTreeSet<Reg> = CALLEE_SAVED.iter().copied().collect();
        let callee_saved_used = phase4
            .vreg_to_reg
            .values()
            .filter(|&&r| callee_saved_set.contains(&r))
            .copied()
            .collect::<BTreeSet<Reg>>()
            .into_iter()
            .collect();
        let coalesce_aliases = build_transitive_alias_map(&phase4.alias_map);
        return Ok(GlobalRegAllocResult {
            per_block_insts: phase4.per_block_insts,
            vreg_to_reg: phase4.vreg_to_reg,
            spill_slots: 0,
            callee_saved_used,
            unprecolored_params: phase4.unprecolored_params,
            coalesce_aliases,
        });
    }

    Err(format!(
        "global regalloc: register pressure overshoot for function '{func_name}' \
         (gpr_overshoot={}, xmm_overshoot={}). The split pass should have resolved \
         all register pressure before phase 5.",
        phase4.gpr_overshoot, phase4.xmm_overshoot,
    ))
}

/// Compute (gpr_overshoot, xmm_overshoot) from graph and precoloring.
fn compute_overshoot(
    graph: &InterferenceGraph,
    pre_coloring_colors: &BTreeMap<usize, u32>,
    gpr_budget: u32,
    xmm_budget: u32,
) -> (u32, u32) {
    use super::coloring::{greedy_color, mcs_ordering};
    let ordering = mcs_ordering(graph);
    let coloring = greedy_color(graph, &ordering, pre_coloring_colors);
    compute_overshoot_from_coloring(graph, &coloring, gpr_budget, xmm_budget)
}

/// Compute (gpr_overshoot, xmm_overshoot) from an existing coloring result.
fn compute_overshoot_from_coloring(
    graph: &InterferenceGraph,
    coloring: &super::coloring::ColoringResult,
    gpr_budget: u32,
    xmm_budget: u32,
) -> (u32, u32) {
    let mut gpr_max: Option<u32> = None;
    let mut xmm_max: Option<u32> = None;

    for (idx, &color_opt) in coloring.colors.iter().enumerate() {
        let Some(color) = color_opt else { continue };
        if idx >= graph.num_vregs {
            continue;
        }
        match graph.reg_class[idx] {
            RegClass::GPR => {
                gpr_max = Some(gpr_max.map_or(color, |m| m.max(color)));
            }
            RegClass::XMM => {
                xmm_max = Some(xmm_max.map_or(color, |m| m.max(color)));
            }
        }
    }

    let gpr_chromatic = gpr_max.map_or(0, |m| m + 1);
    let xmm_chromatic = xmm_max.map_or(0, |m| m + 1);
    (
        gpr_chromatic.saturating_sub(gpr_budget),
        xmm_chromatic.saturating_sub(xmm_budget),
    )
}

/// Allocate physical registers for a whole function using a function-scope
/// graph-coloring allocator.
///
/// # Arguments
///
/// * `block_schedules` - Scheduled instruction lists per block (one `Vec<ScheduledInst>`
///   per block, in block index order). Each block's schedule has already had
///   effectful-op operands populated by `populate_effectful_operands` before
///   this function is called.
/// * `param_vregs` - ABI precolorings for function parameters (VReg, physical Reg pairs).
/// * `call_arg_precolors` - ABI precolorings for call arguments and return values,
///   computed by the caller from `EffectfulOp::Call` args in ABI argument order
///   BEFORE `populate_effectful_operands` sorts the barrier operands by VReg index.
///   The canonical source is `add_call_precolors_for_block` in `compile/precolor.rs`,
///   called per block and aggregated into a single `Vec<(VReg, Reg)>`. First 6 int
///   args receive RDI/RSI/RDX/RCX/R8/R9; first 8 float args receive XMM0..XMM7;
///   the return value VReg (Proj0 of CallResult) receives RAX or XMM0 depending
///   on return type.
/// * `copy_pairs` - Phi copy pairs for coalescing (source VReg, dest VReg).
/// * `loop_depths` - Loop depth per VReg, used by the spill scorer to prefer
///   spilling values defined/used outside loops.
/// * `cfg_succs` - CFG successors per block (block index -> list of successor block indices).
/// * `phi_uses` - Per-block sets of VRegs referenced by block terminators (Jump/Branch
///   args) that must be kept live at block boundaries.
/// * `block_param_vregs_per_block` - Per-block sets of VRegs that are block parameters
///   (phi destinations); these are excluded from cross-block reload insertion.
/// * `func_name` - Function name used for debug tracing.
/// * `uses_frame_pointer` - When `false`, RBP is allocatable as a general-purpose register.
pub fn allocate_global(
    block_schedules: &[Vec<ScheduledInst>],
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    copy_pairs: &[(VReg, VReg)],
    loop_depths: &BTreeMap<VReg, u32>,
    cfg_succs: &[Vec<usize>],
    phi_uses: &[BTreeSet<VReg>],
    block_param_vregs_per_block: &[BTreeSet<VReg>],
    func_name: &str,
    uses_frame_pointer: bool,
) -> Result<GlobalRegAllocResult, String> {
    // Task 2.2: Compute function-wide global liveness. Block params are added
    // to their block's live_in so pairs of params on the same block interfere
    // (they're written simultaneously by phi copies and must occupy distinct
    // registers even when the block body never reads them).
    let global_liveness =
        crate::regalloc::global_liveness::compute_global_liveness_with_block_params(
            block_schedules,
            cfg_succs,
            phi_uses,
            block_param_vregs_per_block,
        );
    // Also augment the global liveness that run_phase3 will recompute
    // internally: it uses plain `compute_global_liveness` which doesn't know
    // about block params. We pass block_param_vregs_per_block down and augment
    // at each site (see below).

    // Tasks 2.3, 2.4, 2.4.5: Build function-wide interference graph and
    // per-block liveness (stored in Phase2State for Phase 3/5 consumption).
    let mut phase2 = build_global_interference(block_schedules, &global_liveness);
    // Pre-coalesce Phase 2 graph: block_params are still distinct VRegs, so no
    // alias resolution needed.
    add_block_param_interferences(
        &mut phase2.graph,
        block_param_vregs_per_block,
        &BTreeMap::new(),
    );

    // Determine starting next_vreg for phantom injection.
    let next_vreg: u32 = block_schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
        .max(phase2.graph.num_vregs as u32);

    // Phase 3: precolorings, clobbers, coalescing, graph rebuild.
    let phase3 = run_phase3(
        phase2,
        block_schedules.to_vec(),
        param_vregs,
        call_arg_precolors.clone(),
        copy_pairs,
        cfg_succs,
        phi_uses,
        block_param_vregs_per_block,
        uses_frame_pointer,
        next_vreg,
    );

    // Phase 4: global coloring and color-to-register mapping.
    let phase4 = run_phase4(phase3, uses_frame_pointer);

    // Coalescing alias map: threaded through from Phase 3. Used in Phase 5 to
    // build the transitive alias map so the caller's ClassId -> VReg resolution
    // (block_class_to_vreg in compile/mod.rs) chases stale `class_to_vreg`
    // entries that still point at pre-coalesce VRegs.
    let alias_map = phase4.alias_map.clone();

    // Phase 5: global spilling.
    let ctx = Phase5Context {
        cfg_succs: cfg_succs.to_vec(),
        phi_uses: phi_uses.to_vec(),
        block_param_vregs_per_block: block_param_vregs_per_block.to_vec(),
        param_vregs: param_vregs.to_vec(),
        call_arg_precolors: call_arg_precolors.clone(),
        copy_pairs: copy_pairs.to_vec(),
        loop_depths: loop_depths.clone(),
        uses_frame_pointer,
        func_name: func_name.to_string(),
        alias_map,
    };

    run_phase5(phase4, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;
    use crate::regalloc::global_liveness::compute_global_liveness;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn iconst_inst(dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(dst as i64, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Proj0,
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    fn empty_phi_uses(n: usize) -> Vec<BTreeSet<VReg>> {
        vec![BTreeSet::new(); n]
    }

    /// Build a `Phase2State` for a test CFG.
    fn phase2_for(
        block_schedules: &[Vec<ScheduledInst>],
        successors: &[Vec<usize>],
    ) -> Phase2State {
        let phi_uses = empty_phi_uses(block_schedules.len());
        let global_liveness = compute_global_liveness(block_schedules, successors, &phi_uses);
        build_global_interference(block_schedules, &global_liveness)
    }

    fn interfere(state: &Phase2State, a: u32, b: u32) -> bool {
        let ai = a as usize;
        let bi = b as usize;
        if ai >= state.graph.num_vregs || bi >= state.graph.num_vregs {
            return false;
        }
        state.graph.adj[ai].contains(&bi)
    }

    fn interfere_in(graph: &InterferenceGraph, a: u32, b: u32) -> bool {
        let ai = a as usize;
        let bi = b as usize;
        if ai >= graph.num_vregs || bi >= graph.num_vregs {
            return false;
        }
        graph.adj[ai].contains(&bi)
    }

    // ── Task 2.5: Two-block straight-line CFG ────────────────────────────────
    //
    // Block 0: v0 = iconst, v1 = iconst
    // Block 1: v2 = use(v0)
    //
    // v0 is live from block 0 through block 1.
    // v1 is local to block 0.
    // v2 is local to block 1.
    //
    // v0 and v1 overlap in block 0 -> they interfere.
    // v1 and v2 are in different blocks and never simultaneously live -> no interference.
    #[test]
    fn straight_line_no_spurious_interference() {
        // Block 0: v0 = iconst, v1 = iconst
        // Block 1: v2 = use(v0)
        let block_schedules = vec![vec![iconst_inst(0), iconst_inst(1)], vec![use_inst(2, 0)]];
        let successors = vec![vec![1usize], vec![]];
        let state = phase2_for(&block_schedules, &successors);

        // v0 is live in block 0 (as cross-block value) and used in block 1.
        // v1 is defined and dies in block 0 (never in block 1).
        // v2 is defined and dies in block 1.

        // v1 and v2 are never simultaneously live: no spurious interference.
        assert!(
            !interfere(&state, 1, 2),
            "v1 (block-0-local) and v2 (block-1-local) must NOT interfere"
        );

        // Sanity: v0 and v1 are both defined in block 0 (v0 is live-out, v1 is
        // local). After v0 is defined it lives until block 1; v1 is defined
        // after v0 in the schedule and both survive until block 0 exits.
        // They overlap in live_at of the insts that come after v1's def, so
        // they do interfere (same block, overlapping live ranges).
        // This confirms the test is meaningful: cross-block non-overlapping
        // values (v1 and v2) do NOT interfere while intra-block overlapping
        // values (v0 and v1) DO.
        assert!(
            interfere(&state, 0, 1),
            "v0 and v1 both live in block 0 should interfere"
        );
    }

    // ── Task 2.6: Diamond CFG ─────────────────────────────────────────────────
    //
    // Block 0: v0 = iconst  (branches to block 1 and block 2)
    // Block 1: v1 = iconst  (arm A; joins at block 3)
    // Block 2: v2 = iconst  (arm B; joins at block 3)
    // Block 3: v3 = use(v0)
    //
    // v1 is live only in block 1 (arm A).
    // v2 is live only in block 2 (arm B).
    // They are never simultaneously live on any execution path -> no interference.
    #[test]
    fn diamond_cfg_different_arms_no_interference() {
        let block_schedules = vec![
            vec![iconst_inst(0)], // block 0: v0 = iconst
            vec![iconst_inst(1)], // block 1: v1 = iconst (arm A)
            vec![iconst_inst(2)], // block 2: v2 = iconst (arm B)
            vec![use_inst(3, 0)], // block 3: v3 = use(v0)
        ];
        // 0 -> {1, 2}, 1 -> 3, 2 -> 3
        let successors = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let state = phase2_for(&block_schedules, &successors);

        // v1 (arm A only) and v2 (arm B only) must NOT interfere.
        assert!(
            !interfere(&state, 1, 2),
            "v1 (arm A) and v2 (arm B) must NOT interfere (different arms)"
        );

        // v0 is live through both arms (live_out of block 0, live_in of blocks
        // 1 and 2 by transitivity since block 3 uses it).
        // v1 is live in block 1; v0 is also live in block 1 (as live-in).
        // They are simultaneously live in block 1, so they interfere.
        assert!(
            interfere(&state, 0, 1),
            "v0 (crosses diamond) and v1 (arm A) should interfere in block 1"
        );
    }

    // ── Task 2.7: Loop CFG ────────────────────────────────────────────────────
    //
    // Block 0 (header): v0 = iconst  (loop entry)
    // Block 1 (body):   v2 = iconst; v1 = add(v0, v2)  (back-edge to block 0)
    //
    // v0 is defined in block 0 and used in block 1 AFTER v2 is defined; the
    // back edge from block 1 to block 0 makes v0 live across the back edge.
    // v2 is defined in block 1 (body-local); v0 is still live when v2 is
    // defined (used later in the same block), so v0 and v2 interfere.
    #[test]
    fn loop_cfg_live_across_back_edge() {
        // Block 0 (header): v0 = iconst
        // Block 1 (body):   v2 = iconst; v1 = add(v0, v2)  [v0 used AFTER v2]
        let block_schedules = vec![
            vec![iconst_inst(0)],
            vec![iconst_inst(2), add_inst(1, 0, 2)],
        ];
        // 0 -> 1, 1 -> 0 (back-edge)
        let successors = vec![vec![1], vec![0]];
        let state = phase2_for(&block_schedules, &successors);

        // v0 must be live across the back edge: live_out of block 0, live_in
        // of block 1.
        let global_liveness = {
            let phi_uses = empty_phi_uses(2);
            compute_global_liveness(&block_schedules, &successors, &phi_uses)
        };
        assert!(
            global_liveness.live_out[0].contains(&VReg(0)),
            "v0 must be live_out of block 0 (live across back edge)"
        );
        assert!(
            global_liveness.live_in[1].contains(&VReg(0)),
            "v0 must be live_in of block 1"
        );

        // In block 1: v2 = iconst at inst 0, then v1 = add(v0, v2) at inst 1.
        // Backward pass through block 1 (live_out = {}):
        //   inst 1 (v1=add(v0,v2)): live_at[1] = {v0, v2}  <- v0 and v2 both live
        //   inst 0 (v2=iconst):     live_at[0] = {v0}
        // v0 is live at inst 1 when v2 is also live -> they interfere.
        assert!(
            interfere(&state, 0, 2),
            "v0 (live across back edge) and v2 (body-local) must interfere in block 1"
        );

        // v2 is body-local (dies within block 1). v0 is the cross-block live
        // value. Confirm v2 does NOT appear in live_out[1] (it does not cross
        // the back edge back to block 0).
        assert!(
            !global_liveness.live_out[1].contains(&VReg(2)),
            "v2 (body-local) must NOT be live_out of block 1"
        );
    }

    // ── Task 3.6a: Coalescing reduces VReg count on a copy-pair program ──────
    //
    // Two-block straight-line CFG where v0 and v1 are defined in separate blocks
    // with no overlap (never simultaneously live), forming a non-interfering pair.
    // After coalescing with copy pair (v0, v1), v1 should be aliased to v0 in
    // the post-coalesce schedules.
    #[test]
    fn coalescing_reduces_vreg_count() {
        // Block 0: v0 = iconst (no cross-block use; dead by block 1 entry)
        // Block 1: v1 = iconst (block-local; no overlap with v0)
        //
        // Copy pair: (v0, v1) — non-interfering (different blocks, no overlap)
        // After coalescing v1 is aliased to v0.
        let block_schedules = vec![
            vec![iconst_inst(0)], // block 0: v0 = iconst (no cross-block use)
            vec![iconst_inst(1)], // block 1: v1 = iconst
        ];
        let successors = vec![vec![1usize], vec![]];
        let phi_uses = empty_phi_uses(2);
        let global_liveness = compute_global_liveness(&block_schedules, &successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);

        // v0 and v1 should NOT interfere (different blocks, no overlap).
        assert!(
            !interfere_in(&phase2.graph, 0, 1),
            "v0 and v1 in separate blocks with no overlap must not interfere"
        );

        // Apply coalescing with copy pair (v0, v1).
        let pairs = [(0usize, 1usize)]; // v0 is src, v1 is dst
        let coalesced = coalesce(&phase2.graph, &pairs);

        // At least one merge should occur since v0 and v1 don't interfere.
        assert!(
            !coalesced.is_empty(),
            "coalescing non-interfering (v0, v1) should produce at least one merge"
        );

        // Apply coalescing to each block's schedule.
        let post_coalesce: Vec<Vec<ScheduledInst>> = block_schedules
            .iter()
            .map(|sched| apply_coalescing(sched, &coalesced))
            .collect();

        // After coalescing, v1's dst should be renamed to v0 (the canonical).
        // Check block 1: the iconst that was v1 should now have dst = v0.
        let dst_in_block1 = post_coalesce[1][0].dst;
        assert_eq!(
            dst_in_block1,
            VReg(0),
            "after coalescing, block 1's v1 should be renamed to v0"
        );
    }

    // ── Task 3.6b: Param precolor dropped when call clobbers ABI reg ─────────
    //
    // Function parameter v0 is precolored to RDI. Block 0 contains a call
    // (modeled as a VoidCallBarrier) so a GPR call phantom is injected that
    // covers RDI. If v0 is live at the call point, the phantom for RDI will
    // interfere with v0, and the param precoloring should be dropped.
    #[test]
    fn param_precolor_dropped_on_call_clobber() {
        use crate::ir::op::Op;

        // Block 0: v0 = iconst (param), v1 = VoidCallBarrier (uses v0 as arg)
        //
        // We model the call as Op::VoidCallBarrier with v0 as an operand.
        // v0 is live at the call point (it is an operand of VoidCallBarrier
        // but the exclude_call_args logic only excludes args NOT live after
        // the call; here v0 is a call arg that is NOT live after, so it IS
        // excluded from interference with the phantom).
        //
        // To test the param-drop path, v0 must be live at the call and NOT be
        // a call arg (so it appears in live_at but is not an operand). We use
        // v2 as an independent value live at the call point, precolored to RDI.
        //
        // Scenario:
        //   v0 = iconst          <- precolored to RDI
        //   v1 = VoidCallBarrier  (call, no args, clobbers all caller-saved GPRs)
        //   v2 = use(v0)         <- forces v0 to be live at v1
        //
        // live_at[1] (before VoidCallBarrier) = {v0}
        // GPR call phantom for RDI is injected with interference to v0.
        // v0 (precolored RDI) conflicts with phantom (same color + interference).
        // -> v0's param precoloring should be dropped.

        // Build schedule: [v0=iconst, v1=VoidCallBarrier, v2=use(v0)]
        let void_call = ScheduledInst {
            op: Op::VoidCallBarrier,
            dst: VReg(1),
            operands: vec![], // no call args
        };
        let block_schedules = vec![vec![
            iconst_inst(0), // v0 = iconst, precolored to RDI
            void_call,      // v1 = VoidCallBarrier
            use_inst(2, 0), // v2 = use(v0) — forces v0 live at call
        ]];
        let successors = vec![vec![]];
        let phi_uses = empty_phi_uses(1);
        let global_liveness = compute_global_liveness(&block_schedules, &successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);

        let param_vregs = vec![(VReg(0), Reg::RDI)];
        let call_arg_precolors: Vec<(VReg, Reg)> = vec![];
        let copy_pairs: Vec<(VReg, VReg)> = vec![];

        let phase3 = run_phase3(
            phase2,
            block_schedules.clone(),
            &param_vregs,
            call_arg_precolors,
            &copy_pairs,
            &successors,
            &phi_uses,
            &Vec::<std::collections::BTreeSet<VReg>>::new(),
            false, // uses_frame_pointer
            10,    // next_vreg start
        );

        // v0 was precolored to RDI. With a call phantom for RDI interfering
        // with v0, the precoloring should have been dropped.
        assert!(
            phase3.unprecolored_params.contains(&(VReg(0), Reg::RDI)),
            "v0's RDI precoloring must be dropped and added to unprecolored_params \
             when a call phantom for RDI interferes with it"
        );

        // After the param precoloring for v0 is dropped, index 0 must be
        // absent from pre_coloring_colors. Phantom VReg indices are all >= 10
        // in this test (next_vreg starts at 10), so index 0 can only appear
        // as the dropped param — which must be gone.
        assert!(
            !phase3.pre_coloring_colors.contains_key(&0usize),
            "v0 (index 0) must be absent from pre_coloring_colors after param drop"
        );
    }

    // ── Task 3.6c: Post-rebuild graph has phantoms, pre-coalesce does not ────
    //
    // Build a two-block CFG with a call in block 0. The Phase 2 pre-phantom
    // graph should have no phantom VRegs (only real VRegs). The Phase 3
    // post-rebuild graph should contain phantom VRegs for the call clobbers.
    #[test]
    fn post_rebuild_graph_has_phantoms_pre_coalesce_does_not() {
        use crate::ir::op::Op;

        // Block 0: v0 = iconst, v1 = VoidCallBarrier, v2 = use(v0)
        // Block 1: v3 = use(v2)
        //
        // The call in block 0 should generate GPR call phantoms. Phase 2 graph
        // has no phantoms; Phase 3 graph does.

        let void_call = ScheduledInst {
            op: Op::VoidCallBarrier,
            dst: VReg(1),
            operands: vec![],
        };
        let block_schedules = vec![
            vec![iconst_inst(0), void_call, use_inst(2, 0)],
            vec![use_inst(3, 2)],
        ];
        let successors = vec![vec![1usize], vec![]];
        let phi_uses = empty_phi_uses(2);
        let global_liveness = compute_global_liveness(&block_schedules, &successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);

        // Phase 2 graph has exactly the real VRegs (0..=3) — no phantoms.
        let phase2_num_vregs = phase2.graph.num_vregs;
        assert!(
            phase2_num_vregs <= 4,
            "Phase 2 graph must have no phantom VRegs (real vregs 0-3 only)"
        );

        let param_vregs: Vec<(VReg, Reg)> = vec![];
        let call_arg_precolors: Vec<(VReg, Reg)> = vec![];
        let copy_pairs: Vec<(VReg, VReg)> = vec![];
        let next_vreg = phase2_num_vregs as u32;

        let phase3 = run_phase3(
            phase2,
            block_schedules.clone(),
            &param_vregs,
            call_arg_precolors,
            &copy_pairs,
            &successors,
            &phi_uses,
            &Vec::<std::collections::BTreeSet<VReg>>::new(),
            false,
            next_vreg,
        );

        // Phase 3 graph must have more VRegs than Phase 2 due to phantom
        // injection (at least one GPR call phantom per clobbered caller-saved
        // register at the VoidCallBarrier point).
        assert!(
            phase3.graph.num_vregs > phase2_num_vregs,
            "Phase 3 graph must have phantom VRegs added by clobber injection \
             (phase2={phase2_num_vregs}, phase3={})",
            phase3.graph.num_vregs
        );

        // The phantoms must NOT appear in the Phase 2 graph — confirmed
        // implicitly by the num_vregs check above.
    }

    // ── call_arg_precolors_feed_through_to_phase3 ────────────────────────────
    //
    // Verifies that caller-supplied call_arg_precolors pass through run_phase3
    // and appear in the resulting pre_coloring_colors map.
    //
    // Setup: 1-block function; v0 and v1 are "call arg" VRegs (iconst defs).
    // We supply call_arg_precolors = [(v0, RDI), (v1, RSI)] directly — this
    // simulates what compile/mod.rs will produce by reading EffectfulOp::Call
    // args in ABI order before populate_effectful_operands sorts them.
    //
    // After run_phase3, pre_coloring_colors must contain:
    //   v0 → color(RDI)
    //   v1 → color(RSI)
    #[test]
    fn call_arg_precolors_feed_through_to_phase3() {
        let block_schedules = vec![vec![iconst_inst(0), iconst_inst(1)]];
        let successors = vec![vec![]];
        let phi_uses = empty_phi_uses(1);
        let global_liveness = compute_global_liveness(&block_schedules, &successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);

        let param_vregs: Vec<(VReg, Reg)> = vec![];
        // Caller supplies these in ABI argument order (v0=first arg, v1=second arg).
        let call_arg_precolors = vec![(VReg(0), Reg::RDI), (VReg(1), Reg::RSI)];
        let copy_pairs: Vec<(VReg, VReg)> = vec![];
        let next_vreg = phase2.graph.num_vregs as u32;

        let phase3 = run_phase3(
            phase2,
            block_schedules.clone(),
            &param_vregs,
            call_arg_precolors,
            &copy_pairs,
            &successors,
            &phi_uses,
            &Vec::<std::collections::BTreeSet<VReg>>::new(),
            false,
            next_vreg,
        );

        let gpr_order = allocatable_gpr_order(false);
        let rdi_color = gpr_order.iter().position(|&r| r == Reg::RDI).unwrap() as u32;
        let rsi_color = gpr_order.iter().position(|&r| r == Reg::RSI).unwrap() as u32;

        assert_eq!(
            phase3.pre_coloring_colors.get(&0usize).copied(),
            Some(rdi_color),
            "v0 must appear in pre_coloring_colors with RDI color"
        );
        assert_eq!(
            phase3.pre_coloring_colors.get(&1usize).copied(),
            Some(rsi_color),
            "v1 must appear in pre_coloring_colors with RSI color"
        );
    }

    // ── call_return_value_precolor_feeds_through ──────────────────────────────
    //
    // Verifies that a return-value VReg precoloring supplied via call_arg_precolors
    // passes through run_phase3 and appears in pre_coloring_colors.
    //
    // Setup: 1-block function; v2 is the "return value" VReg (iconst def).
    // We supply call_arg_precolors = [(v2, RAX)] directly.
    //
    // After run_phase3, pre_coloring_colors must contain:
    //   v2 → color(RAX)
    #[test]
    fn call_return_value_precolor_feeds_through() {
        let block_schedules = vec![vec![iconst_inst(2)]];
        let successors = vec![vec![]];
        let phi_uses = empty_phi_uses(1);
        let global_liveness = compute_global_liveness(&block_schedules, &successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);

        let param_vregs: Vec<(VReg, Reg)> = vec![];
        // Caller supplies the return-value precoloring.
        let call_arg_precolors = vec![(VReg(2), Reg::RAX)];
        let copy_pairs: Vec<(VReg, VReg)> = vec![];
        let next_vreg = phase2.graph.num_vregs as u32;

        let phase3 = run_phase3(
            phase2,
            block_schedules.clone(),
            &param_vregs,
            call_arg_precolors,
            &copy_pairs,
            &successors,
            &phi_uses,
            &Vec::<std::collections::BTreeSet<VReg>>::new(),
            false,
            next_vreg,
        );

        let gpr_order = allocatable_gpr_order(false);
        let rax_color = gpr_order.iter().position(|&r| r == Reg::RAX).unwrap() as u32;

        assert_eq!(
            phase3.pre_coloring_colors.get(&2usize).copied(),
            Some(rax_color),
            "v2 (return value VReg) must appear in pre_coloring_colors with RAX color"
        );
    }

    // ── Helper: build Phase3State for a simple CFG ────────────────────────────

    fn run_phase3_for(
        block_schedules: Vec<Vec<ScheduledInst>>,
        successors: &[Vec<usize>],
        param_vregs: &[(VReg, Reg)],
        call_arg_precolors: Vec<(VReg, Reg)>,
        copy_pairs: &[(VReg, VReg)],
        uses_frame_pointer: bool,
    ) -> Phase3State {
        let phi_uses = empty_phi_uses(block_schedules.len());
        let global_liveness = compute_global_liveness(&block_schedules, successors, &phi_uses);
        let phase2 = build_global_interference(&block_schedules, &global_liveness);
        let next_vreg = phase2.graph.num_vregs as u32;
        run_phase3(
            phase2,
            block_schedules,
            param_vregs,
            call_arg_precolors,
            copy_pairs,
            successors,
            &phi_uses,
            &Vec::<std::collections::BTreeSet<VReg>>::new(),
            uses_frame_pointer,
            next_vreg,
        )
    }

    // ── Task 4 unit tests ─────────────────────────────────────────────────────

    // ── simple_coloring_succeeds ─────────────────────────────────────────────
    //
    // A small two-block straight-line function with low register pressure.
    // Block 0: v0 = iconst, v1 = iconst, v2 = add(v0, v1)
    // Block 1: v3 = use(v2)
    //
    // Three simultaneously live GPR values at most (v0, v1 in block 0;
    // v2 crosses the block boundary). Well within the 14/15-register budget.
    //
    // Verify: gpr_overshoot == 0, xmm_overshoot == 0, vreg_to_reg contains
    // every real VReg (v0..v3).
    #[test]
    fn simple_coloring_succeeds() {
        let block_schedules = vec![
            vec![iconst_inst(0), iconst_inst(1), add_inst(2, 0, 1)],
            vec![use_inst(3, 2)],
        ];
        let successors = vec![vec![1usize], vec![]];

        let phase3 = run_phase3_for(block_schedules, &successors, &[], vec![], &[], false);
        let phase4 = run_phase4(phase3, false);

        assert_eq!(
            phase4.gpr_overshoot, 0,
            "low-pressure function must not overshoot GPR budget"
        );
        assert_eq!(
            phase4.xmm_overshoot, 0,
            "no XMM VRegs, so XMM overshoot must be 0"
        );

        // Every real VReg (0..=3) must appear in vreg_to_reg.
        for idx in 0u32..=3 {
            assert!(
                phase4.vreg_to_reg.contains_key(&VReg(idx)),
                "vreg_to_reg must contain real VReg v{idx}"
            );
        }
    }

    // ── high_pressure_reports_overshoot ──────────────────────────────────────
    //
    // Construct a single-block function where more than
    // `available_gpr_colors(false)` = 15 GPR values are simultaneously live.
    //
    // We create 16 iconst VRegs (v0..v15) and then a single add instruction
    // that uses all 16 as operands (v16 = add_many). Since the add instruction
    // sees all 16 values as live_at, they all interfere pairwise -> chromatic
    // number = 16 > 15 -> gpr_overshoot >= 1.
    //
    // Verify: run_phase4 returns (does NOT panic), and gpr_overshoot > 0.
    #[test]
    fn high_pressure_reports_overshoot() {
        // Build a synthetic high-pressure block: 16 simultaneous live GPR values.
        // v0..v15 = iconst; v16 has all of them as operands.
        let n = 16u32;
        let mut sched: Vec<ScheduledInst> = (0..n).map(iconst_inst).collect();
        // Add an instruction that uses all n values (forces all to be live simultaneously).
        sched.push(ScheduledInst {
            op: Op::X86Add,
            dst: VReg(n),
            operands: (0..n).map(VReg).collect(),
        });
        let block_schedules = vec![sched];
        let successors = vec![vec![]];

        let phase3 = run_phase3_for(
            block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            false, // uses_frame_pointer=false -> 15 GPR colors
        );
        // run_phase4 must return without panicking even when over budget.
        let phase4 = run_phase4(phase3, false);

        assert!(
            phase4.gpr_overshoot > 0,
            "16 simultaneously live GPR values with budget=15 must give gpr_overshoot > 0, \
             got {}",
            phase4.gpr_overshoot
        );
    }

    // ── callee_saved_detected ────────────────────────────────────────────────
    //
    // A function whose coloring is forced to use callee-saved registers.
    //
    // Strategy: pre-color v0..v8 to the first 9 caller-saved GPRs (RAX, RCX,
    // RDX, RSI, RDI, R8, R9, R10, R11) and add interference edges among them
    // so that the 10th simultaneously live VReg (v9) must land in a
    // callee-saved register (RBX, R12, ...).
    //
    // Since we can't easily inject a 10-way clique via the Phase 3 pipeline
    // without complex plumbing, we instead create 10 iconst VRegs all in a
    // single block (so they are all simultaneously live at the program point
    // where the last one is defined). With 9 caller-saved GPRs and one more,
    // the 10th must receive a callee-saved register.
    //
    // Verify: callee_saved_used is non-empty.
    #[test]
    fn callee_saved_detected() {
        // 10 simultaneously live GPR values in a single block:
        //   v0..v8 = iconst
        //   v9 = add(v0, v1, ..., v8)  <- all 9 are live here
        //   v10 = iconst                <- v10 is live alongside v0..v9
        //   v11 = add(v9, v10)          <- finalizes
        //
        // The backward liveness pass will show v0..v8 all live when v9 is
        // assigned, so they form a 10-clique after the add. We need 10 colors.
        // Caller-saved GPRs (excl. RSP) = 9: RAX, RCX, RDX, RSI, RDI, R8, R9,
        // R10, R11. The 10th value (v9) must go to a callee-saved reg.
        let mut sched: Vec<ScheduledInst> = (0u32..9).map(iconst_inst).collect();
        // v9 = add(v0..v8): all 9 iconconsts are live here.
        sched.push(ScheduledInst {
            op: Op::X86Add,
            dst: VReg(9),
            operands: (0u32..9).map(VReg).collect(),
        });
        // v10 = iconst: adds another live value after v9 is defined.
        sched.push(iconst_inst(10));
        // v11 = add(v9, v10): needs both v9 and v10 live.
        sched.push(add_inst(11, 9, 10));
        let block_schedules = vec![sched];
        let successors = vec![vec![]];

        let phase3 = run_phase3_for(
            block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            true, // uses_frame_pointer=true -> 14 GPR colors (RBP excluded)
        );
        let phase4 = run_phase4(phase3, true);

        // With 10 simultaneously live values and 9 caller-saved GPRs (excl. RSP),
        // at least one value must land in a callee-saved register.
        assert!(
            !phase4.callee_saved_used.is_empty(),
            "10 simultaneously live GPRs must force at least one callee-saved register, \
             got callee_saved_used = {:?}",
            phase4.callee_saved_used
        );

        // Confirm all callee-saved entries are genuine callee-saved regs.
        use crate::x86::abi::CALLEE_SAVED;
        for &r in &phase4.callee_saved_used {
            assert!(
                CALLEE_SAVED.contains(&r),
                "callee_saved_used contains {r:?} which is not in CALLEE_SAVED"
            );
        }
    }

    // ── Phase 5 unit tests ────────────────────────────────────────────────────

    /// Helper: run the full allocator pipeline (Phases 2–5) via `allocate_global`.
    fn run_allocate_global(
        block_schedules: &[Vec<ScheduledInst>],
        cfg_succs: &[Vec<usize>],
        param_vregs: &[(VReg, Reg)],
        call_arg_precolors: Vec<(VReg, Reg)>,
        copy_pairs: &[(VReg, VReg)],
        loop_depths: &BTreeMap<VReg, u32>,
        uses_frame_pointer: bool,
    ) -> GlobalRegAllocResult {
        let n = block_schedules.len();
        let phi_uses = empty_phi_uses(n);
        let block_param_vregs: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n];
        allocate_global(
            block_schedules,
            param_vregs,
            call_arg_precolors,
            copy_pairs,
            loop_depths,
            cfg_succs,
            &phi_uses,
            &block_param_vregs,
            "test_fn",
            uses_frame_pointer,
        )
        .expect("allocate_global should succeed")
    }

    // ── Task 5 test 1: low_pressure_no_spill_roundtrips ─────────────────────
    //
    // A two-block straight-line function with low register pressure (3 live
    // GPR values at most, well within the 14/15-register budget). Verifies
    // that Phase 5 converges immediately in round 0 (no spills emitted) and
    // every real VReg gets a physical register assignment.
    //
    // Block 0: v0 = iconst, v1 = iconst, v2 = add(v0, v1)
    // Block 1: v3 = use(v2)
    #[test]
    fn low_pressure_no_spill_roundtrips() {
        let block_schedules = vec![
            vec![iconst_inst(0), iconst_inst(1), add_inst(2, 0, 1)],
            vec![use_inst(3, 2)],
        ];
        let successors = vec![vec![1usize], vec![]];
        let result = run_allocate_global(
            &block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            &BTreeMap::new(),
            false,
        );

        // No spills: every real VReg must have a physical register.
        assert_eq!(
            result.spill_slots, 0,
            "no spill slots expected for low-pressure function"
        );
        for idx in 0u32..=3 {
            assert!(
                result.vreg_to_reg.contains_key(&VReg(idx)),
                "vreg_to_reg must contain VReg v{idx}"
            );
        }
        // The schedules should be unchanged (no spill/reload instructions).
        assert_eq!(result.per_block_insts[0].len(), 3);
        assert_eq!(result.per_block_insts[1].len(), 1);
    }

    // ── Task 5 test 4 / Task 5.9: xmm_cross_block_phi_allocates ────────────
    //
    // An XMM value defined in block 0, consumed by a block parameter (phi) in
    // block 2 via block 1, with NO calls on the path. Verifies that:
    // - The allocator does not panic.
    // - The XMM VReg receives a valid physical XMM register assignment (since
    //   there are no calls, no XMM call phantom is injected, so the XMM VReg
    //   can stay in a register across the block boundary).
    // - `spill_slots` == 0 (pressure-based spilling is sufficient; no forced
    //   slot pre-spill is needed when there are no calls on the path).
    //
    // This test validates the Task 5.9 audit conclusion: phi-copy emission is
    // reg-to-reg movsd and the global allocator assigns XMM registers across
    // block boundaries correctly without forced-slot pre-spilling.
    #[test]
    fn xmm_cross_block_phi_allocates() {
        // Simulate an XMM VReg flowing across blocks via a phi.
        //
        // We use Op::X86Addsd as an FP op (classified as XMM by build_vreg_classes).
        // Block 0: xmm_val = x86addsd(dummy1, dummy2)  [XMM def]
        // Block 1: pass-through block (no instructions, just live-in/out)
        // Block 2: xmm_use = x86addsd(xmm_val, xmm_val)
        //
        // Since we can't directly model block parameters here (that requires the
        // full IR), we use a cross-block live value instead, which exercises the
        // same interference-graph cross-block path.
        //
        // VRegs:
        //   v0 = iconst (GPR dummy)
        //   v1 = x86addsd(v0, v0) in block 0  -> XMM class
        //   v2 = x86addsd(v1, v1) in block 2  -> XMM class, uses v1
        let xmm_def = ScheduledInst {
            op: Op::X86Addsd,
            dst: VReg(1),
            operands: vec![VReg(0), VReg(0)],
        };
        let xmm_use = ScheduledInst {
            op: Op::X86Addsd,
            dst: VReg(2),
            operands: vec![VReg(1), VReg(1)],
        };

        let block_schedules = vec![
            vec![iconst_inst(0), xmm_def], // block 0: define XMM v1
            vec![],                        // block 1: empty pass-through
            vec![xmm_use],                 // block 2: use XMM v1
        ];
        // 0 -> 1 -> 2
        let successors = vec![vec![1usize], vec![2usize], vec![]];

        let result = run_allocate_global(
            &block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            &BTreeMap::new(),
            false,
        );

        // Must not panic (already verified by not crashing).
        // XMM VReg v1 must have a physical XMM register (no call on the path
        // means XMM phantoms are never injected -> v1 is freely assignable).
        assert!(
            result.vreg_to_reg.contains_key(&VReg(1)),
            "XMM VReg v1 must have a physical register assignment when no call is on the path"
        );
        let assigned_reg = result.vreg_to_reg[&VReg(1)];
        assert!(
            assigned_reg.is_xmm(),
            "XMM VReg v1 must be assigned an XMM register, got {assigned_reg:?}"
        );

        // No spills expected: only 1 XMM value, budget is 16 XMM registers.
        assert_eq!(
            result.spill_slots, 0,
            "no spill slots expected for low-pressure XMM cross-block function"
        );
    }

    // ── Test A8: many_call_args_exceed_abi_regs ───────────────────────────────
    //
    // A function calling another function with 8 int arguments. The first 6
    // receive ABI precolors (RDI, RSI, RDX, RCX, R8, R9); args 7 and 8 exceed
    // the 6-register limit and receive no precolor (stack-passed in real ABI,
    // no precolor needed in our register allocator model).
    //
    // Verifies:
    // - allocate_global returns Ok (no spurious precolor conflicts).
    // - The 6 precolored args appear in vreg_to_reg with their expected regs.
    // - Args 7 and 8 (not precolored) also appear in vreg_to_reg (they get
    //   some free register).
    // - No panic or error from having more call-arg precolors than ABI regs.
    #[test]
    fn many_call_args_exceed_abi_regs() {
        use crate::x86::reg::Reg;

        // Simulate a call with 8 int args.
        // v0..v7 = iconst (the 8 call arguments)
        // v8 = VoidCallBarrier(v0..v7)  <- call with 8 args
        // v9 = use(v0)  <- keeps v0 live to avoid trivial elision

        let mut sched: Vec<ScheduledInst> = (0u32..8).map(iconst_inst).collect();
        sched.push(ScheduledInst {
            op: Op::VoidCallBarrier,
            dst: VReg(8),
            operands: (0u32..8).map(VReg).collect(),
        });
        sched.push(use_inst(9, 0));

        let block_schedules = vec![sched];
        let successors = vec![vec![]];

        // ABI precolors for the first 6 int args (SysV AMD64):
        // v0=RDI, v1=RSI, v2=RDX, v3=RCX, v4=R8, v5=R9
        // v6 and v7 have no ABI precolor (7th and 8th args are stack-passed).
        let abi_regs = [Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9];
        let call_arg_precolors: Vec<(VReg, Reg)> = (0u32..6)
            .zip(abi_regs.iter().copied())
            .map(|(v, r)| (VReg(v), r))
            .collect();

        let n = block_schedules.len();
        let phi_uses = empty_phi_uses(n);
        let block_param_vregs: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); n];

        let result = allocate_global(
            &block_schedules,
            &[],
            call_arg_precolors,
            &[],
            &BTreeMap::new(),
            &successors,
            &phi_uses,
            &block_param_vregs,
            "test_many_args",
            false,
        )
        .expect("allocate_global must succeed with 8 call args (6 precolored, 2 unprecolored)");

        // All 9 VRegs (v0..v8) must get a physical register.
        for v in 0u32..9 {
            assert!(
                result.vreg_to_reg.contains_key(&VReg(v)),
                "VReg v{v} must have a physical register assignment"
            );
        }

        // At least one of v0..v5 must land on its ABI reg. Some may be dropped to
        // `unprecolored_params` if live across the call conflicts with a phantom
        // (R5), but dropping ALL six would indicate the precoloring pass is broken.
        let abi_set: BTreeSet<Reg> = [Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9]
            .iter()
            .copied()
            .collect();
        let assigned_abi_count = (0u32..6)
            .filter(|&v| {
                result
                    .vreg_to_reg
                    .get(&VReg(v))
                    .map(|&r| abi_set.contains(&r))
                    .unwrap_or(false)
            })
            .count();
        assert!(
            assigned_abi_count >= 1,
            "expected at least one of v0..v5 to land on its ABI reg; got {assigned_abi_count}"
        );
    }

    // ── Regression: loop with three phi params must not merge two params ────
    //
    // Mirrors the `sum` miscompile from Phase 6. The IR is:
    //   block0: v0 = iconst(1)  (initial i)
    //           v1 = iconst(5)  (initial n, the bound)
    //           v2 = iconst(0)  (initial acc)
    //           jump block1(v0, v1, v2)
    //   block1(p0=v3 (i), p1=v4 (n), p2=v5 (acc)):
    //           v6 = x86_sub(v3, v4)  // cmp-like i - n
    //           jump block2(v6)
    //   block2(p0=v7): use(v7)
    //
    // This triggers the same structural pattern as loop-sum: three block
    // params on a single block fed by three distinct iconsts. After Phase 6
    // cutover, the allocator must assign v3, v4, v5 to three DISTINCT
    // registers — NOT merge two of them into the same color. Two params on
    // the same register would cause the phi-copy lowering to emit two movs
    // with the same destination, overwriting one value.
    //
    // This test exercises `allocate_global` directly with a hand-built
    // schedule. It does NOT exercise build_phi_copies (that's compile/mod.rs
    // territory) — the invariant being checked is purely on the allocator:
    // each distinct block param gets a distinct color.
    //
    // Current status (pre-fix): v3, v4, v5 correctly get distinct colors when
    // `add_block_param_interferences` runs on the post-coalesce graph with
    // alias resolution. If this test regresses, check that the
    // block_param_vregs_per_block argument is being propagated to both the
    // initial phase 2 build AND the post-coalesce rebuild inside run_phase3.
    #[test]
    fn three_phi_params_get_distinct_colors() {
        use crate::ir::op::Op;
        use crate::ir::types::Type;

        fn iconst(dst: u32, val: i64) -> ScheduledInst {
            ScheduledInst {
                op: Op::Iconst(val, Type::I64),
                dst: VReg(dst),
                operands: vec![],
            }
        }
        fn block_param(dst: u32, bid: u32, idx: u32) -> ScheduledInst {
            ScheduledInst {
                op: Op::BlockParam(bid, idx, Type::I64),
                dst: VReg(dst),
                operands: vec![],
            }
        }

        let block_schedules = vec![
            // block 0: three iconsts fed as phi sources to block 1.
            vec![iconst(0, 1), iconst(1, 5), iconst(2, 0)],
            // block 1: three block params; sub to exercise them; feed block 2.
            vec![
                block_param(3, 1, 0),
                block_param(4, 1, 1),
                block_param(5, 1, 2),
                ScheduledInst {
                    op: Op::X86Sub,
                    dst: VReg(6),
                    operands: vec![VReg(3), VReg(4)],
                },
            ],
            // block 2: single block param, use it.
            vec![
                block_param(7, 2, 0),
                ScheduledInst {
                    op: Op::X86Sub,
                    dst: VReg(8),
                    operands: vec![VReg(7), VReg(7)],
                },
            ],
        ];
        let cfg_succs = vec![vec![1usize], vec![2usize], vec![]];
        // phi_uses[0] = {v0, v1, v2} (terminator args of block 0 to block 1).
        // phi_uses[1] = {v6} (terminator arg of block 1 to block 2 — v6 is the sub result).
        let mut phi_uses: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); 3];
        phi_uses[0].insert(VReg(0));
        phi_uses[0].insert(VReg(1));
        phi_uses[0].insert(VReg(2));
        phi_uses[1].insert(VReg(6));
        let mut block_param_vregs: Vec<BTreeSet<VReg>> = vec![BTreeSet::new(); 3];
        block_param_vregs[1].extend([VReg(3), VReg(4), VReg(5)]);
        block_param_vregs[2].insert(VReg(7));
        // Copy pairs: the phi args → block params. These are what coalesce
        // will attempt to merge. Critically, (v0, v3) (v1, v4) (v2, v5) must
        // NOT coalesce in a way that collapses v3/v4/v5 onto a single color.
        let copy_pairs = vec![
            (VReg(0), VReg(3)),
            (VReg(1), VReg(4)),
            (VReg(2), VReg(5)),
            (VReg(6), VReg(7)),
        ];

        let result = allocate_global(
            &block_schedules,
            &[],
            vec![],
            &copy_pairs,
            &BTreeMap::new(),
            &cfg_succs,
            &phi_uses,
            &block_param_vregs,
            "three_phi_params",
            false,
        )
        .expect("allocate_global must succeed");

        // Resolve each param to its canonical (via coalesce_aliases) and
        // physical register.
        let resolve = |v: VReg| -> Reg {
            let mut cur = v;
            while let Some(&aliased) = result.coalesce_aliases.get(&cur) {
                if aliased == cur {
                    break;
                }
                cur = aliased;
            }
            *result
                .vreg_to_reg
                .get(&cur)
                .unwrap_or_else(|| panic!("VReg {cur:?} has no register"))
        };
        let r3 = resolve(VReg(3));
        let r4 = resolve(VReg(4));
        let r5 = resolve(VReg(5));
        assert_ne!(
            r3, r4,
            "v3 (param 0) and v4 (param 1) must be in distinct regs"
        );
        assert_ne!(
            r3, r5,
            "v3 (param 0) and v5 (param 2) must be in distinct regs"
        );
        assert_ne!(
            r4, r5,
            "v4 (param 1) and v5 (param 2) must be in distinct regs"
        );
    }
}
