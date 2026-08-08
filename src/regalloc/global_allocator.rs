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

use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::{MachOp, Op, PseudoOp, PureOp};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::ArgLoc;
use crate::x86::abi::{CALLER_SAVED_GPR, CALLER_SAVED_XMM};
use crate::x86::reg::{Reg, RegClass};

use super::GlobalRegAllocResult;
use super::build_vreg_classes_from_all_blocks;
use super::coalesce::coalesce;
use super::coloring::{allocatable_gpr_order, allocatable_xmm_order};
use super::interference::{
    InterferenceGraph, VRegSet, build_interference_into, dying_clobber_operands,
    resolve_precoloring_conflicts,
};
use super::liveness::LivenessInfo;
use super::rewrite::CoalesceAliases;
use super::slots::SlotAllocator;

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

    /// Merged precoloring map: VReg index -> color. Covers param precolors,
    /// shift/div precolors, and all three phantom precolor sets.
    pre_coloring_colors: BTreeMap<usize, u32>,

    /// Param VRegs whose ABI precoloring was dropped because a call clobber
    /// phantom interferes with the same color. The lowering must emit a mov
    /// from the ABI register to the allocated register at function entry.
    unprecolored_params: Vec<(VReg, Reg)>,

    /// Coalescing alias map: `from_idx -> into_idx`. When two VRegs are coalesced,
    /// the "from" VReg no longer exists in the post-coalesce schedules; its uses
    /// have been rewritten to "into". Downstream callers (e.g. lowering's
    /// `block_class_to_vreg`) must apply this map when resolving ClassId -> VReg
    /// so that stale `class_to_vreg` entries pointing at `from` VRegs are chased
    /// to their live canonical counterparts.
    alias_map: BTreeMap<u32, u32>,
}

/// Add the interferences a block's parameters have that their marker
/// instructions do not express.
///
/// Every parameter of a block holds its register before the block's first
/// instruction runs -- a block parameter because the phi copies on the edge
/// wrote it, a function parameter because the caller did. The marker is only a
/// name for the value, and the scheduler puts markers wherever the dependence
/// order allows -- so the schedule's own live ranges understate what has
/// already been placed, in two ways.
///
/// **Between parameters.** Two parameters unused in the block body have disjoint
/// schedule-level ranges, yet the copy writes both, so they need distinct
/// registers.
///
/// **Between a parameter and anything the block does before that parameter's
/// marker.** A value defined and dead again ahead of the marker looks free to
/// take the parameter's register, and the register already holds the parameter:
/// a splitter store/reload pair inserted after the first parameter's marker read
/// a slot into RAX while RAX held the seventeenth parameter, whose marker came
/// twelve instructions later. Nothing downstream could see it -- RAX was written
/// before it was read, and no two *modelled* ranges overlapped. The same shape
/// with function parameters gave a splitter store its own copy of a value in
/// RCX, which was the fourth argument, and the callee read the wrong one.
fn add_param_interferences(
    graph: &mut InterferenceGraph,
    block_param_vregs_per_block: &[VRegSet],
    block_schedules: &[Vec<ScheduledInst>],
    alias_map: &BTreeMap<u32, u32>,
    arg_locs: &[ArgLoc],
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
    let interfere = |a: VReg, b: VReg, graph: &mut InterferenceGraph| {
        let (a, b) = (a.0 as usize, b.0 as usize);
        if a == b || a >= graph.num_vregs || b >= graph.num_vregs {
            return;
        }
        if graph.reg_class[a] == graph.reg_class[b] {
            graph.adj[a].insert(b);
            graph.adj[b].insert(a);
        }
    };
    let no_params = VRegSet::default();
    for (bi, sched) in block_schedules.iter().enumerate() {
        // The block's parameters, from the CFG's record plus the entry block's
        // function parameters. Only the second comes from the schedule:
        // `collect_block_param_vregs_per_block` collects `BlockParam` markers
        // and is the authority on those, while a `Param` marker is the only
        // place a function parameter is stated.
        let mut seen: BTreeSet<VReg> = BTreeSet::new();
        let unique: Vec<VReg> = block_param_vregs_per_block
            .get(bi)
            .unwrap_or(&no_params)
            .iter()
            .map(|p| VReg(p as u32))
            .chain(
                sched
                    .iter()
                    .filter(|inst| {
                        matches!(inst.op, Op::Pure(PureOp::Param(..)))
                            && crate::x86::abi::marker_is_entry_resident(&inst.op, arg_locs)
                    })
                    .map(|inst| inst.dst),
            )
            .map(resolve)
            .filter(|&v| seen.insert(v))
            .collect();
        if unique.is_empty() {
            continue;
        }
        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                interfere(unique[i], unique[j], graph);
            }
        }

        // Everything named before the block's last parameter marker is inside
        // the shadow. A pseudo-op's `dst` is excluded: it names no value and
        // takes no register.
        let Some(last_marker) = sched.iter().rposition(|inst| inst.op.is_param_marker()) else {
            continue;
        };
        let mut in_shadow: BTreeSet<VReg> = BTreeSet::new();
        for inst in &sched[..=last_marker] {
            // A stack-passed parameter is not resident, so its marker is an
            // ordinary definition inside the shadow rather than one of the values
            // the shadow is drawn around.
            if !crate::x86::abi::marker_is_entry_resident(&inst.op, arg_locs)
                && !inst.op.has_no_result()
            {
                in_shadow.insert(resolve(inst.dst));
            }
            in_shadow.extend(inst.operands.iter().map(|&v| resolve(v)));
        }
        for &param in &unique {
            for &other in &in_shadow {
                interfere(param, other, graph);
            }
        }
    }
}

/// Build the function-wide interference graph (Phase 2).
///
/// Steps:
/// 1. Build a function-wide VReg class map from all blocks.
/// 2. Determine `num_vregs` and allocate the shared `InterferenceGraph` with
///    `reg_class` pre-populated.
/// 3. For each block, compute per-block `LivenessInfo` using the global
///    `live_out` set, call `build_interference_into` to add edges, and store
///    the `LivenessInfo` in `per_block_liveness` (Tasks 2.4, 2.4.5).
/// 4. Add cross-block boundary interferences: all pairs in `live_in[b]` of the
///    Same class interfere; same for `live_out[b]`.
fn build_global_interference(
    block_schedules: &[Vec<ScheduledInst>],
    global_liveness: &crate::regalloc::global_liveness::GlobalLiveness,
    slot_resident: &VRegSet,
) -> Phase2State {
    // Function-wide class map (must complete before graph init).
    let vreg_class_map = build_vreg_classes_from_all_blocks(block_schedules);

    // Determine num_vregs across all blocks + global liveness sets.
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
            for idx in live_set {
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
                live_in: VRegSet::new(),
                live_out: VRegSet::new(),
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
        adj: vec![VRegSet::new(); num_vregs],
        reg_class,
    };

    // Collect per-block liveness.
    let mut per_block_liveness: Vec<LivenessInfo> = Vec::with_capacity(block_schedules.len());

    // For each block: compute liveness, add edges, then add boundary interferences.
    for (b, sched) in block_schedules.iter().enumerate() {
        let block_live_out = &global_liveness.live_out[b];
        let liveness = crate::regalloc::liveness::compute_liveness_excluding(
            sched,
            block_live_out,
            slot_resident,
        );

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
fn add_boundary_interferences(graph: &mut InterferenceGraph, boundary_set: &VRegSet) {
    let live: Vec<usize> = boundary_set
        .iter()
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

// ── Function-wide precoloring ──────────────────────────────────────

/// Pre-color shift count operands to RCX for variable-shift instructions.
///
/// Mirrors `add_shift_precolors` from `compile/precolor.rs`.
fn add_shift_precolors_global(insts: &[ScheduledInst], precolors: &mut Vec<(VReg, Reg)>) {
    for inst in insts {
        if matches!(
            inst.op,
            Op::Mach(MachOp::X86Shl) | Op::Mach(MachOp::X86Shr) | Op::Mach(MachOp::X86Sar)
        ) && inst.operands.len() >= 2
        {
            let count_vreg = inst.operands[1];
            if !precolors.iter().any(|&(v, _)| v == count_vreg) {
                precolors.push((count_vreg, Reg::RCX));
            }
        }
    }
}

/// Build a function-wide precoloring list covering params, shifts, and
/// caller-supplied call-argument/return-value VRegs.
///
/// Division contributes nothing. Neither operand nor projection is pinned: the
/// lowering emits `mov rax, <dividend>`, `mov dst, rax` and `mov dst, rdx`
/// around the instruction, and coalescing removes each copy when the register
/// is free. The quotient used to be pinned to RAX here, and in
/// `compile/precolor.rs` until it was removed there; two quotients live at once
/// both got RAX and the second division destroyed the first.
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

// ── Call/div point collection ─────────────────────────────────────

/// The `(block_idx, inst_idx)` positions of every call and every div.
type CallDivPoints = (Vec<(usize, usize)>, Vec<(usize, usize)>);

/// An interference graph with clobber phantoms injected, and the phantom node
/// index each call or div point contributed: GPR call, XMM call, then div.
type PhantomGraph = (
    InterferenceGraph,
    BTreeMap<usize, u32>,
    BTreeMap<usize, u32>,
    BTreeMap<usize, u32>,
);

/// Collect all call and div program points across all blocks.
///
/// Returns `(call_points, div_points)` where each entry is `(block_idx, inst_idx)`.
fn collect_call_div_points(block_schedules: &[Vec<ScheduledInst>]) -> CallDivPoints {
    let mut call_points: Vec<(usize, usize)> = Vec::new();
    let mut div_points: Vec<(usize, usize)> = Vec::new();

    for (b, sched) in block_schedules.iter().enumerate() {
        for (i, inst) in sched.iter().enumerate() {
            if matches!(
                inst.op,
                Op::Pseudo(PseudoOp::CallResult(_, _)) | Op::Pseudo(PseudoOp::VoidCallBarrier)
            ) {
                call_points.push((b, i));
            }
            if matches!(
                inst.op,
                Op::Mach(MachOp::X86Idiv(..)) | Op::Mach(MachOp::X86Div(..))
            ) {
                div_points.push((b, i));
            }
        }
    }

    (call_points, div_points)
}

// ── Global clobber interference injection ───────────────────

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

        let live_at_cp: &VRegSet = if inst_idx < n {
            &liveness.live_at[inst_idx]
        } else {
            &liveness.live_out
        };

        // Operands the clobbering instruction itself consumes, which are in the
        // clobbered register on purpose. Only those that do not survive the
        // point: an argument live past its call must still interfere.
        let consumed: BTreeSet<usize> = if inst_idx < sched.len() {
            let live_after: &VRegSet = if inst_idx + 1 < n {
                &liveness.live_at[inst_idx + 1]
            } else {
                &liveness.live_out
            };
            dying_clobber_operands(&sched[inst_idx], live_after)
        } else {
            BTreeSet::new()
        };

        // Early-out: skip if no VRegs of the target class are live besides the
        // instruction's own operands.
        if config.skip_if_no_live {
            let has_live = live_at_cp.iter().any(|idx| {
                idx < graph.num_vregs
                    && graph.reg_class[idx] == config.reg_class
                    && !consumed.contains(&idx)
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
                graph.adj.resize(new_n, VRegSet::new());
                graph.reg_class.resize(new_n, config.reg_class);
                graph.num_vregs = new_n;
            }
            graph.reg_class[phantom_idx] = config.reg_class;

            phantom_precolors.insert(phantom_idx, color);

            for live_idx in live_at_cp {
                if live_idx < graph.num_vregs
                    && graph.reg_class[live_idx] == config.reg_class
                    && !consumed.contains(&live_idx)
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
) -> PhantomGraph {
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
            skip_if_no_live: false,
        },
        next_vreg,
    );

    (graph, gpr_call_phantoms, xmm_call_phantoms, div_phantoms)
}

// ── Global merge_precolorings ──────────────────────────────────────

/// Merge phantom precolorings with param precolorings into one map.
///
/// When a VReg is precolored to the same color as a clobber phantom AND the
/// graph has an interference edge between them, its precoloring is dropped (it
/// will receive a free register instead). Dropped params are appended to
/// `unprecolored_params` so the lowering emits an entry move.
///
/// Every phantom is checked, not only the call ones. A phantom stands for a
/// register the hardware overwrites at a point some value is live across, and
/// a division clobbers RAX and RDX exactly as a call clobbers the caller-saved
/// set. Checking calls alone left a parameter pinned to RDX holding that
/// register across an `idiv` that overwrites it -- three of forty generated
/// programs, reported by `check_precolorings` as two interfering VRegs
/// pre-colored alike.
///
/// Mirrors `merge_precolorings` from `allocator.rs` but operates at function
/// scope with the global `param_vreg_to_reg` map and `unprecolored_params`.
#[allow(clippy::too_many_arguments)] // Four phantom/precolor maps that have no other owner.
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

    // For each phantom, check if any precoloring conflicts (same color +
    // interference edge). Drop conflicting precolorings: the VReg will get a
    // free register, and the lowering will emit a mov to the ABI register at
    // the use site (call arg setup or function prologue).
    //
    // This covers both function params AND call-arg VRegs: a call-arg VReg
    // whose value is live across OTHER calls that clobber the target register
    // cannot be precolored to that register, or its value is destroyed by the
    // intervening call. The `setup_call_args` lowering handles non-precolored
    // arg VRegs by emitting `mov rdi, <arg_reg>` before the call.
    let phantoms = gpr_call_phantoms
        .iter()
        .chain(xmm_call_phantoms)
        .chain(div_phantoms);
    for (&phantom_vreg, &phantom_color) in phantoms {
        let conflicting: Vec<usize> = merged
            .iter()
            .filter(|&(&pv, &pc)| {
                pc == phantom_color
                    && phantom_vreg < graph.num_vregs
                    && pv < graph.num_vregs
                    && graph.adj[phantom_vreg].contains(pv)
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

    // Then the pre-colorings that collide with each other rather than with a
    // phantom. Done after the phantom pass and before the phantoms are merged
    // in, so it sees exactly the real pre-colorings that survived.
    for pv in resolve_precoloring_conflicts(&merged, graph, param_vreg_indices) {
        merged.remove(&pv);
        let vreg = VReg(pv as u32);
        if let Some(reg) = param_vreg_to_reg.remove(&vreg)
            && param_vreg_indices.contains(&pv)
        {
            unprecolored_params.push((vreg, reg));
        }
    }

    // Inject phantom precolorings. Phantoms represent hard hardware constraints
    // and override any remaining param precolorings at the same index.
    merged.extend(gpr_call_phantoms);
    merged.extend(xmm_call_phantoms);
    merged.extend(div_phantoms);

    merged
}

// ── Coalescing and post-rebuild ──────────────────────────────

/// Follow `from -> into` to the end of its chain.
///
/// The self-entry check is what makes this terminate: a map with `x -> x` in it
/// is a fixpoint, not a step, and a loop that only asks whether a key is present
/// spins on one forever. The bound is a second guard, for a cycle no single
/// entry reveals.
fn chase_u32(mut idx: u32, aliases: &BTreeMap<u32, u32>) -> u32 {
    for _ in 0..aliases.len() + 1 {
        match aliases.get(&idx) {
            Some(&target) if target != idx => idx = target,
            _ => break,
        }
    }
    idx
}

/// Run Phase 3: precoloring, clobber phantoms, coalescing, and graph rebuild.
///
/// # Order of operations (matches the per-block allocator)
///
/// 1. Build function-wide precoloring from params + shifts + divs.
/// 2. Collect call/div points.
/// 3. **Coalesce on the PRE-phantom graph** produced by Phase 2.
/// 4. Apply coalescing aliases to each block's schedule.
/// 5. Rebuild interference graph from scratch on post-coalesce schedules.
/// 6. Inject clobber phantoms into the rebuilt graph (Tasks 3.3/3.4/3.7).
/// 7. Rebuild precolorings on the post-coalesce VReg set; apply
///    `merge_precolorings_global` to detect and drop conflicting param
///    precolorings (Tasks 3.5/3.7).
#[allow(clippy::too_many_arguments)] // A phase's inputs are the previous phase's outputs.
fn run_phase3(
    phase2: Phase2State,
    block_schedules: Vec<Vec<ScheduledInst>>,
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    copy_pairs: &[(VReg, VReg)],
    cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    uses_frame_pointer: bool,
    mut next_vreg: u32,
    coalesce_now: bool,
    coalesce_deny: &BTreeSet<(usize, usize)>,
    slot_resident: &VRegSet,
    arg_locs: &[ArgLoc],
) -> Phase3State {
    // Build function-wide precoloring (params + shifts + divs +
    // caller-supplied call-arg precolors).
    let (precolors, param_vreg_indices) =
        build_function_wide_precoloring(param_vregs, &block_schedules, call_arg_precolors);
    let mut param_vreg_to_reg: BTreeMap<VReg, Reg> = precolors.iter().copied().collect();

    // Coalesce on the PRE-phantom graph from Phase 2, once for the function.
    //
    // `coalesce_now` is what makes "once" true. The spill loop runs this phase
    // every round, and `copy_pairs` is derived from the schedules the loop
    // started with -- so a second round would coalesce an already-coalesced
    // list against pairs naming VRegs the first round merged away. That is not
    // a smaller version of the same decision, it is a different one: on `args`
    // seed 81 a round whose spilling changed nothing still took the overshoot
    // from 1 to 9, purely from being coalesced twice, and the loop then read its
    // own re-coalescing as proof that spilling cannot help.
    let coalesced = if coalesce_now {
        let pairs: Vec<(usize, usize)> = copy_pairs
            .iter()
            .map(|(src, dst)| (src.0 as usize, dst.0 as usize))
            .filter(|&(src, dst)| src < phase2.graph.num_vregs && dst < phase2.graph.num_vregs)
            .collect();
        coalesce(&phase2.graph, &pairs, coalesce_deny)
    } else {
        Default::default()
    };

    // Apply coalescing aliases to each block's schedule individually,
    // preserving block boundaries. The alias table is built once for the
    // function: every block resolves against the same merges, and rebuilding it
    // per block was a quarter of the compile on a 3763-block function.
    let coalesce_aliases_table = CoalesceAliases::new(&coalesced);
    let post_coalesce_schedules: Vec<Vec<ScheduledInst>> = block_schedules
        .iter()
        .map(|sched| coalesce_aliases_table.apply(sched))
        .collect();

    // Build the coalescing alias map early so it can be used to resolve
    // block_param VRegs to their post-coalesce canonicals before the rebuild's
    // interference injection. (The later declaration of `alias_map` is reused.)
    let alias_map_early: BTreeMap<u32, u32> = coalesced
        .iter()
        .map(|&(into, from)| (from as u32, into as u32))
        .collect();

    // Rebuild the interference graph from scratch on the post-coalesce
    // schedules: rebuild the function-wide VReg class map, re-initialize the
    // graph with `reg_class` pre-populated, re-run `build_interference_into`
    // per block, and re-add the cross-block boundary interferences.
    //
    // The CFG topology (cfg_succs) is unchanged by coalescing — only VReg
    // names change, so liveness has to be rebuilt in the new names. A
    // terminator's uses are read back off the coalesced schedules, which
    // `apply_coalescing` has already renamed; block params are not written down
    // in any instruction, so those go through the alias map. Seeding live_out
    // with a pre-coalesce name instead would name a VReg the schedule never
    // mentions, and the canonical one (e.g. v1 for n) would not read as live —
    // a new def in the block could then land on the same register as n.
    let resolve_vreg_early = |v: VReg| -> VReg { VReg(chase_u32(v.0, &alias_map_early)) };
    let post_coalesce_phi_uses = crate::compile::barrier::terminator_uses(&post_coalesce_schedules);
    let renamed_block_param_vregs: Vec<VRegSet> = block_param_vregs_per_block
        .iter()
        .map(|set| {
            set.iter()
                .map(|v| resolve_vreg_early(VReg(v as u32)).0 as usize)
                .collect()
        })
        .collect();
    let rebuild_global_liveness =
        crate::regalloc::global_liveness::compute_global_liveness_excluding(
            &post_coalesce_schedules,
            cfg_succs,
            &post_coalesce_phi_uses,
            &renamed_block_param_vregs,
            slot_resident,
        );

    let mut rebuilt = build_global_interference(
        &post_coalesce_schedules,
        &rebuild_global_liveness,
        slot_resident,
    );
    add_param_interferences(
        &mut rebuilt.graph,
        &renamed_block_param_vregs,
        &post_coalesce_schedules,
        &alias_map_early,
        arg_locs,
    );

    // Re-inject clobber phantoms into the rebuilt graph.
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
    let alias_map = alias_map_early;

    let resolve_vreg = |v: VReg| -> VReg { VReg(chase_u32(v.0, &alias_map)) };

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

    // merge_precolorings_global — detect and drop param precolorings
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
        pre_coloring_colors,
        unprecolored_params,
        alias_map,
    }
}

// ── Phase 4: Global coloring and mapping ─────────────────────────────────────

/// Result of Phase 4: the VReg-to-register assignment, what the coloring could
/// not place, and the per-class overshoot the spill loop reads.
pub(crate) struct Phase4State {
    /// Final function-wide VReg -> physical register assignment.
    ///
    /// Contains only real VRegs (those appearing as `dst` or `operands` in
    /// `per_block_insts`). Phantom VRegs injected by Phase 3 clobber injection
    /// are excluded.
    pub vreg_to_reg: BTreeMap<VReg, Reg>,

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

    /// Real VRegs the coloring could not fit in their class's budget.
    ///
    /// The spill loop's candidates: a VReg with a color at or above its budget
    /// is one no register exists for. Precolored VRegs are excluded, since
    /// spilling one would break the ABI constraint that gave it its color.
    pub over_budget: Vec<VReg>,

    /// Inherited from Phase 3: coalesce alias map (`from_idx -> into_idx`).
    pub alias_map: BTreeMap<u32, u32>,
}

/// Run Phase 4: global coloring and color-to-register mapping.
///
/// # Steps
///
/// 1. `mcs_ordering` on the Phase 3 graph.
/// 2. `greedy_color` with Phase 3's merged precoloring.
/// 3. compute per-class chromatic numbers and overshoot counts.
/// 4. interval-color fallback is intentionally omitted (see module
///    doc `# Coloring strategy`). If greedy fails, Phase 5 handles it.
/// 5. `map_colors_to_regs` per class; build `vreg_to_reg` from
///    real VRegs only (phantoms are excluded).
/// 6. compute `callee_saved_used` as the union of assigned physical
///    registers that appear in `CALLEE_SAVED` / `CALLEE_SAVED_XMM`.
pub(crate) fn run_phase4(phase3: Phase3State, uses_frame_pointer: bool) -> Phase4State {
    use super::coloring::{available_gpr_colors, greedy_color, map_colors_to_regs, mcs_ordering};
    use crate::x86::abi::CALLEE_SAVED;

    // MCS ordering on the Phase 3 graph.
    let ordering = mcs_ordering(&phase3.graph);

    // Greedy coloring with merged precoloring from Phase 3, biased toward the
    // register each two-address result wants so lowering emits no copy for it.
    let hints = two_address_hints(&phase3.per_block_insts);
    let mut coloring = greedy_color(
        &phase3.graph,
        &ordering,
        &phase3.pre_coloring_colors,
        &hints,
    );

    // Reverse MCS order is optimal on chordal graphs, and a single block's
    // interference graph is one. A function-scope graph is not: a value live
    // through several blocks interferes with values that never coexist, and the
    // result has cycles no elimination order makes simplicial. Greedy then
    // overshoots by a color or two on graphs that are colorable within budget --
    // the signature is one VReg of very high degree whose neighbours happen to
    // occupy every color below the budget.
    //
    // Retry in descending-degree order when that happens, and keep whichever
    // coloring is better. This runs only on the path that would otherwise fail
    // allocation outright, so it costs nothing on programs that already fit.
    let gpr_budget_for_retry = available_gpr_colors(uses_frame_pointer);
    if exceeds_budget(&coloring, &phase3.graph, gpr_budget_for_retry) {
        let mut by_degree: Vec<usize> = (0..phase3.graph.num_vregs).collect();
        by_degree.sort_by_key(|&v| std::cmp::Reverse(phase3.graph.adj[v].len()));
        // `greedy_color` walks its ordering in reverse, so hand it the ascending
        // list to have it colour the highest-degree nodes first.
        by_degree.reverse();
        let retry = greedy_color(
            &phase3.graph,
            &by_degree,
            &phase3.pre_coloring_colors,
            &hints,
        );
        if retry.chromatic_number < coloring.chromatic_number {
            coloring = retry;
        }
    }
    let coloring = coloring;

    // Per-class chromatic numbers and overshoot counts.
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
            // Flags occupy no register file, so they widen neither.
            RegClass::Flags => {}
        }
    }

    let gpr_chromatic = gpr_max_color.map_or(0, |m| m + 1);
    let xmm_chromatic = xmm_max_color.map_or(0, |m| m + 1);

    let gpr_budget = available_gpr_colors(uses_frame_pointer);
    let xmm_budget = super::coloring::AVAILABLE_XMM_COLORS;

    let gpr_overshoot = gpr_chromatic.saturating_sub(gpr_budget);
    let xmm_overshoot = xmm_chromatic.saturating_sub(xmm_budget);

    if (gpr_overshoot > 0 || xmm_overshoot > 0) && crate::trace::is_enabled("split") {
        tracing::debug!(
            target: "blitz::split",
            "allocator disagrees with the splitter: gpr_chromatic={gpr_chromatic} \
             (budget {gpr_budget}), xmm_chromatic={xmm_chromatic} (budget {xmm_budget})\n{}",
            format_overshoot(&phase3, &coloring, gpr_budget, xmm_budget),
        );
    }

    // Map colors to physical registers per class.
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
                RegClass::Flags => None,
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
            // A flags value gets no machine register: the comparison leaves it
            // where its consumer reads it, and lowering emits nothing for the
            // projection naming it.
            RegClass::Flags => None,
        };
        if let Some(r) = reg {
            vreg_to_reg.insert(VReg(idx as u32), r);
        }
    }

    // The VRegs no register exists for: a color at or above the class budget.
    // `map_colors_to_regs` gives them no entry in `vreg_to_reg` either, so
    // without a spill loop they reach lowering with no register at all.
    let mut over_budget: Vec<VReg> = real_vreg_indices
        .iter()
        .copied()
        .filter(|&idx| idx < phase3.graph.num_vregs)
        .filter(|idx| !phase3.pre_coloring_colors.contains_key(idx))
        .filter(|&idx| {
            let Some(color) = coloring.colors.get(idx).copied().flatten() else {
                return false;
            };
            let budget = match phase3.graph.reg_class[idx] {
                RegClass::GPR => gpr_budget,
                RegClass::XMM => xmm_budget,
                // One EFLAGS, and the schedules never put two flags values live
                // at once -- so a colour of 1 or more here is a real conflict
                // and belongs over budget like any other.
                RegClass::Flags => 1,
            };
            color >= budget
        })
        .map(|idx| VReg(idx as u32))
        .collect();
    over_budget.sort_unstable();

    // Compute callee_saved_used.
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
        vreg_to_reg,
        over_budget,
        gpr_overshoot,
        xmm_overshoot,
        unprecolored_params: phase3.unprecolored_params,
        per_block_insts: phase3.per_block_insts,
        alias_map: phase3.alias_map,
    }
}

// ── Phase 5: Register assignment ─────────────────────────────────────────────

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

/// Run Phase 5: global spilling with iterative spill-and-recolor.
///
/// # XMM cross-block-phi audit
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
///
/// `spill_rounds` bounds the allocator's own spill loop. `MAX_GLOBAL_SPILL_ROUNDS`
/// is the working value; **zero makes this a probe** -- colour once and report
/// whether it fits, without spilling anything. That is what lets the caller
/// relieve pressure with the splitter first, which places a value better than
/// this loop does, and fall back to spilling here only when the splitter has
/// nothing left to offer.
///
/// `slots` is the function's slot allocator, shared with the passes that spilled
/// before this one. A spill loop here takes its slots from it: numbering its own
/// from zero would name cells those passes already hold.
pub(crate) fn run_phase5(
    phase4: Phase4State,
    func_name: &str,
    accumulated_aliases: &BTreeMap<u32, u32>,
    slot_resident: &BTreeMap<VReg, u32>,
    slots: &mut SlotAllocator,
) -> Result<GlobalRegAllocResult, String> {
    use crate::x86::abi::CALLEE_SAVED;

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
        // Every round's aliases, not the converging round's. Only the first
        // round coalesces, so a function needing two would otherwise report none
        // and leave every stale `class_to_vreg` entry pointing at a VReg that no
        // longer exists.
        let coalesce_aliases = build_transitive_alias_map(accumulated_aliases);
        // A slot-resident call argument holds no register, and the colouring gave
        // it none: it is excluded from every live set, so it is not a node the
        // graph coloured. Its assignment is the slot, which is what lowering
        // reads to push the argument out of memory.
        let assignment = phase4
            .vreg_to_reg
            .into_iter()
            .map(|(v, r)| (v, super::Assignment::Reg(r)))
            .chain(
                slot_resident
                    .iter()
                    .map(|(&v, &slot)| (v, super::Assignment::Slot(slot))),
            )
            .collect();
        return Ok(GlobalRegAllocResult {
            per_block_insts: phase4.per_block_insts,
            assignment,
            callee_saved_used,
            unprecolored_params: phase4.unprecolored_params,
            coalesce_aliases,
        });
    }

    Err(format!(
        "global regalloc: register pressure overshoot for function '{func_name}' \
         (gpr_overshoot={}, xmm_overshoot={}, spill slots already committed={}). \
         The split pass should have resolved all register pressure before phase 5.",
        phase4.gpr_overshoot,
        phase4.xmm_overshoot,
        slots.count(),
    ))
}

/// Whether any VReg of either class took a color its budget does not have.
fn exceeds_budget(
    coloring: &super::coloring::ColoringResult,
    graph: &InterferenceGraph,
    gpr_budget: u32,
) -> bool {
    coloring
        .colors
        .iter()
        .enumerate()
        .take(graph.num_vregs)
        .any(|(idx, &c)| {
            let Some(color) = c else { return false };
            let budget = match graph.reg_class[idx] {
                RegClass::GPR => gpr_budget,
                RegClass::XMM => super::coloring::AVAILABLE_XMM_COLORS,
                RegClass::Flags => 1,
            };
            color >= budget
        })
}

/// Describe why the coloring needed more colors than the budget.
///
/// The splitter's job is to bring pressure within budget before this point, so
/// an overshoot here means the two models disagree. Max live values is a lower
/// bound on colors, not the count: precolored ABI nodes and clobber phantoms
/// constrain *which* color each neighbour may take, so a set of values that fits
/// in the budget by pressure can still fail to color. This prints each
/// over-budget VReg with the op that defines it, the colors its neighbours
/// already hold, and a clique it belongs to.
///
/// The clique is the part that separates the two causes. Neighbours holding every
/// color below the budget does *not* mean they all conflict with each other, so
/// it is not on its own evidence of real pressure; a found clique larger than the
/// budget is, because no coloring can do better. Below that, the coloring order
/// is the suspect.
/// Size of a clique containing `idx`, found greedily among its neighbours of the
/// same class in descending degree order.
///
/// A lower bound, not the maximum -- that is NP-hard and this runs inside an
/// error path. It is still decisive in one direction: any clique it reports is
/// real, so one larger than the budget proves the coloring could not have
/// succeeded and the pressure has to be split rather than colored better.
fn greedy_clique_containing(phase3: &Phase3State, idx: usize, class: RegClass) -> Vec<usize> {
    let mut candidates: Vec<usize> = phase3.graph.adj[idx]
        .iter()
        .filter(|&n| phase3.graph.reg_class[n] == class)
        .collect();
    candidates.sort_by_key(|&n| std::cmp::Reverse(phase3.graph.adj[n].len()));

    let mut clique = vec![idx];
    for cand in candidates {
        if clique.iter().all(|&m| phase3.graph.adj[cand].contains(m)) {
            clique.push(cand);
        }
    }
    clique
}

fn format_overshoot(
    phase3: &Phase3State,
    coloring: &super::coloring::ColoringResult,
    gpr_budget: u32,
    xmm_budget: u32,
) -> String {
    use std::fmt::Write;

    // VReg index -> defining op, for naming the offenders.
    let mut def_op: BTreeMap<u32, String> = BTreeMap::new();
    for (bi, insts) in phase3.per_block_insts.iter().enumerate() {
        for inst in insts {
            def_op
                .entry(inst.dst.0)
                .or_insert_with(|| format!("b{bi} {:?}", inst.op));
        }
    }

    let mut out = String::new();
    for (idx, &color) in coloring.colors.iter().enumerate() {
        let Some(color) = color else { continue };
        if idx >= phase3.graph.num_vregs {
            continue;
        }
        let class = phase3.graph.reg_class[idx];
        let budget = match class {
            RegClass::GPR => gpr_budget,
            RegClass::XMM => xmm_budget,
            RegClass::Flags => 1,
        };
        if color < budget {
            continue;
        }
        let mut neighbor_colors: Vec<u32> = phase3.graph.adj[idx]
            .iter()
            .filter(|&n| phase3.graph.reg_class[n] == class)
            .filter_map(|n| coloring.colors[n])
            .collect();
        neighbor_colors.sort_unstable();
        neighbor_colors.dedup();
        let precolored = phase3.pre_coloring_colors.contains_key(&idx);
        let clique = greedy_clique_containing(phase3, idx, class);
        let members: Vec<String> = clique
            .iter()
            .map(|&m| match def_op.get(&(m as u32)) {
                Some(d) => format!("v{m}={d}"),
                None => format!("v{m}"),
            })
            .collect();
        writeln!(
            out,
            "  v{idx} {class:?} color={color} precolored={precolored} degree={} \
             clique>={} [{}] neighbor_colors={neighbor_colors:?} def={}",
            phase3.graph.adj[idx].len(),
            clique.len(),
            members.join(", "),
            def_op
                .get(&(idx as u32))
                .map(String::as_str)
                .unwrap_or("<phantom or coalesced away>"),
        )
        .unwrap();
    }
    out
}

/// Which VRegs want the colour of an operand, from the two-address ops.
///
/// See [`super::coloring::ColorHints`]: `lower.rs` emits `mov dst, operand[0]`
/// for every op in [`Op::two_address_src`] whose result did not get its first
/// operand's register, and that copy is created after allocation, so nothing
/// downstream can remove it. Built from the post-coalesce lists, so the VRegs
/// named here are the ones the graph is coloured over.
fn two_address_hints(per_block_insts: &[Vec<ScheduledInst>]) -> super::coloring::ColorHints {
    let mut hints: super::coloring::ColorHints = BTreeMap::new();
    for block in per_block_insts {
        for inst in block {
            if let Some(idx) = inst.op.two_address_src()
                && let Some(&src) = inst.operands.get(idx)
                && src != inst.dst
            {
                hints
                    .entry(inst.dst.0 as usize)
                    .or_default()
                    .push(src.0 as usize);
            }
        }
    }
    hints
}

/// A colouring that did not fit, and the merges to try without it.
struct AllocFailure {
    /// What the caller is told if no retry helps.
    message: String,
    /// Copy pairs, in their pre-coalesce names, whose merge the failed
    /// colouring blames. Empty when nothing it can undo is implicated.
    blame: Vec<(usize, usize)>,
}

/// The copy pairs that built the groups a failed colouring could not place.
///
/// `over_budget` names VRegs in the post-coalesce numbering, so each is the
/// representative of a whole group; every copy with an endpoint in that group is
/// what put a member there. Undoing all of them splits the group back apart,
/// which is coarser than re-colouring its members individually but is the only
/// version whose result the next attempt can simply be re-run to check.
fn blame_coalescing(
    over_budget: &[VReg],
    copy_pairs: &[(VReg, VReg)],
    aliases: &BTreeMap<u32, u32>,
) -> Vec<(usize, usize)> {
    if over_budget.is_empty() {
        return Vec::new();
    }
    let blamed: BTreeSet<u32> = over_budget
        .iter()
        .map(|v| chase_u32(v.0, aliases))
        .collect();
    copy_pairs
        .iter()
        .filter(|(src, dst)| {
            blamed.contains(&chase_u32(src.0, aliases))
                || blamed.contains(&chase_u32(dst.0, aliases))
        })
        .map(|&(src, dst)| (src.0 as usize, dst.0 as usize))
        .collect()
}

/// Colour the function, undoing coalescing where the colouring blames it.
///
/// **This is the "optimistic" half of optimistic coalescing.** `coalesce` merges
/// every copy whose endpoints do not interfere, which is more than the graph can
/// always take: the merged node's neighbourhood is the union of its members'.
/// Where that overflows the register budget, the attempt is thrown away whole
/// and re-run with the implicated merges denied, so the copies come back only on
/// the values that could not do without them.
///
/// Retrying from the original schedules rather than patching the failed attempt
/// is what makes it sound. A merge is not a local edit -- it renames a VReg in
/// every schedule, every block-parameter set and every precoloring, and the
/// spill rounds after it are stated in the post-coalesce numbering. There is no
/// "undo one merge" that leaves the rest of that consistent.
///
/// Bounded, because each attempt is a full allocation of the function and the
/// denylist only grows. A function that still does not fit falls through to the
/// same error the conservative allocator raised, with the same text.
#[allow(clippy::too_many_arguments)]
pub fn allocate_global(
    block_schedules: &[Vec<ScheduledInst>],
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    copy_pairs: &[(VReg, VReg)],
    loop_depths: &BTreeMap<VReg, u32>,
    cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    func_name: &str,
    uses_frame_pointer: bool,
    arg_locs: &[ArgLoc],
    stack_args: &BTreeSet<VReg>,
    slots: &mut SlotAllocator,
    spill_rounds: usize,
) -> Result<GlobalRegAllocResult, String> {
    /// How many times a failed colouring may blame coalescing before the
    /// function is simply reported as not fitting. Each attempt denies at least
    /// one merge, and the corpus has never needed a third.
    const MAX_UNCOALESCE_ATTEMPTS: usize = 4;

    let mut deny: BTreeSet<(usize, usize)> = BTreeSet::new();
    let slot_mark = slots.count();
    let mut last: Option<String> = None;

    for attempt in 0..MAX_UNCOALESCE_ATTEMPTS {
        // Every attempt starts from the frame the caller handed over. A failed
        // one committed slots for spills whose instructions are being discarded
        // with it, and leaving those reserved grows the frame for nothing.
        slots.rollback_to(slot_mark);
        let result = allocate_with_denylist(
            block_schedules,
            param_vregs,
            call_arg_precolors.clone(),
            copy_pairs,
            loop_depths,
            cfg_succs,
            block_param_vregs_per_block,
            func_name,
            uses_frame_pointer,
            arg_locs,
            stack_args,
            slots,
            spill_rounds,
            &deny,
        );
        let failure = match result {
            Ok(ok) => return Ok(ok),
            Err(f) => f,
        };

        let fresh: Vec<(usize, usize)> = failure
            .blame
            .iter()
            .copied()
            .filter(|p| !deny.contains(p))
            .collect();
        last = Some(failure.message);
        if fresh.is_empty() {
            break;
        }
        if crate::trace::is_enabled("coalesce") && crate::trace::fn_matches(func_name) {
            tracing::debug!(
                target: "blitz::coalesce",
                "[{func_name}] attempt {attempt} did not colour; denying {} merge(s) and \
                 retrying with {} denied in total",
                fresh.len(),
                deny.len() + fresh.len(),
            );
        }
        deny.extend(fresh);
    }

    slots.rollback_to(slot_mark);
    // The last attempt is the one the caller is told about, and it is the most
    // constrained -- so the message describes the allocation that came closest.
    let message = last.expect("the loop runs at least once");
    allocate_with_denylist(
        block_schedules,
        param_vregs,
        call_arg_precolors,
        copy_pairs,
        loop_depths,
        cfg_succs,
        block_param_vregs_per_block,
        func_name,
        uses_frame_pointer,
        arg_locs,
        stack_args,
        slots,
        spill_rounds,
        &deny,
    )
    .map_err(|_| message)
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
/// * `block_param_vregs_per_block` - Per-block sets of VRegs that are block parameters
///   (phi destinations); these are excluded from cross-block reload insertion.
/// * `func_name` - Function name used for debug tracing.
/// * `uses_frame_pointer` - When `false`, RBP is allocatable as a general-purpose register.
///
/// `coalesce_deny` is the merges a previous attempt of `allocate_global`'s
/// blamed for not fitting; this function makes one attempt with exactly those
/// withheld.
#[allow(clippy::too_many_arguments)] // The allocator's entry point; bundling these hides what each phase reads.
fn allocate_with_denylist(
    block_schedules: &[Vec<ScheduledInst>],
    param_vregs: &[(VReg, Reg)],
    call_arg_precolors: Vec<(VReg, Reg)>,
    copy_pairs: &[(VReg, VReg)],
    loop_depths: &BTreeMap<VReg, u32>,
    cfg_succs: &[Vec<usize>],
    block_param_vregs_per_block: &[VRegSet],
    func_name: &str,
    uses_frame_pointer: bool,
    arg_locs: &[ArgLoc],
    stack_args: &BTreeSet<VReg>,
    slots: &mut SlotAllocator,
    spill_rounds: usize,
    coalesce_deny: &BTreeSet<(usize, usize)>,
) -> Result<GlobalRegAllocResult, AllocFailure> {
    let mut schedules: Vec<Vec<ScheduledInst>> = block_schedules.to_vec();
    let mut spill_next_vreg: u32 = schedules
        .iter()
        .flatten()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    // A register allocator's contract is that it always succeeds, in the worst
    // case with a great deal of spill code. Before this loop existed the
    // contract here was "the splitter must be perfect": an overshoot was a
    // compile error, which made every splitter change a correctness gamble
    // rather than a quality knob.
    //
    // Nothing changes for a function that colors on the first round, which is
    // every function the splitter already handles -- the loop's body only runs
    // where the old code returned `Err`.
    // The overshoot the previous round left, so a round that does not reduce it
    // can stop rather than run the limit out. Spilling is not always able to
    // help: where the pressure point is one instruction with more operands than
    // the budget has registers -- a terminator passing 28 block arguments, say --
    // every reload lands immediately before that instruction and is live there
    // too, so the overshoot rises instead. Measured on `pressure` seed 15: 3, 9,
    // 12, 15, 18, 21, 23, and then flat for the remaining rounds, having
    // committed 207 slots to buy nothing.
    let mut prev_overshoot: Option<u32> = None;
    // Everything the loop has already spilled. A spilled value is live only
    // from its definition to the store, so spilling it a second time moves
    // nothing and costs another slot; without this the loop can keep picking
    // the cheapest value in the function forever.
    let mut already_spilled: BTreeSet<VReg> = BTreeSet::new();
    // Every coalescing decision made for this function, which the first round
    // makes all of.
    let mut coalesce_aliases: BTreeMap<u32, u32> = BTreeMap::new();
    // Which VRegs are block parameters, in the names the round's schedules use.
    //
    // The schedules a round colours are the previous round's post-coalesce
    // lists, and this set is the one input that cannot be read back off them --
    // a parameter is written by a phi copy on the edge, not by any instruction
    // in the block. So it is renamed as the aliases are made instead. Left in
    // its pre-coalesce names it would name VRegs the schedules no longer
    // mention, `add_block_param_interferences` would draw no edge between two
    // parameters of one block, and the colouring would be free to give them the
    // same register -- which the phi copies then detect as a parallel copy with
    // a repeated destination, one value short.
    let mut block_params_now: Vec<VRegSet> = block_param_vregs_per_block.to_vec();
    // Call arguments taken out of the register file entirely, and the slot each
    // stands for. A call reads every argument at one program point, so a call
    // with more arguments than there are registers cannot be coloured however
    // much is spilled -- see `spill::route_call_args_to_slots`. Populated only
    // after a round has failed, so a function that colours today is untouched.
    let mut slot_resident: BTreeMap<VReg, u32> = BTreeMap::new();
    let mut slot_resident_set = VRegSet::new();
    let mut routed_stack_args = false;
    // Coalescing runs once per function, and "once" cannot be spelled `round == 0`
    // any more: routing the stack arguments through slots does not consume a
    // round, so round 0 can be entered twice. The second entry would coalesce
    // schedules that are already post-coalesce, against copy pairs stated in the
    // pre-coalesce numbering.
    let mut coalesced = false;

    // `round` counts *spill* rounds, and taking the stack arguments out of the
    // register file is not one: it does not choose a value to spill, it removes a
    // register requirement the program never had. Counting it would spend the
    // only round the splitter's probe allows (`spill_rounds` of 0) on the
    // measurement that made it necessary.
    let mut round = 0usize;
    while round <= spill_rounds {
        let block_schedules: &[Vec<ScheduledInst>] = &schedules;

        // What each terminator consumes, off the schedules this round colors.
        //
        // A round rewrites them: `insert_spills_global` rematerializes a value
        // and renames every use, `Op::TerminatorArgs` among them. Carried in
        // from before the loop instead, the set names the pre-spill VReg -- and
        // since it feeds `live_out`, a value whose only remaining mention was
        // that stale entry stays live from the entry block to the last return
        // with no definition anywhere. It then interferes with everything and
        // takes a colour no register exists for, so the round after a
        // successful spill reports a *higher* overshoot than the one before it
        // and the loop stops, blaming a program that in fact fits.
        let phi_uses = crate::compile::barrier::terminator_uses(block_schedules);
        let phi_uses: &[VRegSet] = &phi_uses;

        // Compute function-wide global liveness. Block params are added
        // to their block's live_in so pairs of params on the same block interfere
        // (they're written simultaneously by phi copies and must occupy distinct
        // registers even when the block body never reads them).
        let global_liveness = crate::regalloc::global_liveness::compute_global_liveness_excluding(
            block_schedules,
            cfg_succs,
            phi_uses,
            &block_params_now,
            &slot_resident_set,
        );
        // Also augment the global liveness that run_phase3 will recompute
        // internally: it uses plain `compute_global_liveness` which doesn't know
        // about block params. We pass block_param_vregs_per_block down and augment
        // at each site (see below).

        // The liveness the allocator believes, per block, before anything acts on
        // it. Two values the graph lets share a register are values this dump shows
        // no block holding at once -- so it is where a missing interference edge is
        // read off, against the emitted schedules rather than argued about.
        if crate::trace::is_enabled("liveness") && crate::trace::fn_matches(func_name) {
            for (bi, sched) in block_schedules.iter().enumerate() {
                let fmt = |s: &VRegSet| {
                    let v: Vec<usize> = s.iter().collect();
                    format!("{v:?}")
                };
                tracing::debug!(
                    target: "blitz::liveness",
                    "[{func_name}] block {bi}: succs={:?} live_in={} live_out={} params={}\n{}",
                    cfg_succs[bi],
                    fmt(&global_liveness.live_in[bi]),
                    fmt(&global_liveness.live_out[bi]),
                    fmt(&block_params_now[bi]),
                    crate::trace::format_liveness(
                        sched,
                        &crate::regalloc::liveness::compute_liveness_excluding(
                            sched,
                            &global_liveness.live_out[bi],
                            &slot_resident_set,
                        )
                        .live_at,
                        &global_liveness.live_out[bi],
                    ),
                );
            }
        }

        // Tasks 2.3, 2.4, 2.4.5: Build function-wide interference graph and
        // per-block liveness (stored in Phase2State for Phase 3/5 consumption).
        let mut phase2 =
            build_global_interference(block_schedules, &global_liveness, &slot_resident_set);
        // Pre-coalesce Phase 2 graph: block_params are still distinct VRegs, so no
        // alias resolution needed.
        add_param_interferences(
            &mut phase2.graph,
            &block_params_now,
            block_schedules,
            &BTreeMap::new(),
            arg_locs,
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
            &block_params_now,
            uses_frame_pointer,
            next_vreg,
            !coalesced,
            coalesce_deny,
            &slot_resident_set,
            arg_locs,
        );
        coalesced = true;

        // Phase 4: global coloring and color-to-register mapping.
        let phase4 = run_phase4(phase3, uses_frame_pointer);

        // Coalescing alias map: threaded through from Phase 3. Used in Phase 5 to
        // build the transitive alias map so the caller's ClassId -> VReg resolution
        // (block_class_to_vreg in compile/mod.rs) chases stale `class_to_vreg`
        // entries that still point at pre-coalesce VRegs.
        //
        // Carried across rounds rather than taken from the round that happens to
        // converge: only the first round coalesces, so a function needing two
        // rounds would otherwise report no aliases at all and leave every stale
        // entry pointing at a VReg that no longer exists.
        coalesce_aliases.extend(phase4.alias_map.iter().map(|(&k, &v)| (k, v)));
        let alias_map = coalesce_aliases.clone();

        // The schedules from here on are the post-coalesce ones, so the block
        // parameters have to be named the same way.
        if !phase4.alias_map.is_empty() {
            let resolve = |v: VReg| -> VReg { VReg(chase_u32(v.0, &phase4.alias_map)) };
            for set in block_params_now.iter_mut() {
                *set = set
                    .iter()
                    .map(|v| resolve(VReg(v as u32)).0 as usize)
                    .collect();
            }
        }

        if phase4.gpr_overshoot == 0 && phase4.xmm_overshoot == 0 {
            return run_phase5(phase4, func_name, &alias_map, &slot_resident, slots).map_err(
                |message| AllocFailure {
                    message,
                    blame: Vec::new(),
                },
            );
        }

        // Before spilling: a call argument that goes on the stack needs no
        // register at all, and no amount of spilling can relieve a point where
        // the instruction itself reads more values than there are registers.
        // Taking those out of the register file is the only relief that shape
        // has, so it is tried once, ahead of the first spill round.
        if !routed_stack_args && !stack_args.is_empty() {
            routed_stack_args = true;
            let mut next = spill_next_vreg.max(
                phase4
                    .per_block_insts
                    .iter()
                    .flatten()
                    .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
                    .max()
                    .map(|m| m + 1)
                    .unwrap_or(0),
            );
            let mut next_schedules = phase4.per_block_insts.clone();
            let vreg_classes = super::build_vreg_classes_from_all_blocks(&next_schedules);
            // In the names this round's schedules use: coalescing renamed some of
            // them, and the CFG's list is in the pre-coalesce numbering.
            let targets: BTreeSet<VReg> = stack_args
                .iter()
                .map(|&v| VReg(chase_u32(v.0, &coalesce_aliases)))
                .collect();
            let routed = crate::regalloc::spill::route_call_args_to_slots(
                &mut next_schedules,
                &targets,
                slots,
                &mut next,
                &vreg_classes,
            );
            if !routed.is_empty() {
                if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
                    tracing::debug!(
                        target: "blitz::regalloc",
                        "[{func_name}] routed {} stack call argument(s) through slots after \
                         round {round} (gpr_overshoot={}, xmm_overshoot={})",
                        routed.len(),
                        phase4.gpr_overshoot,
                        phase4.xmm_overshoot,
                    );
                }
                for (&v, &slot) in &routed {
                    slot_resident_set.insert(v.0 as usize);
                    slot_resident.insert(v, slot);
                }
                spill_next_vreg = next;
                schedules = next_schedules;
                continue;
            }
        }

        // Spill and try again. A block parameter is not a candidate: its value
        // is written by a phi copy on the edge, so a store placed in the block
        // would store whatever the register held before the copy. Routing one
        // through a slot is the splitter's `slot_spilled_params`, which decides
        // it before the schedules are built.
        let block_params: BTreeSet<VReg> = block_params_now
            .iter()
            .flatten()
            .map(|v| VReg(v as u32))
            .collect();
        // Nor is a division's pair, for the reason `Op::result_in_fixed_regs`
        // gives: the store would save a register that never held it.
        let fixed_reg_results: BTreeSet<VReg> = phase4
            .per_block_insts
            .iter()
            .flatten()
            .filter(|inst| inst.op.result_in_fixed_regs())
            .map(|inst| inst.dst)
            .collect();
        let class = if phase4.gpr_overshoot > 0 {
            RegClass::GPR
        } else {
            RegClass::XMM
        };
        let overshoot = phase4.gpr_overshoot + phase4.xmm_overshoot;
        let ineligible: BTreeSet<VReg> = block_params
            .iter()
            .chain(fixed_reg_results.iter())
            .chain(already_spilled.iter())
            .copied()
            .collect();
        let candidates = choose_spill_candidates(
            &phase4.per_block_insts,
            cfg_succs,
            &block_params,
            &ineligible,
            class,
            loop_depths,
            &phase4.over_budget,
            &slot_resident_set,
            func_name,
        );

        // A round that leaves the overshoot where it was has still moved: the
        // choice covers every point that overflowed and never repeats a value,
        // so the next round faces a strictly smaller problem. What cannot make
        // progress is a round that raises the overshoot.
        let no_progress = prev_overshoot.is_some_and(|prev| overshoot > prev);
        if round == spill_rounds || candidates.is_empty() || no_progress {
            let why = if no_progress {
                "spilling did not reduce it, so every value live at the pressure \
                 point is one the instruction there reads or one no store can move"
            } else if candidates.is_empty() {
                "nothing live at the pressure point can be spilled: every value \
                 there is read by the instruction, a block parameter, or a result \
                 the hardware pins to a register. A phi seam wider than the \
                 register file has this shape, and only the splitter can relieve \
                 it, by routing a parameter through a slot before the schedules \
                 are built"
            } else {
                "the round limit ran out"
            };
            let peak = pressure_peak(
                &phase4.per_block_insts,
                &global_liveness,
                &super::build_vreg_classes_from_all_blocks(&phase4.per_block_insts),
                class,
                &slot_resident_set,
            )
            .unwrap_or_else(|| "no instructions".to_string());
            let defs = over_budget_defs(&phase4.over_budget, &phase4.per_block_insts);
            // Which values could not be placed, and what defines them. A count
            // says the loop failed; this says what it failed on, which is the
            // difference between "the program is genuinely too wide here" and
            // "the loop kept choosing something that does not help".
            if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
                let over: Vec<String> = phase4
                    .over_budget
                    .iter()
                    .take(24)
                    .map(|v| {
                        let def = phase4
                            .per_block_insts
                            .iter()
                            .flatten()
                            .find(|i| i.dst == *v)
                            .map(|i| format!("{:?}", i.op))
                            .unwrap_or_else(|| "no def".to_string());
                        format!("v{}={def}", v.0)
                    })
                    .collect();
                tracing::debug!(
                    target: "blitz::regalloc",
                    "[{func_name}] over budget: [{}]", over.join(", "),
                );
            }
            return Err(AllocFailure {
                message: format!(
                    "global regalloc: register pressure overshoot for function '{func_name}' \
                     after {round} spill round(s): {why} (gpr_overshoot={}, xmm_overshoot={}, \
                     over-budget VRegs={} defined by {defs}, of which spillable={}, \
                     spill slots committed={}); {peak}",
                    phase4.gpr_overshoot,
                    phase4.xmm_overshoot,
                    phase4.over_budget.len(),
                    candidates.len(),
                    slots.count(),
                ),
                blame: blame_coalescing(&phase4.over_budget, copy_pairs, &coalesce_aliases),
            });
        }
        prev_overshoot = Some(overshoot);
        already_spilled.extend(candidates.iter().map(|&v| VReg(v as u32)));

        // What is about to be spilled, read off the list the coloring ran on --
        // not off the result, where a rematerialized VReg's defining
        // instruction has already been dropped and every candidate reads as
        // undefined.
        let chosen: Vec<String> =
            if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
                candidates
                    .iter()
                    .take(16)
                    .map(|&v| {
                        let def = phase4
                            .per_block_insts
                            .iter()
                            .flatten()
                            .find(|i| i.dst.0 as usize == v)
                            .map(|i| format!("{:?}", i.op))
                            .unwrap_or_else(|| "no def".to_string());
                        format!("v{v}={def}")
                    })
                    .collect()
            } else {
                Vec::new()
            };

        // Spill on the schedules the coloring was computed from, which are
        // Phase 3's post-coalesce lists rather than the ones this round
        // started with: the candidate VRegs are named in that numbering.
        let mut next = schedules
            .iter()
            .chain(phase4.per_block_insts.iter())
            .flatten()
            .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
            .max(spill_next_vreg);
        let mut next_schedules = phase4.per_block_insts;
        let vreg_classes = super::build_vreg_classes_from_all_blocks(&next_schedules);
        crate::regalloc::spill::insert_spills_global(
            &mut next_schedules,
            &candidates,
            slots,
            &mut next,
            &vreg_classes,
        );
        spill_next_vreg = next;
        schedules = next_schedules;

        if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
            // What was spilled, not just how many: a round that raises the
            // overshoot is the loop's own choice going wrong, and the choice is
            // the VReg and the op that defines it.
            tracing::debug!(
                target: "blitz::regalloc",
                "[{func_name}] global spill round {round}: spilled {} VReg(s), \
                 gpr_overshoot={}, xmm_overshoot={}, chose [{}]",
                candidates.len(),
                phase4.gpr_overshoot,
                phase4.xmm_overshoot,
                chosen.join(", "),
            );
        }
        round += 1;
    }

    unreachable!("the spill loop returns before exhausting its rounds")
}

/// Which VRegs to spill when the colouring did not fit.
///
/// Spilling relieves a value that is live *across* a point. It cannot relieve
/// one the instruction at that point reads: the reload lands immediately before
/// that instruction and is live there too, so the pressure is unchanged and a
/// slot has been spent for nothing. The VRegs the colouring failed on are
/// typically exactly those values -- the ones being computed where the pressure
/// is -- which is why choosing the spill set from `over_budget` itself could
/// leave the loop spilling the same useless value every round and then report
/// that the program does not fit.
///
/// So `over_budget` names the *trouble*, not the cure. Every instruction where
/// one of those VRegs is live is a point the colouring could not satisfy, and
/// the values live there that the instruction does not itself read or write are
/// what spilling can actually remove. Taking the points from the colouring
/// rather than from a live-count threshold is what makes this hold for an
/// instruction that pins registers of its own: `idiv` keeps the dividend in RAX
/// and clobbers RDX, so it overflows with two live values fewer than the budget
/// and no count of what is live can see it, but the VReg left uncoloured is
/// still live exactly there.
///
/// Among the candidates, cheapest first: Chaitin's ratio, the loop-weighted
/// count of references divided by how many such points the value crosses, so a
/// value that relieves many points for few reloads goes first.
#[allow(clippy::too_many_arguments)]
fn choose_spill_candidates(
    block_schedules: &[Vec<ScheduledInst>],
    cfg_succs: &[Vec<usize>],
    block_params: &BTreeSet<VReg>,
    ineligible: &BTreeSet<VReg>,
    class: RegClass,
    loop_depths: &BTreeMap<VReg, u32>,
    over_budget: &[VReg],
    slot_resident: &VRegSet,
    func_name: &str,
) -> BTreeSet<usize> {
    let classes = super::build_vreg_classes_from_all_blocks(block_schedules);
    let of_class = |v: &VReg| classes.get(v).copied() == Some(class);
    let uncolored: BTreeSet<VReg> = over_budget.iter().copied().collect();

    let phi_uses = crate::compile::barrier::terminator_uses(block_schedules);
    let block_param_sets: Vec<VRegSet> = block_schedules
        .iter()
        .map(|sched| {
            sched
                .iter()
                .map(|inst| inst.dst)
                .filter(|v| block_params.contains(v))
                .map(|v| v.0 as usize)
                .collect()
        })
        .collect();
    let liveness = crate::regalloc::global_liveness::compute_global_liveness_with_block_params(
        block_schedules,
        cfg_succs,
        &phi_uses,
        &block_param_sets,
    );

    // The points the colouring could not satisfy, and which candidates cross
    // each one without being read there -- the values a spill can remove from it.
    let mut relief: BTreeMap<(usize, usize), BTreeSet<VReg>> = BTreeMap::new();
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        let live_out = liveness
            .live_out
            .get(block_idx)
            .cloned()
            .unwrap_or_default();
        let per_inst =
            crate::regalloc::liveness::compute_liveness_excluding(sched, &live_out, slot_resident);
        for (i, inst) in sched.iter().enumerate() {
            if !per_inst.live_at[i]
                .iter()
                .any(|idx| uncolored.contains(&VReg(idx as u32)))
            {
                continue;
            }
            let touched: BTreeSet<VReg> = inst
                .operands
                .iter()
                .copied()
                .chain(std::iter::once(inst.dst))
                .collect();
            let here: BTreeSet<VReg> = per_inst.live_at[i]
                .iter()
                .map(|idx| VReg(idx as u32))
                .filter(of_class)
                .filter(|v| !touched.contains(v) && !ineligible.contains(v))
                .collect();
            relief.insert((block_idx, i), here);
        }
    }

    // Loop-weighted references per candidate: what the spill costs to execute.
    let candidates: BTreeSet<VReg> = relief.values().flatten().copied().collect();
    let mut refs: BTreeMap<VReg, u64> = BTreeMap::new();
    for inst in block_schedules.iter().flatten() {
        for v in inst
            .operands
            .iter()
            .copied()
            .chain(std::iter::once(inst.dst))
        {
            if !candidates.contains(&v) {
                continue;
            }
            let depth = loop_depths.get(&v).copied().unwrap_or(0).min(16);
            *refs.entry(v).or_insert(0) += 1u64 << depth;
        }
    }

    let mut uncovered: BTreeSet<(usize, usize)> = relief
        .iter()
        .filter(|(_, here)| !here.is_empty())
        .map(|(&point, _)| point)
        .collect();
    // A point with nothing to take is the shape that ends the loop, so name a
    // few: it is the difference between a program too wide to compile and a
    // choice this pass could have made and did not.
    if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
        let barren: Vec<String> = relief
            .iter()
            .filter(|(_, here)| here.is_empty())
            .take(8)
            .map(|(&(b, i), _)| format!("b{b}[{i}]={:?}", block_schedules[b][i].op))
            .collect();
        tracing::debug!(
            target: "blitz::regalloc",
            "[{func_name}] spill choice: {} uncolourable point(s), {} relievable, \
             {} candidate(s); nothing to take at [{}]",
            relief.len(),
            uncovered.len(),
            candidates.len(),
            barren.join(", "),
        );
    }
    // One value taken out of one point is not enough: several points can be
    // uncolourable at once and a value only relieves the ones it crosses. So
    // cover them, cheapest-per-point first -- Chaitin's ratio, the loop-weighted
    // cost of the spill over the number of still-uncovered points it relieves.
    // A round then lowers the pressure at *every* point that overflowed, which
    // is what makes the next round's overshoot a verdict on the program rather
    // than on which value this one happened to pick.
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    while !uncovered.is_empty() {
        let best = candidates
            .iter()
            .filter(|v| !chosen.contains(&(v.0 as usize)))
            .filter_map(|&v| {
                let covers = uncovered
                    .iter()
                    .filter(|point| relief[point].contains(&v))
                    .count() as u64;
                if covers == 0 {
                    return None;
                }
                let cost = refs.get(&v).copied().unwrap_or(1);
                // Scaled so the division keeps its ordering in integers, and the
                // VReg index breaks ties so the choice does not move with hash order.
                Some((cost * 1024 / covers, v.0, v))
            })
            .min();
        let Some((_, _, v)) = best else { break };
        chosen.insert(v.0 as usize);
        uncovered.retain(|point| !relief[point].contains(&v));
    }
    chosen
}

/// An op's variant path without its payload: `Pure(Iconst(5, I32))` reads as
/// `Pure(Iconst)`, so every value defined by the same kind of op groups under
/// one name.
fn op_kind(op: &Op) -> String {
    let debug = format!("{op:?}");
    let mut kind = String::new();
    let mut open = 0usize;
    for c in debug.chars() {
        if c == '(' {
            if open == 1 {
                break;
            }
            open += 1;
            kind.push(c);
        } else if c.is_alphanumeric() || c == '_' {
            kind.push(c);
        } else {
            break;
        }
    }
    for _ in 0..open {
        kind.push(')');
    }
    kind
}

/// The three kinds of op that define the most over-budget VRegs, with counts.
///
/// The count alone says the coloring failed; this says what it failed on.
/// `Pure(BlockParam)x21` is the block-parameter wall, which only the splitter
/// can relieve, and reads nothing like a spill loop that kept choosing values
/// that do not help.
fn over_budget_defs(over_budget: &[VReg], per_block_insts: &[Vec<ScheduledInst>]) -> String {
    if over_budget.is_empty() {
        return "nothing".to_string();
    }
    let mut defs: BTreeMap<VReg, &Op> = BTreeMap::new();
    for inst in per_block_insts.iter().flatten() {
        defs.entry(inst.dst).or_insert(&inst.op);
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for v in over_budget {
        let kind = defs
            .get(v)
            .map_or_else(|| "NoDef".to_string(), |op| op_kind(op));
        *counts.entry(kind).or_default() += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .iter()
        .take(3)
        .map(|(kind, n)| format!("{kind}x{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Where the most values of one class are live at once, and what instruction is
/// there.
///
/// The question the spill loop's failure has to answer: spilling relieves a
/// value that is live *across* a point, and cannot relieve one the instruction
/// at that point is reading. So the report names the instruction, how many of
/// its own operands are of the class that overflowed, and how many values are
/// live there -- which is what separates "too many values in flight" from "one
/// instruction wants more registers than exist".
fn pressure_peak(
    block_schedules: &[Vec<ScheduledInst>],
    global_liveness: &crate::regalloc::global_liveness::GlobalLiveness,
    vreg_classes: &BTreeMap<VReg, RegClass>,
    class: RegClass,
    slot_resident: &VRegSet,
) -> Option<String> {
    let of_class = |v: &&VReg| vreg_classes.get(*v).copied() == Some(class);
    let mut best: Option<(usize, usize, usize, &ScheduledInst)> = None;
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        let live_out = global_liveness
            .live_out
            .get(block_idx)
            .cloned()
            .unwrap_or_default();
        let liveness =
            crate::regalloc::liveness::compute_liveness_excluding(sched, &live_out, slot_resident);
        for (i, inst) in sched.iter().enumerate() {
            let live = liveness.live_at[i]
                .iter()
                .map(|idx| VReg(idx as u32))
                .filter(|v| of_class(&v))
                .count();
            if best.is_none_or(|(b, _, _, _)| live > b) {
                let operands = inst.operands.iter().filter(of_class).count();
                best = Some((live, operands, block_idx, inst));
            }
        }
    }
    let (live, operands, block_idx, inst) = best?;
    Some(format!(
        "peak {live} {class:?} value(s) live at block {block_idx}'s {:?}, which reads \
         {operands} of them as its own operands ({} operands in total)",
        inst.op,
        inst.operands.len(),
    ))
}

/// How many times the global allocator will spill and recolor before giving up.
///
/// Each round shatters a value's live range into one store and one reload per
/// use, so pressure at the point that overflowed strictly falls; the limit is a
/// backstop against a shape that does not converge, not a budget the allocator
/// is expected to use.
pub const MAX_GLOBAL_SPILL_ROUNDS: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;
    use crate::regalloc::global_liveness::compute_global_liveness;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn iconst_inst(dst: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Iconst(dst as i64, Type::I64)),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn use_inst(dst: u32, src: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Pure(PureOp::Proj0),
            dst: VReg(dst),
            operands: vec![VReg(src)],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::Mach(MachOp::X86Add),
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    #[test]
    fn op_kind_drops_the_payload() {
        assert_eq!(
            op_kind(&Op::Pure(PureOp::Iconst(5, Type::I64))),
            "Pure(Iconst)"
        );
        assert_eq!(
            op_kind(&Op::Pure(PureOp::BlockParam(0, 0, Type::I64))),
            "Pure(BlockParam)"
        );
        assert_eq!(op_kind(&Op::Mach(MachOp::X86Add)), "Mach(X86Add)");
    }

    #[test]
    fn over_budget_defs_ranks_by_count() {
        let insts = vec![vec![
            iconst_inst(0),
            iconst_inst(1),
            add_inst(2, 0, 1),
            add_inst(3, 0, 1),
            add_inst(4, 0, 1),
        ]];
        let over: Vec<VReg> = (0..5).map(VReg).collect();
        assert_eq!(
            over_budget_defs(&over, &insts),
            "Mach(X86Add)x3, Pure(Iconst)x2"
        );
    }

    #[test]
    fn over_budget_defs_names_undefined_vregs() {
        assert_eq!(over_budget_defs(&[], &[]), "nothing");
        assert_eq!(over_budget_defs(&[VReg(9)], &[]), "NoDefx1");
    }

    fn empty_phi_uses(n: usize) -> Vec<VRegSet> {
        vec![VRegSet::new(); n]
    }

    /// Build a `Phase2State` for a test CFG.
    fn phase2_for(
        block_schedules: &[Vec<ScheduledInst>],
        successors: &[Vec<usize>],
    ) -> Phase2State {
        let phi_uses = empty_phi_uses(block_schedules.len());
        let global_liveness = compute_global_liveness(block_schedules, successors, &phi_uses);
        build_global_interference(block_schedules, &global_liveness, &VRegSet::new())
    }

    fn interfere(state: &Phase2State, a: u32, b: u32) -> bool {
        let ai = a as usize;
        let bi = b as usize;
        if ai >= state.graph.num_vregs || bi >= state.graph.num_vregs {
            return false;
        }
        state.graph.adj[ai].contains(bi)
    }

    fn interfere_in(graph: &InterferenceGraph, a: u32, b: u32) -> bool {
        let ai = a as usize;
        let bi = b as usize;
        if ai >= graph.num_vregs || bi >= graph.num_vregs {
            return false;
        }
        graph.adj[ai].contains(bi)
    }

    // ── Two-block straight-line CFG ────────────────────────────────
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

    // ── Diamond CFG ─────────────────────────────────────────────────
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

    // ── Loop CFG ────────────────────────────────────────────────────
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
            global_liveness.live_out[0].contains(0),
            "v0 must be live_out of block 0 (live across back edge)"
        );
        assert!(
            global_liveness.live_in[1].contains(0),
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
            !global_liveness.live_out[1].contains(2),
            "v2 (body-local) must NOT be live_out of block 1"
        );
    }

    // ── Coalescing reduces VReg count on a copy-pair program ──────
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());

        // v0 and v1 should NOT interfere (different blocks, no overlap).
        assert!(
            !interfere_in(&phase2.graph, 0, 1),
            "v0 and v1 in separate blocks with no overlap must not interfere"
        );

        // Apply coalescing with copy pair (v0, v1).
        let pairs = [(0usize, 1usize)]; // v0 is src, v1 is dst
        let coalesced = coalesce(&phase2.graph, &pairs, &BTreeSet::new());

        // At least one merge should occur since v0 and v1 don't interfere.
        assert!(
            !coalesced.is_empty(),
            "coalescing non-interfering (v0, v1) should produce at least one merge"
        );

        // Apply coalescing to each block's schedule.
        let coalesce_aliases_table = CoalesceAliases::new(&coalesced);
        let post_coalesce: Vec<Vec<ScheduledInst>> = block_schedules
            .iter()
            .map(|sched| coalesce_aliases_table.apply(sched))
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

    // ── Param precolor dropped when call clobbers ABI reg ─────────
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
        // We model the call as Op::Pseudo(PseudoOp::VoidCallBarrier) with v0 as an operand.
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
            op: Op::Pseudo(PseudoOp::VoidCallBarrier),
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());

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
            &Vec::<VRegSet>::new(),
            false, // uses_frame_pointer
            10,    // next_vreg start
            true,  // coalesce_now
            &BTreeSet::new(),
            &VRegSet::new(),
            &[],
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

    // ── Post-rebuild graph has phantoms, pre-coalesce does not ────
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
            op: Op::Pseudo(PseudoOp::VoidCallBarrier),
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());

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
            &Vec::<VRegSet>::new(),
            false,
            next_vreg,
            true, // coalesce_now,
            &BTreeSet::new(),
            &VRegSet::new(),
            &[],
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());

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
            &Vec::<VRegSet>::new(),
            false,
            next_vreg,
            true, // coalesce_now,
            &BTreeSet::new(),
            &VRegSet::new(),
            &[],
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());

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
            &Vec::<VRegSet>::new(),
            false,
            next_vreg,
            true, // coalesce_now,
            &BTreeSet::new(),
            &VRegSet::new(),
            &[],
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
        let phase2 = build_global_interference(&block_schedules, &global_liveness, &VRegSet::new());
        let next_vreg = phase2.graph.num_vregs as u32;
        run_phase3(
            phase2,
            block_schedules,
            param_vregs,
            call_arg_precolors,
            copy_pairs,
            successors,
            &Vec::<VRegSet>::new(),
            uses_frame_pointer,
            next_vreg,
            true,
            &BTreeSet::new(),
            &VRegSet::new(),
            &[],
        )
    }

    // ── Coloring ──────────────────────────────────────────────────────────────

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
            op: Op::Mach(MachOp::X86Add),
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
            op: Op::Mach(MachOp::X86Add),
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

        // Read off the assignment, which is what `run_phase5` does to build the
        // list the prologue preserves. Asking the assignment rather than a field
        // means the property is checked against what the allocator decided.
        use crate::x86::abi::CALLEE_SAVED;
        let callee_saved_used: Vec<Reg> = phase4
            .vreg_to_reg
            .values()
            .copied()
            .filter(|r| CALLEE_SAVED.contains(r))
            .collect();

        // With 10 simultaneously live values and 9 caller-saved GPRs (excl. RSP),
        // at least one value must land in a callee-saved register.
        assert!(
            !callee_saved_used.is_empty(),
            "10 simultaneously live GPRs must force at least one callee-saved register, \
             got assignment = {:?}",
            phase4.vreg_to_reg
        );
    }

    // ── Phase 5 unit tests ────────────────────────────────────────────────────

    /// Helper: run the full allocator pipeline (Phases 2–5) via `allocate_global`.
    #[allow(clippy::too_many_arguments)]
    fn run_allocate_global(
        block_schedules: &[Vec<ScheduledInst>],
        cfg_succs: &[Vec<usize>],
        param_vregs: &[(VReg, Reg)],
        call_arg_precolors: Vec<(VReg, Reg)>,
        copy_pairs: &[(VReg, VReg)],
        loop_depths: &BTreeMap<VReg, u32>,
        uses_frame_pointer: bool,
        slots: &mut SlotAllocator,
    ) -> GlobalRegAllocResult {
        let n = block_schedules.len();
        let block_param_vregs: Vec<VRegSet> = vec![VRegSet::new(); n];
        allocate_global(
            block_schedules,
            param_vregs,
            call_arg_precolors,
            copy_pairs,
            loop_depths,
            cfg_succs,
            &block_param_vregs,
            "test_fn",
            uses_frame_pointer,
            &[],
            &BTreeSet::new(),
            slots,
            MAX_GLOBAL_SPILL_ROUNDS,
        )
        .expect("allocate_global should succeed")
    }

    // ── Spilling ──────────────────────────────────────────────────────────────
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
        let mut slots = SlotAllocator::new();
        let result = run_allocate_global(
            &block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            &BTreeMap::new(),
            false,
            &mut slots,
        );

        // No spills: every real VReg must have a physical register.
        assert_eq!(
            slots.count(),
            0,
            "no spill slots expected for low-pressure function"
        );
        for idx in 0u32..=3 {
            assert!(
                result.assignment.contains_key(&VReg(idx)),
                "vreg_to_reg must contain VReg v{idx}"
            );
        }
        // The schedules should be unchanged (no spill/reload instructions).
        assert_eq!(result.per_block_insts[0].len(), 3);
        assert_eq!(result.per_block_insts[1].len(), 1);
    }

    // ── An XMM value crossing a block boundary through a phi ──────────────────
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
    // This is the case `run_phase5`'s `# XMM cross-block-phi audit` covers, and
    // what it concludes: phi-copy emission is
    // reg-to-reg movsd and the global allocator assigns XMM registers across
    // block boundaries correctly without forced-slot pre-spilling.
    #[test]
    fn xmm_cross_block_phi_allocates() {
        // Simulate an XMM VReg flowing across blocks via a phi.
        //
        // We use Op::Mach(MachOp::X86Addsd) as an FP op (classified as XMM by build_vreg_classes).
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
            op: Op::Mach(MachOp::X86Addsd),
            dst: VReg(1),
            operands: vec![VReg(0), VReg(0)],
        };
        let xmm_use = ScheduledInst {
            op: Op::Mach(MachOp::X86Addsd),
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

        let mut slots = SlotAllocator::new();
        let result = run_allocate_global(
            &block_schedules,
            &successors,
            &[],
            vec![],
            &[],
            &BTreeMap::new(),
            false,
            &mut slots,
        );

        // Must not panic (already verified by not crashing).
        // XMM VReg v1 must have a physical XMM register (no call on the path
        // means XMM phantoms are never injected -> v1 is freely assignable).
        assert!(
            result.assignment.contains_key(&VReg(1)),
            "XMM VReg v1 must have a physical register assignment when no call is on the path"
        );
        let assigned_reg = result.assignment[&VReg(1)].reg().unwrap();
        assert!(
            assigned_reg.is_xmm(),
            "XMM VReg v1 must be assigned an XMM register, got {assigned_reg:?}"
        );

        // No spills expected: only 1 XMM value, budget is 16 XMM registers.
        assert_eq!(
            slots.count(),
            0,
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
            op: Op::Pseudo(PseudoOp::VoidCallBarrier),
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
        let block_param_vregs: Vec<VRegSet> = vec![VRegSet::new(); n];

        let result = allocate_global(
            &block_schedules,
            &[],
            call_arg_precolors,
            &[],
            &BTreeMap::new(),
            &successors,
            &block_param_vregs,
            "test_many_args",
            false,
            &[],
            &BTreeSet::new(),
            &mut SlotAllocator::new(),
            MAX_GLOBAL_SPILL_ROUNDS,
        )
        .expect("allocate_global must succeed with 8 call args (6 precolored, 2 unprecolored)");

        // All 9 VRegs (v0..v8) must get a physical register.
        for v in 0u32..9 {
            assert!(
                result.assignment.contains_key(&VReg(v)),
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
                    .assignment
                    .get(&VReg(v))
                    .and_then(|a| a.reg())
                    .map(|r| abi_set.contains(&r))
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
                op: Op::Pure(PureOp::Iconst(val, Type::I64)),
                dst: VReg(dst),
                operands: vec![],
            }
        }
        fn block_param(dst: u32, bid: u32, idx: u32) -> ScheduledInst {
            ScheduledInst {
                op: Op::Pure(PureOp::BlockParam(bid, idx, Type::I64)),
                dst: VReg(dst),
                operands: vec![],
            }
        }

        // A terminator's arguments are the operands of its `TerminatorArgs`,
        // which is where the allocator reads them from: `phi_uses` is derived,
        // not supplied, so a schedule without one passes no arguments at all.
        fn terminator_args(dst: u32, args: &[u32]) -> ScheduledInst {
            ScheduledInst {
                op: Op::Pseudo(PseudoOp::TerminatorArgs((0..args.len() as u32).collect())),
                dst: VReg(dst),
                operands: args.iter().map(|&v| VReg(v)).collect(),
            }
        }

        let block_schedules = vec![
            // block 0: three iconsts fed as phi sources to block 1.
            vec![
                iconst(0, 1),
                iconst(1, 5),
                iconst(2, 0),
                terminator_args(9, &[0, 1, 2]),
            ],
            // block 1: three block params; sub to exercise them; feed block 2.
            vec![
                block_param(3, 1, 0),
                block_param(4, 1, 1),
                block_param(5, 1, 2),
                ScheduledInst {
                    op: Op::Mach(MachOp::X86Sub),
                    dst: VReg(6),
                    operands: vec![VReg(3), VReg(4)],
                },
                terminator_args(10, &[6]),
            ],
            // block 2: single block param, use it.
            vec![
                block_param(7, 2, 0),
                ScheduledInst {
                    op: Op::Mach(MachOp::X86Sub),
                    dst: VReg(8),
                    operands: vec![VReg(7), VReg(7)],
                },
            ],
        ];
        let cfg_succs = vec![vec![1usize], vec![2usize], vec![]];
        let mut block_param_vregs: Vec<VRegSet> = vec![VRegSet::new(); 3];
        block_param_vregs[1] = [3usize, 4, 5].into_iter().collect();
        block_param_vregs[2].insert(7);
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
            &block_param_vregs,
            "three_phi_params",
            false,
            &[],
            &BTreeSet::new(),
            &mut SlotAllocator::new(),
            MAX_GLOBAL_SPILL_ROUNDS,
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
            result
                .assignment
                .get(&cur)
                .and_then(|a| a.reg())
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
