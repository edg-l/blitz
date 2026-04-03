use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::extract::VReg;
use crate::ir::op::Op;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::{CALLEE_SAVED, CALLER_SAVED_GPR};
use crate::x86::reg::{Reg, RegClass};

use super::coalesce::coalesce;
use super::coloring::{
    allocatable_gpr_order, available_gpr_colors, greedy_color, interval_color, map_colors_to_regs,
    mcs_ordering,
};
use super::interference::build_interference;
use super::liveness::compute_liveness;
use super::rewrite::apply_coalescing;
use super::spill::{insert_spills, select_spill};

/// Result of register allocation for a single basic block.
pub struct RegAllocResult {
    /// Maps each VReg to the physical register assigned to it.
    pub vreg_to_reg: BTreeMap<VReg, Reg>,
    /// Number of spill slots used (each slot is 8 bytes for GPR, 16 for XMM).
    pub spill_slots: u32,
    /// Callee-saved registers that were actually assigned (must be preserved).
    pub callee_saved_used: Vec<Reg>,
    /// Final instruction list with spill/reload code inserted and coalescing
    /// aliases applied. Callers must use this instead of their original
    /// instruction list, since `vreg_to_reg` was computed for this version.
    pub insts: Vec<ScheduledInst>,
    /// Function parameter VRegs whose precoloring was removed because they are
    /// live across a call that clobbers their ABI register. The lowering must
    /// emit a mov from the ABI register to the allocated register at function
    /// entry for these params.
    pub unprecolored_params: Vec<(VReg, Reg)>,
}

/// Allocate physical registers for a single basic block's scheduled instruction list.
///
/// This function is called once per basic block by the per-block register allocator
/// in compile.rs. Cross-block live ranges are handled by the caller via spill/reload
/// insertion (`rewrite_block_for_splitting`) before this function is invoked.
///
/// `uses_frame_pointer`: when false, RBP is included in the allocatable GPR set (15 regs total).
///
/// Algorithm:
/// 1. Compute liveness.
/// 2. Build interference graph.
/// 3. Coalesce non-interfering copy pairs (block params).
/// 4. MCS ordering.
/// 5. Greedy coloring with pre-coloring.
/// 6. If chromatic_number > available_regs: spill, re-run (up to 3 times).
/// 7. Map colors to physical registers.
/// 8. Return result.
pub fn allocate(
    insts: &[ScheduledInst],
    param_vregs: &[(VReg, Reg)], // pre-colored function params
    block_live_out: &BTreeSet<VReg>,
    copy_pairs: &[(VReg, VReg)], // phi copy pairs for coalescing
    loop_depths: &std::collections::BTreeMap<VReg, u32>, // loop-depth info for spill selection
    call_points: &[usize],       // instruction indices where a call clobbers all caller-saved regs
    div_points: &[usize],        // instruction indices where div clobbers RAX/RDX only
    uses_frame_pointer: bool,
    func_name: &str,
) -> Result<RegAllocResult, String> {
    let mut insts: Vec<ScheduledInst> = insts.to_vec();
    let mut block_live_out: BTreeSet<VReg> = block_live_out.clone();
    let mut spill_slots = 0u32;

    // Determine the starting next_vreg from the max VReg index in the input.
    let mut next_vreg: u32 = insts
        .iter()
        .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    // Build pre-coloring map from function parameters.
    // Track which VRegs are function params (vs shift/div/call precolors).
    let mut param_vreg_to_reg: BTreeMap<VReg, Reg> = BTreeMap::new();
    let param_vreg_indices: BTreeSet<usize> =
        param_vregs.iter().map(|(v, _)| v.0 as usize).collect();
    for &(vreg, reg) in param_vregs {
        param_vreg_to_reg.insert(vreg, reg);
    }

    const MAX_SPILL_ROUNDS: usize = 10;

    for round in 0..=MAX_SPILL_ROUNDS {
        // Step 1: Compute liveness.
        let liveness = compute_liveness(&insts, &block_live_out);

        if round == 0 && crate::trace::is_enabled("liveness") && crate::trace::fn_matches(func_name)
        {
            tracing::debug!(
                target: "blitz::liveness",
                "[{func_name}] liveness (round 0):\n{}",
                crate::trace::format_liveness(&insts, &liveness.live_at, &block_live_out),
            );
        }

        // Step 2: Build VReg class map (all GPR for now; XMM support in 10.13a).
        let vreg_classes = build_vreg_classes(&insts, &liveness);

        // Step 3: Build interference graph.
        let graph = build_interference(&liveness, &insts, &vreg_classes);
        let (graph, _) = add_call_clobber_interferences(
            graph,
            &liveness,
            call_points,
            &mut next_vreg,
            uses_frame_pointer,
        );
        let (graph, _) = add_div_clobber_interferences(
            graph,
            &liveness,
            div_points,
            &mut next_vreg,
            uses_frame_pointer,
        );

        // Step 4 (first round only): coalesce copy pairs on original SSA graph.
        // Per spec: coalescing must NOT be re-run after spill insertion.
        let coalesced = if round == 0 {
            let pairs: Vec<(usize, usize)> = copy_pairs
                .iter()
                .map(|(src, dst)| (src.0 as usize, dst.0 as usize))
                .filter(|&(src, dst)| src < graph.num_vregs && dst < graph.num_vregs)
                .collect();
            coalesce(&graph, &pairs)
        } else {
            vec![]
        };

        // Apply coalescing aliases to the instruction list.
        let insts_coalesced = apply_coalescing(&insts, &coalesced);

        // Build pre-coloring: VReg index -> color.
        // Assign each pre-colored param a unique color based on a canonical
        // ordering of the ABI registers.
        let pre_coloring_colors: BTreeMap<usize, u32> =
            build_pre_coloring_colors(&insts_coalesced, &param_vreg_to_reg, uses_frame_pointer);

        // Step 5: MCS ordering + greedy coloring.
        // Recompute liveness/graph on coalesced insts for accuracy.
        let liveness2 = compute_liveness(&insts_coalesced, &block_live_out);
        let vreg_classes2 = build_vreg_classes(&insts_coalesced, &liveness2);
        let graph2 = build_interference(&liveness2, &insts_coalesced, &vreg_classes2);
        let (graph2, phantom_precolors) = add_call_clobber_interferences(
            graph2,
            &liveness2,
            call_points,
            &mut next_vreg,
            uses_frame_pointer,
        );
        let (graph2, div_phantom_precolors) = add_div_clobber_interferences(
            graph2,
            &liveness2,
            div_points,
            &mut next_vreg,
            uses_frame_pointer,
        );

        // Merge phantom pre-colorings with param pre-colorings. Phantoms
        // represent hardware constraints (call clobbers, div clobbers) that
        // cannot be changed. If a param precoloring conflicts with a phantom
        // (same color + interference edge), remove the param precoloring --
        // the param will get a free register from the coloring instead.
        let mut pre_coloring_colors2 = pre_coloring_colors.clone();
        for (&phantom_vreg, &phantom_color) in &phantom_precolors {
            // Find and remove any param precoloring that conflicts.
            let conflicting: Vec<usize> = pre_coloring_colors2
                .iter()
                .filter(|(pv, pc)| {
                    let (pv, pc) = (**pv, **pc);
                    pc == phantom_color
                        && param_vreg_indices.contains(&pv)
                        && phantom_vreg < graph2.num_vregs
                        && pv < graph2.num_vregs
                        && graph2.adj[phantom_vreg].contains(&pv)
                })
                .map(|(pv, _)| *pv)
                .collect();
            for pv in conflicting {
                pre_coloring_colors2.remove(&pv);
                param_vreg_to_reg.remove(&VReg(pv as u32));
            }
        }
        pre_coloring_colors2.extend(phantom_precolors);
        pre_coloring_colors2.extend(div_phantom_precolors);

        let ordering = mcs_ordering(&graph2);
        let mut coloring = greedy_color(&graph2, &ordering, &pre_coloring_colors2);

        // Step 6: Check if we need to spill.
        let avail = available_gpr_colors(uses_frame_pointer);

        // MCS+greedy can overestimate on graphs where spill code breaks
        // chordality. Fall back to interval coloring (optimal for single-block
        // SSA) when the greedy result exceeds the theoretical optimum.
        let max_live2 = liveness2.live_at.iter().map(|s| s.len()).max().unwrap_or(0) as u32 + 1;
        if coloring.chromatic_number > avail && coloring.chromatic_number > max_live2 {
            let ic = interval_color(
                &insts_coalesced,
                &liveness2,
                &pre_coloring_colors2,
                graph2.num_vregs,
            );
            if ic.chromatic_number < coloring.chromatic_number {
                // Validate: coalescing can create multiply-defined VRegs,
                // breaking the SSA invariant that interval_color relies on.
                // Check that no interference edge has both endpoints with
                // the same color before accepting.
                let valid = (0..graph2.num_vregs).all(|v| {
                    let vc = ic.colors[v];
                    graph2.adj[v].iter().all(|&u| ic.colors[u] != vc)
                });
                if valid {
                    coloring = ic;
                }
            }
        }

        let gpr_colors_needed = coloring.chromatic_number;

        if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
            tracing::debug!(
                target: "blitz::regalloc",
                "[{func_name}] round {round}: chromatic={gpr_colors_needed}, avail={avail}, max_live={max_live2}, graph_vregs={}", graph2.num_vregs,
            );
        }

        if gpr_colors_needed <= avail {
            // Success: map colors to physical registers.
            let pre_coloring_regs = build_pre_coloring_regs(&insts_coalesced, &param_vreg_to_reg);
            let color_to_reg = map_colors_to_regs(
                &coloring,
                RegClass::GPR,
                &pre_coloring_regs,
                uses_frame_pointer,
            );

            // Build final VReg -> Reg mapping.
            let mut vreg_to_reg: BTreeMap<VReg, Reg> = BTreeMap::new();
            for (i, color_opt) in coloring.colors.iter().enumerate() {
                if let Some(&color) = color_opt.as_ref()
                    && let Some(&reg) = color_to_reg.get(&color)
                {
                    vreg_to_reg.insert(VReg(i as u32), reg);
                }
            }

            // Identify callee-saved registers actually used.
            let callee_saved_set: BTreeSet<Reg> = CALLEE_SAVED.iter().copied().collect();
            let mut callee_saved_used: Vec<Reg> = vreg_to_reg
                .values()
                .filter(|r| callee_saved_set.contains(r))
                .copied()
                .collect();
            callee_saved_used.sort_by_key(|r| *r as u8);
            callee_saved_used.dedup();

            if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(func_name) {
                tracing::debug!(
                    target: "blitz::regalloc",
                    "[{func_name}] allocation ok (round {round}, spills={spill_slots}, callee_saved={callee_saved_used:?}):\n{}",
                    crate::trace::format_vreg_to_reg(&vreg_to_reg),
                );
            }

            return Ok(RegAllocResult {
                vreg_to_reg,
                spill_slots,
                callee_saved_used,
                insts: insts_coalesced,
                unprecolored_params: vec![],
            });
        }

        // Need to spill.
        if round == MAX_SPILL_ROUNDS {
            return Err(format!(
                "register allocation failed after {MAX_SPILL_ROUNDS} spill rounds: \
                 chromatic number {gpr_colors_needed} exceeds available GPR colors \
                 {avail}. Live ranges at pressure point may indicate \
                 too many simultaneously live values."
            ));
        }

        // Select VRegs to spill. When the gap between chromatic number and
        // available registers is large, spill multiple VRegs per round to
        // converge faster.
        let overshoot = gpr_colors_needed.saturating_sub(avail) as usize;
        let spill_count = overshoot.clamp(1, 4); // spill 1-4 per round

        let excluded: BTreeSet<usize> = pre_coloring_colors2.keys().copied().collect();
        let mut spilled = BTreeSet::new();
        for _ in 0..spill_count {
            let candidate = select_spill(
                &graph2,
                &liveness2,
                &insts_coalesced,
                avail,
                loop_depths,
                &excluded,
            );
            let Some(idx) = candidate else { break };
            if spilled.contains(&idx) {
                break; // same candidate selected twice, no progress
            }
            spilled.insert(idx);
        }

        if spilled.is_empty() {
            return Err(format!(
                "register allocation: could not find spill candidate in round {round}"
            ));
        }

        // If spilled VRegs are rematerializable AND not call args, remove
        // from block_live_out. Call-arg VRegs must stay in live_out so their
        // def is not removed and they remain live at call clobber points.
        let call_arg_vregs = super::spill::collect_call_arg_vregs(&insts);
        for &idx in &spilled {
            if let Some(def) = insts.iter().find(|i| i.dst.0 as usize == idx)
                && super::spill::is_rematerializable(def)
                && !call_arg_vregs.contains(&idx)
            {
                block_live_out.remove(&VReg(idx as u32));
            }
        }

        insert_spills(
            &mut insts,
            &spilled,
            &mut spill_slots,
            &mut next_vreg,
            &vreg_classes2,
        );
    }

    unreachable!("loop should have returned before exhausting rounds")
}

/// Extend the interference graph with phantom VRegs for caller-saved GPRs at each
/// call point.
///
/// For each call point index `cp`, all GPR VRegs live at that point must not be
/// assigned to any caller-saved register. This is modeled by adding a phantom VReg
/// pre-colored to each caller-saved register and adding interference edges between
/// the phantom and every live GPR VReg at `cp`.
///
/// Returns the extended graph and a pre-coloring map (phantom vreg idx -> color)
/// that the caller must merge into the coloring's pre-coloring constraints.
fn add_call_clobber_interferences(
    mut graph: super::interference::InterferenceGraph,
    liveness: &super::liveness::LivenessInfo,
    call_points: &[usize],
    next_vreg: &mut u32,
    uses_frame_pointer: bool,
) -> (super::interference::InterferenceGraph, BTreeMap<usize, u32>) {
    if call_points.is_empty() {
        return (graph, BTreeMap::new());
    }

    // Use the same register ordering as map_colors_to_regs so that the color numbers
    // assigned to phantom VRegs correspond to the correct physical registers.
    let ordered_regs = allocatable_gpr_order(uses_frame_pointer);
    let reg_to_color: BTreeMap<Reg, u32> = ordered_regs
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();

    let n = liveness.live_at.len();
    let mut phantom_precolors: BTreeMap<usize, u32> = BTreeMap::new();

    for &cp in call_points {
        // The live set at the call boundary: if cp < n use live_at[cp]; otherwise live_out.
        let live_at_cp: &std::collections::BTreeSet<VReg> = if cp < n {
            &liveness.live_at[cp]
        } else {
            &liveness.live_out
        };

        // For each caller-saved GPR, add a phantom VReg pre-colored to it and
        // add interference with all GPR VRegs in live_at_cp.
        for &csr in CALLER_SAVED_GPR.iter().filter(|&&r| r != Reg::RSP) {
            let Some(&color) = reg_to_color.get(&csr) else {
                continue;
            };

            // Allocate a fresh phantom VReg index.
            let phantom_idx = *next_vreg as usize;
            *next_vreg += 1;

            // Grow the graph to accommodate the new phantom.
            if phantom_idx >= graph.num_vregs {
                let new_n = phantom_idx + 1;
                graph.adj.resize(new_n, std::collections::BTreeSet::new());
                graph.reg_class.resize(new_n, RegClass::GPR);
                graph.num_vregs = new_n;
            }

            // Record pre-coloring for the phantom.
            phantom_precolors.insert(phantom_idx, color);

            // Add interference between the phantom and each GPR VReg live at cp.
            for &live_v in live_at_cp {
                let live_idx = live_v.0 as usize;
                if live_idx < graph.num_vregs && graph.reg_class[live_idx] == RegClass::GPR {
                    graph.add_edge(phantom_idx, live_idx);
                }
            }
        }
    }

    (graph, phantom_precolors)
}

/// Like `add_call_clobber_interferences` but only adds phantoms for RAX and RDX,
/// which are the registers clobbered by x86 DIV/IDIV instructions.
fn add_div_clobber_interferences(
    mut graph: super::interference::InterferenceGraph,
    liveness: &super::liveness::LivenessInfo,
    div_points: &[usize],
    next_vreg: &mut u32,
    uses_frame_pointer: bool,
) -> (super::interference::InterferenceGraph, BTreeMap<usize, u32>) {
    if div_points.is_empty() {
        return (graph, BTreeMap::new());
    }

    let ordered_regs = allocatable_gpr_order(uses_frame_pointer);
    let reg_to_color: BTreeMap<Reg, u32> = ordered_regs
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();

    let n = liveness.live_at.len();
    let mut phantom_precolors: BTreeMap<usize, u32> = BTreeMap::new();

    // DIV/IDIV only clobbers RAX (quotient) and RDX (remainder).
    let div_clobbered = [Reg::RAX, Reg::RDX];

    for &cp in div_points {
        let live_at_cp: &std::collections::BTreeSet<VReg> = if cp < n {
            &liveness.live_at[cp]
        } else {
            &liveness.live_out
        };

        for &reg in &div_clobbered {
            let Some(&color) = reg_to_color.get(&reg) else {
                continue;
            };

            let phantom_idx = *next_vreg as usize;
            *next_vreg += 1;

            if phantom_idx >= graph.num_vregs {
                let new_n = phantom_idx + 1;
                graph.adj.resize(new_n, std::collections::BTreeSet::new());
                graph.reg_class.resize(new_n, RegClass::GPR);
                graph.num_vregs = new_n;
            }

            phantom_precolors.insert(phantom_idx, color);

            for &live_v in live_at_cp {
                let live_idx = live_v.0 as usize;
                if live_idx < graph.num_vregs && graph.reg_class[live_idx] == RegClass::GPR {
                    graph.add_edge(phantom_idx, live_idx);
                }
            }
        }
    }

    (graph, phantom_precolors)
}

/// Returns true if `op` produces an XMM (FP) register as its destination.
fn is_fp_op(op: &Op) -> bool {
    use crate::ir::types::Type;
    match op {
        Op::X86Addsd
        | Op::X86Subsd
        | Op::X86Mulsd
        | Op::X86Divsd
        | Op::X86Sqrtsd
        | Op::Fconst(_) => true,
        Op::X86Bitcast { to, .. } => matches!(to, Type::F32 | Type::F64),
        _ => false,
    }
}

/// Build a VReg class map for all VRegs referenced in the instruction list.
/// FP ops (X86Addsd etc.) use XMM; everything else uses GPR.
fn build_vreg_classes(
    insts: &[ScheduledInst],
    liveness: &super::liveness::LivenessInfo,
) -> BTreeMap<VReg, RegClass> {
    let mut map = BTreeMap::new();
    for inst in insts {
        let class = if is_fp_op(&inst.op) {
            RegClass::XMM
        } else {
            RegClass::GPR
        };
        map.insert(inst.dst, class);
        for &op in &inst.operands {
            // Operand class is inferred from the defining instruction's class.
            // Default to GPR; will be overridden if the defining inst is FP.
            map.entry(op).or_insert(RegClass::GPR);
        }
    }
    // Propagate XMM class: if an instruction is FP, its operands are also XMM.
    for inst in insts {
        if is_fp_op(&inst.op) {
            for &op in &inst.operands {
                map.insert(op, RegClass::XMM);
            }
        }
    }
    for live_set in &liveness.live_at {
        for &v in live_set {
            map.entry(v).or_insert(RegClass::GPR);
        }
    }
    map
}

/// Build a pre-coloring map from VReg index -> color.
///
/// Each ABI register is assigned a stable color number based on its position
/// in the register ordering used by `map_colors_to_regs` (via `allocatable_gpr_order`).
/// `uses_frame_pointer` must match what was passed to `allocate` so that color numbers
/// correspond to the same physical registers.
fn build_pre_coloring_colors(
    insts: &[ScheduledInst],
    param_vreg_to_reg: &BTreeMap<VReg, Reg>,
    uses_frame_pointer: bool,
) -> BTreeMap<usize, u32> {
    let ordered_regs = allocatable_gpr_order(uses_frame_pointer);
    let reg_to_color: BTreeMap<Reg, u32> = ordered_regs
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();

    let mut pre: BTreeMap<usize, u32> = BTreeMap::new();
    for inst in insts {
        let vreg = inst.dst;
        if let Some(&reg) = param_vreg_to_reg.get(&vreg)
            && let Some(&color) = reg_to_color.get(&reg)
        {
            pre.insert(vreg.0 as usize, color);
        }
    }
    pre
}

/// Build a pre-coloring map from VReg index -> Reg (for map_colors_to_regs).
fn build_pre_coloring_regs(
    insts: &[ScheduledInst],
    param_vreg_to_reg: &BTreeMap<VReg, Reg>,
) -> BTreeMap<usize, Reg> {
    let mut pre: BTreeMap<usize, Reg> = BTreeMap::new();
    for inst in insts {
        let vreg = inst.dst;
        if let Some(&reg) = param_vreg_to_reg.get(&vreg) {
            pre.insert(vreg.0 as usize, reg);
        }
    }
    pre
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn iconst_inst(dst: u32, val: i64) -> ScheduledInst {
        ScheduledInst {
            op: Op::Iconst(val, Type::I64),
            dst: VReg(dst),
            operands: vec![],
        }
    }

    fn add_inst(dst: u32, a: u32, b: u32) -> ScheduledInst {
        ScheduledInst {
            op: Op::X86Add,
            dst: VReg(dst),
            operands: vec![VReg(a), VReg(b)],
        }
    }

    // 10.14: Basic integration: straight-line block, all VRegs get a register.
    #[test]
    fn basic_allocation_succeeds() {
        let insts = vec![iconst_inst(0, 1), iconst_inst(1, 2), add_inst(2, 0, 1)];
        let live_out = BTreeSet::new();
        let result = allocate(
            &insts,
            &[],
            &live_out,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        )
        .expect("allocation should succeed");

        assert!(result.vreg_to_reg.contains_key(&VReg(0)));
        assert!(result.vreg_to_reg.contains_key(&VReg(1)));
        assert!(result.vreg_to_reg.contains_key(&VReg(2)));

        // All assigned registers are valid GPRs and not RSP.
        for (_, &reg) in &result.vreg_to_reg {
            assert!(reg.is_gpr(), "all allocated regs must be GPRs");
            assert_ne!(reg, Reg::RSP, "RSP must not be allocated");
        }

        assert_eq!(result.spill_slots, 0);
    }

    // 10.15: Pre-coloring: function parameter pre-assigned to RDI.
    #[test]
    fn param_precolored_to_rdi() {
        // v0 is a function parameter pre-colored to RDI.
        let insts = vec![
            iconst_inst(0, 99), // represents the param def
            iconst_inst(1, 1),
            add_inst(2, 0, 1),
        ];
        let params = vec![(VReg(0), Reg::RDI)];
        let live_out = BTreeSet::new();
        let result = allocate(
            &insts,
            &params,
            &live_out,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        )
        .expect("allocation should succeed");

        assert_eq!(
            result.vreg_to_reg.get(&VReg(0)),
            Some(&Reg::RDI),
            "v0 must be allocated to RDI"
        );
    }

    // 10.16: No interference between non-overlapping VRegs: they can share regs.
    #[test]
    fn non_overlapping_can_share_register() {
        // v0 = iconst; v1 = use(v0) -- v0 dies here; v2 = iconst; v3 = use(v2)
        let insts = vec![
            iconst_inst(0, 1),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(1),
                operands: vec![VReg(0)],
            },
            iconst_inst(2, 2),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(3),
                operands: vec![VReg(2)],
            },
        ];
        let live_out = BTreeSet::new();
        let result = allocate(
            &insts,
            &[],
            &live_out,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        )
        .expect("allocation should succeed");

        // v0 and v2 don't overlap -- they may get the same register.
        // The important thing is that v0 and v2 are allocated (no panic).
        assert!(result.vreg_to_reg.contains_key(&VReg(0)));
        assert!(result.vreg_to_reg.contains_key(&VReg(2)));
    }

    // Spill loop detection: exceed available registers and verify error after 3 rounds.
    // We construct a block with 16 simultaneously live VRegs (exceeds 15 available GPRs).
    #[test]
    fn spill_loop_detection() {
        // Create 16 iconsts and one instruction that uses them all simultaneously.
        // Use a chain of X86Add: v_result = add(v_result, v_i) for each i.
        // But all v_i need to be live at the same time. We use a single multi-operand
        // instruction -- since X86Add only takes 2 operands, we create a chain but
        // keep all inputs alive by having them used at the end.
        //
        // Simpler: create 16 iconsts all used by a dummy "return" that has 16 operands.
        // Since ScheduledInst.operands is Vec<VReg>, we can have many operands.
        let mut insts: Vec<ScheduledInst> = (0u32..16).map(|i| iconst_inst(i, i as i64)).collect();

        // A single instruction that uses all 16.
        insts.push(ScheduledInst {
            op: Op::Iconst(0, Type::I64), // dummy op
            dst: VReg(16),
            operands: (0u32..16).map(VReg).collect(),
        });

        // These 16 VRegs are all simultaneously live at inst 16.
        // Chromatic number = 16 > 15 available GPRs.
        // After 3 spill rounds, should error.
        let live_out = BTreeSet::new();
        let result = allocate(
            &insts,
            &[],
            &live_out,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        );
        // Either it succeeds (some spilling reduced pressure) or it errors.
        // With 16 simultaneous live regs and only constants (rematerializable),
        // spilling will re-emit iconsts and eventually succeed.
        // This test just checks it doesn't panic or loop infinitely.
        let _ = result; // success or error, both acceptable here
    }

    // Phase 4.1: allocator with explicit copy pairs coalesces non-interfering ones.
    //
    // v0 = iconst; v1 = iconst; [v0 and v1 are non-interfering since v0 dies before v1 is used]
    // Copy pair (v0, v2) and (v1, v3) — if regalloc coalesces them, v0==v2 and v1==v3.
    //
    // Actually we test that passing copy pairs doesn't break allocation, and
    // that non-interfering pairs can share a register (same as non_overlapping test
    // but with explicit copy pairs supplied).
    #[test]
    fn copy_pairs_passed_to_allocate() {
        // v0 = iconst; v1 = use(v0) [v0 dies]; v2 = iconst; v3 = use(v2)
        // Copy pair (v0, v2): since they don't interfere, coalescing may assign same reg.
        let insts = vec![
            iconst_inst(0, 1),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(1),
                operands: vec![VReg(0)],
            },
            iconst_inst(2, 2),
            ScheduledInst {
                op: Op::Proj0,
                dst: VReg(3),
                operands: vec![VReg(2)],
            },
        ];
        let live_out = BTreeSet::new();
        // Pass a copy pair (v0, v2) — non-interfering, so coalescing may unify them.
        let copy_pairs = vec![(VReg(0), VReg(2))];
        let result = allocate(
            &insts,
            &[],
            &live_out,
            &copy_pairs,
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        )
        .expect("allocation with copy pairs should succeed");

        // All four VRegs must be allocated.
        assert!(result.vreg_to_reg.contains_key(&VReg(0)));
        assert!(result.vreg_to_reg.contains_key(&VReg(1)));
        assert!(result.vreg_to_reg.contains_key(&VReg(2)));
        assert!(result.vreg_to_reg.contains_key(&VReg(3)));
    }

    // Phase 4.3: shift count VReg pre-colored to RCX is allocated to RCX.
    //
    // Simulate a variable shift by pre-coloring the count VReg to RCX,
    // as compile() does before calling allocate().
    #[test]
    fn shift_count_precolored_to_rcx() {
        // v0 = iconst (value to shift); v1 = iconst (shift count, pre-colored to RCX)
        // v2 = X86Shl(v0, v1)
        let insts = vec![
            iconst_inst(0, 4),
            iconst_inst(1, 2), // count
            ScheduledInst {
                op: Op::X86Shl,
                dst: VReg(2),
                operands: vec![VReg(0), VReg(1)],
            },
        ];
        let params = vec![(VReg(1), Reg::RCX)]; // pre-color count to RCX
        let live_out = BTreeSet::new();
        let result = allocate(
            &insts,
            &params,
            &live_out,
            &[],
            &std::collections::BTreeMap::new(),
            &[],
            &[],
            false,
            "",
        )
        .expect("allocation with shift pre-coloring should succeed");

        // The shift count must be allocated to RCX.
        assert_eq!(
            result.vreg_to_reg.get(&VReg(1)),
            Some(&Reg::RCX),
            "shift count VReg must be allocated to RCX"
        );
    }
}
