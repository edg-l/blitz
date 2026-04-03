use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::{ArgLoc, FP_RETURN_REG, GPR_RETURN_REG, assign_args};
use crate::x86::reg::Reg;

/// Map function parameters to (VReg, Reg) pairs for pre-coloring.
///
/// Uses `func.param_class_ids` (populated by the builder) to look up the
/// corresponding VRegs in the ClassId -> VReg map from extraction.
pub(super) fn assign_param_vregs_from_map(
    func: &Function,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    egraph: &EGraph,
    block_has_calls: bool,
) -> Vec<(VReg, Reg)> {
    use crate::x86::abi::CALLER_SAVED_GPR;

    if func.param_class_ids.is_empty() {
        return vec![];
    }

    let arg_locs = assign_args(&func.param_types);
    let mut pairs: Vec<(VReg, Reg)> = Vec::new();

    for (param_idx, &class_id) in func.param_class_ids.iter().enumerate() {
        // Canonicalize the class_id after run_phases merges.
        let canon = egraph.unionfind.find_immutable(class_id);
        if let Some(&vreg) = class_to_vreg.get(&canon)
            && let ArgLoc::Reg(reg) = arg_locs[param_idx]
        {
            // Don't precolor params to caller-saved registers when the block
            // has calls -- the call clobbers the register and the regalloc
            // can't resolve the conflict. The param will get a callee-saved
            // register and a mov will be emitted at function entry.
            if block_has_calls && CALLER_SAVED_GPR.contains(&reg) {
                continue;
            }
            // All XMM registers are caller-saved in SystemV AMD64 ABI.
            if block_has_calls && reg.is_xmm() {
                continue;
            }
            pairs.push((vreg, reg));
        }
    }

    pairs
}

/// Pre-color shift count operands to RCX for variable-shift instructions.
pub(super) fn add_shift_precolors(insts: &[ScheduledInst], param_vregs: &mut Vec<(VReg, Reg)>) {
    for inst in insts {
        if matches!(inst.op, Op::X86Shl | Op::X86Shr | Op::X86Sar) && inst.operands.len() >= 2 {
            let count_vreg = inst.operands[1];
            if !param_vregs.iter().any(|&(v, _)| v == count_vreg) {
                param_vregs.push((count_vreg, Reg::RCX));
            }
        }
    }
}

/// Pre-color division operands and projections to RAX/RDX.
///
/// - For each X86Idiv/X86Div in the schedule: operand 0 (dividend) → RAX.
/// - For each Proj0 projecting from an X86Idiv/X86Div VReg: Proj0 dst → RAX (quotient).
/// - For each Proj1 projecting from an X86Idiv/X86Div VReg: Proj1 dst → RDX (remainder).
/// - The X86Idiv/X86Div Pair node itself is NOT pre-colored.
pub(super) fn add_div_precolors(insts: &[ScheduledInst], param_vregs: &mut Vec<(VReg, Reg)>) {
    // Collect VRegs defined by X86Idiv/X86Div instructions.
    let mut div_dst_vregs: BTreeSet<VReg> = BTreeSet::new();
    for inst in insts {
        if !matches!(inst.op, Op::X86Idiv | Op::X86Div) {
            continue;
        }
        div_dst_vregs.insert(inst.dst);
        // Pre-color dividend (operand 0) to RAX.
        if let Some(&dividend) = inst.operands.first()
            && !param_vregs.iter().any(|&(v, _)| v == dividend)
        {
            param_vregs.push((dividend, Reg::RAX));
        }
    }

    // Pre-color Proj0 nodes that project from a div result to RAX (quotient).
    // Proj1 (remainder) is NOT pre-colored to RDX: the lowering emits
    // `mov dst, rdx` so the remainder can live in any register, which avoids
    // conflicts when the remainder flows into a loop back edge as a divisor.
    for inst in insts {
        if inst.op == Op::Proj0
            && let Some(&src) = inst.operands.first()
            && div_dst_vregs.contains(&src)
            && !param_vregs.iter().any(|&(v, _)| v == inst.dst)
        {
            param_vregs.push((inst.dst, Reg::RAX));
        }
    }
}

/// Collect the schedule indices of X86Idiv/X86Div instructions.
///
/// Each such index is used as a "div point" so that RDX is modeled as clobbered
/// at that position (same mechanism as call clobber points).
pub(super) fn collect_div_clobber_points(insts: &[ScheduledInst]) -> Vec<usize> {
    insts
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| {
            if matches!(inst.op, Op::X86Idiv | Op::X86Div) {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Compute call point indices (local to a block's instruction list) for
/// caller-saved clobber modeling.
///
/// Returns one entry per call in the block. Each entry is the schedule
/// position *after* which the call logically occurs -- i.e. the first
/// schedule index at which the call result VRegs become live.
///
/// For a single call, the sentinel `block_sched.len()` is returned (meaning
/// "clobber spans the entire block"). For multiple calls, we compute the
/// earliest schedule position at which each call's arguments are fully
/// defined, so that values live between calls are forced to callee-saved
/// registers by the interference graph.
pub(super) fn collect_call_points_for_block(
    func: &Function,
    block_idx: usize,
    block_sched: &[ScheduledInst],
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    egraph: &EGraph,
) -> Vec<usize> {
    let block = &func.blocks[block_idx];
    let non_term_count = if block.ops.is_empty() {
        0
    } else {
        block.ops.len() - 1
    };

    // Collect all calls (in order) along with their args and result ClassIds.
    let calls: Vec<(&Vec<ClassId>, &Vec<ClassId>)> = block.ops[..non_term_count]
        .iter()
        .filter_map(|op| {
            if let EffectfulOp::Call { args, results, .. } = op {
                Some((args, results))
            } else {
                None
            }
        })
        .collect();

    if calls.is_empty() {
        return vec![];
    }

    // For each call, find the schedule position that represents the call point.
    // For non-void calls: use the CallResult node's position in block_sched.
    // The liveness at that index captures VRegs live across the call.
    // For void calls: use the position after the last argument in block_sched,
    // since the call happens after all arguments are computed.
    let mut call_points: Vec<usize> = Vec::with_capacity(calls.len());
    for (arg_cids, result_cids) in &calls {
        let mut cp = block_sched.len(); // sentinel fallback

        // Try to find via CallResult first (non-void calls).
        if let Some(&first_result_cid) = result_cids.first() {
            let canon = egraph.unionfind.find_immutable(first_result_cid);
            if let Some(&result_vreg) = class_to_vreg.get(&canon)
                && let Some(pos) = block_sched.iter().position(|inst| inst.dst == result_vreg)
            {
                cp = pos;
            }
        }

        // For void calls (no results or result not found), use the
        // VoidCallBarrier's position if one exists with matching arg VRegs.
        // Fall back to position after last argument otherwise.
        if cp == block_sched.len() {
            // Try VoidCallBarrier first: its operands are the call's arg VRegs.
            let arg_vregs: Vec<VReg> = arg_cids
                .iter()
                .filter_map(|&cid| {
                    let canon = egraph.unionfind.find_immutable(cid);
                    class_to_vreg.get(&canon).copied()
                })
                .collect();
            if !arg_vregs.is_empty()
                && let Some(pos) = block_sched.iter().position(|inst| {
                    matches!(inst.op, Op::VoidCallBarrier)
                        && arg_vregs.iter().all(|v| inst.operands.contains(v))
                })
            {
                cp = pos;
            } else if !arg_cids.is_empty() {
                // Fallback: position after the last argument.
                let mut max_arg_pos: Option<usize> = None;
                for &arg_cid in arg_cids.iter() {
                    let canon = egraph.unionfind.find_immutable(arg_cid);
                    if let Some(&arg_vreg) = class_to_vreg.get(&canon)
                        && let Some(pos) = block_sched.iter().position(|inst| inst.dst == arg_vreg)
                    {
                        max_arg_pos = Some(max_arg_pos.map_or(pos, |m: usize| m.max(pos)));
                    }
                }
                if let Some(pos) = max_arg_pos {
                    cp = (pos + 1).min(block_sched.len());
                }
            }
        }

        call_points.push(cp);
    }
    call_points
}

/// Pre-color call argument and result VRegs for a single block.
///
/// Same logic as `add_call_precolors` but scoped to one block, preventing
/// call-arg precolorings from one block from leaking into another block's
/// register allocation.
pub(super) fn add_call_precolors_for_block(
    block: &crate::ir::function::BasicBlock,
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    param_vregs: &mut Vec<(VReg, Reg)>,
    live_out: &mut BTreeSet<VReg>,
) {
    let non_term_count = if block.ops.is_empty() {
        0
    } else {
        block.ops.len() - 1
    };
    let call_count = block.ops[..non_term_count]
        .iter()
        .filter(|op| matches!(op, EffectfulOp::Call { .. }))
        .count();

    for op in &block.ops {
        if let EffectfulOp::Call {
            args,
            arg_tys,
            ret_tys,
            results,
            ..
        } = op
        {
            let locs = assign_args(arg_tys);
            for (&cid, loc) in args.iter().zip(locs.iter()) {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    match loc {
                        ArgLoc::Reg(reg) => {
                            if call_count == 1
                                && !param_vregs.iter().any(|&(v, _)| v == vreg)
                                && !param_vregs.iter().any(|&(_, r)| r == *reg)
                            {
                                param_vregs.push((vreg, *reg));
                            }
                        }
                        ArgLoc::Stack { .. } => {
                            live_out.insert(vreg);
                        }
                    }
                }
            }
            if call_count == 1
                && let Some(&first_result_cid) = results.first()
            {
                let canon = egraph.unionfind.find_immutable(first_result_cid);
                if let Some(&vreg) = class_to_vreg.get(&canon)
                    && !param_vregs.iter().any(|&(v, _)| v == vreg)
                {
                    let is_float_ret = ret_tys.first().is_some_and(|t| t.is_float());
                    let abi_reg = if is_float_ret {
                        FP_RETURN_REG
                    } else {
                        GPR_RETURN_REG
                    };
                    param_vregs.push((vreg, abi_reg));
                }
            }
        }
    }
}
