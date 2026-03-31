use std::collections::{HashMap, HashSet};

use crate::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::{ArgLoc, GPR_RETURN_REG, assign_args};
use crate::x86::reg::Reg;

/// Map function parameters to (VReg, Reg) pairs for pre-coloring.
///
/// Uses `func.param_class_ids` (populated by the builder) to look up the
/// corresponding VRegs in the ClassId -> VReg map from extraction.
pub(super) fn assign_param_vregs_from_map(
    func: &Function,
    class_to_vreg: &HashMap<ClassId, VReg>,
    egraph: &EGraph,
) -> Vec<(VReg, Reg)> {
    if func.param_class_ids.is_empty() {
        return vec![];
    }

    let arg_locs = assign_args(&func.param_types);
    let mut pairs: Vec<(VReg, Reg)> = Vec::new();

    for (param_idx, &class_id) in func.param_class_ids.iter().enumerate() {
        // Canonicalize the class_id after run_phases merges.
        let canon = egraph.unionfind.find_immutable(class_id);
        if let Some(&vreg) = class_to_vreg.get(&canon) {
            if let ArgLoc::Reg(reg) = arg_locs[param_idx] {
                pairs.push((vreg, reg));
            }
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

/// Pre-color call argument and call result VRegs to their ABI registers.
///
/// For register args (first 6 GPR), pre-color directly.
/// For stack args, add to `live_out` to force them to interfere with each other.
/// For call results, pre-color to RAX.
pub(super) fn add_call_precolors(
    func: &Function,
    egraph: &EGraph,
    class_to_vreg: &HashMap<ClassId, VReg>,
    param_vregs: &mut Vec<(VReg, Reg)>,
    live_out: &mut HashSet<VReg>,
) {
    for block in &func.blocks {
        // Count how many calls are in this block (excluding the terminator).
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
            if let EffectfulOp::Call { args, results, .. } = op {
                let arg_types: Vec<Type> = vec![Type::I64; args.len()];
                let locs = assign_args(&arg_types);
                for (&cid, loc) in args.iter().zip(locs.iter()) {
                    let canon = egraph.unionfind.find_immutable(cid);
                    if let Some(&vreg) = class_to_vreg.get(&canon) {
                        match loc {
                            ArgLoc::Reg(reg) => {
                                if !param_vregs.iter().any(|&(v, _)| v == vreg) {
                                    param_vregs.push((vreg, *reg));
                                }
                            }
                            ArgLoc::Stack { .. } => {
                                live_out.insert(vreg);
                            }
                        }
                    }
                }
                // Only precolor the call result to RAX when there is exactly one call
                // in this block. With multiple calls, result VRegs from earlier calls
                // must survive subsequent call clobbers, so we let the allocator freely
                // assign them to callee-saved registers. The lowering will emit a
                // `mov allocated_reg, rax` to capture the return value when needed.
                if call_count == 1 {
                    if let Some(&first_result_cid) = results.first() {
                        let canon = egraph.unionfind.find_immutable(first_result_cid);
                        if let Some(&vreg) = class_to_vreg.get(&canon) {
                            if !param_vregs.iter().any(|&(v, _)| v == vreg) {
                                param_vregs.push((vreg, GPR_RETURN_REG));
                            }
                        }
                    }
                }
            }
        }
    }
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
    class_to_vreg: &HashMap<ClassId, VReg>,
    egraph: &EGraph,
) -> Vec<usize> {
    let block = &func.blocks[block_idx];
    let non_term_count = if block.ops.is_empty() {
        0
    } else {
        block.ops.len() - 1
    };

    // Collect all calls (in order) along with their result ClassIds.
    let calls: Vec<&Vec<ClassId>> = block.ops[..non_term_count]
        .iter()
        .filter_map(|op| {
            if let EffectfulOp::Call { results, .. } = op {
                Some(results)
            } else {
                None
            }
        })
        .collect();

    if calls.is_empty() {
        return vec![];
    }

    if calls.len() == 1 {
        // Single call: use the sentinel (entire block is the clobber region).
        return vec![block_sched.len()];
    }

    // For blocks with multiple calls, find the schedule index of each call's
    // CallResult node. The call point for call K is the index of its CallResult
    // in block_sched. The liveness analysis at that index captures VRegs that
    // are live "just before" the CallResult is defined, which represents VRegs
    // that are live immediately after call K completes (i.e., they need to
    // survive the call's clobber).
    //
    // For each call, we look up the first result's VReg and find its position
    // in block_sched. If not found, fall back to the sentinel.
    let mut call_points: Vec<usize> = Vec::with_capacity(calls.len());
    for result_cids in &calls {
        let mut cp = block_sched.len(); // sentinel fallback
        if let Some(&first_result_cid) = result_cids.first() {
            let canon = egraph.unionfind.find_immutable(first_result_cid);
            if let Some(&result_vreg) = class_to_vreg.get(&canon) {
                // Find the schedule index of this CallResult node.
                if let Some(pos) = block_sched.iter().position(|inst| inst.dst == result_vreg) {
                    cp = pos;
                }
            }
        }
        call_points.push(cp);
    }
    call_points
}
