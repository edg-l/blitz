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

/// Compute call point indices (local to a block's instruction list) for
/// caller-saved clobber modeling. Returns `block_sched.len()` as a sentinel
/// meaning "clobber spans through end of block" -- the allocator treats
/// these as exclusive upper bounds, not instruction indices.
pub(super) fn collect_call_points_for_block(
    func: &Function,
    block_idx: usize,
    block_sched: &[ScheduledInst],
) -> Vec<usize> {
    let block = &func.blocks[block_idx];
    let non_term_count = if block.ops.is_empty() {
        0
    } else {
        block.ops.len() - 1
    };
    let has_call = block.ops[..non_term_count]
        .iter()
        .any(|op| matches!(op, EffectfulOp::Call { .. }));
    if has_call {
        vec![block_sched.len()]
    } else {
        vec![]
    }
}
