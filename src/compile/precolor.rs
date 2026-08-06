use std::collections::BTreeSet;

use crate::compile::program_point::ProgramPoint;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::ir::effectful::{EffOperand, EffectfulOp};
use crate::ir::function::Function;
use crate::x86::abi::{ArgLoc, FP_RETURN_REG, GPR_RETURN_REG, assign_args};
use crate::x86::reg::Reg;

/// Map function parameters to (VReg, Reg) pairs for pre-coloring.
///
/// Uses `func.param_class_ids` (populated by the builder) to look up the
/// corresponding VRegs in the ClassId -> VReg map from extraction.
pub(super) fn assign_param_vregs_from_map(
    func: &Function,
    class_to_vreg: &ClassVRegMap,
    egraph: &EGraph,
    block_has_calls: bool,
    arg_locs: &[ArgLoc],
) -> Vec<(VReg, Reg)> {
    use crate::x86::abi::CALLER_SAVED_GPR;

    if func.param_class_ids.is_empty() {
        return vec![];
    }
    let mut pairs: Vec<(VReg, Reg)> = Vec::new();

    // Params are always in the entry block (block 0).
    let entry_point = ProgramPoint::block_entry(0);
    for (param_idx, &class_id) in func.param_class_ids.iter().enumerate() {
        // Canonicalize the class_id after run_phases merges.
        let canon = egraph.unionfind.find_immutable(class_id);
        if let Some(vreg) = class_to_vreg.lookup(canon, entry_point)
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

/// Collect the schedule indices of X86Idiv/X86Div instructions.
/// Pre-color call argument and result VRegs for a single block.
///
/// Same logic as `add_call_precolors` but scoped to one block, preventing
/// call-arg precolorings from one block from leaking into another block's
/// register allocation.
pub(super) fn add_call_precolors_for_block(
    block: &crate::ir::function::BasicBlock,
    param_vregs: &mut Vec<(VReg, Reg)>,
    live_out: &mut BTreeSet<VReg>,
) {
    let non_term_count = block.non_term_count();
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
            for (arg, loc) in args.iter().zip(locs.iter()) {
                // The VReg the CFG states. This pass and
                // `populate_effectful_operands` used to derive that answer
                // separately, from the function-wide map at the block's entry
                // and from the block's own snapshot at the barrier -- and a
                // per-block question answered through the function-wide map is
                // the shape most wrong-code bugs here have had. One derivation
                // now, made by linearization and committed into the CFG.
                if let Some(vreg) = arg.vreg() {
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
            if call_count == 1 {
                // The result's VReg, as the CFG states it: the same VReg the
                // barrier instruction defining it carries.
                if let Some(vreg) = results.first().and_then(EffOperand::vreg)
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
