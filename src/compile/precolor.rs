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

/// Every VReg that is a stack-position call argument and is never a
/// register-position one.
///
/// A stack-position argument is pushed, and `abi::setup_call_args` reads the
/// push's source from memory when the allocator left the value in a slot -- so
/// such a value needs no register at the call, which is what lets a call with
/// more arguments than the machine has registers be allocated at all. The
/// register-position arguments of the same call still need theirs.
///
/// A VReg used at both kinds of position is excluded rather than handled: it has
/// to be in a register at the call that wants it there, and one answer per VReg
/// is what both allocators can act on. Nothing in the generated or saved corpora
/// hits that case, and being conservative there costs only the pressure relief.
///
/// Stated here, from the CFG's own `EffOperand` VRegs and `assign_args`, because
/// this is where the ABI positions of a call are already known. Deriving it in
/// the allocators from the absence of a pre-coloring would be the same fact
/// inferred twice.
pub(crate) fn stack_arg_vregs(func: &Function) -> BTreeSet<VReg> {
    let mut on_stack = BTreeSet::new();
    let mut in_reg = BTreeSet::new();
    for block in &func.blocks {
        for op in &block.ops {
            let EffectfulOp::Call { args, arg_tys, .. } = op else {
                continue;
            };
            for (arg, loc) in args.iter().zip(assign_args(arg_tys).iter()) {
                if let Some(vreg) = arg.vreg() {
                    match loc {
                        ArgLoc::Stack { .. } => on_stack.insert(vreg),
                        ArgLoc::Reg(_) => in_reg.insert(vreg),
                    };
                }
            }
        }
    }
    on_stack.retain(|v| !in_reg.contains(v));
    on_stack
}

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::BasicBlock;
    use crate::ir::op::ClassId;
    use crate::ir::types::Type;

    fn operand(n: u32) -> EffOperand {
        EffOperand::Committed {
            class: ClassId(n),
            vreg: VReg(n),
        }
    }

    /// One call, `arg_tys` integers unless `float_ret`, results in VReg 100.
    fn call(arg_count: u32, float_ret: bool) -> EffectfulOp {
        EffectfulOp::Call {
            func: "callee".to_string(),
            args: (0..arg_count).map(operand).collect(),
            arg_tys: vec![Type::I64; arg_count as usize],
            ret_tys: vec![if float_ret { Type::F64 } else { Type::I64 }],
            results: vec![operand(100)],
            variadic: false,
        }
    }

    fn block_of(ops: Vec<EffectfulOp>) -> BasicBlock {
        let mut ops = ops;
        ops.push(EffectfulOp::Ret { val: None });
        BasicBlock {
            id: 0,
            param_types: vec![],
            param_vregs: vec![],
            ops,
        }
    }

    fn run(block: &BasicBlock) -> (Vec<(VReg, Reg)>, BTreeSet<VReg>) {
        let mut pairs = Vec::new();
        let mut live_out = BTreeSet::new();
        add_call_precolors_for_block(block, &mut pairs, &mut live_out);
        (pairs, live_out)
    }

    /// Register arguments take their ABI register, in `assign_args` order, and
    /// the result takes the ABI return register.
    #[test]
    fn single_call_precolors_args_and_result() {
        let (pairs, live_out) = run(&block_of(vec![call(3, false)]));
        assert_eq!(
            pairs,
            vec![
                (VReg(0), Reg::RDI),
                (VReg(1), Reg::RSI),
                (VReg(2), Reg::RDX),
                (VReg(100), GPR_RETURN_REG),
            ]
        );
        assert!(live_out.is_empty());
    }

    /// A float return goes to xmm0, not rax.
    #[test]
    fn float_result_takes_the_fp_return_reg() {
        let (pairs, _) = run(&block_of(vec![call(0, true)]));
        assert_eq!(pairs, vec![(VReg(100), FP_RETURN_REG)]);
    }

    /// Past the six argument registers an argument is passed on the stack, so
    /// it gets no color; it is live out instead, since the caller's stores read
    /// it after the barrier.
    #[test]
    fn stack_args_are_live_out_rather_than_precolored() {
        let (pairs, live_out) = run(&block_of(vec![call(8, false)]));
        assert_eq!(pairs.len(), 7, "6 register args plus the result");
        assert!(!pairs.iter().any(|&(v, _)| v == VReg(6) || v == VReg(7)));
        assert_eq!(live_out, BTreeSet::from([VReg(6), VReg(7)]));
    }

    /// Two calls in one block share the argument registers, so a fixed color
    /// per VReg cannot satisfy both. The pass colors nothing and leaves the
    /// allocator to place the copies; stack arguments are still live out.
    #[test]
    fn two_calls_in_a_block_precolor_nothing() {
        let (pairs, live_out) = run(&block_of(vec![call(8, false), call(2, false)]));
        assert!(pairs.is_empty());
        assert_eq!(live_out, BTreeSet::from([VReg(6), VReg(7)]));
    }

    /// A register already claimed by an earlier pairing is not handed out
    /// twice, and a VReg already colored keeps its first color.
    #[test]
    fn existing_pairings_win() {
        let mut pairs = vec![(VReg(50), Reg::RSI)];
        let mut live_out = BTreeSet::new();
        add_call_precolors_for_block(&block_of(vec![call(2, false)]), &mut pairs, &mut live_out);
        assert_eq!(
            pairs,
            vec![
                (VReg(50), Reg::RSI),
                (VReg(0), Reg::RDI),
                (VReg(100), GPR_RETURN_REG)
            ],
            "VReg 1 wanted rsi, which VReg 50 already holds"
        );
    }
}
