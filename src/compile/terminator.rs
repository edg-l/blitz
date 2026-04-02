use std::collections::BTreeMap;

use crate::egraph::EGraph;
use crate::egraph::extract::VReg;
use crate::egraph::isel::find_cc_in_class;
use crate::emit::phi_elim::phi_copies;
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::GPR_RETURN_REG;
use crate::x86::inst::{LabelId, MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

use super::{BlockItem, CompileError, IrLocation};

/// Negate a CondCode.
fn negate_cc(cc: CondCode) -> CondCode {
    match cc {
        CondCode::Eq => CondCode::Ne,
        CondCode::Ne => CondCode::Eq,
        CondCode::Slt => CondCode::Sge,
        CondCode::Sle => CondCode::Sgt,
        CondCode::Sgt => CondCode::Sle,
        CondCode::Sge => CondCode::Slt,
        CondCode::Ult => CondCode::Uge,
        CondCode::Ule => CondCode::Ugt,
        CondCode::Ugt => CondCode::Ule,
        CondCode::Uge => CondCode::Ult,
    }
}

/// Rewrite branch targets to skip through empty trampoline blocks.
///
/// A block is "empty" if its items contain only a single `Jmp { target }` (no phi
/// copies, no labels). For any such block, we record `block_id -> target` and then
/// rewrite all `Jcc` and `Jmp` instructions that point to it to jump directly to
/// the final destination. Repeated until no changes occur (handles chains).
pub(super) fn thread_branches(
    block_items: &mut [Vec<BlockItem>],
    func: &Function,
    rpo_order: &[usize],
) {
    loop {
        // Build a map: block_id -> jump_target for blocks that are just a Jmp.
        let mut redirect: BTreeMap<LabelId, LabelId> = BTreeMap::new();
        for (rpo_pos, items) in block_items.iter().enumerate() {
            let block_id = func.blocks[rpo_order[rpo_pos]].id as LabelId;
            // Count real instructions (not BindLabel).
            let real: Vec<&MachInst> = items
                .iter()
                .filter_map(|item| {
                    if let BlockItem::Inst(inst) = item {
                        Some(inst)
                    } else {
                        None
                    }
                })
                .collect();
            if real.len() == 1
                && let MachInst::Jmp { target } = real[0]
            {
                redirect.insert(block_id, *target);
            }
        }

        if redirect.is_empty() {
            break;
        }

        // Resolve chains: if A -> B -> C, make A -> C directly.
        let keys: Vec<LabelId> = redirect.keys().copied().collect();
        for k in keys {
            let mut dest = redirect[&k];
            let mut seen = std::collections::BTreeSet::new();
            seen.insert(k);
            while let Some(&next) = redirect.get(&dest) {
                if seen.contains(&next) {
                    break; // cycle guard
                }
                seen.insert(dest);
                dest = next;
            }
            redirect.insert(k, dest);
        }

        // Rewrite Jcc/Jmp targets in all blocks.
        let mut changed = false;
        for items in block_items.iter_mut() {
            for item in items.iter_mut() {
                if let BlockItem::Inst(inst) = item {
                    match inst {
                        MachInst::Jmp { target } => {
                            if let Some(&new_target) = redirect.get(target) {
                                *target = new_target;
                                changed = true;
                            }
                        }
                        MachInst::Jcc { target, .. } => {
                            if let Some(&new_target) = redirect.get(target) {
                                *target = new_target;
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Lower a block terminator, including phi copies for block-parameter passing.
///
/// Returns a list of `BlockItem`s (instructions and label bindings).
/// Uses `next_label` to allocate extra labels for trampoline code.
/// `MachInst::Ret` is returned as a marker replaced by `emit_epilogue` at encode time.
///
/// `next_block_id` is the block ID of the block that immediately follows this one
/// in emission (RPO) order. When a jump target equals `next_block_id`, the jump
/// can be omitted (fallthrough optimization).
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_terminator(
    op: &EffectfulOp,
    next_block_id: Option<BlockId>,
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    ret_class_to_vreg: &BTreeMap<ClassId, VReg>,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    next_label: &mut LabelId,
) -> Result<Vec<BlockItem>, CompileError> {
    let get_reg = |cid: ClassId, ctv: &BTreeMap<ClassId, VReg>| -> Option<Reg> {
        let canon = egraph.unionfind.find_immutable(cid);
        ctv.get(&canon)
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
    };

    match op {
        EffectfulOp::Ret { val } => {
            let mut items = Vec::new();
            if let Some(&ret_cid) = val.as_ref()
                && let Some(ret_reg) = get_reg(ret_cid, ret_class_to_vreg)
                && ret_reg != GPR_RETURN_REG
            {
                // Use the function's return type for the MOV size.
                let ret_size = func
                    .return_types
                    .first()
                    .map(OpSize::from_type)
                    .unwrap_or(OpSize::S64);
                items.push(BlockItem::Inst(MachInst::MovRR {
                    size: ret_size,
                    dst: Operand::Reg(GPR_RETURN_REG),
                    src: Operand::Reg(ret_reg),
                }));
            }
            // Ret marker: replaced with emit_epilogue() in the encoding loop.
            items.push(BlockItem::Inst(MachInst::Ret));
            Ok(items)
        }

        EffectfulOp::Jump { target, args } => {
            let copies = build_phi_copies(
                *target,
                args,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                regalloc,
                func,
            )?;
            let mut items: Vec<BlockItem> = phi_copies(&copies, Reg::R11)
                .into_iter()
                .map(BlockItem::Inst)
                .collect();
            // Fallthrough optimization: omit the jump if the target is the
            // immediately following block in emission (RPO) order.
            if next_block_id != Some(*target) {
                items.push(BlockItem::Inst(MachInst::Jmp {
                    target: *target as LabelId,
                }));
            }
            Ok(items)
        }

        EffectfulOp::Branch {
            cond,
            bb_true,
            bb_false,
            true_args,
            false_args,
        } => {
            let canon_cond = egraph.unionfind.find_immutable(*cond);
            let cc = find_cc_in_class(egraph, canon_cond).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: format!(
                    "branch condition class {:?} has no Icmp node; cannot determine CondCode",
                    canon_cond
                ),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;

            let true_copies = build_phi_copies(
                *bb_true,
                true_args,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                regalloc,
                func,
            )?;
            let false_copies = build_phi_copies(
                *bb_false,
                false_args,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                regalloc,
                func,
            )?;

            let true_phi = phi_copies(&true_copies, Reg::R11);
            let false_phi = phi_copies(&false_copies, Reg::R11);

            let false_is_fallthrough = next_block_id == Some(*bb_false);
            let true_is_fallthrough = next_block_id == Some(*bb_true);

            let mut items = Vec::new();
            if true_phi.is_empty() {
                // jcc cc, true_block; [false_phi]; jmp false_block
                // If false_block is the fallthrough, omit the final jmp.
                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc,
                    target: *bb_true as LabelId,
                }));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                if !false_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_false as LabelId,
                    }));
                }
            } else if false_phi.is_empty() {
                // jcc !cc, false_block; [true_phi]; jmp true_block
                // The Jcc is always needed (even if false is fallthrough) to skip
                // the true_phi copies when the condition is false.
                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc: negate_cc(cc),
                    target: *bb_false as LabelId,
                }));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                if !true_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_true as LabelId,
                    }));
                }
            } else {
                // Both sides have copies. Use trampoline labels:
                //   jcc !cc, L_false_copies
                //   [true_phi]
                //   jmp true_block         (omit if true_block is fallthrough)
                //   L_false_copies:
                //   [false_phi]
                //   jmp false_block        (omit if false_block is fallthrough)
                let l_false = *next_label;
                *next_label += 1;

                items.push(BlockItem::Inst(MachInst::Jcc {
                    cc: negate_cc(cc),
                    target: l_false,
                }));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                if !true_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_true as LabelId,
                    }));
                }
                items.push(BlockItem::BindLabel(l_false));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                if !false_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_false as LabelId,
                    }));
                }
            }
            Ok(items)
        }

        EffectfulOp::Load { .. } | EffectfulOp::Store { .. } | EffectfulOp::Call { .. } => {
            unreachable!("non-terminators handled by lower_effectful_op")
        }
    }
}

/// Build (src_reg, dst_reg, size) phi copy triples for a jump to `target` with `args`.
fn build_phi_copies(
    target: BlockId,
    args: &[ClassId],
    egraph: &EGraph,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    block_class_to_vreg: &BTreeMap<ClassId, VReg>,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
) -> Result<Vec<(Reg, Reg, OpSize)>, CompileError> {
    if args.is_empty() {
        return Ok(vec![]);
    }
    let target_block = func
        .blocks
        .iter()
        .find(|b| b.id == target)
        .ok_or_else(|| CompileError {
            phase: "phi-elim".into(),
            message: format!("jump target block {target} not found"),
            location: None,
        })?;
    let n_params = target_block.param_types.len();
    if n_params == 0 {
        return Ok(vec![]);
    }

    let mut copies = Vec::new();
    for (param_idx, &arg_cid) in args.iter().enumerate() {
        let param_cid = block_param_map
            .get(&(target, param_idx as u32))
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!(
                    "block param ({target}, {param_idx}) not found in block_param_map"
                ),
                location: None,
            })?;

        let canon_arg = egraph.unionfind.find_immutable(arg_cid);
        let arg_vreg = block_class_to_vreg
            .get(&canon_arg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("arg class {:?} not in class_to_vreg", canon_arg),
                location: None,
            })?;
        let src_reg = regalloc
            .vreg_to_reg
            .get(&arg_vreg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("arg vreg {:?} not in regalloc", arg_vreg),
                location: None,
            })?;

        let param_vreg = param_vreg_overrides
            .get(&(target, param_idx as u32))
            .copied()
            .or_else(|| class_to_vreg.get(&param_cid).copied())
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("param class {:?} not in class_to_vreg", param_cid),
                location: None,
            })?;
        let dst_reg = regalloc
            .vreg_to_reg
            .get(&param_vreg)
            .copied()
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("param vreg {:?} not in regalloc", param_vreg),
                location: None,
            })?;

        // Derive OpSize from the block parameter's type.
        let size = OpSize::from_type(&target_block.param_types[param_idx]);

        copies.push((src_reg, dst_reg, size));
    }
    Ok(copies)
}
