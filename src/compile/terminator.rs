use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::compile::split::BlockParamSlotMap;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::emit::phi_elim::phi_copies;
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::{FP_RETURN_REG, FrameLayout, GPR_RETURN_REG};
use crate::x86::addr::Addr;
use crate::x86::inst::{LabelId, MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

use super::{BlockItem, CompileError};

/// Emit a conditional jump, expanding OrdEq/UnordNe into multi-instruction sequences.
/// Returns the items and updates `next_label` if an internal label was needed.
fn emit_jcc(cc: CondCode, target: LabelId, next_label: &mut LabelId) -> Vec<BlockItem> {
    match cc {
        CondCode::OrdEq => {
            // Jump to target if ZF=1 AND PF=0:
            //   jp skip; je target; skip:
            let skip = *next_label;
            *next_label += 1;
            vec![
                BlockItem::Inst(MachInst::Jcc {
                    cc: CondCode::Parity,
                    target: skip,
                }),
                BlockItem::Inst(MachInst::Jcc {
                    cc: CondCode::Eq,
                    target,
                }),
                BlockItem::BindLabel(skip),
            ]
        }
        CondCode::UnordNe => {
            // Jump to target if ZF=0 OR PF=1:
            //   jp target; jne target
            vec![
                BlockItem::Inst(MachInst::Jcc {
                    cc: CondCode::Parity,
                    target,
                }),
                BlockItem::Inst(MachInst::Jcc {
                    cc: CondCode::Ne,
                    target,
                }),
            ]
        }
        _ => vec![BlockItem::Inst(MachInst::Jcc { cc, target })],
    }
}

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
        CondCode::Parity => CondCode::NotParity,
        CondCode::NotParity => CondCode::Parity,
        CondCode::OrdEq => CondCode::UnordNe,
        CondCode::UnordNe => CondCode::OrdEq,
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

/// A phi copy entry: either a register-to-register copy or a slot store.
///
/// Used internally in `lower_terminator` to handle Phase 6 block-param
/// slot spilling: when a block param is slot-spilled, the predecessor emits
/// a `Slot` copy (stores the arg reg to the spill slot) instead of the
/// normal register-to-register phi copy.
#[derive(Debug, Clone)]
enum PhiCopy {
    /// Normal register copy: `src -> dst`.
    Reg(Reg, Reg, OpSize),
    /// Slot store: store `src_reg` to spill slot `slot` with size `size`.
    Slot {
        src_reg: Reg,
        slot: i64,
        size: OpSize,
    },
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
///
/// `slot_spilled_params` is populated by Phase 6 when block params are slot-spilled.
/// When a jump to `target` has a slot-spilled param at index `k`, `lower_terminator`
/// emits a `SpillStore`/`XmmSpillStore` before the phi copies instead of a register copy.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_terminator(
    op: &EffectfulOp,
    block_idx: usize,
    next_block_id: Option<BlockId>,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    ret_class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    next_label: &mut LabelId,
    slot_spilled_params: &BlockParamSlotMap,
    frame_layout: &FrameLayout,
) -> Result<Vec<BlockItem>, CompileError> {
    let exit_point = ProgramPoint::block_exit(block_idx);
    let get_reg = |cid: ClassId, ctv: &ClassVRegMap| -> Option<Reg> {
        let canon = egraph.unionfind.find_immutable(cid);
        ctv.lookup(canon, exit_point)
            .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
    };

    match op {
        EffectfulOp::Ret { val } => {
            let mut items = Vec::new();
            if let Some(&ret_cid) = val.as_ref()
                && let Some(ret_reg) = get_reg(ret_cid, ret_class_to_vreg)
            {
                let is_float_ret = func.return_types.first().is_some_and(|t| t.is_float());
                let abi_reg = if is_float_ret {
                    FP_RETURN_REG
                } else {
                    GPR_RETURN_REG
                };
                if ret_reg != abi_reg {
                    if is_float_ret {
                        items.push(BlockItem::Inst(MachInst::MovsdRR {
                            dst: Operand::Reg(abi_reg),
                            src: Operand::Reg(ret_reg),
                        }));
                    } else {
                        let ret_size = func
                            .return_types
                            .first()
                            .map(OpSize::from_int_type)
                            .unwrap_or(OpSize::S64);
                        items.push(BlockItem::Inst(MachInst::MovRR {
                            size: ret_size,
                            dst: Operand::Reg(abi_reg),
                            src: Operand::Reg(ret_reg),
                        }));
                    }
                }
            }
            // Ret marker: replaced with emit_epilogue() in the encoding loop.
            items.push(BlockItem::Inst(MachInst::Ret));
            Ok(items)
        }

        EffectfulOp::Jump { target, args } => {
            let copies = build_phi_copies(
                *target,
                args,
                block_idx,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
            )?;
            let mut items: Vec<BlockItem> = emit_phi_copies(&copies, Reg::R11, frame_layout)
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
            cond: _,
            cc,
            bb_true,
            bb_false,
            true_args,
            false_args,
        } => {
            let cc = *cc;

            let true_copies = build_phi_copies(
                *bb_true,
                true_args,
                block_idx,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
            )?;
            let false_copies = build_phi_copies(
                *bb_false,
                false_args,
                block_idx,
                egraph,
                class_to_vreg,
                ret_class_to_vreg,
                block_param_map,
                param_vreg_overrides,
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
            )?;

            let true_phi = emit_phi_copies(&true_copies, Reg::R11, frame_layout);
            let false_phi = emit_phi_copies(&false_copies, Reg::R11, frame_layout);

            let false_is_fallthrough = next_block_id == Some(*bb_false);
            let true_is_fallthrough = next_block_id == Some(*bb_true);

            let mut items = Vec::new();
            if true_phi.is_empty() {
                // jcc cc, true_block; [false_phi]; jmp false_block
                items.extend(emit_jcc(cc, *bb_true as LabelId, next_label));
                items.extend(false_phi.into_iter().map(BlockItem::Inst));
                if !false_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_false as LabelId,
                    }));
                }
            } else if false_phi.is_empty() {
                // jcc !cc, false_block; [true_phi]; jmp true_block
                items.extend(emit_jcc(negate_cc(cc), *bb_false as LabelId, next_label));
                items.extend(true_phi.into_iter().map(BlockItem::Inst));
                if !true_is_fallthrough {
                    items.push(BlockItem::Inst(MachInst::Jmp {
                        target: *bb_true as LabelId,
                    }));
                }
            } else {
                // Both sides have copies. Use trampoline labels.
                let l_false = *next_label;
                *next_label += 1;

                items.extend(emit_jcc(negate_cc(cc), l_false, next_label));
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

/// Build phi copy entries for a jump to `target` with `args`.
///
/// Returns a list of `PhiCopy` values: either register-to-register copies
/// (`Reg`) or slot stores (`Slot`) for Phase 6 slot-spilled block params.
///
/// For each param:
/// - If the param VReg has a register in `regalloc`, emit `PhiCopy::Reg`.
/// - If the param VReg has NO register BUT `slot_spilled_params` has an entry
///   for `(target, param_idx)`, emit `PhiCopy::Slot` (store arg reg to slot).
/// - Otherwise skip (legacy: "flow through cross-block spill slots" path).
fn build_phi_copies(
    target: BlockId,
    args: &[ClassId],
    src_block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    slot_spilled_params: &BlockParamSlotMap,
) -> Result<Vec<PhiCopy>, CompileError> {
    if args.is_empty() {
        return Ok(vec![]);
    }
    let (target_block_idx, target_block) = func
        .blocks
        .iter()
        .enumerate()
        .find(|(_, b)| b.id == target)
        .ok_or_else(|| CompileError {
            phase: "phi-elim".into(),
            message: format!("jump target block {target} not found"),
            location: None,
        })?;
    let n_params = target_block.param_types.len();
    if n_params == 0 {
        return Ok(vec![]);
    }

    let src_exit = ProgramPoint::block_exit(src_block_idx);
    let tgt_entry = ProgramPoint::block_entry(target_block_idx);

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
            .lookup(canon_arg, src_exit)
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("arg class {:?} not in class_to_vreg", canon_arg),
                location: None,
            })?;
        let src_reg = match regalloc.vreg_to_reg.get(&arg_vreg).copied() {
            Some(r) => r,
            None => {
                // XMM values that flow through cross-block spill slots
                // are not assigned registers. Skip the phi copy; the
                // successor will load from the spill slot at block entry.
                continue;
            }
        };

        // Derive OpSize from the block parameter's type.
        // Float types use S64 here; phi_copies detects XMM registers and
        // emits MovsdRR/MovssRR instead of MovRR.
        let param_ty = &target_block.param_types[param_idx];
        let size = if param_ty.is_float() {
            OpSize::S64
        } else {
            OpSize::from_int_type(param_ty)
        };

        // Phase 6: if this param is slot-spilled, emit a slot store directly.
        // The param's segment was truncated to start after block_entry so the
        // class_to_vreg lookup at tgt_entry would fail -- we skip it entirely.
        //
        // Back-edge optimisation: if the argument class IS the same as the
        // param class (e.g. an immutable loop-carried value like `base`), the
        // slot already contains the correct value from the forward-edge store.
        // Skip the store; re-storing from an incorrect register would clobber it.
        if let Some(info) = slot_spilled_params.get(&(target, param_idx as u32)) {
            let canon_param = egraph.unionfind.find_immutable(param_cid);
            if canon_arg != canon_param {
                // Arg differs from param: emit slot store with current src_reg.
                copies.push(PhiCopy::Slot {
                    src_reg,
                    slot: info.slot,
                    size,
                });
            }
            // If canon_arg == canon_param: back-edge with unchanged value; slot
            // already has the right value from the forward edge. Skip.
            continue;
        }

        let mut param_vreg = param_vreg_overrides
            .get(&(target, param_idx as u32))
            .copied()
            .or_else(|| class_to_vreg.lookup(param_cid, tgt_entry))
            .ok_or_else(|| CompileError {
                phase: "phi-elim".into(),
                message: format!("param class {:?} not in class_to_vreg", param_cid),
                location: None,
            })?;
        // Apply coalesce aliases so a dest VReg merged away by Phase 3 resolves
        // to its canonical. Without this, vreg_to_reg lookup fails and the copy
        // is silently dropped, dropping the back-edge and miscompiling loops.
        // Source-side aliasing is already done via block_class_to_vreg (see
        // compile/mod.rs:963-1004); this is the symmetric fix for the dest side.
        while let Some(&aliased) = coalesce_aliases.get(&param_vreg) {
            if aliased == param_vreg {
                break;
            }
            param_vreg = aliased;
        }

        match regalloc.vreg_to_reg.get(&param_vreg).copied() {
            Some(dst_reg) => {
                copies.push(PhiCopy::Reg(src_reg, dst_reg, size));
            }
            None => {
                // Legacy path: param flows through cross-block spill slot.
                // Skip; the successor reloads at block entry.
            }
        }
    }
    Ok(copies)
}

/// Emit `MachInst`s for a list of `PhiCopy` entries.
///
/// Slot copies are emitted first (as spill stores), then register copies
/// are handed to `phi_copies` for Briggs-style permutation resolution.
/// This ordering is safe because slot stores write to memory (never to any
/// phi-copy destination register), so they commute with register copies.
fn emit_phi_copies(copies: &[PhiCopy], temp: Reg, frame_layout: &FrameLayout) -> Vec<MachInst> {
    let mut result = Vec::new();

    // Emit slot stores first.
    for copy in copies {
        if let PhiCopy::Slot {
            src_reg,
            slot,
            size,
        } = copy
        {
            let addr = Addr {
                base: Some(frame_layout.spill_base),
                index: None,
                scale: 1,
                disp: frame_layout.spill_offset + (*slot as i32) * 8,
            };
            // Float params use movsd (S64); integer params use mov (S64 for slots).
            if *size == OpSize::S64 && src_reg.is_xmm() {
                result.push(MachInst::MovsdMR {
                    addr,
                    src: Operand::Reg(*src_reg),
                });
            } else {
                result.push(MachInst::MovMR {
                    size: OpSize::S64,
                    addr,
                    src: Operand::Reg(*src_reg),
                });
            }
        }
    }

    // Collect register copies and run through phi_copies for permutation.
    let reg_copies: Vec<(Reg, Reg, OpSize)> = copies
        .iter()
        .filter_map(|c| {
            if let PhiCopy::Reg(src, dst, size) = c {
                Some((*src, *dst, *size))
            } else {
                None
            }
        })
        .collect();

    result.extend(phi_copies(&reg_copies, temp));
    result
}
