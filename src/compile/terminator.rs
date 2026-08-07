use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::compile::split::BlockParamSlotMap;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::emit::phi_elim::phi_copies;
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp, TermArgs};
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::regalloc::allocator::RegAllocResult;
use crate::regalloc::coalesce::chase_alias;
use crate::schedule::scheduler::ScheduledInst;
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

        // Rewrite Jcc/Jmp targets in all blocks. A rewrite that lands on the
        // target the jump already had is not a change: a cycle of trampolines
        // redirects onto itself, and counting that as progress means the outer
        // loop rebuilds the same map forever.
        let mut changed = false;
        for items in block_items.iter_mut() {
            for item in items.iter_mut() {
                if let BlockItem::Inst(MachInst::Jmp { target } | MachInst::Jcc { target, .. }) =
                    item
                    && let Some(&new_target) = redirect.get(target)
                    && new_target != *target
                {
                    *target = new_target;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Delete a jump whose target is the instruction that already follows it.
///
/// `lower_terminator` omits a jump to the block that comes next in emission
/// order, but that is only the one-step case. A jump can become a fallthrough
/// afterwards: `thread_branches` retargets a jump onto a trampoline's
/// destination, and a run of blocks that emit nothing can sit between a block
/// and the block it names. Both leave a jump over zero bytes.
///
/// The scan is over emission order, so a label bound after the jump with no
/// instruction in between names the same point the jump would reach. A `Jcc`
/// counts for the same reason a `Jmp` does: control continues at the next
/// instruction whether it is taken or not.
pub(super) fn remove_fallthrough_jumps(
    block_items: &mut [Vec<BlockItem>],
    func: &Function,
    rpo_order: &[usize],
) {
    // Backwards, so a jump made redundant by a later deletion is seen in the
    // same pass: what follows a jump can only shrink.
    for rpo_pos in (0..block_items.len()).rev() {
        for item_idx in (0..block_items[rpo_pos].len()).rev() {
            let BlockItem::Inst(MachInst::Jmp { target } | MachInst::Jcc { target, .. }) =
                block_items[rpo_pos][item_idx]
            else {
                continue;
            };
            if binds_at_next_instruction(block_items, func, rpo_order, rpo_pos, item_idx, target) {
                block_items[rpo_pos].remove(item_idx);
            }
        }
    }
}

/// True when `target` is bound at the first instruction after the item at
/// (`rpo_pos`, `item_idx`), with only label bindings in between.
fn binds_at_next_instruction(
    block_items: &[Vec<BlockItem>],
    func: &Function,
    rpo_order: &[usize],
    rpo_pos: usize,
    item_idx: usize,
    target: LabelId,
) -> bool {
    let binds = |items: &[BlockItem]| -> Option<bool> {
        for item in items {
            match item {
                BlockItem::BindLabel(label) if *label == target => return Some(true),
                BlockItem::BindLabel(_) => {}
                BlockItem::Inst(_) => return Some(false),
            }
        }
        None
    };

    if let Some(found) = binds(&block_items[rpo_pos][item_idx + 1..]) {
        return found;
    }
    for next_pos in rpo_pos + 1..block_items.len() {
        // A block's own label is bound before its first item.
        if func.blocks[rpo_order[next_pos]].id as LabelId == target {
            return true;
        }
        if let Some(found) = binds(&block_items[next_pos]) {
            return found;
        }
    }
    false
}

/// A phi copy entry: either a register-to-register copy or a slot store.
///
/// Used internally in `lower_terminator` to handle slot-routed block-param
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

/// The register holding a `Ret`'s value.
///
/// `Op::TerminatorArgs` numbers a `Ret`'s value as argument 0, so the schedule
/// names the VReg directly: it is post-split, post-coalesce, and the one the
/// allocator assigned a register to. Resolving the value's *class* instead is the
/// same reconstruction that produced seven wrong-code bugs at this seam, and here
/// it fails silently -- no register means no move, so the function returns
/// whatever the ABI register already held.
///
/// The VReg the CFG committed answers where the schedule has no operand, which
/// is the single-block path: `append_terminator_args` never runs there, so
/// `term_args` is empty. It is linearization's choice and the splitter does not
/// run on that path, so it needs only the coalesce aliases the schedule's own
/// answer needs. Measured over the whole lit corpus and 30 generated programs at
/// both levels: 463 fallbacks, all naming the register the class map named.
fn ret_value_reg(
    committed: Option<VReg>,
    term_args: &BTreeMap<u32, VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
) -> Option<Reg> {
    term_args
        .get(&0)
        .copied()
        .or(committed)
        .map(|v| chase_alias(v, coalesce_aliases))
        .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
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
/// `slot_spilled_params` is populated by the splitter's block-param slot routing.
/// When a jump to `target` has a slot-spilled param at index `k`, `lower_terminator`
/// emits a `SpillStore`/`XmmSpillStore` before the phi copies instead of a register copy.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_terminator(
    op: &EffectfulOp,
    block_idx: usize,
    next_block_id: Option<BlockId>,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    term_args: &BTreeMap<u32, VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    next_label: &mut LabelId,
    slot_spilled_params: &BlockParamSlotMap,
    frame_layout: &FrameLayout,
    block_schedules: &[Vec<ScheduledInst>],
) -> Result<Vec<BlockItem>, CompileError> {
    match op {
        EffectfulOp::Ret { val } => {
            let mut items = Vec::new();
            // A constant return value in a function that makes calls is
            // materialized straight into the ABI register instead of being
            // trusted to a register assignment.
            //
            // The assignment cannot be trusted there: the class map may resolve
            // the constant to a VReg the allocator placed in RAX, and the code
            // below then emits nothing because the value looks like it is
            // already in place. For `return 0` after a call that is exactly
            // wrong -- RAX holds the callee's return value, so main returned
            // whatever printf did.
            //
            // Restricted to functions containing a call so call-free code keeps
            // the single `mov rax, K; ret` it already emits; with no call there
            // is nothing that could have clobbered RAX behind our back.
            let func_has_calls = func
                .blocks
                .iter()
                .any(|b| b.ops.iter().any(|o| matches!(o, EffectfulOp::Call { .. })));
            let const_ret = val
                .as_ref()
                .filter(|_| func_has_calls)
                .filter(|_| !func.return_types.first().is_some_and(|t| t.is_float()))
                .and_then(|&cid| egraph.get_constant(cid.class()));
            if let Some((value, ty)) = const_ret {
                items.push(BlockItem::Inst(MachInst::MovRI {
                    size: OpSize::from_int_type(&ty),
                    dst: Operand::Reg(GPR_RETURN_REG),
                    imm: value,
                }));
            } else if let Some(&ret_cid) = val.as_ref() {
                // A returned value that resolves to no register used to emit no
                // move at all, so the function returned whatever the ABI register
                // happened to hold -- a wrong answer with nothing downstream able
                // to see it. There is no correct code to emit here, so say so.
                let ret_reg = ret_value_reg(ret_cid.vreg(), term_args, coalesce_aliases, regalloc)
                    .ok_or_else(|| CompileError {
                        phase: "lowering".into(),
                        message: format!(
                            "Ret: no register for value class {:?}; the terminator names {:?} \
                         and the CFG states {:?}",
                            egraph.unionfind.find_immutable(ret_cid.class()),
                            term_args.get(&0),
                            ret_cid.vreg(),
                        ),
                        location: None,
                    })?;
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
                block_param_map,
                term_args,
                0,
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
                block_schedules,
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
                block_param_map,
                term_args,
                0,
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
                block_schedules,
            )?;
            let false_copies = build_phi_copies(
                *bb_false,
                false_args,
                block_idx,
                egraph,
                class_to_vreg,
                block_param_map,
                term_args,
                true_args.len(),
                coalesce_aliases,
                regalloc,
                func,
                slot_spilled_params,
                block_schedules,
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
/// (`Reg`) or slot stores (`Slot`) for slot-spilled block params.
///
/// For each param:
/// - If the param VReg has a register in `regalloc`, emit `PhiCopy::Reg`.
/// - If the param VReg has NO register BUT `slot_spilled_params` has an entry
///   for `(target, param_idx)`, emit `PhiCopy::Slot` (store arg reg to slot).
/// - Otherwise skip (legacy: "flow through cross-block spill slots" path).
///
/// `term_args` maps a terminator argument index to the VReg the schedule
/// carries for it, and `arg_base` is where this edge's arguments start in that
/// numbering -- 0 for a Jump or a Branch's true edge, the true edge's argument
/// count for a Branch's false edge. An argument with no entry was routed through
/// a stack slot and needs no copy.
///
/// `block_schedules` is indexed by block position and holds the final schedules,
/// so the target block's own `BlockParam` instruction can name the destination.
#[allow(clippy::too_many_arguments)]
fn build_phi_copies(
    target: BlockId,
    args: &TermArgs,
    src_block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    term_args: &BTreeMap<u32, VReg>,
    arg_base: usize,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    slot_spilled_params: &BlockParamSlotMap,
    block_schedules: &[Vec<ScheduledInst>],
) -> Result<Vec<PhiCopy>, CompileError> {
    // The VReg beside each argument's class is what linearization chose; the
    // authority at this point is `term_args`, which the splitter rewrote and
    // coalescing renamed. The class is read for one question a VReg cannot
    // answer: whether an argument *is* the parameter it feeds.
    let args = args.as_committed().ok_or_else(|| CompileError {
        phase: "phi-elim".into(),
        message: format!("block {src_block_idx}: terminator arguments were never committed"),
        location: None,
    })?;
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

    let tgt_entry = ProgramPoint::block_entry(target_block_idx);
    // The target block's final schedule. Empty when the caller has none for it,
    // which leaves the lookups below exactly as they were.
    let target_schedule: &[ScheduledInst] = block_schedules
        .get(target_block_idx)
        .map_or(&[], |s| s.as_slice());

    let trace = crate::trace::is_enabled("phi") && crate::trace::fn_matches(&func.name);

    let mut copies = Vec::new();
    // Two parameters of one block can be the same e-class, and then they are one
    // value in one register: `propagate_block_params` merges a parameter with its
    // incoming argument for single-predecessor blocks when that argument is
    // constant, so two parameters carrying the same constant collapse onto the
    // same class. The second copy would write the value its destination already
    // holds, and a parallel copy cannot express two writes to one register.
    let mut params_copied: BTreeMap<ClassId, usize> = BTreeMap::new();
    for (param_idx, arg) in args.iter().enumerate() {
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
        let canon_param = egraph.unionfind.find_immutable(param_cid);
        if let Some(&first_idx) = params_copied.get(&canon_param) {
            if trace {
                tracing::debug!(
                    target: "blitz::phi",
                    "[{}] b{src_block_idx} -> {target} p{param_idx}: param {canon_param:?} \
                     is also p{first_idx} -- one value, one copy, SKIPPED",
                    func.name,
                );
            }
            continue;
        }

        let canon_arg = egraph.unionfind.find_immutable(arg.class);
        // The operand the schedule carries is the answer, not a hint. It is the
        // one the splitter rewrote and coalescing renamed, so it names the VReg
        // that actually holds the value at this point; re-resolving the class
        // through a map those passes mutated is what produced the wrong answers
        // this op exists to stop.
        //
        // No operand means the argument travels through a stack slot, and the
        // slot store the splitter placed in this block already wrote it -- which
        // is only true when the parameter reads that slot rather than a
        // register. Where it does not, nothing on this edge writes the
        // parameter at all: the copy is not emitted here and no store stands in
        // for it, so the block reads whatever its register last held.
        let Some(&arg_vreg) = term_args.get(&((arg_base + param_idx) as u32)) else {
            if !slot_spilled_params.contains_key(&(target, param_idx as u32)) {
                return Err(CompileError {
                    phase: "phi-elim".into(),
                    message: format!(
                        "block {src_block_idx} -> {target} p{param_idx}: argument has no \
                         operand, so nothing writes the parameter on this edge, and the \
                         parameter is not slot-routed either (param class {param_cid:?})"
                    ),
                    location: None,
                });
            }
            continue;
        };
        // Chase the alias chain, exactly as the destination side below does.
        // Without this the register lookup misses and the copy is silently
        // dropped -- which is a dropped back edge, so the loop never terminates.
        let arg_vreg = chase_alias(arg_vreg, coalesce_aliases);
        // `k=<n>` is the argument class's constant value when it has one. Two
        // params with different `k` reading the same `src` is the signature of
        // an argument resolved to the wrong VReg.
        let arg_const = match egraph.get_constant(canon_arg) {
            Some((v, _)) => format!(" k={v}"),
            None => String::new(),
        };
        let src_reg = match regalloc.vreg_to_reg.get(&arg_vreg).copied() {
            Some(r) => r,
            None => {
                // XMM values that flow through cross-block spill slots
                // are not assigned registers. Skip the phi copy; the
                // successor will load from the spill slot at block entry.
                if trace {
                    tracing::debug!(
                        target: "blitz::phi",
                        "[{}] b{src_block_idx} -> {target} p{param_idx}: arg {canon_arg:?}{arg_const} \
                         {arg_vreg:?} has no register -- SKIPPED",
                        func.name,
                    );
                }
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

        // If this param is slot-spilled, emit a slot store directly.
        // The param's segment was truncated to start after block_entry so the
        // class_to_vreg lookup at tgt_entry would fail -- we skip it entirely.
        //
        // Back-edge optimisation: if the argument class IS the same as the
        // param class (e.g. an immutable loop-carried value like `base`), the
        // slot already contains the correct value from the forward-edge store.
        // Skip the store; re-storing from an incorrect register would clobber it.
        //
        // Unless the parameter names the value it carries rather than storage of
        // its own: then that equality is the signature of the edge that feeds the
        // value in, and it is the one edge that must store.
        if let Some(info) = slot_spilled_params.get(&(target, param_idx as u32)) {
            let store = canon_arg != canon_param || info.value_alias;
            if trace {
                tracing::debug!(
                    target: "blitz::phi",
                    "[{}] b{src_block_idx} -> {target} p{param_idx}: arg {canon_arg:?}{arg_const} \
                     {arg_vreg:?} src={src_reg:?} -> slot {} {}",
                    func.name,
                    info.slot,
                    if store { "" } else { "(skipped: back-edge identity)" },
                );
            }
            if store {
                // Arg differs from param: emit slot store with current src_reg.
                copies.push(PhiCopy::Slot {
                    src_reg,
                    slot: info.slot,
                    size,
                });
                params_copied.insert(canon_param, param_idx);
            }
            // If canon_arg == canon_param: back-edge with unchanged value; slot
            // already has the right value from the forward edge. Skip.
            continue;
        }

        // The destination, from the three places that can name it -- see
        // `cfg::resolve_block_param_vreg`, which every pass touching a phi copy
        // resolves through so they cannot disagree about which VReg the copy
        // writes.
        let param_vreg = super::cfg::resolve_block_param_vreg(
            target_block,
            param_idx as u32,
            target_block_idx,
            target_schedule,
            egraph,
            class_to_vreg,
            block_param_map,
        )
        .ok_or_else(|| CompileError {
            phase: "phi-elim".into(),
            message: format!(
                "param class {param_cid:?} of ({target}, {param_idx}) not in \
                     class_to_vreg at {tgt_entry:?}, jumping from block \
                     {src_block_idx}; segments for the class: [{}]",
                class_to_vreg
                    .iter_segments()
                    .filter(|&(c, _, _, _)| c == param_cid)
                    .map(|(_, v, s, e)| format!("{v:?} {s:?}..={e:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            location: None,
        })?;
        // Apply coalesce aliases so a dest VReg coalescing merged away resolves
        // to its canonical. Without this, vreg_to_reg lookup fails and the copy
        // is silently dropped, dropping the back-edge and miscompiling loops.
        // The source side chases the same chain above.
        let param_vreg = chase_alias(param_vreg, coalesce_aliases);

        match regalloc.vreg_to_reg.get(&param_vreg).copied() {
            Some(dst_reg) => {
                if trace {
                    tracing::debug!(
                        target: "blitz::phi",
                        "[{}] b{src_block_idx} -> {target} p{param_idx}: arg {canon_arg:?}{arg_const} \
                         {arg_vreg:?} src={src_reg:?} -> param {param_cid:?} {param_vreg:?} dst={dst_reg:?}",
                        func.name,
                    );
                }
                copies.push(PhiCopy::Reg(src_reg, dst_reg, size));
                params_copied.insert(canon_param, param_idx);
            }
            None => {
                // Legacy path: param flows through cross-block spill slot.
                // Skip; the successor reloads at block entry.
                if trace {
                    tracing::debug!(
                        target: "blitz::phi",
                        "[{}] b{src_block_idx} -> {target} p{param_idx}: arg {canon_arg:?}{arg_const} \
                         {arg_vreg:?} src={src_reg:?} -> param {param_cid:?} {param_vreg:?} has no register -- SKIPPED",
                        func.name,
                    );
                }
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::BasicBlock;

    /// Blocks 0..n with ids equal to indices, which is what lets `rpo_order` be
    /// the identity below and a `LabelId` be a block index.
    fn func_of(n: usize) -> Function {
        Function {
            name: "f".to_string(),
            param_types: vec![],
            return_types: vec![],
            next_block_id: n as BlockId,
            blocks: (0..n)
                .map(|i| BasicBlock {
                    id: i as BlockId,
                    param_types: vec![],
                    param_vregs: vec![],
                    ops: vec![EffectfulOp::Ret { val: None }],
                })
                .collect(),
            param_class_ids: vec![],
            egraph: None,
            stack_slots: vec![],
            noinline: false,
        }
    }

    fn jmp(target: LabelId) -> BlockItem {
        BlockItem::Inst(MachInst::Jmp { target })
    }

    fn mov() -> BlockItem {
        BlockItem::Inst(MachInst::MovRR {
            size: OpSize::S64,
            dst: Operand::Reg(Reg::RAX),
            src: Operand::Reg(Reg::RCX),
        })
    }

    /// `thread_branches` over blocks in index order, returning the target of
    /// each block's jump.
    fn thread(mut items: Vec<Vec<BlockItem>>) -> Vec<Option<LabelId>> {
        let func = func_of(items.len());
        let rpo: Vec<usize> = (0..items.len()).collect();
        thread_branches(&mut items, &func, &rpo);
        items
            .iter()
            .map(|block| {
                block.iter().find_map(|item| match item {
                    BlockItem::Inst(MachInst::Jmp { target }) => Some(*target),
                    _ => None,
                })
            })
            .collect()
    }

    /// A chain of trampolines collapses in one call: every jump into it lands on
    /// the block that does real work, not on the next hop.
    #[test]
    fn a_chain_of_empty_blocks_is_threaded_to_its_end() {
        let targets = thread(vec![
            vec![jmp(1)],
            vec![jmp(2)],
            vec![jmp(3)],
            vec![BlockItem::BindLabel(3), mov()],
        ]);
        assert_eq!(targets[0], Some(3));
        assert_eq!(targets[1], Some(3));
    }

    /// A `Jcc` is threaded exactly as a `Jmp` is; both name a block, and a block
    /// that only jumps on is not a destination either way.
    #[test]
    fn a_conditional_target_is_threaded_too() {
        let mut items = vec![
            vec![BlockItem::Inst(MachInst::Jcc {
                cc: CondCode::Ne,
                target: 1,
            })],
            vec![jmp(2)],
            vec![mov()],
        ];
        let func = func_of(3);
        thread_branches(&mut items, &func, &[0, 1, 2]);
        assert!(matches!(
            items[0][0],
            BlockItem::Inst(MachInst::Jcc { target: 2, .. }),
        ));
    }

    /// Only a block whose whole body is one jump is a trampoline. A block that
    /// carries a phi copy has work to do, so a jump to it must survive.
    #[test]
    fn a_block_that_does_work_is_not_threaded_through() {
        let targets = thread(vec![vec![jmp(1)], vec![mov(), jmp(2)], vec![mov()]]);
        assert_eq!(targets[0], Some(1));
    }

    /// A loop of empty blocks is a cycle in the redirect map, and resolving it
    /// must terminate rather than chase the cycle forever.
    #[test]
    fn a_cycle_of_empty_blocks_terminates() {
        let targets = thread(vec![vec![jmp(1)], vec![jmp(2)], vec![jmp(1)]]);
        assert!(targets[0].is_some());
    }

    /// `remove_fallthrough_jumps` over blocks in index order, returning the
    /// instruction count of each block.
    fn fallthrough(items: &mut Vec<Vec<BlockItem>>) -> Vec<usize> {
        let func = func_of(items.len());
        let rpo: Vec<usize> = (0..items.len()).collect();
        remove_fallthrough_jumps(items, &func, &rpo);
        items
            .iter()
            .map(|block| {
                block
                    .iter()
                    .filter(|item| matches!(item, BlockItem::Inst(_)))
                    .count()
            })
            .collect()
    }

    /// A jump to the block that comes next emits no bytes and moves nothing.
    #[test]
    fn a_jump_to_the_following_block_is_deleted() {
        let counts = fallthrough(&mut vec![vec![mov(), jmp(1)], vec![mov()]]);
        assert_eq!(counts, vec![1, 1]);
    }

    /// Blocks that emit nothing lie between a jump and its target without
    /// separating them, so the jump is still a fallthrough.
    #[test]
    fn empty_blocks_between_a_jump_and_its_target_do_not_save_it() {
        let counts = fallthrough(&mut vec![vec![jmp(3)], vec![], vec![], vec![mov()]]);
        assert_eq!(counts, vec![0, 0, 0, 1]);
    }

    /// A jump over a block that emits an instruction goes somewhere.
    #[test]
    fn a_jump_over_real_work_is_kept() {
        let counts = fallthrough(&mut vec![vec![jmp(2)], vec![mov()], vec![mov()]]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    /// A conditional jump continues at the next instruction whether it is taken
    /// or not, so one that names that instruction is deleted too.
    #[test]
    fn a_conditional_jump_to_the_next_instruction_is_deleted() {
        let counts = fallthrough(&mut vec![
            vec![BlockItem::Inst(MachInst::Jcc {
                cc: CondCode::Ne,
                target: 1,
            })],
            vec![mov()],
        ]);
        assert_eq!(counts, vec![0, 1]);
    }

    /// A trampoline label bound in the middle of a block names the instruction
    /// that follows it, exactly as a block label does.
    #[test]
    fn a_jump_to_a_trampoline_label_just_ahead_is_deleted() {
        let counts = fallthrough(&mut vec![vec![jmp(7), BlockItem::BindLabel(7), mov()]]);
        assert_eq!(counts, vec![1]);
    }

    /// Deleting one jump can leave the one in front of it jumping over nothing.
    /// The backward pass sees that in the same run.
    #[test]
    fn a_jump_made_redundant_by_a_deletion_is_deleted_too() {
        let counts = fallthrough(&mut vec![vec![jmp(1), jmp(1)], vec![mov()]]);
        assert_eq!(counts, vec![0, 1]);
    }

    /// The last block has nothing after it, so its jump names a point that is
    /// not the next instruction.
    #[test]
    fn a_jump_out_of_the_last_block_is_kept() {
        let counts = fallthrough(&mut vec![vec![mov()], vec![jmp(0)]]);
        assert_eq!(counts, vec![1, 1]);
    }

    fn regalloc_of(pairs: &[(u32, Reg)]) -> RegAllocResult {
        RegAllocResult {
            vreg_to_reg: pairs.iter().map(|&(v, r)| (VReg(v), r)).collect(),
            spill_slots: 0,
            callee_saved_used: vec![],
            insts: vec![],
            unprecolored_params: vec![],
        }
    }

    /// A `Ret`'s value is argument 0 of `Op::TerminatorArgs`, and the schedule's
    /// answer is the post-split, post-coalesce one -- so it answers ahead of the
    /// VReg the CFG committed.
    #[test]
    fn ret_value_prefers_the_schedules_argument() {
        let regalloc = regalloc_of(&[(1, Reg::RAX), (2, Reg::RCX)]);
        let term_args = BTreeMap::from([(0, VReg(2))]);
        assert_eq!(
            ret_value_reg(Some(VReg(1)), &term_args, &BTreeMap::new(), &regalloc),
            Some(Reg::RCX),
        );
    }

    /// The single-block path has no terminator arguments at all --
    /// `append_terminator_args` never runs there -- so the CFG's committed VReg
    /// is what names the register, chased through the coalesce aliases as the
    /// schedule's own answer is.
    #[test]
    fn ret_value_falls_back_to_the_committed_vreg_through_its_alias() {
        let regalloc = regalloc_of(&[(3, Reg::RDX)]);
        let aliases = BTreeMap::from([(VReg(1), VReg(3))]);
        assert_eq!(
            ret_value_reg(Some(VReg(1)), &BTreeMap::new(), &aliases, &regalloc),
            Some(Reg::RDX),
        );
        assert_eq!(
            ret_value_reg(None, &BTreeMap::new(), &aliases, &regalloc),
            None,
            "no value and no argument is a void return",
        );
    }
}
