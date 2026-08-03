use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::compile::split::BlockParamSlotMap;
use crate::egraph::EGraph;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::emit::phi_elim::phi_copies;
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::regalloc::allocator::RegAllocResult;
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
/// The class map stays as the fallback for the single-block path, where
/// `append_terminator_args` never ran and `term_args` is empty.
fn ret_value_reg(
    ret_cid: ClassId,
    term_args: &BTreeMap<u32, VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    ret_class_to_vreg: &ClassVRegMap,
    get_reg: &impl Fn(ClassId, &ClassVRegMap) -> Option<Reg>,
) -> Option<Reg> {
    term_args
        .get(&0)
        .copied()
        .map(|v| chase_alias(v, coalesce_aliases))
        .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
        .or_else(|| get_reg(ret_cid, ret_class_to_vreg))
}

/// Follow a coalescing alias chain to the VReg that survived the merge.
///
/// The map is transitive, and a single step leaves a VReg with no register
/// assignment -- which reads as "no answer" and drops whatever copy was being
/// emitted.
pub(super) fn chase_alias(mut vreg: VReg, coalesce_aliases: &BTreeMap<VReg, VReg>) -> VReg {
    while let Some(&aliased) = coalesce_aliases.get(&vreg) {
        if aliased == vreg {
            break;
        }
        vreg = aliased;
    }
    vreg
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
    ret_class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    block_param_vregs: &BTreeMap<(BlockId, u32), VReg>,
    term_args: &BTreeMap<u32, VReg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    next_label: &mut LabelId,
    slot_spilled_params: &BlockParamSlotMap,
    frame_layout: &FrameLayout,
    block_schedules: &[Vec<ScheduledInst>],
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
                .and_then(|&cid| egraph.get_constant(cid));
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
                let ret_reg = ret_value_reg(
                    ret_cid,
                    term_args,
                    coalesce_aliases,
                    regalloc,
                    ret_class_to_vreg,
                    &get_reg,
                )
                .ok_or_else(|| CompileError {
                    phase: "lowering".into(),
                    message: format!(
                        "Ret: no register for value class {:?}; the terminator names {:?} \
                         and the class map says {:?} at {exit_point:?}",
                        egraph.unionfind.find_immutable(ret_cid),
                        term_args.get(&0),
                        ret_class_to_vreg
                            .lookup(egraph.unionfind.find_immutable(ret_cid), exit_point),
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
                param_vreg_overrides,
                block_param_vregs,
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
                param_vreg_overrides,
                block_param_vregs,
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
                param_vreg_overrides,
                block_param_vregs,
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
/// numbering -- 0 for a Jump or a Branch's true edge, `true_args.len()` for a
/// Branch's false edge. An argument with no entry was routed through a stack
/// slot and needs no copy.
///
/// `block_schedules` is indexed by block position and holds the final schedules,
/// so the target block's own `BlockParam` instruction can name the destination.
#[allow(clippy::too_many_arguments)]
fn build_phi_copies(
    target: BlockId,
    args: &[ClassId],
    src_block_idx: usize,
    egraph: &EGraph,
    class_to_vreg: &ClassVRegMap,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
    param_vreg_overrides: &BTreeMap<(BlockId, u32), VReg>,
    block_param_vregs: &BTreeMap<(BlockId, u32), VReg>,
    term_args: &BTreeMap<u32, VReg>,
    arg_base: usize,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    regalloc: &RegAllocResult,
    func: &Function,
    slot_spilled_params: &BlockParamSlotMap,
    block_schedules: &[Vec<ScheduledInst>],
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

        let canon_arg = egraph.unionfind.find_immutable(arg_cid);
        // The operand the schedule carries is the answer, not a hint. It is the
        // one the splitter rewrote and coalescing renamed, so it names the VReg
        // that actually holds the value at this point; re-resolving the class
        // through a map those passes mutated is what produced the wrong answers
        // this op exists to stop.
        //
        // No operand means the argument travels through a stack slot, and the
        // slot store the splitter placed in this block already wrote it.
        let Some(&arg_vreg) = term_args.get(&((arg_base + param_idx) as u32)) else {
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
        if let Some(info) = slot_spilled_params.get(&(target, param_idx as u32)) {
            if trace {
                tracing::debug!(
                    target: "blitz::phi",
                    "[{}] b{src_block_idx} -> {target} p{param_idx}: arg {canon_arg:?}{arg_const} \
                     {arg_vreg:?} src={src_reg:?} -> slot {} {}",
                    func.name,
                    info.slot,
                    if canon_arg == canon_param { "(skipped: back-edge identity)" } else { "" },
                );
            }
            if canon_arg != canon_param {
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

        // The target block's own `BlockParam` instruction is asked first: that is
        // the VReg the block reads, so a copy into anything else writes a register
        // nobody looks at. One class can name several VRegs, and here the two
        // answers came apart -- the class resolved to a VReg in RAX while the
        // block's schedule read RSI, so a loop counter started at whatever RSI
        // held and the loop was skipped entirely.
        //
        // `class_to_vreg` comes next, because where a reload covers the target's
        // entry that reload is what the block reads.
        //
        // `block_param_vregs` backs it up with what linearization decided. A
        // param that passes a dominating definition straight through gets no
        // BlockParam of its own, so once the splitter cross-block-spills that
        // definition and truncates its segment to the defining block, nothing
        // else names the value here -- 9 of 40 generated programs failed to
        // compile at -O1 on exactly that.
        let param_vreg = target_schedule
            .iter()
            .find_map(|inst| match inst.op {
                Op::BlockParam(bid, pidx, _) if bid == target && pidx == param_idx as u32 => {
                    Some(inst.dst)
                }
                _ => None,
            })
            .or_else(|| {
                param_vreg_overrides
                    .get(&(target, param_idx as u32))
                    .copied()
            })
            .or_else(|| class_to_vreg.lookup(param_cid, tgt_entry))
            .or_else(|| block_param_vregs.get(&(target, param_idx as u32)).copied())
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
