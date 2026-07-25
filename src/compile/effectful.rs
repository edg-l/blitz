use std::collections::BTreeMap;

use crate::compile::program_point::ProgramPoint;
use crate::egraph::extract::{ClassVRegMap, ExtractionResult, VReg};
use crate::egraph::unionfind::UnionFind;
use crate::ir::Type;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::{ArgLoc, FP_RETURN_REG, GPR_RETURN_REG, assign_args, setup_call_args};
use crate::x86::addr::Addr;
use crate::x86::inst::{MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

use crate::schedule::scheduler::ScheduledInst;

use super::{CompileError, IrLocation};

/// Build an `Addr` for Load/Store by checking if `addr_cid` extracted to an Addr node
/// AND the addr VReg is an actual Addr instruction in the current schedule.
///
/// Addr folding replaces the LEA with a complex addressing mode `[base + index*scale + disp]`,
/// using the Addr's children registers directly. This is only valid when those children's
/// registers hold the correct values at the load/store point. If the addr VReg came from a
/// SpillLoad or cross-block import, the children's registers may be stale.
fn build_mem_addr(
    addr_cid: ClassId,
    addr_reg: Reg,
    block_idx: usize,
    barrier_pos: Option<usize>,
    _extraction: &ExtractionResult,
    class_to_vreg: &ClassVRegMap,
    regalloc: &RegAllocResult,
    conflict_reg: Option<Reg>,
    schedule: &[ScheduledInst],
) -> Addr {
    let point = ProgramPoint::block_exit(block_idx);
    // Only fold if the addr VReg's scheduled instruction is an Addr op.
    // When it's a SpillLoad, BlockParam, or other non-Addr op, the extraction
    // may show an Addr node for the class, but the children's registers aren't
    // guaranteed live at the load/store point.
    // Fold only when the Addr's own scheduled instruction is available: its
    // operands are the VRegs the LEA would have used, which is what the folded
    // addressing mode has to reproduce.
    //
    // Re-resolving `ext.children` through the class map instead is what
    // produced `mov [rax+rax*1]`: the map returned a VReg for the index class
    // that the allocator had placed in RAX, while the instruction that computed
    // the address had used a different VReg in R13. One class can have several
    // VRegs once values are rematerialized, and only the instruction knows
    // which one it read.
    let addr_vreg = class_to_vreg.lookup(addr_cid, point);
    let addr_inst = addr_vreg.and_then(|v| {
        schedule
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.dst == v && matches!(inst.op, Op::Addr { .. }))
    });

    if let Some((addr_pos, inst)) = addr_inst
        && let Op::Addr { scale, disp } = &inst.op
    {
        let reg_of = |v: VReg| regalloc.vreg_to_reg.get(&v).copied();
        let base_reg = inst.operands.first().copied().and_then(reg_of);
        let index_reg = inst
            .operands
            .get(1)
            .copied()
            .filter(|_| *scale != 0)
            .and_then(reg_of);

        // Those registers only still hold the address components if nothing
        // between the Addr and this access overwrote them. The allocator is
        // free to reuse them as soon as the components die, which they do
        // immediately: the Addr was their only consumer.
        let clobbered = barrier_pos.is_none_or(|end| {
            addr_pos >= end
                || schedule[addr_pos + 1..end].iter().any(|between| {
                    reg_of(between.dst).is_some_and(|r| Some(r) == base_reg || Some(r) == index_reg)
                })
        });

        if let Some(base) = base_reg
            && !clobbered
        {
            // If the folded base or index register conflicts with an operand
            // read simultaneously (e.g. the Store value), fall back to the
            // pre-computed addr_reg to avoid clobbering.
            if let Some(cr) = conflict_reg
                && (base == cr || index_reg == Some(cr))
            {
                return Addr {
                    base: Some(addr_reg),
                    index: None,
                    scale: 1,
                    disp: 0,
                };
            }
            return Addr {
                base: Some(base),
                index: index_reg,
                scale: *scale,
                disp: *disp,
            };
        }
    }
    Addr {
        base: Some(addr_reg),
        index: None,
        scale: 1,
        disp: 0,
    }
}

/// Lower a non-terminator effectful op (Load, Store, Call) to MachInsts.
pub(super) fn lower_effectful_op(
    op: &EffectfulOp,
    block_idx: usize,
    barrier_idx: usize,
    class_to_vreg: &ClassVRegMap,
    regalloc: &RegAllocResult,
    extraction: &ExtractionResult,
    func: &Function,
    uf: &UnionFind,
    schedule: &[ScheduledInst],
) -> Result<Vec<MachInst>, CompileError> {
    // Schedule position of this barrier, used both to resolve operands at the
    // right point and to bound the clobber scan in `build_mem_addr`.
    let barrier_pos = schedule
        .iter()
        .enumerate()
        .filter(|(_, i)| {
            matches!(
                i.op,
                Op::CallResult(_, _)
                    | Op::LoadResult(_, _)
                    | Op::VoidCallBarrier
                    | Op::StoreBarrier
            )
        })
        .map(|(idx, _)| idx)
        .nth(barrier_idx);
    // Resolve at this barrier's own program point, not the block exit. Once the
    // splitter has divided a live range, a class maps to different VRegs at
    // different points, and a block-exit lookup hands back the register the
    // value occupied before it was spilled -- so a load reads the stale
    // register instead of the reload.
    let point = match barrier_pos {
        Some(pos) if pos > 0 => ProgramPoint::inst_point(block_idx, pos),
        Some(_) => ProgramPoint::block_entry(block_idx),
        None => ProgramPoint::block_exit(block_idx),
    };
    let get_reg = |cid: ClassId| -> Option<Reg> {
        let canon = uf.find_immutable(cid);
        class_to_vreg
            .lookup(canon, point)
            .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
    };

    // The address of a load or store is a ClassId in the CFG, not a VReg in the
    // schedule, so the splitter's operand rewriting never reaches it. When the
    // address has been spilled, resolving the class gives the register it lived
    // in *before* the spill while the reload sits in a different one. Go
    // through the barrier's operands, which the splitter does rewrite.
    let resolve_addr = |cid: ClassId| -> Option<Reg> {
        resolve_arg_regs_after_spilling(
            &[cid],
            barrier_pos.and_then(|p| schedule.get(p)),
            point,
            class_to_vreg,
            regalloc,
            uf,
            schedule,
        )
        .first()
        .copied()
        .flatten()
        .or_else(|| get_reg(cid))
    };

    match op {
        EffectfulOp::Load { addr, result, ty } => {
            let is_float = matches!(ty, Type::F32 | Type::F64);
            let canon_addr = uf.find_immutable(*addr);
            let addr_reg = resolve_addr(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Load: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let canon_result = uf.find_immutable(*result);
            let result_reg = class_to_vreg
                .lookup(canon_result, point)
                .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
                .ok_or_else(|| CompileError {
                    phase: "lowering".into(),
                    message: "Load: no register for result".into(),
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: None,
                        inst: None,
                    }),
                })?;
            let addr = build_mem_addr(
                canon_addr,
                addr_reg,
                block_idx,
                barrier_pos,
                extraction,
                class_to_vreg,
                regalloc,
                None,
                schedule,
            );
            let inst = if is_float {
                match ty {
                    Type::F64 => MachInst::MovsdRM {
                        dst: Operand::Reg(result_reg),
                        addr,
                    },
                    _ => MachInst::MovssRM {
                        dst: Operand::Reg(result_reg),
                        addr,
                    },
                }
            } else {
                let load_size = OpSize::from_int_type(ty);
                // S8/S16 loads must use zero-extending loads (MovzxBRM/MovzxWRM) to
                // avoid partial register writes that leave upper bits unchanged.
                match load_size {
                    OpSize::S8 => MachInst::MovzxBRM {
                        dst: Operand::Reg(result_reg),
                        addr,
                    },
                    OpSize::S16 => MachInst::MovzxWRM {
                        dst: Operand::Reg(result_reg),
                        addr,
                    },
                    _ => MachInst::MovRM {
                        size: load_size,
                        dst: Operand::Reg(result_reg),
                        addr,
                    },
                }
            };
            Ok(vec![inst])
        }
        EffectfulOp::Store { addr, val, ty } => {
            let is_float = matches!(ty, Type::F32 | Type::F64);
            let canon_addr = uf.find_immutable(*addr);
            let addr_reg = resolve_addr(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Store: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            // If val was spilled, its original VReg has no register. Find the
            // StoreBarrier for this Store and read the (possibly renamed) val
            // operand from it — that VReg points at the reload/remat copy.
            let canon_val = uf.find_immutable(*val);
            let val_reg = get_reg(canon_val)
                .or_else(|| {
                    resolve_store_val_reg_after_spilling(
                        canon_addr,
                        canon_val,
                        block_idx,
                        class_to_vreg,
                        regalloc,
                        extraction,
                        schedule,
                    )
                })
                .ok_or_else(|| CompileError {
                    phase: "lowering".into(),
                    message: "Store: no register for val".into(),
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: None,
                        inst: None,
                    }),
                })?;
            let addr = build_mem_addr(
                canon_addr,
                addr_reg,
                block_idx,
                barrier_pos,
                extraction,
                class_to_vreg,
                regalloc,
                Some(val_reg),
                schedule,
            );
            let inst = if is_float {
                match ty {
                    Type::F64 => MachInst::MovsdMR {
                        addr,
                        src: Operand::Reg(val_reg),
                    },
                    _ => MachInst::MovssMR {
                        addr,
                        src: Operand::Reg(val_reg),
                    },
                }
            } else {
                let store_size = OpSize::from_int_type(ty);
                MachInst::MovMR {
                    size: store_size,
                    addr,
                    src: Operand::Reg(val_reg),
                }
            };
            Ok(vec![inst])
        }
        EffectfulOp::Call {
            func: callee,
            args,
            arg_tys,
            ret_tys,
            results,
        } => {
            // After spilling, the original arg vregs may share a register
            // (their defs are short-lived after the spill store). The actual
            // values at the call point live in SpillLoad vregs, which have
            // distinct registers. Find those registers by tracing spill slots.
            let spill_reload_regs = resolve_arg_regs_after_spilling(
                args,
                barrier_pos.and_then(|p| schedule.get(p)),
                point,
                class_to_vreg,
                regalloc,
                uf,
                schedule,
            );

            let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
            for (i, &cid) in args.iter().enumerate() {
                let r = spill_reload_regs
                    .get(i)
                    .copied()
                    .flatten()
                    .or_else(|| get_reg(cid))
                    .ok_or_else(|| CompileError {
                        phase: "lowering".into(),
                        message: format!("Call: no register for argument class {cid:?}"),
                        location: Some(IrLocation {
                            function: func.name.clone(),
                            block: None,
                            inst: None,
                        }),
                    })?;
                arg_regs.push(r);
            }

            let mut insts = setup_call_args(arg_tys, &arg_regs, Reg::R11);

            // Count stack args so we can clean up RSP after the call.
            let locs = assign_args(arg_tys);
            let n_stack = locs
                .iter()
                .filter(|l| matches!(l, ArgLoc::Stack { .. }))
                .count();

            // SysV AMD64: AL holds the number of vector registers used to pass
            // arguments. A variadic callee branches on it to decide whether to
            // spill XMM0-7 into its register save area, so a stale AL of zero
            // makes `printf("%f", x)` read a save area that was never written.
            //
            // Blitz cannot tell a variadic callee from a fixed one -- tinyc
            // prototypes carry no `...` -- so set AL on every call. A
            // non-variadic callee ignores it, and AL is caller-saved, so this
            // costs 2 bytes and clobbers nothing live. `mov` is deliberate:
            // `xor al, al` would be shorter for the zero case but writes
            // EFLAGS, which may be live across the argument setup.
            let n_xmm_args = locs
                .iter()
                .filter(|l| matches!(l, ArgLoc::Reg(r) if r.is_xmm()))
                .count();
            insts.push(MachInst::MovRI {
                size: OpSize::S8,
                dst: Operand::Reg(Reg::RAX),
                imm: n_xmm_args as i64,
            });

            insts.push(MachInst::CallDirect {
                target: callee.clone(),
            });

            // Clean up stack arguments after the call.
            if n_stack > 0 {
                insts.push(MachInst::AddRI {
                    size: OpSize::S64,
                    dst: Operand::Reg(Reg::RSP),
                    imm: (n_stack as i32) * 8,
                });
            }

            // After the call, the first GPR return value is in RAX.
            // If a CallResult ClassId was allocated to a different register, emit a MOV.
            //
            // Known limitation: caller-saved registers (RAX, RCX, RDX, RSI, RDI, R8-R11)
            // are not modeled as clobbered by the call. VRegs live across the call may
            // be incorrectly assigned to caller-saved registers and corrupted.
            if let Some(&result_cid) = results.first()
                && let Some(result_reg) = get_reg(result_cid)
            {
                let is_float_ret = ret_tys.first().is_some_and(|t| t.is_float());
                let abi_reg = if is_float_ret {
                    FP_RETURN_REG
                } else {
                    GPR_RETURN_REG
                };
                if result_reg != abi_reg {
                    if is_float_ret {
                        insts.push(MachInst::MovsdRR {
                            dst: Operand::Reg(result_reg),
                            src: Operand::Reg(abi_reg),
                        });
                    } else {
                        let ret_size = ret_tys
                            .first()
                            .map(OpSize::from_int_type)
                            .unwrap_or(OpSize::S64);
                        insts.push(MachInst::MovRR {
                            size: ret_size,
                            dst: Operand::Reg(result_reg),
                            src: Operand::Reg(abi_reg),
                        });
                    }
                }
            }
            Ok(insts)
        }
        EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => {
            unreachable!("terminators must be handled separately")
        }
    }
}

/// Resolve the physical register for each call argument after spilling.
///
/// After spilling, the original arg VRegs may have been replaced in the
/// CallResult/VoidCallBarrier operands by SpillLoad or rematerialized VRegs.
/// This function finds the replacement VRegs and returns their physical
/// registers.
///
/// Returns a Vec with one Option<Reg> per arg.
fn resolve_arg_regs_after_spilling(
    args: &[ClassId],
    barrier: Option<&ScheduledInst>,
    point: ProgramPoint,
    class_to_vreg: &ClassVRegMap,
    regalloc: &RegAllocResult,
    uf: &UnionFind,
    schedule: &[ScheduledInst],
) -> Vec<Option<Reg>> {
    let fallback = || -> Vec<Option<Reg>> {
        args.iter()
            .map(|&cid| {
                let canon = uf.find_immutable(cid);
                class_to_vreg
                    .lookup(canon, point)
                    .and_then(|v| regalloc.vreg_to_reg.get(&v).copied())
            })
            .collect()
    };

    let Some(barrier) = barrier else {
        return fallback();
    };

    // Build a lookup from VReg -> defining instruction op for barrier operands.
    let barrier_op_defs: BTreeMap<VReg, &ScheduledInst> = schedule
        .iter()
        .filter(|inst| barrier.operands.contains(&inst.dst))
        .map(|inst| (inst.dst, inst))
        .collect();

    // Build spill slot -> original VReg mapping.
    let mut slot_to_original_vreg: BTreeMap<i64, VReg> = BTreeMap::new();
    for inst in schedule {
        if let Op::SpillStore(slot) | Op::XmmSpillStore(slot) = &inst.op
            && let Some(&original_vreg) = inst.operands.first()
        {
            slot_to_original_vreg.insert(*slot, original_vreg);
        }
    }

    // Build original VReg -> defining instruction.
    let mut original_def_insts: BTreeMap<VReg, &ScheduledInst> = BTreeMap::new();
    for inst in schedule {
        original_def_insts.entry(inst.dst).or_insert(inst);
    }

    // For each original arg VReg, find its replacement in the barrier operands.
    // The replacement may be:
    // 1. The original VReg itself (not spilled)
    // 2. A SpillLoad/XmmSpillLoad from the same slot
    // 3. A rematerialized VReg with the same op
    let mut original_to_replacement: BTreeMap<VReg, VReg> = BTreeMap::new();
    for &cid in args {
        let canon = uf.find_immutable(cid);
        let Some(original_vreg) = class_to_vreg.lookup(canon, point) else {
            continue;
        };

        // Case 1: original VReg is still in barrier operands.
        if barrier.operands.contains(&original_vreg) {
            original_to_replacement.insert(original_vreg, original_vreg);
            continue;
        }

        // Case 2: find SpillLoad for same slot.
        let mut found = false;
        for (&op_vreg, &def_inst) in &barrier_op_defs {
            if let Op::SpillLoad(slot) | Op::XmmSpillLoad(slot) = &def_inst.op
                && let Some(&stored_vreg) = slot_to_original_vreg.get(slot)
                && stored_vreg == original_vreg
            {
                original_to_replacement.insert(original_vreg, op_vreg);
                found = true;
                break;
            }
        }
        if found {
            continue;
        }

        // Case 3: find a rematerialized VReg computing the same value.
        //
        // Both the op and its operands must match. Matching on the op alone
        // picks any instruction of the same shape, and a block full of
        // `arr[i]` addresses has one `Op::Addr { scale: 1, disp: 0 }` per
        // element -- so a store to arr[6] would happily take the address of
        // some other element, or of an uninitialized register.
        if let Some(orig_inst) = original_def_insts.get(&original_vreg) {
            for (&op_vreg, &def_inst) in &barrier_op_defs {
                if op_vreg != original_vreg
                    && def_inst.op == orig_inst.op
                    && def_inst.operands == orig_inst.operands
                {
                    original_to_replacement.insert(original_vreg, op_vreg);
                    break;
                }
            }
        }
    }

    // Build the result: for each arg, use the replacement VReg's register.
    args.iter()
        .map(|&cid| {
            let canon = uf.find_immutable(cid);
            let original_vreg = class_to_vreg.lookup(canon, point)?;
            if let Some(&replacement) = original_to_replacement.get(&original_vreg) {
                return regalloc.vreg_to_reg.get(&replacement).copied();
            }
            regalloc.vreg_to_reg.get(&original_vreg).copied()
        })
        .collect()
}

/// Resolve the physical register for a Store's `val` operand after spilling.
///
/// When the val VReg is spilled, its original VReg has no register
/// assignment. The StoreBarrier for this Store has the operand renamed to a
/// SpillLoad/remat VReg which does have a register. Find the matching
/// StoreBarrier (identified by containing this Store's addr VReg in its
/// operand set) and locate the replacement val VReg in the barrier operands.
fn resolve_store_val_reg_after_spilling(
    canon_addr: ClassId,
    canon_val: ClassId,
    block_idx: usize,
    class_to_vreg: &ClassVRegMap,
    regalloc: &RegAllocResult,
    extraction: &ExtractionResult,
    schedule: &[ScheduledInst],
) -> Option<Reg> {
    let point = ProgramPoint::block_exit(block_idx);
    let addr_vreg = class_to_vreg.lookup(canon_addr, point)?;
    let original_val_vreg = class_to_vreg.lookup(canon_val, point)?;

    // `populate_effectful_operands` sorts StoreBarrier operands by VReg index
    // and may add Addr children, so we cannot rely on positional lookup.
    // Match by membership: find the StoreBarrier whose operands contain the
    // addr VReg for this specific store.
    let barrier = schedule
        .iter()
        .find(|inst| matches!(inst.op, Op::StoreBarrier) && inst.operands.contains(&addr_vreg))?;

    // Build a VReg -> defining instruction lookup for this barrier's operands.
    let barrier_op_defs: BTreeMap<VReg, &ScheduledInst> = schedule
        .iter()
        .filter(|inst| barrier.operands.contains(&inst.dst))
        .map(|inst| (inst.dst, inst))
        .collect();

    // Build a spill-slot -> original-VReg map so we can trace reloads back to
    // the VReg whose value they reload.
    let mut slot_to_original_vreg: BTreeMap<i64, VReg> = BTreeMap::new();
    for inst in schedule {
        if let Op::SpillStore(slot) | Op::XmmSpillStore(slot) = &inst.op
            && let Some(&original_vreg) = inst.operands.first()
        {
            slot_to_original_vreg.insert(*slot, original_vreg);
        }
    }

    // If the original val is still among the barrier's operands, use it.
    if barrier.operands.contains(&original_val_vreg) {
        return regalloc.vreg_to_reg.get(&original_val_vreg).copied();
    }

    // A SpillLoad in the barrier operands that reloads val's slot.
    for (&op_vreg, def_inst) in &barrier_op_defs {
        if let Op::SpillLoad(slot) | Op::XmmSpillLoad(slot) = &def_inst.op
            && slot_to_original_vreg.get(slot) == Some(&original_val_vreg)
        {
            return regalloc.vreg_to_reg.get(&op_vreg).copied();
        }
    }

    // Remat: val was rematerialized before each use and its original def was
    // dropped from the schedule. Match by op against extraction.choices[val].
    let val_choice = &extraction.choices.get(&canon_val)?.op;
    for (&op_vreg, def_inst) in &barrier_op_defs {
        if op_vreg != original_val_vreg && &def_inst.op == val_choice {
            return regalloc.vreg_to_reg.get(&op_vreg).copied();
        }
    }
    None
}
