use crate::compile::program_point::ProgramPoint;
use crate::egraph::extract::VReg;
use crate::egraph::unionfind::UnionFind;
use crate::ir::Type;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{Op, PseudoOp, PureOp};
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::{
    ArgLoc, FP_RETURN_REG, GPR_RETURN_REG, assign_args, setup_call_args, stack_arg_bytes,
};
use crate::x86::addr::Addr;
use crate::x86::inst::{MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

use crate::schedule::scheduler::ScheduledInst;

use super::{CompileError, IrLocation, barrier};

/// Build an `Addr` for Load/Store, folding the address computation into the
/// addressing mode when `addr_vreg` is the destination of an `Addr` instruction
/// in this schedule.
///
/// Addr folding replaces the LEA with a complex addressing mode `[base + index*scale + disp]`,
/// using the Addr's children registers directly. This is only valid when those children's
/// registers hold the correct values at the load/store point. If the addr VReg came from a
/// SpillLoad or cross-block import, the children's registers may be stale.
///
/// `addr_vreg` is the VReg the barrier declares for the address, not a class
/// resolved here: one class can have several VRegs once values are spilled or
/// rematerialized, and only the instruction that computed the address knows
/// which one it read.
fn build_mem_addr(
    addr_vreg: Option<VReg>,
    addr_reg: Reg,
    barrier_pos: Option<usize>,
    regalloc: &RegAllocResult,
    conflict_reg: Option<Reg>,
    schedule: &[ScheduledInst],
) -> Addr {
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
    // the address had used a different VReg in R13.
    let addr_inst = addr_vreg.and_then(|v| {
        schedule
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.dst == v && matches!(inst.op, Op::Pure(PureOp::Addr { .. })))
    });

    if let Some((addr_pos, inst)) = addr_inst
        && let Op::Pure(PureOp::Addr { scale, disp }) = &inst.op
    {
        let reg_of = |v: VReg| {
            regalloc
                .assignment
                .get(&v)
                .copied()
                .and_then(crate::regalloc::Assignment::reg)
        };
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
    regalloc: &RegAllocResult,
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
                Op::Pseudo(PseudoOp::CallResult(_, _))
                    | Op::Pseudo(PseudoOp::LoadResult(_, _))
                    | Op::Pseudo(PseudoOp::VoidCallBarrier)
                    | Op::Pseudo(PseudoOp::StoreBarrier)
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

    // The register holding a role operand, read straight off the barrier.
    //
    // An effectful op's operands are ClassIds in the CFG, and the whole class of
    // bugs here came from Phase 7 trying to work back from a ClassId to the VReg
    // that holds it: the splitter's rewrites, spill slots and rematerialized
    // copies all move that answer, and a snapshot of the class map taken during
    // linearization does not know about any of them.
    //
    // The barrier already records the answer. Its leading operands are the role
    // operands in a fixed order (see `barrier::role_operand_count`), it is taken
    // from the post-coalesce, post-allocation schedule, and every pass that
    // rewrites operands does so by index -- so operand `i` is the VReg this op
    // actually reads, and `assignment` gives the register the allocator put it
    // in. No reconstruction, nothing to go stale.
    let barrier = barrier_pos.and_then(|p| schedule.get(p));
    // A role operand carries a register only if the allocator gave it one; where
    // it did not, the caller falls back to the class map, so both forms have to
    // agree on when the barrier has an answer.
    let role_vreg = |i: usize| -> Option<VReg> {
        barrier
            .filter(|_| i < barrier::role_operand_count(op))
            .and_then(|b| b.operands.get(i))
            .copied()
            .filter(|v| regalloc.assignment.contains_key(v))
    };
    let role_reg = |i: usize| -> Option<Reg> {
        role_vreg(i).and_then(|v| {
            regalloc
                .assignment
                .get(&v)
                .copied()
                .and_then(crate::regalloc::Assignment::reg)
        })
    };

    // Under BLITZ_VERIFY, hold this seam to its own invariant: the register an
    // effectful op reads must be one the barrier consuming that op declares as
    // an operand. Every resolution path here -- the class map at a program
    // point, the spill-slot trace, the remat match -- is a *reconstruction* of
    // what the barrier already records, and the failure mode is silent: a
    // plausible register holding some other value, which no def-before-use check
    // can see. Seven wrong-code bugs came out of this seam; this catches the ones
    // that resolve outside the operand list, such as an address resolving to the
    // register it occupied before a spill.
    //
    // What it cannot catch: a resolution landing on the *wrong* barrier operand.
    // `populate_effectful_operands` adds the folded `Addr`'s children next to the
    // address and sorts by VReg index, so the barrier records a set of VRegs and
    // not which one fills which role -- an address resolving to its own index
    // constant passes this check. See `tests/fuzz/findings/
    // seed4_load_addr_is_index.c`; role-tagged operands are the fix.
    let check_barrier_operand = |what: &str, reg: Reg| {
        if !crate::verify::is_enabled() {
            return;
        }
        let Some(barrier) = barrier_pos.and_then(|p| schedule.get(p)) else {
            return;
        };
        let declared: Vec<Reg> = barrier
            .operands
            .iter()
            .filter_map(|v| {
                regalloc
                    .assignment
                    .get(v)
                    .copied()
                    .and_then(crate::regalloc::Assignment::reg)
            })
            .collect();
        // An empty operand list means the barrier records nothing to check
        // against (a load whose address is a block param, say).
        if declared.is_empty() || declared.contains(&reg) {
            return;
        }
        panic!(
            "BLITZ_VERIFY: in function '{}', {what} resolved to {reg:?} at {point:?}, \
             which the barrier {:?} does not declare. Barrier operands {:?} are in {declared:?}. \
             The resolved register is not one this effectful op was scheduled to read.",
            func.name, barrier.op, barrier.operands,
        );
    };

    match op {
        EffectfulOp::Load { addr, ty, .. } => {
            let is_float = matches!(ty, Type::F32 | Type::F64);
            let canon_addr = uf.find_immutable(addr.class());
            let addr_reg = role_reg(0).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: format!(
                    "Load: no register for addr class {canon_addr:?} at {point:?} \
                     (barrier {:?})",
                    barrier.map(|b| &b.operands),
                ),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            check_barrier_operand("Load address", addr_reg);
            // The `LoadResult` barrier's own dst names the register this load must
            // write, for the same reason a call's result does (see the Call arm):
            // the schedule's VReg is the one the splitter rewrote, coalescing
            // renamed and the consumers read, while the class can name a different
            // VReg whose register nobody is reading.
            let result_reg = barrier
                .and_then(|inst| {
                    regalloc
                        .assignment
                        .get(&inst.dst)
                        .copied()
                        .and_then(crate::regalloc::Assignment::reg)
                })
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
                role_vreg(0),
                addr_reg,
                barrier_pos,
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
            let canon_addr = uf.find_immutable(addr.class());
            let addr_reg = role_reg(0).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: format!(
                    "Store: no register for addr class {canon_addr:?} at {point:?} \
                     (barrier {:?})",
                    barrier.map(|b| &b.operands),
                ),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            check_barrier_operand("Store address", addr_reg);
            // If val was spilled, its original VReg has no register. Find the
            // StoreBarrier for this Store and read the (possibly renamed) val
            // operand from it — that VReg points at the reload/remat copy.
            let canon_val = uf.find_immutable(val.class());
            let val_reg = role_reg(1).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: format!(
                    "Store: no register for val class {canon_val:?} at {point:?} \
                         (barrier {:?})",
                    barrier.map(|b| &b.operands)
                ),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let addr = build_mem_addr(
                role_vreg(0),
                addr_reg,
                barrier_pos,
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
            variadic,
        } => {
            // After spilling, the original arg vregs may share a register
            // (their defs are short-lived after the spill store). The actual
            // values at the call point live in SpillLoad vregs, which have
            // distinct registers. Find those registers by tracing spill slots.
            let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
            for (i, arg) in args.iter().enumerate() {
                let cid = arg.class();
                let r = role_reg(i).ok_or_else(|| CompileError {
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

            let locs = assign_args(arg_tys);

            // SysV AMD64: AL holds the number of vector registers used to pass
            // arguments. A variadic callee branches on it to decide whether to
            // spill XMM0-7 into its register save area, so a stale AL of zero
            // makes `printf("%f", x)` read a save area that was never written.
            //
            // Only for a callee declared `...`. A fixed callee never reads AL,
            // and setting it on every call was 2.18% of the instructions blitz
            // emitted. The declaration is load-bearing rather than advisory:
            // real `printf` reads AL whatever a prototype claims, so declaring
            // a variadic function with a fixed signature and passing it a
            // double reads a register save area that was never written.
            // `mov` is deliberate:
            // `xor al, al` would be shorter for the zero case but writes
            // EFLAGS, which may be live across the argument setup.
            if *variadic {
                let n_xmm_args = locs
                    .iter()
                    .filter(|l| matches!(l, ArgLoc::Reg(r) if r.is_xmm()))
                    .count();
                insts.push(MachInst::MovRI {
                    size: OpSize::S8,
                    dst: Operand::Reg(Reg::RAX),
                    imm: n_xmm_args as i64,
                });
            }

            insts.push(MachInst::CallDirect {
                target: callee.clone(),
            });

            // Clean up stack arguments after the call, alignment padding and all.
            let stack_bytes = stack_arg_bytes(arg_tys);
            if stack_bytes > 0 {
                insts.push(MachInst::AddRI {
                    size: OpSize::S64,
                    dst: Operand::Reg(Reg::RSP),
                    imm: stack_bytes,
                });
            }

            // After the call, the first GPR return value is in RAX.
            // If a CallResult ClassId was allocated to a different register, emit a MOV.
            //
            // Known limitation: caller-saved registers (RAX, RCX, RDX, RSI, RDI, R8-R11)
            // are not modeled as clobbered by the call. VRegs live across the call may
            // be incorrectly assigned to caller-saved registers and corrupted.
            // Which register the result must land in is answered by the barrier
            // instruction's own dst, not by resolving the result CLASS. The
            // schedule operand is the one the splitter rewrote and coalescing
            // renamed, so it is what the consumers read; a class lookup can hand
            // back a different VReg's register, and then this copy writes somewhere
            // nobody reads while the consumer reads a register the call never
            // wrote. Measured: with two calls in one block, `d0 = f1(...)` spilled
            // XMM1 -- a leftover argument -- immediately after a call whose result
            // was in XMM0, because no copy was emitted at all.
            let result_reg = barrier_pos
                .and_then(|pos| schedule.get(pos))
                .and_then(|inst| {
                    regalloc
                        .assignment
                        .get(&inst.dst)
                        .copied()
                        .and_then(crate::regalloc::Assignment::reg)
                });
            if let Some(&_result_cid) = results.first()
                && let Some(result_reg) = result_reg
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
