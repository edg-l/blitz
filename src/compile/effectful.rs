use std::collections::BTreeMap;

use crate::egraph::extract::{ExtractionResult, VReg};
use crate::egraph::unionfind::UnionFind;
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
    extraction: &ExtractionResult,
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
    conflict_reg: Option<Reg>,
    schedule: &[ScheduledInst],
) -> Addr {
    // Only fold if the addr VReg's scheduled instruction is an Addr op.
    // When it's a SpillLoad, BlockParam, or other non-Addr op, the extraction
    // may show an Addr node for the class, but the children's registers aren't
    // guaranteed live at the load/store point.
    let addr_vreg = class_to_vreg.get(&addr_cid);
    let is_addr_inst = addr_vreg.is_some_and(|v| {
        schedule
            .iter()
            .any(|inst| inst.dst == *v && matches!(inst.op, Op::Addr { .. }))
    });

    if is_addr_inst
        && let Some(ext) = extraction.choices.get(&addr_cid)
        && let Op::Addr { scale, disp } = &ext.op
    {
        // children[0] = base ClassId, children[1] = index ClassId (may be NONE).
        let base_reg = ext
            .children
            .first()
            .and_then(|&c| class_to_vreg.get(&c))
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied());
        let index_reg = ext
            .children
            .get(1)
            .filter(|&&c| c != ClassId::NONE)
            .and_then(|&c| class_to_vreg.get(&c))
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied());
        if let Some(base) = base_reg {
            // If the folded base or index register conflicts with an
            // operand that is read simultaneously (e.g. the Store value),
            // fall back to the pre-computed addr_reg to avoid clobbering.
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
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
    extraction: &ExtractionResult,
    func: &Function,
    uf: &UnionFind,
    schedule: &[ScheduledInst],
) -> Result<Vec<MachInst>, CompileError> {
    let get_reg = |cid: ClassId| -> Option<Reg> {
        let canon = uf.find_immutable(cid);
        class_to_vreg
            .get(&canon)
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
    };

    match op {
        EffectfulOp::Load { addr, result, ty } => {
            let load_size = OpSize::from_type(ty);
            let canon_addr = uf.find_immutable(*addr);
            let addr_reg = get_reg(canon_addr).ok_or_else(|| CompileError {
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
                .get(&canon_result)
                .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
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
                extraction,
                class_to_vreg,
                regalloc,
                None,
                schedule,
            );
            // S8/S16 loads must use zero-extending loads (MovzxBRM/MovzxWRM) to
            // avoid partial register writes that leave upper bits unchanged.
            let inst = match load_size {
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
            };
            Ok(vec![inst])
        }
        EffectfulOp::Store { addr, val, ty } => {
            let canon_addr = uf.find_immutable(*addr);
            let addr_reg = get_reg(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Store: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let val_reg = get_reg(*val).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Store: no register for val".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let store_size = OpSize::from_type(ty);
            let addr = build_mem_addr(
                canon_addr,
                addr_reg,
                extraction,
                class_to_vreg,
                regalloc,
                Some(val_reg),
                schedule,
            );
            Ok(vec![MachInst::MovMR {
                size: store_size,
                addr,
                src: Operand::Reg(val_reg),
            }])
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
            let spill_reload_regs = resolve_call_arg_regs_after_spilling(
                args,
                results,
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
                            .map(OpSize::from_type)
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
fn resolve_call_arg_regs_after_spilling(
    args: &[ClassId],
    results: &[ClassId],
    class_to_vreg: &BTreeMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
    uf: &UnionFind,
    schedule: &[ScheduledInst],
) -> Vec<Option<Reg>> {
    // Find the barrier instruction (CallResult or VoidCallBarrier).
    let barrier_inst = if let Some(&first_result_cid) = results.first() {
        let canon = uf.find_immutable(first_result_cid);
        class_to_vreg.get(&canon).and_then(|result_vreg| {
            schedule
                .iter()
                .find(|inst| inst.dst == *result_vreg && matches!(inst.op, Op::CallResult(_, _)))
        })
    } else {
        let arg_vregs: Vec<VReg> = args
            .iter()
            .filter_map(|&cid| {
                let canon = uf.find_immutable(cid);
                class_to_vreg.get(&canon).copied()
            })
            .collect();
        schedule.iter().find(|inst| {
            matches!(inst.op, Op::VoidCallBarrier)
                && arg_vregs.iter().all(|v| inst.operands.contains(v))
        })
    };

    let Some(barrier) = barrier_inst else {
        // No barrier found; fall back to original VReg registers.
        return args
            .iter()
            .map(|&cid| {
                let canon = uf.find_immutable(cid);
                class_to_vreg
                    .get(&canon)
                    .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
            })
            .collect();
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
        if let Op::SpillStore(slot) | Op::XmmSpillStore(slot) = &inst.op {
            if let Some(&original_vreg) = inst.operands.first() {
                slot_to_original_vreg.insert(*slot, original_vreg);
            }
        }
    }

    // Build original VReg -> defining instruction op.
    let mut original_def_ops: BTreeMap<VReg, &Op> = BTreeMap::new();
    for inst in schedule {
        original_def_ops.entry(inst.dst).or_insert(&inst.op);
    }

    // For each original arg VReg, find its replacement in the barrier operands.
    // The replacement may be:
    // 1. The original VReg itself (not spilled)
    // 2. A SpillLoad/XmmSpillLoad from the same slot
    // 3. A rematerialized VReg with the same op
    let mut original_to_replacement: BTreeMap<VReg, VReg> = BTreeMap::new();
    for &cid in args {
        let canon = uf.find_immutable(cid);
        let Some(&original_vreg) = class_to_vreg.get(&canon) else {
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
            if let Op::SpillLoad(slot) | Op::XmmSpillLoad(slot) = &def_inst.op {
                if let Some(&stored_vreg) = slot_to_original_vreg.get(slot) {
                    if stored_vreg == original_vreg {
                        original_to_replacement.insert(original_vreg, op_vreg);
                        found = true;
                        break;
                    }
                }
            }
        }
        if found {
            continue;
        }

        // Case 3: find rematerialized VReg with same op.
        let orig_op = original_def_ops.get(&original_vreg);
        if let Some(orig_op) = orig_op {
            for (&op_vreg, &def_inst) in &barrier_op_defs {
                if op_vreg != original_vreg && &def_inst.op == *orig_op {
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
            let original_vreg = class_to_vreg.get(&canon)?;
            if let Some(&replacement) = original_to_replacement.get(original_vreg) {
                return regalloc.vreg_to_reg.get(&replacement).copied();
            }
            regalloc.vreg_to_reg.get(original_vreg).copied()
        })
        .collect()
}
