use std::collections::BTreeMap;

use crate::egraph::extract::{ExtractionResult, VReg};
use crate::egraph::unionfind::UnionFind;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::{ArgLoc, GPR_RETURN_REG, assign_args, setup_call_args};
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
            // Collect a register for each argument. Missing registers are an error.
            let mut arg_regs: Vec<Reg> = Vec::with_capacity(args.len());
            for &cid in args {
                let r = get_reg(cid).ok_or_else(|| CompileError {
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
                && result_reg != GPR_RETURN_REG
            {
                let ret_size = ret_tys
                    .first()
                    .map(OpSize::from_type)
                    .unwrap_or(OpSize::S64);
                insts.push(MachInst::MovRR {
                    size: ret_size,
                    dst: Operand::Reg(result_reg),
                    src: Operand::Reg(GPR_RETURN_REG),
                });
            }
            Ok(insts)
        }
        EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => {
            unreachable!("terminators must be handled separately")
        }
    }
}
