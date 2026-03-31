use std::collections::HashMap;

use crate::egraph::extract::{ExtractionResult, VReg};
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;
use crate::regalloc::allocator::RegAllocResult;
use crate::x86::abi::{ArgLoc, GPR_RETURN_REG, assign_args, setup_call_args};
use crate::x86::addr::Addr;
use crate::x86::inst::{MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

use super::{CompileError, IrLocation};

/// Build an `Addr` for Load/Store by checking if `addr_cid` extracted to an Addr node.
///
/// If the extraction result for `addr_cid` is an `Op::Addr { scale, disp }` node,
/// fuse the addressing mode directly into the memory operand (no separate LEA needed).
/// Otherwise fall back to `[addr_reg + 0]`.
fn build_mem_addr(
    addr_cid: ClassId,
    addr_reg: Reg,
    extraction: &ExtractionResult,
    class_to_vreg: &HashMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
) -> Addr {
    if let Some(ext) = extraction.choices.get(&addr_cid) {
        if let Op::Addr { scale, disp } = &ext.op {
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
                return Addr {
                    base: Some(base),
                    index: index_reg,
                    scale: *scale,
                    disp: *disp,
                };
            }
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
    class_to_vreg: &HashMap<ClassId, VReg>,
    regalloc: &RegAllocResult,
    extraction: &ExtractionResult,
    func: &Function,
) -> Result<Vec<MachInst>, CompileError> {
    let get_reg = |cid: ClassId| -> Option<Reg> {
        class_to_vreg
            .get(&cid)
            .and_then(|v| regalloc.vreg_to_reg.get(v).copied())
    };

    match op {
        EffectfulOp::Load {
            addr,
            result,
            ty: _,
        } => {
            let canon_addr = *addr;
            let addr_reg = get_reg(canon_addr).ok_or_else(|| CompileError {
                phase: "lowering".into(),
                message: "Load: no register for addr".into(),
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
            let result_reg = class_to_vreg
                .get(result)
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
            let addr = build_mem_addr(canon_addr, addr_reg, extraction, class_to_vreg, regalloc);
            Ok(vec![MachInst::MovRM {
                size: OpSize::S64,
                dst: Operand::Reg(result_reg),
                addr,
            }])
        }
        EffectfulOp::Store { addr, val } => {
            let canon_addr = *addr;
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
            let addr = build_mem_addr(canon_addr, addr_reg, extraction, class_to_vreg, regalloc);
            Ok(vec![MachInst::MovMR {
                size: OpSize::S64,
                addr,
                src: Operand::Reg(val_reg),
            }])
        }
        EffectfulOp::Call {
            func: callee,
            args,
            ret_tys: _,
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
            // All args treated as I64 for ABI assignment (correct for GPR-only calls;
            // FP args via XMM will be handled when arg type tracking is added).
            let arg_types: Vec<Type> = vec![Type::I64; arg_regs.len()];
            let mut insts = setup_call_args(&arg_types, &arg_regs, Reg::R11);

            // Count stack args so we can clean up RSP after the call.
            let locs = assign_args(&arg_types);
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
            if let Some(&result_cid) = results.first() {
                if let Some(result_reg) = get_reg(result_cid) {
                    if result_reg != GPR_RETURN_REG {
                        insts.push(MachInst::MovRR {
                            size: OpSize::S64,
                            dst: Operand::Reg(result_reg),
                            src: Operand::Reg(GPR_RETURN_REG),
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
