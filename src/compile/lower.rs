use std::collections::HashSet;

use crate::egraph::extract::VReg;
use crate::ir::function::Function;
use crate::ir::op::Op;
use crate::ir::types::Type;
use crate::regalloc::allocator::RegAllocResult;
use crate::regalloc::spill::{
    is_spill_load, is_spill_store, is_xmm_spill_load, is_xmm_spill_store, spill_slot_of,
};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::abi::FrameLayout;
use crate::x86::addr::Addr;
use crate::x86::inst::{MachInst, Operand};
use crate::x86::reg::Reg;

use super::{CompileError, IrLocation};

fn get_dst(name: &str, dst_reg: Option<Reg>) -> Result<Reg, String> {
    dst_reg.ok_or_else(|| format!("{name}: no register for dst"))
}

fn get_op(name: &str, operand_regs: &[Option<Reg>], i: usize) -> Result<Reg, String> {
    operand_regs
        .get(i)
        .and_then(|r| *r)
        .ok_or_else(|| format!("{name}: no register for operand {i}"))
}

fn lower_binary_alu(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

fn lower_shift_cl(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    // Move value to shift into dst if needed.
    if dst != src_a {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    // The shift count operand is pre-colored to RCX before register allocation
    // when possible. If the count VReg was already pre-colored to another register
    // (e.g., because it is a function parameter in RSI), emit a MOV to RCX now.
    if src_b != Reg::RCX {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(Reg::RCX),
            src: Operand::Reg(src_b),
        });
    }
    insts.push(mk(Operand::Reg(dst)));
    Ok(insts)
}

fn lower_fp_binary(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovsdRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

fn lower_fp_binary_ss(
    name: &str,
    dst_reg: Option<Reg>,
    operand_regs: &[Option<Reg>],
    mk: fn(Operand, Operand) -> MachInst,
) -> Result<Vec<MachInst>, String> {
    let dst = get_dst(name, dst_reg)?;
    let src_a = get_op(name, operand_regs, 0)?;
    let src_b = get_op(name, operand_regs, 1)?;
    let mut insts = Vec::new();
    if dst != src_a {
        insts.push(MachInst::MovssRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src_a),
        });
    }
    insts.push(mk(Operand::Reg(dst), Operand::Reg(src_b)));
    Ok(insts)
}

/// Convert a single Op to a sequence of MachInsts.
///
/// `dst_vreg` is the VReg being defined; `dst_reg` is the physical reg (if allocated).
fn lower_op(
    op: &Op,
    dst_vreg: VReg,
    dst_reg: Option<Reg>,
    _operand_vregs: &[VReg],
    operand_regs: &[Option<Reg>],
) -> Result<Vec<MachInst>, String> {
    let _ = dst_vreg; // used for context in errors
    match op {
        Op::Iconst(val, _ty) => {
            let dst = dst_reg.ok_or_else(|| "Iconst: no register for dst".to_string())?;
            Ok(vec![MachInst::MovRI {
                dst: Operand::Reg(dst),
                imm: *val,
            }])
        }

        // Param nodes represent function parameters. Their value is already in
        // the ABI argument register (via pre-coloring), so no instruction is needed.
        // The lower_insts_with_ret function skips these VRegs, but as a safety net,
        // emit nothing here too.
        Op::Param(_, _) => Ok(vec![]),

        // BlockParam nodes represent block parameters (SSA phi inputs).
        // Their value arrives from predecessor blocks; no instruction is needed here.
        Op::BlockParam(_, _, _) => Ok(vec![]),

        // X86Add produces a Pair (result + flags); Proj0 extracts the value.
        // We emit: mov dst, src_a; add dst, src_b
        Op::X86Add => lower_binary_alu("X86Add", dst_reg, operand_regs, |dst, src| {
            MachInst::AddRR { dst, src }
        }),
        Op::X86Sub => lower_binary_alu("X86Sub", dst_reg, operand_regs, |dst, src| {
            MachInst::SubRR { dst, src }
        }),
        Op::X86And => lower_binary_alu("X86And", dst_reg, operand_regs, |dst, src| {
            MachInst::AndRR { dst, src }
        }),
        Op::X86Or => lower_binary_alu("X86Or", dst_reg, operand_regs, |dst, src| MachInst::OrRR {
            dst,
            src,
        }),
        Op::X86Xor => lower_binary_alu("X86Xor", dst_reg, operand_regs, |dst, src| {
            MachInst::XorRR { dst, src }
        }),
        // Variable shifts use CL (RCX). The shift count VReg is pre-colored to RCX
        // before register allocation, so src_b is guaranteed to be RCX here.
        Op::X86Shl => lower_shift_cl("X86Shl", dst_reg, operand_regs, |dst| MachInst::ShlRCL {
            dst,
        }),
        Op::X86Shr => lower_shift_cl("X86Shr", dst_reg, operand_regs, |dst| MachInst::ShrRCL {
            dst,
        }),
        Op::X86Sar => lower_shift_cl("X86Sar", dst_reg, operand_regs, |dst| MachInst::SarRCL {
            dst,
        }),

        // Immediate-form shifts: no CL constraint, emit mov+shift directly.
        Op::X86ShlImm(imm) => {
            let dst = get_dst("X86ShlImm", dst_reg)?;
            let src = get_op("X86ShlImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::ShlRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }
        Op::X86ShrImm(imm) => {
            let dst = get_dst("X86ShrImm", dst_reg)?;
            let src = get_op("X86ShrImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::ShrRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }
        Op::X86SarImm(imm) => {
            let dst = get_dst("X86SarImm", dst_reg)?;
            let src = get_op("X86SarImm", operand_regs, 0)?;
            let mut insts = Vec::new();
            if dst != src {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                });
            }
            insts.push(MachInst::SarRI {
                dst: Operand::Reg(dst),
                imm: *imm,
            });
            Ok(insts)
        }

        Op::X86Imul3 => {
            let dst = dst_reg.ok_or_else(|| "X86Imul3: no register for dst".to_string())?;
            let src_a = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Imul3: no register for operand 0".to_string())?;
            let src_b = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Imul3: no register for operand 1".to_string())?;
            // For X86Imul3 without an immediate, fall back to Imul2RR.
            let mut insts = Vec::new();
            if dst != src_a {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src_a),
                });
            }
            insts.push(MachInst::Imul2RR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src_b),
            });
            Ok(insts)
        }

        Op::X86Cmov(cc) => {
            let dst = dst_reg.ok_or_else(|| "X86Cmov: no register for dst".to_string())?;
            // operands: [flags_vreg, true_vreg, false_vreg]
            // flags come from a comparison; Cmov selects between true and false.
            if operand_regs.len() < 3 {
                return Err("X86Cmov requires 3 operands".into());
            }
            let true_reg = operand_regs[1]
                .ok_or_else(|| "X86Cmov: no register for true operand".to_string())?;
            let false_reg = operand_regs[2]
                .ok_or_else(|| "X86Cmov: no register for false operand".to_string())?;

            let mut insts = Vec::new();
            // Load the false value into dst first, then conditionally overwrite.
            if dst != false_reg {
                insts.push(MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(false_reg),
                });
            }
            insts.push(MachInst::Cmov {
                cc: *cc,
                dst: Operand::Reg(dst),
                src: Operand::Reg(true_reg),
            });
            Ok(insts)
        }

        Op::X86Setcc(cc) => {
            let dst = dst_reg.ok_or_else(|| "X86Setcc: no register for dst".to_string())?;
            Ok(vec![MachInst::Setcc {
                cc: *cc,
                dst: Operand::Reg(dst),
            }])
        }

        Op::X86Lea2 => {
            let dst = dst_reg.ok_or_else(|| "X86Lea2: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea2: no register for base".to_string())?;
            let idx = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea2: no register for index".to_string())?;
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: Addr {
                    base: Some(base),
                    index: Some(idx),
                    scale: 1,
                    disp: 0,
                },
            }])
        }

        Op::X86Lea3 { scale } => {
            let dst = dst_reg.ok_or_else(|| "X86Lea3: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea3: no register for base".to_string())?;
            let idx = operand_regs
                .get(1)
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea3: no register for index".to_string())?;
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: Addr {
                    base: Some(base),
                    index: Some(idx),
                    scale: *scale,
                    disp: 0,
                },
            }])
        }

        Op::X86Lea4 { scale, disp } => {
            let dst = dst_reg.ok_or_else(|| "X86Lea4: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Lea4: no register for base".to_string())?;
            let idx_reg = operand_regs.get(1).and_then(|r| *r);
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: Addr {
                    base: Some(base),
                    index: idx_reg,
                    scale: *scale,
                    disp: *disp,
                },
            }])
        }

        Op::Addr { scale, disp } => {
            // Addr nodes are "free" in the cost model and get folded into loads/stores.
            // When extracted standalone (e.g., as a root), emit a LEA.
            let dst = dst_reg.ok_or_else(|| "Addr: no register for dst".to_string())?;
            let base = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "Addr: no register for base".to_string())?;
            let idx_reg = operand_regs.get(1).and_then(|r| *r);
            Ok(vec![MachInst::Lea {
                dst: Operand::Reg(dst),
                addr: Addr {
                    base: Some(base),
                    index: idx_reg,
                    scale: *scale,
                    disp: *disp,
                },
            }])
        }

        // Projections: Proj0 and Proj1 extract values from Pairs.
        // Proj0 extracts the first element (the value), Proj1 extracts flags.
        // Since the Pair's register holds the full value (the ADD instruction
        // defines a register that holds the result), Proj0 is effectively a
        // register copy if the src and dst differ.
        Op::Proj0 => {
            if let (Some(dst), Some(Some(src))) = (dst_reg, operand_regs.first()) {
                if dst == *src {
                    Ok(vec![]) // No-op: dst and src are the same register.
                } else {
                    Ok(vec![MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(*src),
                    }])
                }
            } else {
                // Proj0 with no dst register: it might be a flags projection that's unused.
                Ok(vec![])
            }
        }

        Op::Proj1 => {
            // Proj1 extracts the flags component — flags live in the CPU flags register,
            // not in a GPR. No MachInst needed since flags are implicit.
            Ok(vec![])
        }

        // LoadResult nodes are skipped by lower_block_pure_ops; if reached here,
        // that's a bug in the pipeline.
        Op::LoadResult(_, _) => unreachable!(
            "LoadResult must be skipped by lower_block_pure_ops, not passed to lower_op"
        ),

        // CallResult nodes are skipped by lower_block_pure_ops; if reached here,
        // that's a bug in the pipeline.
        Op::CallResult(_, _) => unreachable!(
            "CallResult must be skipped by lower_block_pure_ops, not passed to lower_op"
        ),

        // ── x86 FP machine ops ────────────────────────────────────────────────
        Op::X86Addsd => lower_fp_binary("X86Addsd", dst_reg, operand_regs, |dst, src| {
            MachInst::AddsdRR { dst, src }
        }),
        Op::X86Subsd => lower_fp_binary("X86Subsd", dst_reg, operand_regs, |dst, src| {
            MachInst::SubsdRR { dst, src }
        }),
        Op::X86Mulsd => lower_fp_binary("X86Mulsd", dst_reg, operand_regs, |dst, src| {
            MachInst::MulsdRR { dst, src }
        }),
        Op::X86Divsd => lower_fp_binary("X86Divsd", dst_reg, operand_regs, |dst, src| {
            MachInst::DivsdRR { dst, src }
        }),
        Op::X86Sqrtsd => {
            let dst = get_dst("X86Sqrtsd", dst_reg)?;
            let src = get_op("X86Sqrtsd", operand_regs, 0)?;
            Ok(vec![MachInst::SqrtsdRR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src),
            }])
        }

        // ── x86 F32 machine ops ───────────────────────────────────────────────
        Op::X86Addss => lower_fp_binary_ss("X86Addss", dst_reg, operand_regs, |dst, src| {
            MachInst::AddssRR { dst, src }
        }),
        Op::X86Subss => lower_fp_binary_ss("X86Subss", dst_reg, operand_regs, |dst, src| {
            MachInst::SubssRR { dst, src }
        }),
        Op::X86Mulss => lower_fp_binary_ss("X86Mulss", dst_reg, operand_regs, |dst, src| {
            MachInst::MulssRR { dst, src }
        }),
        Op::X86Divss => lower_fp_binary_ss("X86Divss", dst_reg, operand_regs, |dst, src| {
            MachInst::DivssRR { dst, src }
        }),
        Op::X86Sqrtss => {
            let dst = get_dst("X86Sqrtss", dst_reg)?;
            let src = get_op("X86Sqrtss", operand_regs, 0)?;
            Ok(vec![MachInst::SqrtssRR {
                dst: Operand::Reg(dst),
                src: Operand::Reg(src),
            }])
        }

        // Fconst: load FP constant bits into a scratch GPR (R11), then move to XMM.
        // R11 is caller-saved and not used by regalloc for any persistent value.
        Op::Fconst(bits) => {
            let dst = dst_reg.ok_or_else(|| "Fconst: no register for dst".to_string())?;
            Ok(vec![
                MachInst::MovRI {
                    dst: Operand::Reg(Reg::R11),
                    imm: *bits as i64,
                },
                MachInst::MovqToXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(Reg::R11),
                },
            ])
        }

        // Generic ops that should have been lowered by the e-graph phases.
        // These should not appear after isel.
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::UDiv
        | Op::SDiv
        | Op::URem
        | Op::SRem
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Shl
        | Op::Shr
        | Op::Sar
        | Op::Sext(_)
        | Op::Zext(_)
        | Op::Trunc(_)
        | Op::Bitcast(_)
        | Op::Icmp(_)
        | Op::Fadd
        | Op::Fsub
        | Op::Fmul
        | Op::Fdiv
        | Op::Fsqrt
        | Op::Select => Err(format!(
            "unlowered op {op:?}: generic IR must be lowered by isel phases before lowering"
        )),
        Op::X86Movsx { from, to: _ } => {
            let dst = dst_reg.ok_or_else(|| "X86Movsx: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Movsx: no register for src".to_string())?;
            let inst = match from {
                Type::I8 => MachInst::MovsxBR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I16 => MachInst::MovsxWR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I32 => MachInst::MovsxDR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                other => {
                    return Err(format!("X86Movsx: unsupported source type {other:?}"));
                }
            };
            Ok(vec![inst])
        }

        Op::X86Movzx { from, to: _ } => {
            let dst = dst_reg.ok_or_else(|| "X86Movzx: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Movzx: no register for src".to_string())?;
            let inst = match from {
                Type::I8 => MachInst::MovzxBR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                Type::I16 => MachInst::MovzxWR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                },
                // 32-bit MOV zero-extends to 64-bit on x86-64 implicitly.
                Type::I32 => {
                    if dst == src {
                        return Ok(vec![]);
                    }
                    MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }
                }
                other => {
                    return Err(format!("X86Movzx: unsupported source type {other:?}"));
                }
            };
            Ok(vec![inst])
        }

        Op::X86Trunc { .. } => {
            // Truncation is free on x86-64: upper bits are simply ignored.
            // Emit a MOV only if dst != src (register copy needed).
            let dst = dst_reg.ok_or_else(|| "X86Trunc: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Trunc: no register for src".to_string())?;
            if dst == src {
                Ok(vec![])
            } else {
                Ok(vec![MachInst::MovRR {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            }
        }

        Op::X86Bitcast { from, to } => {
            let dst = dst_reg.ok_or_else(|| "X86Bitcast: no register for dst".to_string())?;
            let src = operand_regs
                .first()
                .and_then(|r| *r)
                .ok_or_else(|| "X86Bitcast: no register for src".to_string())?;
            let int_to_float = from.is_integer() && matches!(to, Type::F32 | Type::F64);
            let float_to_int = matches!(from, Type::F32 | Type::F64) && to.is_integer();
            if int_to_float {
                // MOVQ xmm, gpr
                Ok(vec![MachInst::MovqToXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            } else if float_to_int {
                // MOVQ gpr, xmm
                Ok(vec![MachInst::MovqFromXmm {
                    dst: Operand::Reg(dst),
                    src: Operand::Reg(src),
                }])
            } else if dst == src {
                Ok(vec![])
            } else {
                // Same class (int->int or float->float): register copy.
                if from.is_integer() {
                    Ok(vec![MachInst::MovRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }])
                } else {
                    Ok(vec![MachInst::MovsdRR {
                        dst: Operand::Reg(dst),
                        src: Operand::Reg(src),
                    }])
                }
            }
        }
    }
}

pub(super) fn lower_block_pure_ops(
    insts: &[ScheduledInst],
    regalloc: &RegAllocResult,
    func: &Function,
    param_vreg_set: &HashSet<VReg>,
    frame_layout: &FrameLayout,
) -> Result<Vec<MachInst>, CompileError> {
    let mut result: Vec<MachInst> = Vec::new();
    let get_reg = |vreg: VReg| -> Option<Reg> { regalloc.vreg_to_reg.get(&vreg).copied() };

    for inst in insts {
        // Skip function param VRegs (pre-colored to ABI arg regs).
        if param_vreg_set.contains(&inst.dst) {
            continue;
        }
        // Skip block param VRegs: their values arrive from predecessor phi copies.
        if matches!(inst.op, Op::BlockParam(_, _, _)) {
            continue;
        }
        // Skip LoadResult VRegs: their values are produced by lower_effectful_op.
        if matches!(inst.op, Op::LoadResult(_, _)) {
            continue;
        }
        // Skip CallResult VRegs: their values are captured after CallDirect in lower_effectful_op.
        if matches!(inst.op, Op::CallResult(_, _)) {
            continue;
        }

        // Handle GPR spill sentinels.
        if is_spill_store(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(src_reg) = inst.operands.first().and_then(|&v| get_reg(v)) {
                result.push(MachInst::MovMR {
                    addr: Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                    src: Operand::Reg(src_reg),
                });
            }
            continue;
        }
        if is_spill_load(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(dst_reg) = get_reg(inst.dst) {
                result.push(MachInst::MovRM {
                    dst: Operand::Reg(dst_reg),
                    addr: Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                });
            }
            continue;
        }

        // Handle XMM spill sentinels.
        if is_xmm_spill_store(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(src_reg) = inst.operands.first().and_then(|&v| get_reg(v)) {
                result.push(MachInst::MovsdMR {
                    addr: Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                    src: Operand::Reg(src_reg),
                });
            }
            continue;
        }
        if is_xmm_spill_load(inst) {
            let slot = spill_slot_of(inst) as i32;
            let disp = frame_layout.spill_offset + slot * 8;
            if let Some(dst_reg) = get_reg(inst.dst) {
                result.push(MachInst::MovsdRM {
                    dst: Operand::Reg(dst_reg),
                    addr: Addr {
                        base: Some(Reg::RBP),
                        index: None,
                        scale: 1,
                        disp,
                    },
                });
            }
            continue;
        }

        let dst_reg_opt = get_reg(inst.dst);
        let op_regs: Vec<Option<Reg>> = inst.operands.iter().map(|&v| get_reg(v)).collect();

        let machinsts = lower_op(&inst.op, inst.dst, dst_reg_opt, &inst.operands, &op_regs)
            .map_err(|msg| CompileError {
                phase: "lowering".into(),
                message: msg,
                location: Some(IrLocation {
                    function: func.name.clone(),
                    block: None,
                    inst: None,
                }),
            })?;
        result.extend(machinsts);
    }
    Ok(result)
}
