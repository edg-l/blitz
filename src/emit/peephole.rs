use crate::x86::addr::Addr;
use crate::x86::inst::{MachInst, OpSize, Operand};
use crate::x86::reg::Reg;

/// Returns true if `inst` is a flag-consuming instruction (conditional jump,
/// unconditional jump, or setcc).
fn is_flag_consumer(inst: &MachInst) -> bool {
    matches!(
        inst,
        MachInst::Jcc { .. } | MachInst::Jmp { .. } | MachInst::Setcc { .. }
    )
}

/// Returns true if `inst` writes (defines) the flags register.
fn writes_flags(inst: &MachInst) -> bool {
    matches!(
        inst,
        MachInst::AddRR { .. }
            | MachInst::AddRI { .. }
            | MachInst::AddRM { .. }
            | MachInst::SubRR { .. }
            | MachInst::SubRI { .. }
            | MachInst::AndRR { .. }
            | MachInst::OrRR { .. }
            | MachInst::XorRR { .. }
            | MachInst::ShlRI { .. }
            | MachInst::ShrRI { .. }
            | MachInst::SarRI { .. }
            | MachInst::RolRI { .. }
            | MachInst::ShldRRI { .. }
            | MachInst::BtsRR { .. }
            | MachInst::BtrRR { .. }
            | MachInst::BtcRR { .. }
            | MachInst::BtsRI { .. }
            | MachInst::BtrRI { .. }
            | MachInst::BtcRI { .. }
            | MachInst::ShlRCL { .. }
            | MachInst::ShrRCL { .. }
            | MachInst::SarRCL { .. }
            | MachInst::Imul2RR { .. }
            | MachInst::Imul3RRI { .. }
            | MachInst::CmpRR { .. }
            | MachInst::CmpRI { .. }
            | MachInst::TestRR { .. }
            | MachInst::TestRI { .. }
            | MachInst::Neg { .. }
            | MachInst::Inc { .. }
            | MachInst::Dec { .. }
    )
}

/// Check whether the flags are dead (not read) after instruction at `idx`.
///
/// Scans forward from `idx + 1`:
/// - If a flag-reading instruction is encountered first, flags are live (returns false).
/// - If a flag-writing instruction is encountered first, the old flags are dead (returns true).
/// - At the end of the slice, flags are considered dead (conservative: block boundary).
pub fn flags_dead_after(insts: &[MachInst], idx: usize) -> bool {
    // Jmp is a terminator; treat Jcc/Cmov/Setcc as flag readers.
    // Don't count Jmp itself as a flag reader for this analysis.
    for inst in &insts[idx + 1..] {
        match inst {
            MachInst::Jcc { .. } | MachInst::Cmov { .. } | MachInst::Setcc { .. } => {
                return false;
            }
            _ if writes_flags(inst) => {
                return true;
            }
            _ => {}
        }
    }
    // Reached end of block: no flag reader found.
    true
}

/// The register `inst` overwrites outright, if any: written without being read,
/// unconditionally, and over the whole 64 bits.
///
/// A 32-bit destination zero-extends into the full register, so `S32` kills what
/// the register held just as `S64` does. An 8- or 16-bit write leaves the upper
/// bits in place and `cmov` writes only when its condition holds, so neither
/// ends a value's live range. Two-address forms (`add`, `neg`, `inc`) read their
/// destination and so are reads, not kills.
fn kills_reg(inst: &MachInst) -> Option<Reg> {
    let (size, dst) = match inst {
        MachInst::MovRR { size, dst, src } => {
            if dst == src {
                return None;
            }
            (size, dst)
        }
        // `xor r, r` and `sub r, r` materialize zero; they do not read `r`.
        MachInst::XorRR { size, dst, src } | MachInst::SubRR { size, dst, src } if dst == src => {
            (size, dst)
        }
        MachInst::MovRI { size, dst, .. }
        | MachInst::MovRM { size, dst, .. }
        | MachInst::Lea { size, dst, .. }
        | MachInst::Imul3RRI { size, dst, .. } => (size, dst),
        // Widening moves and the moves out of an XMM register carry no size:
        // each writes the whole destination by construction.
        MachInst::MovzxBR { dst, .. }
        | MachInst::MovzxWR { dst, .. }
        | MachInst::MovsxBR { dst, .. }
        | MachInst::MovsxWR { dst, .. }
        | MachInst::MovsxDR { dst, .. }
        | MachInst::Cvttsd2siRR { dst, .. }
        | MachInst::Cvttss2siRR { dst, .. }
        | MachInst::MovqFromXmm { dst, .. } => {
            return match dst {
                Operand::Reg(r) => Some(*r),
                Operand::VReg(_) => None,
            };
        }
        _ => return None,
    };
    match (size, dst) {
        (OpSize::S32 | OpSize::S64, Operand::Reg(r)) => Some(*r),
        _ => None,
    }
}

/// True when the LEA at `idx` computes an address this block then discards.
///
/// A load or store whose address folded into its own addressing mode leaves the
/// LEA that computed it behind: the fold happens in `compile::effectful`, after
/// the address has already been scheduled and lowered as an instruction of its
/// own, and nothing between the two removes it.
///
/// The scan is block-local and stops at anything it cannot reason about, so the
/// address is dead only where this block overwrites the register outright before
/// reaching a branch, a call, or its own end.
fn lea_result_dead(insts: &[MachInst], idx: usize) -> bool {
    let MachInst::Lea {
        dst: Operand::Reg(dst),
        ..
    } = &insts[idx]
    else {
        return false;
    };
    // The frame registers are read by the epilogue and by every stack access,
    // neither of which this scan sees.
    if matches!(dst, Reg::RSP | Reg::RBP) {
        return false;
    }
    for inst in &insts[idx + 1..] {
        if matches!(
            inst,
            MachInst::CallDirect { .. }
                | MachInst::CallIndirect { .. }
                | MachInst::Jmp { .. }
                | MachInst::Jcc { .. }
                | MachInst::Ret
        ) {
            return false;
        }
        if inst.uses().contains(dst) {
            return false;
        }
        if kills_reg(inst) == Some(*dst) {
            return true;
        }
    }
    false
}

/// The `lea` that computes what a copy into a destination followed by a
/// destructive add computes, if there is one.
///
/// An add whose result goes somewhere that is neither operand has to be
/// `mov dst, a; add dst, b`, because the x86 form overwrites one of the two.
/// `lea` reads two registers and a displacement and writes a third register, so
/// the pair is one instruction, is a byte shorter, and stays a single cycle on
/// the address unit as long as the address is base-plus-index or
/// base-plus-displacement. What it does not do is write EFLAGS, so the fold is
/// only available where nothing reads them.
fn copy_add_to_lea(mov: &MachInst, add: &MachInst) -> Option<MachInst> {
    let MachInst::MovRR {
        size,
        dst: Operand::Reg(d),
        src: Operand::Reg(s),
    } = mov
    else {
        return None;
    };
    // An 8- or 16-bit add leaves the rest of the destination in place; `lea`
    // writes the whole register.
    if !matches!(size, OpSize::S32 | OpSize::S64) {
        return None;
    }
    let addr = match add {
        MachInst::AddRR {
            size: add_size,
            dst: Operand::Reg(add_dst),
            src: Operand::Reg(t),
        } if add_size == size && add_dst == d && t != d => {
            // RSP is the one register an address cannot index by, so it takes
            // the base position; both operands being RSP has nowhere to go.
            if *t == Reg::RSP && *s == Reg::RSP {
                return None;
            }
            let (base, index) = if *t == Reg::RSP { (*t, *s) } else { (*s, *t) };
            Addr::new(Some(base), Some(index), 1, 0)
        }
        MachInst::AddRI {
            size: add_size,
            dst: Operand::Reg(add_dst),
            imm,
        } if add_size == size && add_dst == d => Addr::new(Some(*s), None, 1, *imm),
        MachInst::SubRI {
            size: add_size,
            dst: Operand::Reg(add_dst),
            imm,
        } if add_size == size && add_dst == d => Addr::new(Some(*s), None, 1, imm.checked_neg()?),
        _ => return None,
    };
    Some(MachInst::Lea {
        size: *size,
        dst: Operand::Reg(*d),
        addr,
    })
}

/// Apply peephole optimizations to a sequence of `MachInst`s.
///
/// Optimizations applied (in order of pattern matching):
/// 0. Redundant round-trip mov: `mov rA, rB; mov rB, rA` -> `mov rA, rB` (S64 only).
/// 1. Delete `mov rX, rX` (redundant self-move).
/// 2. `mov rX, 0` -> `xor rX, rX` (zero idiom, shorter encoding).
/// 3. `cmp rX, 0` followed by Jcc/Setcc -> `test rX, rX` followed by Jcc/Setcc.
/// 4. `add rX, 1` -> `inc rX` when flags are dead after the add.
/// 5. `sub rX, 1` -> `dec rX` when flags are dead after the sub.
/// 6. `add rX, -1` -> `dec rX` when flags are dead.
/// 7. `sub rX, -1` -> `inc rX` when flags are dead.
/// 8. Delete a LEA whose address the block overwrites before reading.
/// 9. `lea rX, [rY]` -> `mov rX, rY`.
/// 10. `mov rD, rS; add rD, rT` -> `lea rD, [rS+rT]` when flags are dead, and
///     the same for `add rD, imm` / `sub rD, imm`.
pub fn peephole(insts: Vec<MachInst>) -> Vec<MachInst> {
    let mut result = Vec::with_capacity(insts.len());
    let mut i = 0;

    while i < insts.len() {
        match &insts[i] {
            // 8. An address computation nothing reads: the load or store that
            // wanted it folded it into its own addressing mode.
            MachInst::Lea { .. } if lea_result_dead(&insts, i) => {
                i += 1;
                continue;
            }

            // 9. An address that is a bare register is a copy of it. The move
            // runs on any execution port rather than the address unit, and it
            // is a byte shorter wherever the address needs a SIB byte (an RSP
            // base) or a zero displacement (an RBP or R13 base), which is
            // where this shape comes from: the address of a stack object.
            MachInst::Lea { size, dst, addr }
                if addr.base.is_some() && addr.index.is_none() && addr.disp == 0 =>
            {
                result.push(MachInst::MovRR {
                    size: *size,
                    dst: dst.clone(),
                    src: Operand::Reg(addr.base.expect("base is some")),
                });
                i += 1;
                continue;
            }

            // 0. Redundant round-trip mov elimination: mov rA, rB; mov rB, rA -> mov rA, rB.
            // Only for S64 (S32 zero-extends upper 32 bits).
            MachInst::MovRR {
                size: OpSize::S64,
                dst: dst_a,
                src: src_b,
            } if dst_a != src_b
                && i + 1 < insts.len()
                && matches!(
                    &insts[i + 1],
                    MachInst::MovRR { size: OpSize::S64, dst, src }
                    if dst == src_b && src == dst_a
                ) =>
            {
                result.push(insts[i].clone());
                i += 2;
                continue;
            }

            // 1. Delete mov rX, rX -- but only for S64.
            // A S32 `mov eax, eax` zero-extends the upper 32 bits and is NOT a no-op.
            // S8/S16 partial-register writes also have observable effects.
            MachInst::MovRR {
                size: OpSize::S64,
                dst,
                src,
            } if dst == src => {
                i += 1;
                continue;
            }

            // 10. `mov rD, rS; add rD, rT` -> `lea rD, [rS+rT]`, and the same
            // for a constant addend, where nothing reads the flags the add
            // would have written.
            MachInst::MovRR { .. }
                if i + 1 < insts.len()
                    && flags_dead_after(&insts, i + 1)
                    && copy_add_to_lea(&insts[i], &insts[i + 1]).is_some() =>
            {
                result.push(
                    copy_add_to_lea(&insts[i], &insts[i + 1]).expect("the guard just built it"),
                );
                i += 2;
                continue;
            }

            // 2. mov rX, 0  ->  xor rX, rX (zero idiom, shorter encoding).
            // Only safe when flags are not live (xor clobbers flags).
            MachInst::MovRI { size, dst, imm: 0 } if flags_dead_after(&insts, i) => {
                result.push(MachInst::XorRR {
                    size: *size,
                    dst: dst.clone(),
                    src: dst.clone(),
                });
                i += 1;
                continue;
            }

            // 3. cmp rX, 0 followed by Jcc  ->  test rX, rX followed by Jcc.
            MachInst::CmpRI { size, dst, imm: 0 }
                if i + 1 < insts.len() && is_flag_consumer(&insts[i + 1]) =>
            {
                result.push(MachInst::TestRR {
                    size: *size,
                    dst: dst.clone(),
                    src: dst.clone(),
                });
                // The Jcc itself will be pushed on the next iteration.
                i += 1;
                continue;
            }

            // 4. add rX, 1  ->  inc rX  (when flags are dead).
            MachInst::AddRI { size, dst, imm: 1 } if flags_dead_after(&insts, i) => {
                result.push(MachInst::Inc {
                    size: *size,
                    dst: dst.clone(),
                });
                i += 1;
                continue;
            }

            // 5. sub rX, 1  ->  dec rX  (when flags are dead).
            MachInst::SubRI { size, dst, imm: 1 } if flags_dead_after(&insts, i) => {
                result.push(MachInst::Dec {
                    size: *size,
                    dst: dst.clone(),
                });
                i += 1;
                continue;
            }

            // 6. add rX, -1  ->  dec rX  (when flags are dead).
            MachInst::AddRI { size, dst, imm: -1 } if flags_dead_after(&insts, i) => {
                result.push(MachInst::Dec {
                    size: *size,
                    dst: dst.clone(),
                });
                i += 1;
                continue;
            }

            // 7. sub rX, -1  ->  inc rX  (when flags are dead).
            MachInst::SubRI { size, dst, imm: -1 } if flags_dead_after(&insts, i) => {
                result.push(MachInst::Inc {
                    size: *size,
                    dst: dst.clone(),
                });
                i += 1;
                continue;
            }

            // 8. Store-load forwarding: mov [addr], rX; mov rY, [addr] -> mov [addr], rX; mov rY, rX.
            // Same size, same address. Avoids the redundant memory round-trip.
            MachInst::MovMR { size, addr, src }
                if i + 1 < insts.len()
                    && matches!(
                        &insts[i + 1],
                        MachInst::MovRM { size: s2, dst: _, addr: a2 }
                        if s2 == size && a2 == addr
                    ) =>
            {
                result.push(insts[i].clone());
                // Replace the load with a reg-reg move.
                if let MachInst::MovRM { size: s2, dst, .. } = &insts[i + 1] {
                    result.push(MachInst::MovRR {
                        size: *s2,
                        dst: dst.clone(),
                        src: src.clone(),
                    });
                }
                i += 2;
                continue;
            }

            _ => {
                result.push(insts[i].clone());
                i += 1;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::condcode::CondCode;
    use crate::x86::inst::{LabelId, Operand};
    use crate::x86::reg::Reg;

    fn reg(r: Reg) -> Operand {
        Operand::Reg(r)
    }

    #[test]
    fn mov_rax_rax_deleted() {
        let insts = vec![MachInst::MovRR {
            size: OpSize::S64,
            dst: reg(Reg::RAX),
            src: reg(Reg::RAX),
        }];
        let out = peephole(insts);
        assert!(out.is_empty(), "self-move should be deleted");
    }

    #[test]
    fn mov_eax_eax_kept_s32() {
        // S32 self-move zero-extends upper 32 bits; it is NOT a no-op.
        let insts = vec![MachInst::MovRR {
            size: OpSize::S32,
            dst: reg(Reg::RAX),
            src: reg(Reg::RAX),
        }];
        let out = peephole(insts);
        assert_eq!(out.len(), 1, "S32 self-move must not be deleted");
    }

    #[test]
    fn mov_different_regs_kept() {
        let insts = vec![MachInst::MovRR {
            size: OpSize::S64,
            dst: reg(Reg::RCX),
            src: reg(Reg::RAX),
        }];
        let out = peephole(insts);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn mov_rax_zero_becomes_xor() {
        let insts = vec![MachInst::MovRI {
            size: OpSize::S64,
            dst: reg(Reg::RAX),
            imm: 0,
        }];
        let out = peephole(insts);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0],
            MachInst::XorRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RAX),
            }
        );
    }

    #[test]
    fn mov_nonzero_imm_kept() {
        let insts = vec![MachInst::MovRI {
            size: OpSize::S64,
            dst: reg(Reg::RAX),
            imm: 42,
        }];
        let out = peephole(insts);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], MachInst::MovRI { imm: 42, .. }));
    }

    #[test]
    fn cmp_rax_zero_plus_je_becomes_test() {
        let label: LabelId = 1;
        let insts = vec![
            MachInst::CmpRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 0,
            },
            MachInst::Jcc {
                cc: CondCode::Eq,
                target: label,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::TestRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RAX),
            }
        );
        assert_eq!(
            out[1],
            MachInst::Jcc {
                cc: CondCode::Eq,
                target: label
            }
        );
    }

    #[test]
    fn cmp_rax_zero_not_followed_by_jcc_kept() {
        // cmp rax, 0 not followed by Jcc -> no transformation.
        let insts = vec![
            MachInst::CmpRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 0,
            },
            MachInst::Ret,
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::CmpRI { .. }));
    }

    #[test]
    fn add_one_becomes_inc_when_flags_dead() {
        // add rax, 1 followed by ret -> flags dead -> inc rax.
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 1,
            },
            MachInst::Ret,
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::Inc {
                size: OpSize::S64,
                dst: reg(Reg::RAX)
            }
        );
    }

    #[test]
    fn sub_one_becomes_dec_when_flags_dead() {
        let insts = vec![
            MachInst::SubRI {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                imm: 1,
            },
            MachInst::Ret,
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::Dec {
                size: OpSize::S64,
                dst: reg(Reg::RCX)
            }
        );
    }

    #[test]
    fn add_one_not_converted_when_flags_live() {
        // add rax, 1 followed by je -> flags are live -> keep as AddRI.
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 1,
            },
            MachInst::Jcc {
                cc: CondCode::Eq,
                target: 0,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::AddRI { imm: 1, .. }));
    }

    #[test]
    fn flags_dead_after_flag_writer() {
        // add rax, 1 followed by another flag-writing instruction -> flags dead.
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 1,
            },
            MachInst::SubRI {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                imm: 2,
            },
        ];
        assert!(flags_dead_after(&insts, 0));
    }

    #[test]
    fn flags_live_before_jcc() {
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 1,
            },
            MachInst::Jcc {
                cc: CondCode::Ne,
                target: 0,
            },
        ];
        assert!(!flags_dead_after(&insts, 0));
    }

    #[test]
    fn roundtrip_mov_s64_eliminated() {
        let insts = vec![
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                src: reg(Reg::RAX),
            },
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RCX),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 1, "round-trip mov should be collapsed to one");
        assert_eq!(
            out[0],
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                src: reg(Reg::RAX),
            }
        );
    }

    #[test]
    fn roundtrip_mov_s32_not_eliminated() {
        let insts = vec![
            MachInst::MovRR {
                size: OpSize::S32,
                dst: reg(Reg::RCX),
                src: reg(Reg::RAX),
            },
            MachInst::MovRR {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                src: reg(Reg::RCX),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2, "S32 round-trip mov must not be eliminated");
    }

    #[test]
    fn roundtrip_mov_not_adjacent_not_eliminated() {
        let insts = vec![
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                src: reg(Reg::RAX),
            },
            MachInst::Ret,
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RCX),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 3, "non-adjacent round-trip mov must be kept");
    }

    #[test]
    fn cmp_zero_before_setcc_becomes_test() {
        let insts = vec![
            MachInst::CmpRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 0,
            },
            MachInst::Setcc {
                cc: CondCode::Eq,
                dst: reg(Reg::RAX),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::TestRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RAX),
            }
        );
        assert!(matches!(
            out[1],
            MachInst::Setcc {
                cc: CondCode::Eq,
                ..
            }
        ));
    }

    #[test]
    fn add_neg1_becomes_dec_when_flags_dead() {
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: -1,
            },
            MachInst::Ret,
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::Dec {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
            }
        );
    }

    #[test]
    fn sub_neg1_becomes_inc_when_flags_dead() {
        let insts = vec![
            MachInst::SubRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: -1,
            },
            MachInst::Ret,
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            MachInst::Inc {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
            }
        );
    }

    #[test]
    fn add_neg1_kept_when_flags_live() {
        let insts = vec![
            MachInst::AddRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: -1,
            },
            MachInst::Jcc {
                cc: CondCode::Eq,
                target: 0,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::AddRI { imm: -1, .. }));
    }

    #[test]
    fn store_load_forwarding_same_addr() {
        use crate::x86::addr::Addr;
        let addr = Addr::new(Some(Reg::RSP), None, 1, 0);
        let insts = vec![
            MachInst::MovMR {
                size: OpSize::S64,
                addr: addr.clone(),
                src: reg(Reg::RAX),
            },
            MachInst::MovRM {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                addr: addr.clone(),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        // Store kept as-is.
        assert!(matches!(out[0], MachInst::MovMR { .. }));
        // Load replaced with reg-reg mov.
        assert_eq!(
            out[1],
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                src: reg(Reg::RAX),
            }
        );
    }

    #[test]
    fn store_load_different_addr_not_forwarded() {
        use crate::x86::addr::Addr;
        let addr1 = Addr::new(Some(Reg::RSP), None, 1, 0);
        let addr2 = Addr::new(Some(Reg::RSP), None, 1, 8);
        let insts = vec![
            MachInst::MovMR {
                size: OpSize::S64,
                addr: addr1,
                src: reg(Reg::RAX),
            },
            MachInst::MovRM {
                size: OpSize::S64,
                dst: reg(Reg::RCX),
                addr: addr2,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        // Both kept as-is (different addresses).
        assert!(matches!(out[0], MachInst::MovMR { .. }));
        assert!(matches!(out[1], MachInst::MovRM { .. }));
    }

    #[test]
    fn store_load_different_size_not_forwarded() {
        use crate::x86::addr::Addr;
        let addr = Addr::new(Some(Reg::RSP), None, 1, 0);
        let insts = vec![
            MachInst::MovMR {
                size: OpSize::S64,
                addr: addr.clone(),
                src: reg(Reg::RAX),
            },
            MachInst::MovRM {
                size: OpSize::S32,
                dst: reg(Reg::RCX),
                addr: addr.clone(),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::MovMR { .. }));
        assert!(matches!(out[1], MachInst::MovRM { .. }));
    }

    fn lea(dst: Reg, base: Reg, disp: i32) -> MachInst {
        MachInst::Lea {
            size: OpSize::S64,
            dst: reg(dst),
            addr: crate::x86::addr::Addr::new(Some(base), None, 1, disp),
        }
    }

    /// `lea rX, [rY]` and `mov rX, rY` produce the same value, and the move is
    /// shorter for the stack base this shape comes from.
    #[test]
    fn lea_of_a_bare_register_becomes_a_move() {
        let out = peephole(vec![
            lea(Reg::RAX, Reg::RSP, 0),
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RDX),
                src: reg(Reg::RAX),
            },
        ]);
        assert!(matches!(
            out[0],
            MachInst::MovRR {
                size: OpSize::S64,
                dst: Operand::Reg(Reg::RAX),
                src: Operand::Reg(Reg::RSP),
            }
        ));
    }

    /// A displacement is a real addition, and an index is a real address
    /// computation; neither is a copy.
    #[test]
    fn lea_with_a_displacement_or_an_index_is_kept() {
        let indexed = MachInst::Lea {
            size: OpSize::S64,
            dst: reg(Reg::RAX),
            addr: crate::x86::addr::Addr::new(Some(Reg::RCX), Some(Reg::RDX), 4, 0),
        };
        for inst in [lea(Reg::RAX, Reg::RCX, 8), indexed] {
            let out = peephole(vec![
                inst,
                MachInst::MovRR {
                    size: OpSize::S64,
                    dst: reg(Reg::RDX),
                    src: reg(Reg::RAX),
                },
            ]);
            assert!(matches!(out[0], MachInst::Lea { .. }));
        }
    }

    #[test]
    fn dead_lea_deleted() {
        // The store folded the address into its own operand; nothing reads RAX
        // before the block overwrites it.
        let insts = vec![
            lea(Reg::RAX, Reg::RCX, 8),
            MachInst::MovMR {
                size: OpSize::S64,
                addr: crate::x86::addr::Addr::new(Some(Reg::RCX), None, 1, 8),
                src: reg(Reg::RDX),
            },
            MachInst::MovRI {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                imm: 3,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::MovMR { .. }));
    }

    #[test]
    fn live_lea_kept() {
        let insts = vec![
            lea(Reg::RAX, Reg::RCX, 8),
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RDX),
                src: reg(Reg::RAX),
            },
            MachInst::MovRI {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                imm: 3,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], MachInst::Lea { .. }));
    }

    #[test]
    fn lea_kept_when_block_never_overwrites_it() {
        // No redefinition before the end of the list: a later block may read it.
        let insts = vec![lea(Reg::RAX, Reg::RCX, 8)];
        let out = peephole(insts);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn lea_kept_across_a_call() {
        // A call is where an address is most likely to be an argument already
        // placed in its register; the scan stops rather than guess.
        let insts = vec![
            lea(Reg::RDI, Reg::RSP, 8),
            MachInst::CallDirect {
                target: "f".to_string(),
            },
            MachInst::MovRI {
                size: OpSize::S64,
                dst: reg(Reg::RDI),
                imm: 3,
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], MachInst::Lea { .. }));
    }

    #[test]
    fn byte_write_does_not_kill_a_lea() {
        // `setcc` writes 8 bits; the upper 56 still hold the address.
        let insts = vec![
            lea(Reg::RAX, Reg::RCX, 8),
            MachInst::Setcc {
                cc: CondCode::Eq,
                dst: reg(Reg::RAX),
            },
        ];
        let out = peephole(insts);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], MachInst::Lea { .. }));
    }

    fn mov(dst: Reg, src: Reg) -> MachInst {
        MachInst::MovRR {
            size: OpSize::S32,
            dst: reg(dst),
            src: reg(src),
        }
    }

    /// The copy the destructive add needs is exactly what `lea`'s third operand
    /// removes.
    #[test]
    fn a_copy_and_a_destructive_add_become_one_lea() {
        let out = peephole(vec![
            mov(Reg::RAX, Reg::RDI),
            MachInst::AddRR {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                src: reg(Reg::RDX),
            },
        ]);
        assert_eq!(out.len(), 1);
        let MachInst::Lea { size, dst, addr } = &out[0] else {
            panic!("expected a lea, got {:?}", out[0]);
        };
        assert_eq!(*size, OpSize::S32);
        assert_eq!(*dst, reg(Reg::RAX));
        assert_eq!(*addr, Addr::new(Some(Reg::RDI), Some(Reg::RDX), 1, 0));
    }

    /// A constant addend is a displacement, and a constant subtrahend is a
    /// negative one.
    #[test]
    fn a_copy_and_a_constant_add_or_sub_become_one_lea() {
        let out = peephole(vec![
            mov(Reg::RAX, Reg::RDI),
            MachInst::AddRI {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                imm: 5,
            },
            mov(Reg::RCX, Reg::RSI),
            MachInst::SubRI {
                size: OpSize::S32,
                dst: reg(Reg::RCX),
                imm: 5,
            },
        ]);
        assert_eq!(out.len(), 2);
        assert!(
            matches!(&out[0], MachInst::Lea { addr, .. } if *addr == Addr::new(Some(Reg::RDI), None, 1, 5))
        );
        assert!(
            matches!(&out[1], MachInst::Lea { addr, .. } if *addr == Addr::new(Some(Reg::RSI), None, 1, -5))
        );
    }

    /// `lea` writes no flags, so the fold is unavailable where the add's are read.
    #[test]
    fn a_destructive_add_whose_flags_are_read_is_kept() {
        let out = peephole(vec![
            mov(Reg::RAX, Reg::RDI),
            MachInst::AddRR {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                src: reg(Reg::RDX),
            },
            MachInst::Setcc {
                cc: CondCode::Eq,
                dst: reg(Reg::RCX),
            },
        ]);
        assert_eq!(out.len(), 3);
        assert!(matches!(out[1], MachInst::AddRR { .. }));
    }

    /// The copy has already overwritten the addend, so there is no address that
    /// computes the same sum.
    #[test]
    fn an_add_of_the_copys_own_destination_is_kept() {
        let out = peephole(vec![
            mov(Reg::RAX, Reg::RDI),
            MachInst::AddRR {
                size: OpSize::S32,
                dst: reg(Reg::RAX),
                src: reg(Reg::RAX),
            },
        ]);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[1], MachInst::AddRR { .. }));
    }

    /// RSP cannot be an index, so it takes the base position.
    #[test]
    fn a_stack_pointer_addend_becomes_the_base() {
        let out = peephole(vec![
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RDI),
            },
            MachInst::AddRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RSP),
            },
        ]);
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], MachInst::Lea { addr, .. }
                if *addr == Addr::new(Some(Reg::RSP), Some(Reg::RDI), 1, 0)));
    }
}
