use crate::x86::inst::{MachInst, OpSize};

/// Returns true if `inst` is a conditional or unconditional jump instruction.
fn is_jcc(inst: &MachInst) -> bool {
    matches!(inst, MachInst::Jcc { .. } | MachInst::Jmp { .. })
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

/// Apply peephole optimizations to a sequence of `MachInst`s.
///
/// Optimizations applied (in order of pattern matching):
/// 1. Delete `mov rX, rX` (redundant self-move).
/// 2. `mov rX, 0` -> `xor rX, rX` (zero idiom, shorter encoding).
/// 3. `cmp rX, 0` followed by Jcc -> `test rX, rX` followed by Jcc.
/// 4. `add rX, 1` -> `inc rX` when flags are dead after the add.
/// 5. `sub rX, 1` -> `dec rX` when flags are dead after the sub.
pub fn peephole(insts: Vec<MachInst>) -> Vec<MachInst> {
    let mut result = Vec::with_capacity(insts.len());
    let mut i = 0;

    while i < insts.len() {
        match &insts[i] {
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
                if i + 1 < insts.len() && is_jcc(&insts[i + 1]) =>
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
}
