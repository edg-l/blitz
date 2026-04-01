use crate::x86::inst::{MachInst, OpSize};

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
/// 0. Redundant round-trip mov: `mov rA, rB; mov rB, rA` -> `mov rA, rB` (S64 only).
/// 1. Delete `mov rX, rX` (redundant self-move).
/// 2. `mov rX, 0` -> `xor rX, rX` (zero idiom, shorter encoding).
/// 3. `cmp rX, 0` followed by Jcc/Setcc -> `test rX, rX` followed by Jcc/Setcc.
/// 4. `add rX, 1` -> `inc rX` when flags are dead after the add.
/// 5. `sub rX, 1` -> `dec rX` when flags are dead after the sub.
/// 6. `add rX, -1` -> `dec rX` when flags are dead.
/// 7. `sub rX, -1` -> `inc rX` when flags are dead.
pub fn peephole(insts: Vec<MachInst>) -> Vec<MachInst> {
    let mut result = Vec::with_capacity(insts.len());
    let mut i = 0;

    while i < insts.len() {
        match &insts[i] {
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
}
