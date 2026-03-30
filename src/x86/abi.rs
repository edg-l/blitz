//! SystemV AMD64 ABI calling convention support.

use crate::ir::types::Type;
use crate::x86::encode::Encoder;
use crate::x86::inst::{MachInst, Operand};
use crate::x86::reg::Reg;

// ── 8.1 Calling convention data ───────────────────────────────────────────────

pub const GPR_ARG_REGS: [Reg; 6] = [Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9];

pub const FP_ARG_REGS: [Reg; 8] = [
    Reg::XMM0,
    Reg::XMM1,
    Reg::XMM2,
    Reg::XMM3,
    Reg::XMM4,
    Reg::XMM5,
    Reg::XMM6,
    Reg::XMM7,
];

pub const GPR_RETURN_REG: Reg = Reg::RAX;
/// Second GPR return register for 128-bit integer returns.
pub const GPR_RETURN_REG2: Reg = Reg::RDX;
pub const FP_RETURN_REG: Reg = Reg::XMM0;

/// Registers the callee must preserve across a call.
pub const CALLEE_SAVED: [Reg; 6] = [Reg::RBX, Reg::RBP, Reg::R12, Reg::R13, Reg::R14, Reg::R15];

/// GPR registers clobbered by a call (caller-saved).
pub const CALLER_SAVED_GPR: [Reg; 9] = [
    Reg::RAX,
    Reg::RCX,
    Reg::RDX,
    Reg::RSI,
    Reg::RDI,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
];

// ── 8.2 Argument assignment ───────────────────────────────────────────────────

/// Where a single function argument is passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgLoc {
    Reg(Reg),
    /// Offset from RSP at the call site (before CALL pushes the return address).
    Stack {
        offset: i32,
    },
}

/// Assign argument locations for a function call per the SystemV AMD64 ABI.
///
/// Integer/pointer types consume GPR slots; float types consume XMM slots.
/// Arguments that exceed the available register slots are placed on the stack
/// at 8-byte intervals starting at offset 0.
pub fn assign_args(param_types: &[Type]) -> Vec<ArgLoc> {
    let mut gpr_idx = 0usize;
    let mut fp_idx = 0usize;
    let mut stack_offset = 0i32;
    let mut locs = Vec::with_capacity(param_types.len());

    for ty in param_types {
        let loc = if ty.is_float() {
            if fp_idx < FP_ARG_REGS.len() {
                let r = FP_ARG_REGS[fp_idx];
                fp_idx += 1;
                ArgLoc::Reg(r)
            } else {
                let off = stack_offset;
                stack_offset += 8;
                ArgLoc::Stack { offset: off }
            }
        } else if ty.is_integer() {
            if gpr_idx < GPR_ARG_REGS.len() {
                let r = GPR_ARG_REGS[gpr_idx];
                gpr_idx += 1;
                ArgLoc::Reg(r)
            } else {
                let off = stack_offset;
                stack_offset += 8;
                ArgLoc::Stack { offset: off }
            }
        } else {
            unreachable!("assign_args: unsupported type {:?}", ty)
        };
        locs.push(loc);
    }
    locs
}

// ── 8.3 Stack frame layout ────────────────────────────────────────────────────

/// Describes the stack frame layout for a function.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    /// Total bytes subtracted from RSP in the prologue (after callee-saved pushes).
    pub frame_size: u32,
    /// Offset from RBP to the start of the spill area (always negative).
    pub spill_offset: i32,
    /// Which callee-saved registers are actually used and must be preserved.
    pub callee_saved: Vec<Reg>,
    pub uses_frame_pointer: bool,
    /// Space reserved for stack arguments passed to callees.
    pub outgoing_arg_space: u32,
}

/// Compute the stack frame layout.
///
/// Stack state on entry to the prologue (after the caller's CALL):
///   RSP % 16 == 8  (return address occupies 8 bytes)
///
/// The prologue pushes RBP (8 bytes → RSP % 16 == 0), then pushes each
/// callee-saved register (8 bytes each).  The total pushed callee-saved bytes
/// plus `frame_size` (sub RSP) must keep RSP 16-byte aligned at the point
/// where a nested CALL would execute.
pub fn compute_frame_layout(
    spill_slots: u32,
    callee_saved_used: &[Reg],
    outgoing_arg_space: u32,
) -> FrameLayout {
    // After push RBP: RSP % 16 == 0.
    // Each callee-saved push shifts RSP by 8.
    let n_callee = callee_saved_used.len() as u32;
    let callee_push_bytes = n_callee * 8;

    // Raw bytes needed for spills and outgoing args.
    let raw = spill_slots * 8 + outgoing_arg_space;

    // After the callee-saved pushes RSP % 16 may be 0 or 8.
    // We need (callee_push_bytes + frame_size) % 16 == 0
    // so that RSP is 16-byte aligned after `sub rsp, frame_size`.
    let misalign = (callee_push_bytes + raw) % 16;
    let frame_size = if misalign == 0 {
        raw
    } else {
        raw + (16 - misalign)
    };

    // Spill area starts below the callee-saved register pushes.
    // Callee-saved regs are pushed after `mov rbp, rsp`, so they sit between
    // RBP and the spill area. The correct RBP-relative offset is:
    //   -(n_callee * 8 + spill_slots * 8)
    let spill_offset = -((n_callee as i32 + spill_slots as i32) * 8);

    FrameLayout {
        frame_size,
        spill_offset,
        callee_saved: callee_saved_used.to_vec(),
        uses_frame_pointer: true,
        outgoing_arg_space,
    }
}

// ── 8.4 Prologue emission ─────────────────────────────────────────────────────

/// Emit the function prologue.
///
/// Sequence:
///   push rbp
///   mov rbp, rsp          (if uses_frame_pointer)
///   push <callee-saved>…  (in declaration order)
///   sub rsp, frame_size   (if non-zero)
pub fn emit_prologue(encoder: &mut Encoder, layout: &FrameLayout) {
    // push rbp
    encoder.encode_inst(&MachInst::Push {
        src: Operand::Reg(Reg::RBP),
    });

    if layout.uses_frame_pointer {
        // mov rbp, rsp
        encoder.encode_inst(&MachInst::MovRR {
            dst: Operand::Reg(Reg::RBP),
            src: Operand::Reg(Reg::RSP),
        });
    }

    // Push callee-saved registers in order.
    for &r in &layout.callee_saved {
        encoder.encode_inst(&MachInst::Push {
            src: Operand::Reg(r),
        });
    }

    // Reserve the frame.
    if layout.frame_size > 0 {
        encoder.encode_inst(&MachInst::SubRI {
            dst: Operand::Reg(Reg::RSP),
            imm: layout.frame_size as i32,
        });
    }
}

// ── 8.5 Epilogue emission ─────────────────────────────────────────────────────

/// Emit the function epilogue.
///
/// Sequence:
///   add rsp, frame_size   (if non-zero)
///   pop <callee-saved>…   (reverse order)
///   pop rbp               (if pushed)
///   ret
pub fn emit_epilogue(encoder: &mut Encoder, layout: &FrameLayout) {
    if layout.frame_size > 0 {
        encoder.encode_inst(&MachInst::AddRI {
            dst: Operand::Reg(Reg::RSP),
            imm: layout.frame_size as i32,
        });
    }

    // Pop callee-saved registers in reverse order.
    for &r in layout.callee_saved.iter().rev() {
        encoder.encode_inst(&MachInst::Pop {
            dst: Operand::Reg(r),
        });
    }

    // pop rbp
    encoder.encode_inst(&MachInst::Pop {
        dst: Operand::Reg(Reg::RBP),
    });

    encoder.encode_inst(&MachInst::Ret);
}

// ── 8.6 Parallel copy sequentialization ──────────────────────────────────────

/// Given a set of simultaneous register copies `(src, dst)`, produce a
/// sequential ordering that is correct even when copies form cycles.
///
/// Cycles are broken by routing through `temp`: the cycle head is saved to
/// `temp`, the remaining copies proceed, and then `temp` is moved to the
/// final destination.
pub fn sequentialize_copies(copies: &[(Reg, Reg)], temp: Reg) -> Vec<(Reg, Reg)> {
    // Build adjacency: dst_map[src] = dst
    use std::collections::{HashMap, HashSet};

    let mut pending: Vec<(Reg, Reg)> = copies.to_vec();
    let mut result: Vec<(Reg, Reg)> = Vec::new();

    loop {
        if pending.is_empty() {
            break;
        }

        // Build a set of all sources.
        let srcs: HashSet<Reg> = pending.iter().map(|&(s, _)| s).collect();

        // Find a copy whose dst is not a src of another pending copy
        // (i.e., safe to emit without clobbering a needed value).
        let safe_pos = pending.iter().position(|&(_, d)| !srcs.contains(&d));

        if let Some(pos) = safe_pos {
            let cp = pending.remove(pos);
            result.push(cp);
        } else {
            // All remaining copies form cycles. Break the first cycle.
            // Find cycle starting from pending[0].
            let cycle_start_src = pending[0].0;

            // Walk the cycle: build the chain src0->dst0->dst1->...
            let dst_map: HashMap<Reg, Reg> = pending.iter().map(|&(s, d)| (s, d)).collect();

            let mut cycle: Vec<Reg> = vec![cycle_start_src];
            let mut cur = dst_map[&cycle_start_src];
            while cur != cycle_start_src {
                cycle.push(cur);
                cur = dst_map[&cur];
            }

            // Remove cycle edges from pending.
            for i in 0..cycle.len() {
                let src = cycle[i];
                let dst = cycle[(i + 1) % cycle.len()];
                let pos = pending
                    .iter()
                    .position(|&(s, d)| s == src && d == dst)
                    .unwrap();
                pending.remove(pos);
            }

            // Break the cycle by saving cycle[0] into temp, then copying
            // cycle[n-1]->cycle[0], cycle[n-2]->cycle[n-1], ..., temp->cycle[1].
            //
            // Example swap [A, B] (A->B, B->A):
            //   cycle = [A, B]
            //   (A, temp), (B, A), (temp, B)
            //
            // Example three-way [A, B, C] (A->B, B->C, C->A):
            //   cycle = [A, B, C]
            //   (A, temp), (C, A), (B, C), (temp, B)
            // Save cycle[0] into temp, then unwind the cycle from the back.
            // For cycle [A, B, C] (A->B, B->C, C->A):
            //   (A, temp), (C, A), (B, C), (temp, B)
            result.push((cycle_start_src, temp));
            let n = cycle.len();
            for i in (1..n).rev() {
                result.push((cycle[i], cycle[(i + 1) % n]));
            }
            result.push((temp, cycle[1]));
        }
    }

    result
}

// ── 8.6a Call site setup ──────────────────────────────────────────────────────

/// Build the instruction sequence to set up arguments for a call.
///
/// `arg_types` are the parameter types; `arg_regs` are the current physical
/// registers holding the argument values (one per argument, in order).
/// `temp` is a scratch register available for cycle-breaking during parallel
/// copy resolution (must not overlap with any argument register).
///
/// Returns a `Vec<MachInst>` that should be emitted immediately before the
/// CALL instruction.  Stack arguments are pushed right-to-left.
pub fn setup_call_args(arg_types: &[Type], arg_regs: &[Reg], temp: Reg) -> Vec<MachInst> {
    assert_eq!(arg_types.len(), arg_regs.len());

    let locs = assign_args(arg_types);
    let mut insts: Vec<MachInst> = Vec::new();

    // Collect register-to-register copies.
    let reg_copies: Vec<(Reg, Reg)> = locs
        .iter()
        .zip(arg_regs.iter())
        .filter_map(|(loc, &src)| {
            if let ArgLoc::Reg(dst) = *loc {
                if src != dst { Some((src, dst)) } else { None }
            } else {
                None
            }
        })
        .collect();

    // Sequentialize register copies to handle cycles.
    let seq = sequentialize_copies(&reg_copies, temp);
    for (src, dst) in seq {
        insts.push(MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src),
        });
    }

    // Push stack arguments right-to-left.
    let stack_args: Vec<(i32, Reg)> = locs
        .iter()
        .zip(arg_regs.iter())
        .filter_map(|(loc, &src)| {
            if let ArgLoc::Stack { offset } = *loc {
                Some((offset, src))
            } else {
                None
            }
        })
        .collect();

    // Sort descending by offset (rightmost = highest offset pushed first).
    let mut stack_args = stack_args;
    stack_args.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, src) in stack_args {
        insts.push(MachInst::Push {
            src: Operand::Reg(src),
        });
    }

    insts
}

// ── 8.7 Caller-saved clobber set ──────────────────────────────────────────────

/// Returns all registers clobbered by a CALL instruction (caller-saved).
///
/// All XMM registers are caller-saved in the SystemV AMD64 ABI.
pub fn caller_saved_clobbers() -> Vec<Reg> {
    let mut regs: Vec<Reg> = CALLER_SAVED_GPR.to_vec();
    // All XMM0-XMM15 are caller-saved.
    regs.extend_from_slice(&[
        Reg::XMM0,
        Reg::XMM1,
        Reg::XMM2,
        Reg::XMM3,
        Reg::XMM4,
        Reg::XMM5,
        Reg::XMM6,
        Reg::XMM7,
        Reg::XMM8,
        Reg::XMM9,
        Reg::XMM10,
        Reg::XMM11,
        Reg::XMM12,
        Reg::XMM13,
        Reg::XMM14,
        Reg::XMM15,
    ]);
    regs
}

// ── 8.8 Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::Type;
    use crate::x86::encode::Encoder;

    // ── assign_args ──────────────────────────────────────────────────────────

    #[test]
    fn assign_args_three_i64() {
        let locs = assign_args(&[Type::I64, Type::I64, Type::I64]);
        assert_eq!(locs[0], ArgLoc::Reg(Reg::RDI));
        assert_eq!(locs[1], ArgLoc::Reg(Reg::RSI));
        assert_eq!(locs[2], ArgLoc::Reg(Reg::RDX));
    }

    #[test]
    fn assign_args_eight_i64_two_on_stack() {
        let types = vec![Type::I64; 8];
        let locs = assign_args(&types);
        // First 6 in registers.
        assert_eq!(locs[0], ArgLoc::Reg(Reg::RDI));
        assert_eq!(locs[1], ArgLoc::Reg(Reg::RSI));
        assert_eq!(locs[2], ArgLoc::Reg(Reg::RDX));
        assert_eq!(locs[3], ArgLoc::Reg(Reg::RCX));
        assert_eq!(locs[4], ArgLoc::Reg(Reg::R8));
        assert_eq!(locs[5], ArgLoc::Reg(Reg::R9));
        // Last 2 on stack.
        assert_eq!(locs[6], ArgLoc::Stack { offset: 0 });
        assert_eq!(locs[7], ArgLoc::Stack { offset: 8 });
    }

    #[test]
    fn assign_args_mixed_int_float() {
        // (i64, f64, i64, f64) -> RDI, XMM0, RSI, XMM1
        let types = [Type::I64, Type::F64, Type::I64, Type::F64];
        let locs = assign_args(&types);
        assert_eq!(locs[0], ArgLoc::Reg(Reg::RDI));
        assert_eq!(locs[1], ArgLoc::Reg(Reg::XMM0));
        assert_eq!(locs[2], ArgLoc::Reg(Reg::RSI));
        assert_eq!(locs[3], ArgLoc::Reg(Reg::XMM1));
    }

    // ── compute_frame_layout ─────────────────────────────────────────────────

    #[test]
    fn frame_layout_no_spills_no_callee_saved() {
        let layout = compute_frame_layout(0, &[], 0);
        assert_eq!(layout.frame_size, 0);
        assert!(layout.callee_saved.is_empty());
    }

    #[test]
    fn frame_layout_two_spill_slots() {
        // 2 spill slots = 16 bytes. After push RBP (RSP%16==0), 0 callee-saved pushes.
        // 16 bytes is already 16-byte aligned.
        let layout = compute_frame_layout(2, &[], 0);
        assert_eq!(layout.frame_size, 16);
    }

    #[test]
    fn frame_layout_alignment_with_one_callee_saved() {
        // 1 callee-saved push = 8 bytes (RSP%16 == 8 after push RBP + 1 push).
        // If spill_slots=0, outgoing=0: raw=0, callee_push=8, misalign=8 → pad 8.
        let layout = compute_frame_layout(0, &[Reg::RBX], 0);
        assert_eq!(layout.frame_size, 8);
    }

    #[test]
    fn frame_layout_outgoing_arg_space() {
        // outgoing_arg_space = 16 (two stack args), no spills, no callee-saved.
        // raw=16, callee_push=0, misalign=0 → frame_size=16.
        let layout = compute_frame_layout(0, &[], 16);
        assert_eq!(layout.frame_size, 16);
    }

    #[test]
    fn frame_layout_spill_offset() {
        let layout = compute_frame_layout(3, &[], 0);
        assert_eq!(layout.spill_offset, -24);
    }

    // ── sequentialize_copies ─────────────────────────────────────────────────

    fn apply_copies(copies: &[(Reg, Reg)], state: &mut std::collections::HashMap<Reg, u64>) {
        // Apply copies sequentially (order matters here – this simulates execution).
        for &(src, dst) in copies {
            let v = state[&src];
            state.insert(dst, v);
        }
    }

    fn regs_to_state(pairs: &[(Reg, u64)]) -> std::collections::HashMap<Reg, u64> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn sequentialize_no_conflict() {
        // RAX->RCX, RDX->RSI: no overlap, should work in any order.
        let copies = [(Reg::RAX, Reg::RCX), (Reg::RDX, Reg::RSI)];
        let seq = sequentialize_copies(&copies, Reg::R10);
        let mut state = regs_to_state(&[
            (Reg::RAX, 1),
            (Reg::RDX, 2),
            (Reg::RCX, 99),
            (Reg::RSI, 99),
            (Reg::R10, 0),
        ]);
        apply_copies(&seq, &mut state);
        assert_eq!(state[&Reg::RCX], 1);
        assert_eq!(state[&Reg::RSI], 2);
    }

    #[test]
    fn sequentialize_swap_cycle() {
        // RAX<->RCX swap cycle.
        let copies = [(Reg::RAX, Reg::RCX), (Reg::RCX, Reg::RAX)];
        let seq = sequentialize_copies(&copies, Reg::R10);
        let mut state = regs_to_state(&[(Reg::RAX, 10), (Reg::RCX, 20), (Reg::R10, 0)]);
        apply_copies(&seq, &mut state);
        assert_eq!(state[&Reg::RAX], 20);
        assert_eq!(state[&Reg::RCX], 10);
    }

    #[test]
    fn sequentialize_three_way_cycle() {
        // RAX->RCX->RDX->RAX (three-way rotation).
        let copies = [
            (Reg::RAX, Reg::RCX),
            (Reg::RCX, Reg::RDX),
            (Reg::RDX, Reg::RAX),
        ];
        let seq = sequentialize_copies(&copies, Reg::R10);
        let mut state =
            regs_to_state(&[(Reg::RAX, 1), (Reg::RCX, 2), (Reg::RDX, 3), (Reg::R10, 0)]);
        apply_copies(&seq, &mut state);
        // After rotation: RAX=3, RCX=1, RDX=2
        assert_eq!(state[&Reg::RAX], 3);
        assert_eq!(state[&Reg::RCX], 1);
        assert_eq!(state[&Reg::RDX], 2);
    }

    // ── 8.9 Prologue/epilogue byte output ────────────────────────────────────

    #[test]
    fn prologue_epilogue_bytes() {
        // 3 callee-saved regs (RBX, R12, R13), 32 bytes of spill space.
        // compute_frame_layout(4 spill slots, [RBX, R12, R13], 0):
        //   callee_push = 3*8 = 24, raw = 4*8 = 32
        //   misalign = (24+32) % 16 = 56 % 16 = 8 → frame_size = 32+8 = 40
        let callee_saved = [Reg::RBX, Reg::R12, Reg::R13];
        let layout = compute_frame_layout(4, &callee_saved, 0);
        assert_eq!(layout.frame_size, 40);

        let mut enc = Encoder::new();
        emit_prologue(&mut enc, &layout);
        emit_epilogue(&mut enc, &layout);

        let buf = &enc.buf;

        // Expected prologue bytes:
        //   55                  push rbp
        //   48 89 E5            mov rbp, rsp
        //   53                  push rbx
        //   41 54               push r12
        //   41 55               push r13
        //   48 83 EC 28         sub rsp, 40

        let prologue: &[u8] = &[
            0x55, // push rbp
            0x48, 0x89, 0xE5, // mov rbp, rsp
            0x53, // push rbx
            0x41, 0x54, // push r12
            0x41, 0x55, // push r13
            0x48, 0x83, 0xEC, 0x28, // sub rsp, 40
        ];
        assert_eq!(&buf[..prologue.len()], prologue, "prologue mismatch");

        // Expected epilogue bytes (immediately following):
        //   48 83 C4 28         add rsp, 40
        //   41 5D               pop r13
        //   41 5C               pop r12
        //   5B                  pop rbx
        //   5D                  pop rbp
        //   C3                  ret

        let epilogue: &[u8] = &[
            0x48, 0x83, 0xC4, 0x28, // add rsp, 40
            0x41, 0x5D, // pop r13
            0x41, 0x5C, // pop r12
            0x5B, // pop rbx
            0x5D, // pop rbp
            0xC3, // ret
        ];
        let ep_start = prologue.len();
        assert_eq!(
            &buf[ep_start..ep_start + epilogue.len()],
            epilogue,
            "epilogue mismatch"
        );
        assert_eq!(buf.len(), prologue.len() + epilogue.len());
    }
}
