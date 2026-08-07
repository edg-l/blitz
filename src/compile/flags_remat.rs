//! Re-emit a comparison wherever its flags would otherwise be read after
//! something wrote EFLAGS.
//!
//! Every `Icmp` over one pair of operands is merged onto a single shared
//! compare by `egraph::isel::apply_icmp_isel`, which is sound as an equality --
//! `a == b` and `a != b` set identical flags. What it also does is make one
//! Flags value live across everything between its consumers, and the scheduler
//! sinks each consumer to its own use, so a `call` lands in between:
//!
//! ```text
//!    cmp    eax,0x1
//!    cmove  edx,eax      <- reads the compare
//!    call   printf
//!    cmovne edx,eax      <- reads printf's flags
//! ```
//!
//! EFLAGS is in no register file and no store reaches it, so the spiller cannot
//! rescue this the way it rescues an ordinary value. The answer is the one used
//! for a constant, which is cheap enough to recompute rather than keep: emit
//! the comparison again where it is needed.
//!
//! This runs on the schedules *before* register allocation, so the operands the
//! re-emitted compare reads have their live ranges extended by it and the
//! allocator sees the graph it is going to colour.

use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
use crate::ir::op::{MachOp, Op, PseudoOp, PureOp};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

/// Whether executing this op can leave EFLAGS holding something else.
///
/// Stated as an allowlist of ops that certainly do not write flags, so an op
/// added later is treated as clobbering until someone says otherwise. The cost
/// of being wrong in that direction is one redundant compare; the cost of being
/// wrong in the other is a `cmov` reading a stranger's flags.
fn writes_flags(op: &Op) -> bool {
    !matches!(
        op,
        // No code, or code that only moves bits.
        Op::Pure(
            PureOp::Iconst(..)
                | PureOp::Fconst(..)
                | PureOp::Param(..)
                | PureOp::BlockParam(..)
                | PureOp::Proj0
                | PureOp::Proj1
        ) | Op::Pseudo(
            PseudoOp::StackAddr(..)
                | PseudoOp::GlobalAddr(..)
                | PseudoOp::SpillStore(..)
                | PseudoOp::SpillLoad(..)
                | PseudoOp::XmmSpillStore(..)
                | PseudoOp::XmmSpillLoad(..)
                | PseudoOp::LoadResult(..)
                | PseudoOp::StoreBarrier
                | PseudoOp::TerminatorArgs(..)
        )
        // `lea` computes an address without touching flags, which is most of
        // why it is worth selecting. `cmov` and `setcc` read them and write
        // none. The SSE arithmetic and conversions leave EFLAGS alone --
        // `ucomis[ds]` is the exception and is absent here on purpose.
        | Op::Mach(
            MachOp::X86Lea2
                | MachOp::X86Lea3 { .. }
                | MachOp::X86Lea4 { .. }
                | MachOp::X86Cmov(..)
                | MachOp::X86Setcc(..)
                | MachOp::X86Movsx { .. }
                | MachOp::X86Movzx { .. }
                | MachOp::X86Addsd
                | MachOp::X86Subsd
                | MachOp::X86Mulsd
                | MachOp::X86Divsd
                | MachOp::X86Sqrtsd
                | MachOp::X86Addss
                | MachOp::X86Subss
                | MachOp::X86Mulss
                | MachOp::X86Divss
                | MachOp::X86Sqrtss
                | MachOp::X86Cvtsi2sd(..)
                | MachOp::X86Cvtsi2ss(..)
                | MachOp::X86Cvttsd2si(..)
                | MachOp::X86Cvttss2si(..)
                | MachOp::X86Cvtsd2ss
                | MachOp::X86Cvtss2sd
        )
    )
}

/// Re-emit flags-producing instructions so no consumer reads stale flags.
///
/// Returns the number of compares inserted, which is zero for almost every
/// function: it takes two consumers of one comparison with a flag writer
/// between them, and a comparison used as a *value* rather than as a branch
/// condition, since a terminator ends its own block.
pub(super) fn remat_flags(
    block_schedules: &mut [Vec<ScheduledInst>],
    classes: &BTreeMap<VReg, RegClass>,
    next_vreg: &mut u32,
) -> usize {
    let mut inserted = 0;

    for sched in block_schedules.iter_mut() {
        let mut out: Vec<ScheduledInst> = Vec::with_capacity(sched.len());
        // The flags-defining instruction for each Flags VReg, and whether
        // anything has written EFLAGS since it was emitted.
        let mut def: BTreeMap<VReg, ScheduledInst> = BTreeMap::new();
        let mut stale: BTreeMap<VReg, bool> = BTreeMap::new();

        for inst in sched.iter() {
            let mut inst = inst.clone();

            for operand in inst.operands.iter_mut() {
                if classes.get(operand) != Some(&RegClass::Flags) {
                    continue;
                }
                if stale.get(operand) != Some(&true) {
                    continue;
                }
                let Some(source) = def.get(operand).cloned() else {
                    // Defined in another block. Flags are never a block
                    // parameter, so there is nothing to re-emit from here and
                    // nothing that produces this shape today.
                    continue;
                };
                let fresh = VReg(*next_vreg);
                *next_vreg += 1;
                out.push(ScheduledInst {
                    op: source.op.clone(),
                    dst: fresh,
                    operands: source.operands.clone(),
                });
                // The clone is the live definition from here on, so a third
                // consumer after another clobber re-emits from it rather than
                // from the original.
                def.insert(fresh, out[out.len() - 1].clone());
                stale.insert(fresh, false);
                *operand = fresh;
                inserted += 1;
            }

            if writes_flags(&inst.op) {
                for v in stale.values_mut() {
                    *v = true;
                }
            }
            if classes.get(&inst.dst) == Some(&RegClass::Flags) {
                def.insert(inst.dst, inst.clone());
                stale.insert(inst.dst, false);
            }
            out.push(inst);
        }

        *sched = out;
    }

    inserted
}
