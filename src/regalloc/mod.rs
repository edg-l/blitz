pub mod allocator;
pub mod coalesce;
pub mod coloring;
pub(crate) mod global_allocator;
pub mod global_liveness;
pub mod interference;
pub mod liveness;
pub mod rewrite;
pub mod slots;
pub mod spill;
pub mod vregset;

pub use allocator::RegAllocResult;
pub use global_allocator::allocate_global;
pub use slots::{SlotAllocator, SlotOwner};
pub use vregset::VRegSet;

use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
use crate::ir::op::PseudoOp;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::{Reg, RegClass};

/// Result of function-scope (global) register allocation.
///
/// Returned by `allocate_global` once it is implemented. Each field mirrors the
/// per-block `RegAllocResult` but covers the whole function.
pub struct GlobalRegAllocResult {
    /// Final instruction lists, one `Vec<ScheduledInst>` per block (same block
    /// order as the input `block_schedules`).
    pub per_block_insts: Vec<Vec<ScheduledInst>>,
    /// Maps every VReg in the function to its assigned physical register.
    pub vreg_to_reg: BTreeMap<VReg, Reg>,
    /// Callee-saved registers that were actually assigned and must be preserved
    /// in the function prologue/epilogue.
    pub callee_saved_used: Vec<Reg>,
    /// Function parameter VRegs whose ABI precoloring was dropped because they
    /// are live across a call that clobbers their ABI register. The lowering must
    /// emit a mov from the ABI register to the allocated register at function
    /// entry for each entry here.
    pub unprecolored_params: Vec<(VReg, Reg)>,
    /// Coalesce alias map: `from_idx -> into_idx`. When two VRegs are coalesced
    /// by coalescing, the `from` VReg is rewritten to `into` in every post-coalesce
    /// schedule. The `vreg_to_reg` map contains only `into` keys; `from` has no
    /// register assignment. Callers must apply this map when resolving a
    /// `ClassId -> VReg` (e.g. for terminator or phi-copy lowering) so stale
    /// `class_to_vreg` entries pointing at `from` chase to their canonical
    /// `into` counterpart. Transitively resolve until no further entry exists.
    ///
    /// This field cannot go until `class_to_vreg` itself names canonical VRegs
    /// after coalescing.
    pub coalesce_aliases: BTreeMap<VReg, VReg>,
}

/// Check whether a VReg is in the XMM register class.
pub fn is_xmm_vreg(vreg: VReg, vreg_classes: &BTreeMap<VReg, RegClass>) -> bool {
    vreg_classes.get(&vreg).copied() == Some(RegClass::XMM)
}

/// Build a function-wide VReg class map by scanning all blocks' schedules.
///
/// Iterates over every block's scheduled instructions and merges the per-block
/// class maps into a single `BTreeMap<VReg, RegClass>`. This is the source of
/// truth for `reg_class` when building a function-wide `InterferenceGraph`, so
/// cross-block live-in VRegs whose def is in another block get the correct class
/// from the start rather than defaulting to GPR.
pub fn build_vreg_classes_from_all_blocks(
    block_schedules: &[Vec<ScheduledInst>],
) -> BTreeMap<VReg, RegClass> {
    let mut map: BTreeMap<VReg, RegClass> = BTreeMap::new();
    for sched in block_schedules {
        for (&vreg, &class) in &build_vreg_classes_from_insts(sched) {
            // A block that saw the definition wins over one that saw only a use
            // and fell back to the GPR default: XMM and Flags are both stated by
            // a defining op, GPR is also what "no idea" looks like.
            let entry = map.entry(vreg).or_insert(class);
            if class != RegClass::GPR {
                *entry = class;
            }
        }
    }
    map
}

/// Build a VReg class map: FP ops (X86Addsd etc.) use XMM; everything else uses GPR.
///
/// Propagates XMM class to operands of FP instructions (excluding barrier ops
/// whose operands are call/store args of mixed types).
pub fn build_vreg_classes_from_insts(insts: &[ScheduledInst]) -> BTreeMap<VReg, RegClass> {
    let mut map: BTreeMap<VReg, RegClass> = BTreeMap::new();
    for inst in insts {
        let class = if inst.op.is_fp_op() {
            RegClass::XMM
        } else {
            RegClass::GPR
        };
        map.insert(inst.dst, class);
        for &op in &inst.operands {
            map.entry(op).or_insert(RegClass::GPR);
        }
    }
    for inst in insts {
        // Barrier pseudo-ops carry call/store arguments of every class, so
        // their operand list says nothing about register class.
        if matches!(
            &inst.op,
            crate::ir::op::Op::Pseudo(PseudoOp::CallResult(_, _))
                | crate::ir::op::Op::Pseudo(PseudoOp::VoidCallBarrier)
                | crate::ir::op::Op::Pseudo(PseudoOp::StoreBarrier)
        ) {
            continue;
        }
        // Force the operand class for ops that know it. This is what fixes
        // cross-block live-ins, whose def is in another block and would
        // otherwise keep the GPR default from the loop above.
        //
        // `operand_reg_class` rather than `is_fp_op`: the latter describes the
        // result, and cvtsi2sd/cvttsd2si/movq read the opposite class from the
        // one they write.
        if inst.op.is_fp_op() || inst.op.has_cross_class_operands() {
            let class = inst.op.operand_reg_class();
            for &op in &inst.operands {
                map.insert(op, class);
            }
        }
    }
    // Flags last, so it wins over the GPR default an earlier pass gave a value
    // it saw as an operand before reaching its definition.
    //
    // Two shapes produce one: an op whose result *is* the flags, and `Proj1` of
    // a pair whose second element is flags -- which is every pair-producing op
    // except a division, whose `Proj1` is the remainder and a real register
    // value. That distinction cannot be made from the projection alone, so it
    // is made from the op defining its operand, the same way `lower.rs` makes
    // it.
    let def_op: BTreeMap<VReg, &crate::ir::op::Op> = insts.iter().map(|i| (i.dst, &i.op)).collect();
    for inst in insts {
        if inst.op.produces_flags() {
            map.insert(inst.dst, RegClass::Flags);
            continue;
        }
        if matches!(inst.op, crate::ir::op::Op::Pure(crate::ir::op::PureOp::Proj1))
            && let Some(&src) = inst.operands.first()
            // A projection whose pair is defined in another block is not a
            // flags projection: flags cannot cross a block boundary, and
            // linearization re-emits them per block for that reason.
            && let Some(src_op) = def_op.get(&src)
            && !src_op.result_in_fixed_regs()
        {
            map.insert(inst.dst, RegClass::Flags);
        }
    }
    map
}
