pub mod allocator;
pub mod coalesce;
pub mod coloring;
pub(crate) mod global_allocator;
pub mod global_liveness;
pub mod interference;
pub mod liveness;
pub mod rewrite;
pub mod spill;

pub use allocator::{RegAllocResult, allocate};
pub use global_allocator::allocate_global;

use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
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
    /// Total spill slots used across the entire function (each slot is 8 bytes
    /// for GPR, 16 bytes for XMM).
    pub spill_slots: u32,
    /// Callee-saved registers that were actually assigned and must be preserved
    /// in the function prologue/epilogue.
    pub callee_saved_used: Vec<Reg>,
    /// Function parameter VRegs whose ABI precoloring was dropped because they
    /// are live across a call that clobbers their ABI register. The lowering must
    /// emit a mov from the ABI register to the allocated register at function
    /// entry for each entry here.
    pub unprecolored_params: Vec<(VReg, Reg)>,
    /// Coalesce alias map: `from_idx -> into_idx`. When two VRegs are coalesced
    /// by Phase 3, the `from` VReg is rewritten to `into` in every post-coalesce
    /// schedule. The `vreg_to_reg` map contains only `into` keys; `from` has no
    /// register assignment. Callers must apply this map when resolving a
    /// `ClassId -> VReg` (e.g. for terminator or phi-copy lowering) so stale
    /// `class_to_vreg` entries pointing at `from` chase to their canonical
    /// `into` counterpart. Transitively resolve until no further entry exists.
    ///
    /// NOTE: this field cannot be removed until `class_to_vreg` is updated after
    /// coalescing to reflect canonical VRegs. See Phase 8 follow-up work.
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
            // XMM wins over GPR if a VReg appears in multiple blocks with different classes.
            let entry = map.entry(vreg).or_insert(class);
            if class == RegClass::XMM {
                *entry = RegClass::XMM;
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
        if inst.op.is_fp_op()
            && !matches!(
                &inst.op,
                crate::ir::op::Op::CallResult(_, _)
                    | crate::ir::op::Op::VoidCallBarrier
                    | crate::ir::op::Op::StoreBarrier
            )
        {
            for &op in &inst.operands {
                map.insert(op, RegClass::XMM);
            }
        }
    }
    map
}
