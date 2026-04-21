pub mod allocator;
pub mod coalesce;
pub mod coloring;
pub(crate) mod global_allocator;
pub mod global_liveness;
pub mod interference;
pub mod liveness;
pub mod rewrite;
pub mod spill;
pub mod split;

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
    /// Per-block VReg rename maps produced by spill insertion. Entry `rename_maps[b][old]
    /// = new` means that spill/reload code in block `b` replaced `old` with `new` as
    /// the live name of a value; terminator and effectful-op lowering consult these
    /// maps to resolve ClassId -> VReg after allocation.
    pub per_block_rename_maps: Vec<BTreeMap<VReg, VReg>>,
    /// VRegs that were spilled to a stack slot. Maps `VReg -> slot index`.
    /// The caller uses this to materialize reloads for terminator/effectful-op
    /// ClassIds that resolve to a spilled VReg but whose use was not already
    /// rewritten by `insert_spills_global` (e.g., a Ret value whose only use is
    /// the terminator itself, not a `ScheduledInst` operand).
    pub vreg_slot: BTreeMap<VReg, u32>,
    /// VRegs that were rematerialized (not slot-backed). Maps `VReg -> defining Op`.
    /// The caller uses this to emit a fresh remat copy (with empty operands) for
    /// terminator/effectful-op ClassIds whose def was dropped by
    /// `insert_spills_global`.
    pub vreg_remat_op: BTreeMap<VReg, crate::ir::op::Op>,
    /// Coalesce alias map: `from_idx -> into_idx`. When two VRegs are coalesced
    /// by Phase 3, the `from` VReg is rewritten to `into` in every post-coalesce
    /// schedule. The `vreg_to_reg` map contains only `into` keys; `from` has no
    /// register assignment. Callers must apply this map when resolving a
    /// `ClassId -> VReg` (e.g. for terminator or phi-copy lowering) so stale
    /// `class_to_vreg` entries pointing at `from` chase to their canonical
    /// `into` counterpart. Transitively resolve until no further entry exists.
    pub coalesce_aliases: BTreeMap<VReg, VReg>,
    /// Set to `true` if Phase 5 needed to run the spill-and-recolor loop (i.e.,
    /// Phase 4 reported overshoot). When the split pass is active, this indicates
    /// the splitter missed an infeasibility and the legacy fallback ran.
    pub spill_loop_triggered: bool,
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
