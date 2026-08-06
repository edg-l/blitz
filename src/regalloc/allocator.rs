use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::Reg;

/// Result of register allocation for one function.
pub struct RegAllocResult {
    /// Maps each VReg to the physical register assigned to it.
    pub vreg_to_reg: BTreeMap<VReg, Reg>,
    /// Number of spill slots used (each slot is 8 bytes for GPR, 16 for XMM).
    pub spill_slots: u32,
    /// Callee-saved registers that were actually assigned (must be preserved).
    pub callee_saved_used: Vec<Reg>,
    /// Final instruction list with spill/reload code inserted and coalescing
    /// aliases applied. Empty on the function-scope path, which returns its
    /// instructions per block instead.
    pub insts: Vec<ScheduledInst>,
    /// Function parameter VRegs whose precoloring was removed because they are
    /// live across a call that clobbers their ABI register. The lowering must
    /// emit a mov from the ABI register to the allocated register at function
    /// entry for these params.
    pub unprecolored_params: Vec<(VReg, Reg)>,
}
