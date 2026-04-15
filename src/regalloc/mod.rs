pub mod allocator;
pub mod coalesce;
pub mod coloring;
pub mod global_liveness;
pub mod interference;
pub mod liveness;
pub mod rewrite;
pub mod spill;
pub mod split;

pub use allocator::{RegAllocResult, allocate};

use std::collections::BTreeMap;

use crate::egraph::extract::VReg;
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::reg::RegClass;

/// Check whether a VReg is in the XMM register class.
pub fn is_xmm_vreg(vreg: VReg, vreg_classes: &BTreeMap<VReg, RegClass>) -> bool {
    vreg_classes.get(&vreg).copied() == Some(RegClass::XMM)
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
