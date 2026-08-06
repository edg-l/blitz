//! How many values a region needs live at once, estimated before scheduling.
//!
//! Two passes decide whether a transform fits in the register file: LICM, which
//! hoists a value into a loop's live range, and inlining, which lands a callee's
//! whole live range inside its caller. Both run long before the allocator exists,
//! so neither can ask what the pressure is; both need the same estimate, and it
//! is the same estimate for the same reason.
//!
//! **Concurrent, not cumulative.** Summing the distinct classes a region names
//! counts values that are never live together: a straight run of statements
//! names one value per statement and holds two. Summing reads every long region
//! as full and refuses the transforms that pay on small ones -- measured, it
//! regressed 8 code-size rows, all of them small loops. What a region needs at
//! once is what it carries across the whole region plus the widest single point
//! in it.

use std::collections::BTreeSet;

use crate::egraph::EGraph;
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::ir::types::Type;

/// `(gpr, xmm)` values live at once across `blocks`.
///
/// `carried` are the values live across the whole region by construction -- a
/// loop header's parameters, a callee's own parameters -- and every one of them
/// occupies a register the entire time. The rest is the widest single effectful
/// op, which is the most any one point in the region demands.
pub(crate) fn concurrent_demand(
    func: &Function,
    egraph: &EGraph,
    blocks: impl Iterator<Item = usize>,
    carried: &[Type],
) -> (u32, u32) {
    let mut gpr = 0;
    let mut xmm = 0;
    for ty in carried {
        if ty.is_float() {
            xmm += 1;
        } else {
            gpr += 1;
        }
    }
    let mut widest_gpr = 0;
    let mut widest_xmm = 0;
    for block_idx in blocks {
        let Some(block) = func.blocks.get(block_idx) else {
            continue;
        };
        for op in &block.ops {
            let mut named: BTreeSet<ClassId> = BTreeSet::new();
            op.for_each_class_id(|cid| {
                named.insert(egraph.unionfind.find_immutable(cid));
            });
            let (mut g, mut x) = (0, 0);
            for cid in named {
                if egraph.class(cid).ty.is_float() {
                    x += 1;
                } else {
                    g += 1;
                }
            }
            widest_gpr = widest_gpr.max(g);
            widest_xmm = widest_xmm.max(x);
        }
    }
    (gpr + widest_gpr, xmm + widest_xmm)
}

/// The register file each class is colouring against.
///
/// The frame pointer costs a GPR, and a region a pressure check would refuse is
/// exactly the shape that forces one -- so assume it. The cheaper assumption is
/// the one that transforms more, which is the behaviour these checks exist to
/// correct.
pub(crate) fn budgets() -> (u32, u32) {
    (
        crate::regalloc::coloring::available_gpr_colors(true),
        crate::regalloc::coloring::AVAILABLE_XMM_COLORS,
    )
}
