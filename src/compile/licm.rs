use std::collections::BTreeMap;

use crate::egraph::EGraph;
use crate::ir::function::Function;
use crate::ir::op::ClassId;

/// Extra roots to add to specific blocks during linearization.
/// Maps block_index -> Vec<ClassId> of invariant classes to emit there.
pub type ExtraRoots = BTreeMap<usize, Vec<ClassId>>;

/// Run LICM: detect loops, insert preheaders, identify invariant classes.
/// Returns extra roots that the linearization phase should include.
pub fn run_licm(_func: &mut Function, _egraph: &mut EGraph) -> ExtraRoots {
    BTreeMap::new()
}
