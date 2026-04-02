use crate::egraph::enode::ENode;
use crate::ir::op::ClassId;
use crate::ir::types::Type;

/// An equivalence class of e-nodes.
#[derive(Clone)]
pub struct EClass {
    pub id: ClassId,
    pub nodes: Vec<ENode>,
    pub ty: Type,
    pub best_cost: f64,
    /// Index into `nodes` of the cheapest node, if any.
    pub best_node: Option<usize>,
}
