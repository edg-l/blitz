use crate::egraph::enode::ENode;
use crate::egraph::known_bits::KnownBits;
use crate::ir::op::ClassId;
use crate::ir::types::Type;

/// An equivalence class of e-nodes.
#[derive(Clone)]
pub struct EClass {
    pub id: ClassId,
    pub nodes: Vec<ENode>,
    pub ty: Type,
    /// Constant value for this e-class, if known (set when an Iconst node is added).
    pub constant_value: Option<(i64, Type)>,
    /// Known-bits analysis: which bits are provably 0 or 1.
    pub known_bits: KnownBits,
}
