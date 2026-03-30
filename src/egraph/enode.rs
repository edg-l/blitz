use std::hash::{Hash, Hasher};

use smallvec::SmallVec;

use crate::ir::op::{ClassId, Op};

/// A single e-node: an operation applied to zero or more child e-classes.
///
/// Children must be canonicalized (via `find`) before hashing or comparing,
/// as the caller is responsible for canonicalization before insertion into the memo table.
#[derive(Debug, Clone)]
pub struct ENode {
    pub op: Op,
    pub children: SmallVec<[ClassId; 2]>,
}

impl PartialEq for ENode {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.children == other.children
    }
}

impl Eq for ENode {}

impl Hash for ENode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.children.hash(state);
    }
}
