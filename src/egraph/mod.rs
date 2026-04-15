pub mod addr_mode;
pub mod algebraic;
pub mod cost;
pub mod distributive;
pub mod eclass;
#[allow(clippy::module_inception)]
pub mod egraph;
pub mod enode;
pub mod extract;
pub mod isel;
pub mod known_bits;
pub mod phases;
pub mod rules;
pub mod strength;
pub mod unionfind;

pub use cost::{CostModel, OptGoal};
pub use eclass::EClass;
pub use egraph::EGraph;
pub use enode::ENode;
pub use extract::{
    ExtractionResult, VReg, VRegInst, extract, extraction_to_vreg_insts,
    extraction_to_vreg_insts_with_map,
};
pub use phases::{CompileOptions, run_phases};
pub use unionfind::UnionFind;
