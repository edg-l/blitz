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
