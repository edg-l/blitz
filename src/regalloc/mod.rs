pub mod allocator;
pub mod coalesce;
pub mod coloring;
pub mod interference;
pub mod liveness;
pub mod rewrite;
pub mod spill;

pub use allocator::{RegAllocResult, allocate};
