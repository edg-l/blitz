pub mod align;
pub mod elf;
pub mod object;
pub mod peephole;
pub mod phi_elim;
pub mod relax;

pub use align::align_loop_headers;
pub use object::{FunctionInfo, ObjectFile};
pub use peephole::{flags_dead_after, peephole};
pub use phi_elim::phi_copies;
pub use relax::relax_branches;
