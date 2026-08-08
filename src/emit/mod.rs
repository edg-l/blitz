pub mod align;
pub mod dead_inst;
pub mod elf;
pub mod object;
pub mod peephole;
pub mod phi_elim;
pub mod relax;

pub use align::{apply_pads, loop_header_pads, shift_positions};
pub use object::{FunctionInfo, ObjectFile};
pub use peephole::{flags_dead_after, peephole};
pub use phi_elim::phi_copies;
pub use relax::relax_branches;
