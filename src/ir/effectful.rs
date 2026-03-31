use crate::ir::op::ClassId;
use crate::ir::types::Type;

pub type Symbol = String;
pub type BlockId = u32;

/// Effectful operations that must appear in the CFG skeleton (not the e-graph).
///
/// All operands that are pure computed values are referenced by `ClassId`,
/// pointing into the e-graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectfulOp {
    /// Load a value of type `ty` from the given address e-class.
    /// `result` is the e-graph ClassId of the `Op::LoadResult` node that
    /// represents the loaded value in the pure-op world.
    Load {
        addr: ClassId,
        ty: Type,
        result: ClassId,
    },

    /// Store a value e-class to an address e-class.
    Store { addr: ClassId, val: ClassId },

    /// Call a named function with the given argument e-classes.
    /// `ret_tys` lists the types of the return values.
    /// `results` holds the e-graph ClassIds of the `Op::CallResult` nodes that
    /// represent the return values in the pure-op world (one per ret_ty).
    Call {
        func: Symbol,
        args: Vec<ClassId>,
        ret_tys: Vec<Type>,
        results: Vec<ClassId>,
    },

    /// Conditional branch to `bb_true` or `bb_false` depending on flags.
    /// `true_args` / `false_args` are passed as block parameters.
    Branch {
        cond: ClassId,
        bb_true: BlockId,
        bb_false: BlockId,
        true_args: Vec<ClassId>,
        false_args: Vec<ClassId>,
    },

    /// Unconditional jump to `target` with block arguments.
    Jump { target: BlockId, args: Vec<ClassId> },

    /// Return from the function, optionally with a value.
    Ret { val: Option<ClassId> },
}

impl EffectfulOp {
    /// Returns `true` if this operation is a block terminator.
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. }
        )
    }

    /// Returns a mutable reference to this op if it is a terminator.
    pub fn as_terminator_mut(&mut self) -> Option<&mut Self> {
        if self.is_terminator() {
            Some(self)
        } else {
            None
        }
    }
}
