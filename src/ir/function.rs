use crate::egraph::EGraph;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::op::ClassId;
use crate::ir::types::Type;

/// A basic block in the CFG skeleton.
///
/// The block must end with exactly one terminator (`Branch`, `Jump`, or `Ret`).
/// All preceding operations are non-terminators. Block parameters act as
/// phi-node replacements: predecessors pass arguments when jumping to this block.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    /// Types of this block's parameters (phi-node arguments from predecessors).
    pub param_types: Vec<Type>,
    /// Effectful operations, including the terminator as the final element.
    pub ops: Vec<EffectfulOp>,
}

impl BasicBlock {
    /// Create a new empty `BasicBlock`.
    pub fn new(id: BlockId, param_types: Vec<Type>) -> Self {
        Self {
            id,
            param_types,
            ops: Vec::new(),
        }
    }

    /// Returns `true` if the block's last operation is a terminator.
    pub fn is_well_formed(&self) -> bool {
        if self.ops.is_empty() {
            return false;
        }
        // The last op must be a terminator.
        if !self.ops.last().unwrap().is_terminator() {
            return false;
        }
        // No terminator before the last position.
        let non_last = self.ops.len() - 1;
        !self.ops[..non_last].iter().any(|op| op.is_terminator())
    }

    /// Panics if the block is not well-formed.
    pub fn validate(&self) {
        assert!(
            !self.ops.is_empty(),
            "BasicBlock {} has no operations (needs a terminator)",
            self.id
        );
        assert!(
            self.ops.last().unwrap().is_terminator(),
            "BasicBlock {} last op is not a terminator",
            self.id
        );
        let non_last = self.ops.len() - 1;
        assert!(
            !self.ops[..non_last].iter().any(|op| op.is_terminator()),
            "BasicBlock {} has a terminator before the final position",
            self.id
        );
    }
}

/// A function in the Blitz IR, consisting of a CFG skeleton of `BasicBlock`s.
///
/// The first block is the entry block (no block parameters by convention).
/// Pure operations live in the e-graph (not here); effectful ops reference
/// e-class IDs for their pure operands.
pub struct Function {
    pub name: String,
    pub param_types: Vec<Type>,
    pub return_types: Vec<Type>,
    pub blocks: Vec<BasicBlock>,
    /// The ClassIds of the function's parameters in the e-graph.
    /// Populated by `FunctionBuilder::finalize()`; empty if constructed manually.
    pub param_class_ids: Vec<ClassId>,
    /// The e-graph containing all pure operations for this function.
    /// Populated by `FunctionBuilder::finalize()` and consumed by `compile()`.
    pub egraph: Option<EGraph>,
}

impl std::fmt::Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("name", &self.name)
            .field("param_types", &self.param_types)
            .field("return_types", &self.return_types)
            .field("blocks", &self.blocks)
            .field("param_class_ids", &self.param_class_ids)
            .field("egraph", &self.egraph.as_ref().map(|_| "<EGraph>"))
            .finish()
    }
}

impl Function {
    /// Create a new function with no blocks.
    pub fn new(name: impl Into<String>, param_types: Vec<Type>, return_types: Vec<Type>) -> Self {
        Self {
            name: name.into(),
            param_types,
            return_types,
            blocks: Vec::new(),
            param_class_ids: Vec::new(),
            egraph: None,
        }
    }

    /// Returns `true` if every block is well-formed and there is at least one block.
    pub fn is_well_formed(&self) -> bool {
        !self.blocks.is_empty() && self.blocks.iter().all(|b| b.is_well_formed())
    }

    /// Panics if any block fails validation, or if there are no blocks.
    pub fn validate(&self) {
        assert!(
            !self.blocks.is_empty(),
            "Function '{}' has no blocks",
            self.name
        );
        for block in &self.blocks {
            block.validate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::op::ClassId;
    use crate::ir::types::Type;

    fn ret_none() -> EffectfulOp {
        EffectfulOp::Ret { val: None }
    }

    fn jump_to(target: BlockId) -> EffectfulOp {
        EffectfulOp::Jump {
            target,
            args: vec![],
        }
    }

    // ── BasicBlock ────────────────────────────────────────────────────────────

    #[test]
    fn block_empty_is_not_well_formed() {
        let bb = BasicBlock::new(0, vec![]);
        assert!(!bb.is_well_formed());
    }

    #[test]
    fn block_with_only_ret_is_well_formed() {
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops.push(ret_none());
        assert!(bb.is_well_formed());
    }

    #[test]
    fn block_with_load_then_ret() {
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops.push(EffectfulOp::Load {
            addr: ClassId(0),
            ty: Type::I64,
            result: ClassId::NONE,
        });
        bb.ops.push(ret_none());
        assert!(bb.is_well_formed());
    }

    #[test]
    #[should_panic]
    fn block_missing_terminator_panics_on_validate() {
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops.push(EffectfulOp::Load {
            addr: ClassId(0),
            ty: Type::I64,
            result: ClassId::NONE,
        });
        bb.validate();
    }

    #[test]
    #[should_panic]
    fn block_terminator_not_last_panics_on_validate() {
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops.push(ret_none());
        bb.ops.push(EffectfulOp::Load {
            addr: ClassId(0),
            ty: Type::I64,
            result: ClassId::NONE,
        });
        bb.validate();
    }

    #[test]
    fn block_with_params() {
        let bb = BasicBlock::new(1, vec![Type::I64, Type::I32]);
        assert_eq!(bb.param_types, vec![Type::I64, Type::I32]);
    }

    // ── Function ──────────────────────────────────────────────────────────────

    #[test]
    fn function_no_blocks_is_not_well_formed() {
        let f = Function::new("empty", vec![], vec![]);
        assert!(!f.is_well_formed());
    }

    #[test]
    fn function_single_block() {
        let mut f = Function::new("foo", vec![Type::I64], vec![Type::I64]);
        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(ret_none());
        f.blocks.push(bb0);
        assert!(f.is_well_formed());
    }

    #[test]
    fn function_two_blocks() {
        let mut f = Function::new("two", vec![], vec![]);

        let mut bb0 = BasicBlock::new(0, vec![]);
        bb0.ops.push(jump_to(1));
        f.blocks.push(bb0);

        // BB1 has a block parameter (phi-like value from BB0)
        let mut bb1 = BasicBlock::new(1, vec![Type::I64]);
        bb1.ops.push(ret_none());
        f.blocks.push(bb1);

        assert!(f.is_well_formed());
        f.validate(); // must not panic
    }

    #[test]
    #[should_panic]
    fn function_with_bad_block_panics_on_validate() {
        let mut f = Function::new("bad", vec![], vec![]);
        // Block has no terminator
        f.blocks.push(BasicBlock::new(0, vec![]));
        f.validate();
    }

    #[test]
    fn branch_terminator_test() {
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops.push(EffectfulOp::Store {
            addr: ClassId(1),
            val: ClassId(2),
        });
        bb.ops.push(EffectfulOp::Branch {
            cond: ClassId(3),
            bb_true: 1,
            bb_false: 2,
            true_args: vec![],
            false_args: vec![],
        });
        assert!(bb.is_well_formed());
    }
}
