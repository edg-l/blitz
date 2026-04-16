use smallvec::smallvec;

use crate::egraph::ENode;
use crate::ir::builder::{FunctionBuilder, Value};
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::op::Op;
use crate::ir::types::Type;

// ── Variable handle ──────────────────────────────────────────────────────────

/// A mutable variable handle for the SSA variable API.
///
/// Frontends declare variables with `FunctionBuilder::declare_var()`, define them
/// with `def_var()`, and read them with `use_var()`. The builder automatically
/// constructs SSA block parameters and wires jump/branch arguments using the
/// Braun et al. algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable(pub u32);

// ── Predecessor edge kind ────────────────────────────────────────────────────

/// Records which arm of a terminator produced a particular predecessor edge.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PredEdge {
    Jump,
    BranchTrue,
    BranchFalse,
}

// ── SSA variable API (Braun et al.) ─────────────────────────────────────────

impl FunctionBuilder {
    /// Declare a new mutable variable of the given type.
    pub fn declare_var(&mut self, ty: Type) -> Variable {
        let var = Variable(self.next_var);
        self.next_var += 1;
        self.var_types.push(ty);
        var
    }

    /// Define (or redefine) a variable in the current block.
    pub fn def_var(&mut self, var: Variable, val: Value) {
        let block = self.current_block.expect("no current block set");
        self.var_defs.insert((block, var), val);
    }

    /// Read the current SSA value of a variable.
    ///
    /// If the variable was not defined in the current block, the Braun algorithm
    /// walks predecessors and inserts block parameters (phis) as needed.
    pub fn use_var(&mut self, var: Variable) -> Value {
        let block = self.current_block.expect("no current block set");
        self.read_variable(var, block)
    }

    /// Mark a block as sealed: all its predecessors are now known.
    ///
    /// **Precondition:** every predecessor block must have already emitted its
    /// terminator (jump/branch) before sealing. The Braun algorithm resolves
    /// incomplete phis by walking predecessor terminators.
    pub fn seal_block(&mut self, block: BlockId) {
        assert!(
            !self.sealed_blocks.contains(&block),
            "block {block} is already sealed"
        );

        // Resolve all incomplete phis for this block.
        let incomplete = self.incomplete_phis.remove(&block).unwrap_or_default();
        for (var, _phi) in incomplete {
            let preds: Vec<(BlockId, PredEdge)> =
                self.predecessors.get(&block).cloned().unwrap_or_default();
            for &(pred_block, edge) in &preds {
                let val = self.read_variable(var, pred_block);
                self.append_jump_arg_for_edge(pred_block, edge, val);
            }
        }

        self.sealed_blocks.insert(block);
    }

    /// Braun et al. core: recursively resolve a variable's reaching definition.
    fn read_variable(&mut self, var: Variable, block: BlockId) -> Value {
        // 1. Local definition in this block?
        if let Some(&val) = self.var_defs.get(&(block, var)) {
            return val;
        }

        // 2. Block not sealed? Create incomplete phi placeholder.
        if !self.sealed_blocks.contains(&block) {
            let ty = self.var_types[var.0 as usize].clone();
            let phi = self.add_block_param(block, ty);
            self.var_defs.insert((block, var), phi);
            self.incomplete_phis
                .entry(block)
                .or_default()
                .push((var, phi));
            return phi;
        }

        // 3. Sealed block -- resolve through predecessors.
        let preds: Vec<(BlockId, PredEdge)> =
            self.predecessors.get(&block).cloned().unwrap_or_default();

        let val = if preds.is_empty() {
            panic!(
                "use of undefined variable {:?} in block {} (no predecessors)",
                var, block
            );
        } else if preds.len() == 1 {
            // Single predecessor -- no phi needed, recurse directly.
            self.read_variable(var, preds[0].0)
        } else {
            // Multiple predecessors -- need a phi (block param).
            let ty = self.var_types[var.0 as usize].clone();
            let phi = self.add_block_param(block, ty);

            // Store BEFORE recursing to break cycles (loops).
            self.var_defs.insert((block, var), phi);

            // Wire args from each predecessor.
            for &(pred_block, edge) in &preds {
                let pred_val = self.read_variable(var, pred_block);
                self.append_jump_arg_for_edge(pred_block, edge, pred_val);
            }

            phi
        };

        // Cache result for single-predecessor case.
        self.var_defs.insert((block, var), val);
        val
    }

    /// Incrementally add a block parameter, creating an Op::BlockParam e-node.
    fn add_block_param(&mut self, block: BlockId, ty: Type) -> Value {
        let block_data = self
            .blocks
            .iter()
            .find(|b| b.id == block)
            .expect("block not found");
        let param_idx = block_data.param_types.len() as u32;

        let node = ENode {
            op: Op::BlockParam(block, param_idx, ty.clone()),
            children: smallvec![],
        };
        let cid = self.egraph.add(node);
        let val = Value(cid);

        // Re-borrow block_data mutably after egraph mutation.
        let block_data = self
            .blocks
            .iter_mut()
            .find(|b| b.id == block)
            .expect("block not found");
        block_data.param_types.push(ty);
        block_data.param_values.push(val);
        val
    }

    /// Append a value to a predecessor's terminator args targeting a specific block.
    fn append_jump_arg_for_edge(&mut self, from_block: BlockId, edge: PredEdge, arg: Value) {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| b.id == from_block)
            .expect("from_block not found");
        let term = block
            .ops
            .last_mut()
            .and_then(|op| op.as_terminator_mut())
            .expect("from_block has no terminator");
        match (term, edge) {
            (EffectfulOp::Jump { args, .. }, PredEdge::Jump) => {
                args.push(arg.0);
            }
            (EffectfulOp::Branch { true_args, .. }, PredEdge::BranchTrue) => {
                true_args.push(arg.0);
            }
            (EffectfulOp::Branch { false_args, .. }, PredEdge::BranchFalse) => {
                false_args.push(arg.0);
            }
            _ => panic!("edge kind does not match terminator type"),
        }
    }
}
