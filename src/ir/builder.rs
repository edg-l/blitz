use smallvec::smallvec;

use crate::egraph::{EGraph, ENode};
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

// ── Value handle ──────────────────────────────────────────────────────────────

/// A lightweight value handle (wraps ClassId). Copy + Clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub ClassId);

impl Value {
    pub fn class_id(self) -> ClassId {
        self.0
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum BuildError {
    NoTerminator {
        block: BlockId,
    },
    BlockAlreadyTerminated {
        block: BlockId,
    },
    TypeMismatch {
        expected: Type,
        got: Type,
    },
    ArgCountMismatch {
        block: BlockId,
        expected: usize,
        got: usize,
    },
    NoBlocks,
    UndefinedValue,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::NoTerminator { block } => write!(f, "block {block} has no terminator"),
            BuildError::BlockAlreadyTerminated { block } => {
                write!(f, "block {block} already has a terminator")
            }
            BuildError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected:?}, got {got:?}")
            }
            BuildError::ArgCountMismatch {
                block,
                expected,
                got,
            } => {
                write!(f, "block {block} expects {expected} arguments, got {got}")
            }
            BuildError::NoBlocks => write!(f, "function has no blocks"),
            BuildError::UndefinedValue => write!(f, "undefined value"),
        }
    }
}

impl std::error::Error for BuildError {}

// ── BlockData ─────────────────────────────────────────────────────────────────

struct BlockData {
    id: BlockId,
    param_types: Vec<Type>,
    #[allow(dead_code)]
    param_values: Vec<Value>,
    ops: Vec<EffectfulOp>,
    terminated: bool,
}

// ── FunctionBuilder ───────────────────────────────────────────────────────────

macro_rules! binop {
    ($name:ident, $op:expr) => {
        pub fn $name(&mut self, a: Value, b: Value) -> Value {
            self.add_node(ENode {
                op: $op,
                children: smallvec![a.0, b.0],
            })
        }
    };
}

pub struct FunctionBuilder {
    name: String,
    param_types: Vec<Type>,
    return_types: Vec<Type>,
    egraph: EGraph,
    blocks: Vec<BlockData>,
    current_block: Option<BlockId>,
    next_block_id: BlockId,
    /// Entry block parameter values (function arguments).
    entry_params: Vec<Value>,
}

impl FunctionBuilder {
    /// Create a new FunctionBuilder. The entry block (block 0) is created
    /// automatically, and e-classes for function parameters are inserted into
    /// the e-graph.
    pub fn new(name: &str, param_types: &[Type], return_types: &[Type]) -> Self {
        let mut egraph = EGraph::new();

        // Create e-classes for function parameters using Op::Param(i, ty).
        // Param nodes have cost 0 and are not touched by algebraic rules or isel,
        // so they won't be constant-folded or rewritten.
        let entry_params: Vec<Value> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let node = ENode {
                    op: Op::Param(i as u32, ty.clone()),
                    children: smallvec![],
                };
                Value(egraph.add(node))
            })
            .collect();

        let entry_block = BlockData {
            id: 0,
            param_types: vec![],
            param_values: vec![],
            ops: vec![],
            terminated: false,
        };

        FunctionBuilder {
            name: name.to_string(),
            param_types: param_types.to_vec(),
            return_types: return_types.to_vec(),
            egraph,
            blocks: vec![entry_block],
            current_block: Some(0),
            next_block_id: 1,
            entry_params,
        }
    }

    /// Create a block with no parameters.
    pub fn create_block(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push(BlockData {
            id,
            param_types: vec![],
            param_values: vec![],
            ops: vec![],
            terminated: false,
        });
        id
    }

    /// Create a block with parameters. Returns the block ID and Value handles
    /// for each parameter (in order).
    pub fn create_block_with_params(&mut self, types: &[Type]) -> (BlockId, Vec<Value>) {
        let id = self.next_block_id;
        self.next_block_id += 1;

        let param_values: Vec<Value> = types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                // Use Op::BlockParam with unique (block_id, param_idx) to force distinct
                // classes without colliding with real Iconst constants or Param nodes.
                let node = ENode {
                    op: Op::BlockParam(id, i as u32, ty.clone()),
                    children: smallvec![],
                };
                Value(self.egraph.add(node))
            })
            .collect();

        self.blocks.push(BlockData {
            id,
            param_types: types.to_vec(),
            param_values: param_values.clone(),
            ops: vec![],
            terminated: false,
        });

        (id, param_values)
    }

    /// Set the current insertion block.
    pub fn set_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Get function parameters as Values.
    pub fn params(&self) -> &[Value] {
        &self.entry_params
    }

    /// Get the ClassIds for the function's parameters.
    pub fn param_class_ids(&self) -> Vec<crate::ir::op::ClassId> {
        self.entry_params.iter().map(|v| v.0).collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn current_block_mut(&mut self) -> &mut BlockData {
        let id = self
            .current_block
            .expect("no current block set; call set_block first");
        self.blocks
            .iter_mut()
            .find(|b| b.id == id)
            .expect("current block not found")
    }

    /// Add an e-node to the e-graph and return a Value handle.
    ///
    /// Type validation is performed by `egraph.add()`, which calls
    /// `Op::result_type()` on the node. This panics with a descriptive message
    /// on type mismatches or wrong child counts, satisfying the spec requirement
    /// of "returns an error or panics indicating type mismatch."
    fn add_node(&mut self, node: ENode) -> Value {
        let cid = self.egraph.add(node);
        Value(cid)
    }

    // ── Pure op builders ──────────────────────────────────────────────────────

    binop!(add, Op::Add);
    binop!(sub, Op::Sub);
    binop!(mul, Op::Mul);
    binop!(udiv, Op::UDiv);
    binop!(sdiv, Op::SDiv);
    binop!(urem, Op::URem);
    binop!(srem, Op::SRem);
    binop!(and, Op::And);
    binop!(or, Op::Or);
    binop!(xor, Op::Xor);
    binop!(shl, Op::Shl);
    binop!(shr, Op::Shr);
    binop!(sar, Op::Sar);
    binop!(fadd, Op::Fadd);
    binop!(fsub, Op::Fsub);
    binop!(fmul, Op::Fmul);
    binop!(fdiv, Op::Fdiv);

    pub fn iconst(&mut self, val: i64, ty: Type) -> Value {
        let node = ENode {
            op: Op::Iconst(val, ty),
            children: smallvec![],
        };
        self.add_node(node)
    }

    pub fn fconst(&mut self, val: f64) -> Value {
        let node = ENode {
            op: Op::Fconst(val.to_bits()),
            children: smallvec![],
        };
        self.add_node(node)
    }

    pub fn sext(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::Sext(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn zext(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::Zext(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn trunc(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::Trunc(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn bitcast(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::Bitcast(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn icmp(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let node = ENode {
            op: Op::Icmp(cc),
            children: smallvec![a.0, b.0],
        };
        self.add_node(node)
    }

    pub fn fsqrt(&mut self, val: Value) -> Value {
        let node = ENode {
            op: Op::Fsqrt,
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn select(&mut self, flags: Value, t: Value, f: Value) -> Value {
        let node = ENode {
            op: Op::Select,
            children: smallvec![flags.0, t.0, f.0],
        };
        self.add_node(node)
    }

    // ── Effectful op builders ─────────────────────────────────────────────────

    /// Emit a load from `addr` of type `ty`. Returns the loaded value.
    pub fn load(&mut self, addr: Value, ty: Type) -> Value {
        // Create a fresh e-class for the loaded value using a sentinel constant.
        let block_id = self
            .current_block
            .expect("no current block set; call set_block first");
        let load_count = {
            let block = self
                .blocks
                .iter()
                .find(|b| b.id == block_id)
                .expect("current block not found");
            block.ops.len()
        };
        let sentinel = -((block_id as i64 + 1) * 100_000 + load_count as i64 + 1);
        let node = ENode {
            op: Op::Iconst(sentinel, ty.clone()),
            children: smallvec![],
        };
        let load_val = Value(self.egraph.add(node));
        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Load { addr: addr.0, ty });
        load_val
    }

    /// Emit a store of `val` to `addr`.
    pub fn store(&mut self, addr: Value, val: Value) {
        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Store {
            addr: addr.0,
            val: val.0,
        });
    }

    /// Emit a call to `func` with the given `args`. Returns Values for each
    /// return type.
    pub fn call(&mut self, func: &str, args: &[Value], ret_tys: &[Type]) -> Vec<Value> {
        let block_id = self
            .current_block
            .expect("no current block set; call set_block first");
        let call_idx = {
            let block = self
                .blocks
                .iter()
                .find(|b| b.id == block_id)
                .expect("current block not found");
            block.ops.len()
        };

        let ret_vals: Vec<Value> = ret_tys
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let sentinel =
                    -((block_id as i64 + 1) * 200_000 + call_idx as i64 * 100 + i as i64 + 1);
                let node = ENode {
                    op: Op::Iconst(sentinel, ty.clone()),
                    children: smallvec![],
                };
                Value(self.egraph.add(node))
            })
            .collect();

        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Call {
            func: func.to_string(),
            args: args.iter().map(|v| v.0).collect(),
            ret_tys: ret_tys.to_vec(),
        });
        ret_vals
    }

    /// Emit a conditional branch.
    ///
    /// Panics if the block is already terminated.
    pub fn branch(
        &mut self,
        cond: Value,
        bb_true: BlockId,
        bb_false: BlockId,
        true_args: &[Value],
        false_args: &[Value],
    ) {
        // Validate arg counts against block param counts.
        let true_expected = self
            .blocks
            .iter()
            .find(|b| b.id == bb_true)
            .map(|b| b.param_types.len())
            .unwrap_or(0);
        let false_expected = self
            .blocks
            .iter()
            .find(|b| b.id == bb_false)
            .map(|b| b.param_types.len())
            .unwrap_or(0);

        assert_eq!(
            true_args.len(),
            true_expected,
            "branch to block {bb_true}: expected {true_expected} args, got {}",
            true_args.len()
        );
        assert_eq!(
            false_args.len(),
            false_expected,
            "branch to block {bb_false}: expected {false_expected} args, got {}",
            false_args.len()
        );

        let block = self.current_block_mut();
        assert!(
            !block.terminated,
            "block {} already has a terminator",
            block.id
        );
        block.ops.push(EffectfulOp::Branch {
            cond: cond.0,
            bb_true,
            bb_false,
            true_args: true_args.iter().map(|v| v.0).collect(),
            false_args: false_args.iter().map(|v| v.0).collect(),
        });
        block.terminated = true;
    }

    /// Emit an unconditional jump to `target` with arguments.
    ///
    /// Panics if the block is already terminated.
    pub fn jump(&mut self, target: BlockId, args: &[Value]) {
        let expected = self
            .blocks
            .iter()
            .find(|b| b.id == target)
            .map(|b| b.param_types.len())
            .unwrap_or(0);

        assert_eq!(
            args.len(),
            expected,
            "jump to block {target}: expected {expected} args, got {}",
            args.len()
        );

        let block = self.current_block_mut();
        assert!(
            !block.terminated,
            "block {} already has a terminator",
            block.id
        );
        block.ops.push(EffectfulOp::Jump {
            target,
            args: args.iter().map(|v| v.0).collect(),
        });
        block.terminated = true;
    }

    /// Emit a return with an optional value.
    ///
    /// Panics if the block is already terminated.
    pub fn ret(&mut self, val: Option<Value>) {
        let block = self.current_block_mut();
        assert!(
            !block.terminated,
            "block {} already has a terminator",
            block.id
        );
        block.ops.push(EffectfulOp::Ret {
            val: val.map(|v| v.0),
        });
        block.terminated = true;
    }

    // ── Finalize ──────────────────────────────────────────────────────────────

    /// Validate and finalize the function. Returns the Function and the EGraph
    /// (which is needed for the compilation pipeline).
    pub fn finalize(self) -> Result<(Function, EGraph), BuildError> {
        if self.blocks.is_empty() {
            return Err(BuildError::NoBlocks);
        }

        // Validate every block has a terminator.
        for block in &self.blocks {
            if !block.terminated {
                return Err(BuildError::NoTerminator { block: block.id });
            }
        }

        // Entry block (block 0) must have no params.
        // (We enforce this at construction; assert here as a safety check.)
        debug_assert!(self.blocks[0].param_types.is_empty());

        // Convert BlockData -> BasicBlock.
        let basic_blocks: Vec<BasicBlock> = self
            .blocks
            .into_iter()
            .map(|bd| BasicBlock {
                id: bd.id,
                param_types: bd.param_types,
                ops: bd.ops,
            })
            .collect();

        let param_class_ids: Vec<crate::ir::op::ClassId> =
            self.entry_params.iter().map(|v| v.0).collect();

        let func = Function {
            name: self.name,
            param_types: self.param_types,
            return_types: self.return_types,
            blocks: basic_blocks,
            param_class_ids,
        };

        Ok((func, self.egraph))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::Type;

    // 13.6: Build add(a, b) function, finalize succeeds.
    #[test]
    fn build_add_function() {
        let mut builder = FunctionBuilder::new("add", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];
        let sum = builder.add(a, b);
        builder.ret(Some(sum));

        let result = builder.finalize();
        assert!(
            result.is_ok(),
            "finalize should succeed: {:?}",
            result.err()
        );
        let (func, _egraph) = result.unwrap();
        assert_eq!(func.name, "add");
        assert_eq!(func.param_types, vec![Type::I64, Type::I64]);
        assert_eq!(func.return_types, vec![Type::I64]);
        assert!(func.is_well_formed());
    }

    // 13.6: Build function with conditional branch and block params.
    #[test]
    fn build_conditional_branch() {
        let mut builder = FunctionBuilder::new("max", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let a = params[0];
        let b = params[1];

        // Create merge block that takes the result.
        let (merge_block, merge_params) = builder.create_block_with_params(&[Type::I64]);
        let result_param = merge_params[0];

        // Emit comparison and branch in entry block.
        let flags = builder.icmp(CondCode::Sgt, a, b);
        builder.branch(flags, merge_block, merge_block, &[a], &[b]);

        // Emit return in merge block.
        builder.set_block(merge_block);
        builder.ret(Some(result_param));

        let result = builder.finalize();
        assert!(
            result.is_ok(),
            "finalize should succeed: {:?}",
            result.err()
        );
        let (func, _egraph) = result.unwrap();
        assert!(func.is_well_formed());
        assert_eq!(func.blocks.len(), 2);
    }

    // 13.6: Build invalid function (no terminator) -> error.
    #[test]
    fn no_terminator_returns_error() {
        let builder = FunctionBuilder::new("broken", &[Type::I64], &[]);
        let _p = builder.params()[0];
        // Intentionally omit the terminator.

        let result = builder.finalize();
        assert!(
            result.is_err(),
            "finalize should fail when block has no terminator"
        );
        let err = result.err().unwrap();
        assert!(
            matches!(err, BuildError::NoTerminator { block: 0 }),
            "expected NoTerminator for block 0, got {err}"
        );
    }

    // Double-terminator panics.
    #[test]
    #[should_panic(expected = "already has a terminator")]
    fn double_terminator_panics() {
        let mut builder = FunctionBuilder::new("double_term", &[], &[]);
        builder.ret(None);
        builder.ret(None); // should panic
    }

    // Jump arg count mismatch panics.
    #[test]
    #[should_panic(expected = "expected 2 args")]
    fn jump_arg_count_mismatch_panics() {
        let mut builder = FunctionBuilder::new("mismatch", &[Type::I64], &[]);
        let params = builder.params().to_vec();
        let (_bb2, _params2) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        // Jump with only 1 arg when 2 expected.
        builder.jump(_bb2, &[params[0]]);
    }

    // create_block_with_params returns correct number of values.
    #[test]
    fn block_params_returned() {
        let mut builder = FunctionBuilder::new("params_test", &[], &[]);
        let (bb1, params) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        assert_eq!(params.len(), 2);

        // Finalize entry block with a jump supplying the params.
        let v1 = builder.iconst(1, Type::I64);
        let v2 = builder.iconst(2, Type::I64);
        builder.jump(bb1, &[v1, v2]);

        builder.set_block(bb1);
        let result_val = builder.add(params[0], params[1]);
        builder.ret(Some(result_val));

        let result = builder.finalize();
        assert!(result.is_ok(), "{:?}", result.err());
    }
}
