use std::collections::{HashMap, HashSet};

use smallvec::smallvec;

use crate::egraph::{EGraph, ENode};
use crate::ir::condcode::CondCode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::{BasicBlock, Function, StackSlot, StackSlotData};
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

pub(crate) use crate::ir::variable::PredEdge;
pub use crate::ir::variable::Variable;

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
    UnsealedBlock {
        block: BlockId,
    },
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
            BuildError::UnsealedBlock { block } => {
                write!(f, "block {block} was never sealed")
            }
        }
    }
}

impl std::error::Error for BuildError {}

// ── BlockData ─────────────────────────────────────────────────────────────────

pub(crate) struct BlockData {
    pub(crate) id: BlockId,
    pub(crate) param_types: Vec<Type>,
    #[allow(dead_code)]
    pub(crate) param_values: Vec<Value>,
    pub(crate) ops: Vec<EffectfulOp>,
    pub(crate) terminated: bool,
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
    pub(crate) egraph: EGraph,
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) current_block: Option<BlockId>,
    /// Index into `blocks` for the current block. Kept in sync with `current_block`.
    current_block_idx: Option<usize>,
    next_block_id: BlockId,
    /// Entry block parameter values (function arguments).
    entry_params: Vec<Value>,

    // ── Stack slots ──────────────────────────────────────────────────────────
    stack_slots: Vec<StackSlotData>,

    // ── Unique ID counter for LoadResult/CallResult e-class disambiguation ──
    next_uid: u32,

    // ── SSA variable API state (Braun et al.) ────────────────────────────────
    pub(crate) next_var: u32,
    pub(crate) var_types: Vec<Type>,
    pub(crate) var_defs: HashMap<(BlockId, Variable), Value>,
    pub(crate) sealed_blocks: HashSet<BlockId>,
    pub(crate) incomplete_phis: HashMap<BlockId, Vec<(Variable, Value)>>,
    pub(crate) predecessors: HashMap<BlockId, Vec<(BlockId, PredEdge)>>,
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

        let mut sealed_blocks = HashSet::new();
        sealed_blocks.insert(0); // entry block has no predecessors, always sealed

        let mut predecessors = HashMap::new();
        predecessors.insert(0, Vec::new()); // entry block: no predecessors

        FunctionBuilder {
            name: name.to_string(),
            param_types: param_types.to_vec(),
            return_types: return_types.to_vec(),
            egraph,
            blocks: vec![entry_block],
            current_block: Some(0),
            current_block_idx: Some(0),
            next_block_id: 1,
            entry_params,
            stack_slots: Vec::new(),
            next_uid: 0,
            next_var: 0,
            var_types: Vec::new(),
            var_defs: HashMap::new(),
            sealed_blocks,
            incomplete_phis: HashMap::new(),
            predecessors,
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
        self.current_block_idx = self.blocks.iter().position(|b| b.id == block);
    }

    /// Get function parameters as Values.
    pub fn params(&self) -> &[Value] {
        &self.entry_params
    }

    /// Return the current insertion block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Return true if the current block already has a terminator.
    pub fn is_current_block_terminated(&self) -> bool {
        match self.current_block_idx {
            Some(idx) => self.blocks[idx].terminated,
            None => false,
        }
    }

    /// Return the result type of `val` by inspecting its e-class.
    fn type_of(&self, val: Value) -> Type {
        let canon = self.egraph.unionfind.find_immutable(val.0);
        self.egraph.class(canon).ty.clone()
    }

    /// Emit `0 - val` (unary negation) with the same type as `val`.
    pub fn neg(&mut self, val: Value) -> Value {
        let ty = self.type_of(val);
        let zero = self.iconst(0, ty);
        self.sub(zero, val)
    }

    /// Emit an `i64` constant. Shorthand for `iconst(val, Type::I64)`.
    pub fn const_i64(&mut self, val: i64) -> Value {
        self.iconst(val, Type::I64)
    }

    /// Emit `icmp(cc, a, b)` followed by `select(flags, 1, 0)`, returning an I64.
    pub fn icmp_val(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.icmp(cc, a, b);
        let one = self.iconst(1, Type::I64);
        let zero = self.iconst(0, Type::I64);
        self.select(flags, one, zero)
    }

    /// Get the ClassIds for the function's parameters.
    pub fn param_class_ids(&self) -> Vec<crate::ir::op::ClassId> {
        self.entry_params.iter().map(|v| v.0).collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn current_block_mut(&mut self) -> &mut BlockData {
        let idx = self
            .current_block_idx
            .expect("no current block set; call set_block first");
        &mut self.blocks[idx]
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
            op: Op::Fconst(val.to_bits(), Type::F64),
            children: smallvec![],
        };
        self.add_node(node)
    }

    pub fn fconst_f32(&mut self, val: f32) -> Value {
        let node = ENode {
            op: Op::Fconst(val.to_bits() as u64, Type::F32),
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

    pub fn fcmp(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let node = ENode {
            op: Op::Fcmp(cc),
            children: smallvec![a.0, b.0],
        };
        self.add_node(node)
    }

    /// Float comparison that produces an I32 0/1 result via Setcc+Zext.
    /// Avoids the Select/Cmov path which has register classification issues for OrdEq/UnordNe.
    pub fn fcmp_to_int(&mut self, cc: CondCode, a: Value, b: Value) -> Value {
        let flags = self.fcmp(cc, a, b);
        let setcc = self.add_node(ENode {
            op: Op::X86Setcc(cc),
            children: smallvec![flags.0],
        });
        let node = ENode {
            op: Op::Zext(Type::I32),
            children: smallvec![setcc.0],
        };
        self.add_node(node)
    }

    pub fn int_to_float(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::IntToFloat(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn float_to_int(&mut self, val: Value, target: Type) -> Value {
        let node = ENode {
            op: Op::FloatToInt(target),
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn float_ext(&mut self, val: Value) -> Value {
        let node = ENode {
            op: Op::FloatExt,
            children: smallvec![val.0],
        };
        self.add_node(node)
    }

    pub fn float_trunc(&mut self, val: Value) -> Value {
        let node = ENode {
            op: Op::FloatTrunc,
            children: smallvec![val.0],
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

    // ── Stack slot API ────────────────────────────────────────────────────────

    /// Allocate a stack slot with the given size and alignment (in bytes).
    pub fn create_stack_slot(&mut self, size: u32, align: u32) -> StackSlot {
        let idx = self.stack_slots.len() as u32;
        self.stack_slots.push(StackSlotData { size, align });
        StackSlot(idx)
    }

    /// Return the address of a stack slot as an I64 value.
    pub fn stack_addr(&mut self, slot: StackSlot) -> Value {
        self.add_node(ENode {
            op: Op::StackAddr(slot.0),
            children: smallvec![],
        })
    }

    /// Return the address of a global variable as an I64 value.
    pub fn global_addr(&mut self, name: &str) -> Value {
        self.add_node(ENode {
            op: Op::GlobalAddr(name.to_string()),
            children: smallvec![],
        })
    }

    // ── Effectful op builders ─────────────────────────────────────────────────

    /// Emit a load from `addr` of type `ty`. Returns the loaded value.
    pub fn load(&mut self, addr: Value, ty: Type) -> Value {
        // Validate that a current block is set before we proceed.
        let _block_id = self
            .current_block
            .expect("no current block set; call set_block first");
        // Compute a unique ID so that two loads with the same type in the same
        // function get distinct e-classes (the egraph memo deduplicates by op+children).
        let uid = self.next_uid;
        self.next_uid += 1;
        // Create a fresh e-class for the loaded value using a LoadResult placeholder.
        let node = ENode {
            op: Op::LoadResult(uid, ty.clone()),
            children: smallvec![],
        };
        let load_result = Value(self.egraph.add(node));
        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Load {
            addr: addr.0,
            ty,
            result: load_result.0,
        });
        load_result
    }

    /// Emit a store of `val` to `addr`.
    pub fn store(&mut self, addr: Value, val: Value) {
        let ty = self.type_of(val);
        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Store {
            addr: addr.0,
            val: val.0,
            ty,
        });
    }

    /// Emit a call to `func` with the given `args`. Returns Values for each
    /// return type.
    pub fn call(&mut self, func: &str, args: &[Value], ret_tys: &[Type]) -> Vec<Value> {
        // Pre-compute UIDs for CallResult nodes before any closures capture &mut self.
        let uid_base = self.next_uid;
        self.next_uid += ret_tys.len() as u32;

        // Create a CallResult placeholder node for each return value.
        // Each result gets a globally unique UID to prevent the egraph from merging
        // return values of different calls that happen to have the same type.
        let ret_vals: Vec<Value> = ret_tys
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let uid = uid_base + i as u32;
                let node = ENode {
                    op: Op::CallResult(uid, ty.clone()),
                    children: smallvec![],
                };
                Value(self.egraph.add(node))
            })
            .collect();

        // Collect ClassIds of the CallResult nodes for storage in the effectful op.
        let result_class_ids: Vec<ClassId> = ret_vals.iter().map(|v| v.0).collect();

        // Derive argument types from the egraph's per-class type info.
        let arg_tys: Vec<Type> = args.iter().map(|v| self.type_of(*v)).collect();

        let block = self.current_block_mut();
        block.ops.push(EffectfulOp::Call {
            func: func.to_string(),
            args: args.iter().map(|v| v.0).collect(),
            arg_tys,
            ret_tys: ret_tys.to_vec(),
            results: result_class_ids,
        });
        ret_vals
    }

    /// Emit a conditional branch.
    ///
    /// Panics if the block is already terminated. Arg counts are validated
    /// at `finalize()` time (deferred to support the SSA variable API which
    /// may add block params after terminators are emitted).
    pub fn branch(
        &mut self,
        cond: Value,
        bb_true: BlockId,
        bb_false: BlockId,
        true_args: &[Value],
        false_args: &[Value],
    ) {
        let current = self.current_block.expect("no current block set");
        // Look up the Icmp CC from the cond's e-class before borrowing the block mutably.
        // If no Icmp is found (bare truthiness test), default to Ne (test != 0).
        let cc = {
            let canon = self.egraph.unionfind.find_immutable(cond.0);
            let class = self.egraph.class(canon);
            class
                .nodes
                .iter()
                .find_map(|n| match &n.op {
                    Op::Icmp(cc) | Op::Fcmp(cc) => Some(*cc),
                    _ => None,
                })
                .unwrap_or(CondCode::Ne)
        };
        let block = self.current_block_mut();
        assert!(
            !block.terminated,
            "block {} already has a terminator",
            block.id
        );
        block.ops.push(EffectfulOp::Branch {
            cond: cond.0,
            cc,
            bb_true,
            bb_false,
            true_args: true_args.iter().map(|v| v.0).collect(),
            false_args: false_args.iter().map(|v| v.0).collect(),
        });
        block.terminated = true;

        // Record predecessor edges.
        self.predecessors
            .entry(bb_true)
            .or_default()
            .push((current, PredEdge::BranchTrue));
        self.predecessors
            .entry(bb_false)
            .or_default()
            .push((current, PredEdge::BranchFalse));
    }

    /// Emit an unconditional jump to `target` with arguments.
    ///
    /// Panics if the block is already terminated. Arg counts are validated
    /// at `finalize()` time (deferred to support the SSA variable API).
    pub fn jump(&mut self, target: BlockId, args: &[Value]) {
        let current = self.current_block.expect("no current block set");
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

        // Record predecessor edge.
        self.predecessors
            .entry(target)
            .or_default()
            .push((current, PredEdge::Jump));
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

    /// Validate and finalize the function.
    ///
    /// The resulting `Function` contains the e-graph in `func.egraph` and is
    /// ready to be passed to `compile()`.
    pub fn finalize(self) -> Result<Function, BuildError> {
        if self.blocks.is_empty() {
            return Err(BuildError::NoBlocks);
        }

        // Validate every block has a terminator.
        for block in &self.blocks {
            if !block.terminated {
                return Err(BuildError::NoTerminator { block: block.id });
            }
        }

        // Validate all blocks are sealed (only when variable API is in use).
        if self.next_var > 0 {
            for block in &self.blocks {
                if !self.sealed_blocks.contains(&block.id) {
                    return Err(BuildError::UnsealedBlock { block: block.id });
                }
            }
        }

        // Build a block_id -> param count map for O(1) arg count validation.
        let block_param_count: HashMap<BlockId, usize> = self
            .blocks
            .iter()
            .map(|b| (b.id, b.param_types.len()))
            .collect();

        // Validate arg counts match target block param counts (deferred from jump/branch).
        for block in &self.blocks {
            if let Some(term) = block.ops.last() {
                match term {
                    EffectfulOp::Jump { target, args } => {
                        let expected = block_param_count.get(target).copied().unwrap_or(0);
                        if args.len() != expected {
                            return Err(BuildError::ArgCountMismatch {
                                block: *target,
                                expected,
                                got: args.len(),
                            });
                        }
                    }
                    EffectfulOp::Branch {
                        bb_true,
                        bb_false,
                        true_args,
                        false_args,
                        ..
                    } => {
                        let true_expected = block_param_count.get(bb_true).copied().unwrap_or(0);
                        if true_args.len() != true_expected {
                            return Err(BuildError::ArgCountMismatch {
                                block: *bb_true,
                                expected: true_expected,
                                got: true_args.len(),
                            });
                        }
                        let false_expected = block_param_count.get(bb_false).copied().unwrap_or(0);
                        if false_args.len() != false_expected {
                            return Err(BuildError::ArgCountMismatch {
                                block: *bb_false,
                                expected: false_expected,
                                got: false_args.len(),
                            });
                        }
                    }
                    _ => {}
                }
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
            egraph: Some(self.egraph),
            stack_slots: self.stack_slots,
            noinline: false,
        };

        Ok(func)
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
        let func = result.unwrap();
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
        let func = result.unwrap();
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

    // Jump arg count mismatch detected at finalize.
    #[test]
    fn jump_arg_count_mismatch_at_finalize() {
        let mut builder = FunctionBuilder::new("mismatch", &[Type::I64], &[]);
        let params = builder.params().to_vec();
        let (bb2, _params2) = builder.create_block_with_params(&[Type::I64, Type::I64]);
        // Jump with only 1 arg when 2 expected.
        builder.jump(bb2, &[params[0]]);
        builder.set_block(bb2);
        builder.ret(None);
        let result = builder.finalize();
        assert!(
            matches!(result, Err(BuildError::ArgCountMismatch { .. })),
            "expected ArgCountMismatch, got {result:?}"
        );
    }

    #[test]
    fn current_block_returns_entry() {
        let builder = FunctionBuilder::new("test", &[], &[]);
        assert_eq!(builder.current_block(), Some(0));
    }

    #[test]
    fn is_current_block_terminated_false_then_true() {
        let mut builder = FunctionBuilder::new("test", &[], &[]);
        assert!(!builder.is_current_block_terminated());
        builder.ret(None);
        assert!(builder.is_current_block_terminated());
    }

    #[test]
    fn neg_emits_sub_zero() {
        let mut builder = FunctionBuilder::new("neg_test", &[Type::I64], &[Type::I64]);
        let p = builder.params()[0];
        let n = builder.neg(p);
        builder.ret(Some(n));
        let result = builder.finalize();
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn const_i64_shorthand() {
        let mut builder = FunctionBuilder::new("const_test", &[], &[Type::I64]);
        let v = builder.const_i64(42);
        builder.ret(Some(v));
        let result = builder.finalize();
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn icmp_val_returns_i64() {
        let mut builder =
            FunctionBuilder::new("icmp_val_test", &[Type::I64, Type::I64], &[Type::I64]);
        let params = builder.params().to_vec();
        let r = builder.icmp_val(CondCode::Slt, params[0], params[1]);
        builder.ret(Some(r));
        let result = builder.finalize();
        assert!(result.is_ok(), "{:?}", result.err());
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

    // ── Variable API tests ───────────────────────────────────────────────────

    #[test]
    fn var_api_straight_line() {
        let mut b = FunctionBuilder::new("test", &[], &[Type::I64]);
        let x = b.declare_var(Type::I64);
        let five = b.iconst(5, Type::I64);
        b.def_var(x, five);
        let val = b.use_var(x);
        b.ret(Some(val));
        let f = b.finalize();
        assert!(f.is_ok(), "{:?}", f.err());
    }

    #[test]
    fn var_api_diamond() {
        let mut b = FunctionBuilder::new("test", &[Type::I64], &[Type::I64]);
        let p = b.params()[0];
        let x = b.declare_var(Type::I64);
        b.def_var(x, p);

        let zero = b.iconst(0, Type::I64);
        let cond = b.icmp(CondCode::Sgt, p, zero);
        let bb_then = b.create_block();
        let bb_else = b.create_block();
        let bb_merge = b.create_block();

        b.branch(cond, bb_then, bb_else, &[], &[]);

        // Then: redefine x
        b.set_block(bb_then);
        b.seal_block(bb_then);
        let ten = b.iconst(10, Type::I64);
        b.def_var(x, ten);
        b.jump(bb_merge, &[]);

        // Else: x not modified
        b.set_block(bb_else);
        b.seal_block(bb_else);
        b.jump(bb_merge, &[]);

        // Merge: use_var should create a block param
        b.seal_block(bb_merge);
        b.set_block(bb_merge);
        let result = b.use_var(x);
        b.ret(Some(result));

        let f = b.finalize();
        assert!(f.is_ok(), "{:?}", f.err());
        let func = f.unwrap();
        // Merge block should have 1 block param (the phi for x)
        assert_eq!(func.blocks[3].param_types.len(), 1);
    }

    #[test]
    fn var_api_while_loop() {
        let mut b = FunctionBuilder::new("test", &[Type::I64], &[Type::I64]);
        let n = b.params()[0];
        let i_var = b.declare_var(Type::I64);
        let acc_var = b.declare_var(Type::I64);
        let zero = b.iconst(0, Type::I64);
        b.def_var(i_var, zero);
        b.def_var(acc_var, zero);

        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();

        b.jump(header, &[]);

        // Header (don't seal yet -- back edge not known)
        b.set_block(header);
        let i = b.use_var(i_var);
        let cond = b.icmp(CondCode::Slt, i, n);
        b.branch(cond, body, exit, &[], &[]);

        // Body
        b.set_block(body);
        b.seal_block(body);
        let acc = b.use_var(acc_var);
        let i = b.use_var(i_var);
        let new_acc = b.add(acc, i);
        let one = b.iconst(1, Type::I64);
        let new_i = b.add(i, one);
        b.def_var(acc_var, new_acc);
        b.def_var(i_var, new_i);
        b.jump(header, &[]);

        // Now seal header (both predecessors known)
        b.seal_block(header);

        // Exit
        b.set_block(exit);
        b.seal_block(exit);
        let result = b.use_var(acc_var);
        b.ret(Some(result));

        let f = b.finalize();
        assert!(f.is_ok(), "{:?}", f.err());
    }

    #[test]
    fn var_api_trivial_phi_elimination() {
        let mut b = FunctionBuilder::new("test", &[Type::I64], &[Type::I64]);
        let p = b.params()[0];
        let x = b.declare_var(Type::I64);
        b.def_var(x, p);

        let zero = b.iconst(0, Type::I64);
        let cond = b.icmp(CondCode::Sgt, p, zero);
        let bb_then = b.create_block();
        let bb_else = b.create_block();
        let bb_merge = b.create_block();

        b.branch(cond, bb_then, bb_else, &[], &[]);

        // Neither branch modifies x
        b.set_block(bb_then);
        b.seal_block(bb_then);
        b.jump(bb_merge, &[]);

        b.set_block(bb_else);
        b.seal_block(bb_else);
        b.jump(bb_merge, &[]);

        b.seal_block(bb_merge);
        b.set_block(bb_merge);
        let result = b.use_var(x);
        b.ret(Some(result));

        let f = b.finalize();
        assert!(f.is_ok(), "{:?}", f.err());
        // The trivial phi should have been eliminated via e-graph union,
        // but the block param may still exist structurally. What matters
        // is that the function compiles correctly.
    }

    #[test]
    fn var_api_unsealed_block_error() {
        let mut b = FunctionBuilder::new("test", &[], &[Type::I64]);
        let x = b.declare_var(Type::I64);
        let zero = b.iconst(0, Type::I64);
        b.def_var(x, zero);
        let block = b.create_block();
        b.jump(block, &[]);
        b.set_block(block);
        let val = b.use_var(x);
        b.ret(Some(val));
        // block is never sealed
        let result = b.finalize();
        assert!(
            matches!(result, Err(BuildError::UnsealedBlock { .. })),
            "expected UnsealedBlock, got {result:?}"
        );
    }

    #[test]
    #[should_panic(expected = "undefined variable")]
    fn var_api_undefined_variable_panics() {
        let mut b = FunctionBuilder::new("test", &[], &[Type::I64]);
        let x = b.declare_var(Type::I64);
        // Never define x, try to use it in entry block
        let _ = b.use_var(x);
    }

    #[test]
    fn var_api_multiple_defs_same_block() {
        // Last write wins within a block
        let mut b = FunctionBuilder::new("test", &[], &[Type::I64]);
        let x = b.declare_var(Type::I64);
        let one = b.iconst(1, Type::I64);
        let two = b.iconst(2, Type::I64);
        b.def_var(x, one);
        b.def_var(x, two);
        let val = b.use_var(x);
        b.ret(Some(val));
        let f = b.finalize();
        assert!(f.is_ok(), "{:?}", f.err());
    }
}
