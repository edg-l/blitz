use std::collections::BTreeMap;

use smallvec::SmallVec;

use crate::egraph::EGraph;
use crate::egraph::enode::ENode;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};

/// Context for remapping ClassIds, BlockIds, stack slots, and UIDs
/// when importing a callee's e-graph and blocks into a caller.
pub struct RemapContext {
    pub class_map: BTreeMap<ClassId, ClassId>,
    pub block_map: BTreeMap<BlockId, BlockId>,
    pub slot_offset: u32,
    pub uid_offset: u32,
}

impl RemapContext {
    /// Create a new remap context. Seeds `class_map` with callee param -> call arg mappings.
    pub fn new(caller: &Function, callee: &Function, call_args: &[ClassId]) -> Self {
        let slot_offset = caller.stack_slots.len() as u32;
        let uid_offset = max_uid_in_function(caller) + 1;

        // Map callee block IDs to new IDs starting past the caller's max block ID.
        // Using blocks.len() would collide when IDs are non-sequential (e.g. after LICM).
        let block_base = caller.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;
        let mut block_map = BTreeMap::new();
        for (i, block) in callee.blocks.iter().enumerate() {
            block_map.insert(block.id, block_base + i as u32);
        }

        // Seed class_map: callee param i -> call argument i.
        let mut class_map = BTreeMap::new();
        for (i, &callee_param_id) in callee.param_class_ids.iter().enumerate() {
            if i < call_args.len() {
                class_map.insert(callee_param_id, call_args[i]);
            }
        }

        Self {
            class_map,
            block_map,
            slot_offset,
            uid_offset,
        }
    }

    /// Import all e-nodes from the callee e-graph into the caller e-graph.
    /// Walks ClassIds 0..N in order. Param nodes are skipped (must be pre-seeded).
    pub fn import_egraph(&mut self, caller_egraph: &mut EGraph, callee_egraph: &EGraph) {
        let num_classes = callee_egraph.classes.len();

        // First pass: import all non-empty classes (canonical representatives).
        for class_idx in 0..num_classes {
            let callee_class_id = ClassId(class_idx as u32);

            if self.class_map.contains_key(&callee_class_id) {
                continue;
            }

            let eclass = &callee_egraph.classes[class_idx];
            if eclass.nodes.is_empty() {
                continue; // merged away, handle in second pass
            }

            let enode = &eclass.nodes[0];

            // Param nodes must have been pre-seeded; panic if we encounter one unmapped.
            if matches!(enode.op, Op::Param(..)) {
                panic!(
                    "import_egraph: encountered unmapped Param node at ClassId({})",
                    class_idx
                );
            }

            // Rewrite the op.
            let new_op = match &enode.op {
                Op::BlockParam(block_id, idx, ty) => {
                    let new_block = self.remap_block_id(*block_id);
                    Op::BlockParam(new_block, *idx, ty.clone())
                }
                Op::LoadResult(uid, ty) => Op::LoadResult(uid + self.uid_offset, ty.clone()),
                Op::CallResult(uid, ty) => Op::CallResult(uid + self.uid_offset, ty.clone()),
                Op::StackAddr(slot) => Op::StackAddr(slot + self.slot_offset),
                other => other.clone(),
            };

            // Remap children.
            let new_children: SmallVec<[ClassId; 2]> = enode
                .children
                .iter()
                .map(|&child| {
                    if child == ClassId::NONE {
                        ClassId::NONE
                    } else {
                        debug_assert!(
                            self.class_map.contains_key(&child),
                            "import_egraph: child ClassId({}) not in class_map when importing ClassId({})",
                            child.0,
                            class_idx
                        );
                        self.class_map[&child]
                    }
                })
                .collect();

            let new_enode = ENode {
                op: new_op,
                children: new_children,
            };
            let new_class_id = caller_egraph.add(new_enode);
            self.class_map.insert(callee_class_id, new_class_id);
        }

        // Second pass: map empty (merged-away) classes to their canonical representative's mapping.
        for class_idx in 0..num_classes {
            let callee_class_id = ClassId(class_idx as u32);
            if self.class_map.contains_key(&callee_class_id) {
                continue;
            }
            let canonical = callee_egraph.unionfind.find_immutable(callee_class_id);
            if let Some(&mapped) = self.class_map.get(&canonical) {
                self.class_map.insert(callee_class_id, mapped);
            }
        }
    }

    /// Look up a remapped ClassId. Panics if not found.
    pub fn remap_class_id(&self, id: ClassId) -> ClassId {
        if id == ClassId::NONE {
            return ClassId::NONE;
        }
        *self
            .class_map
            .get(&id)
            .unwrap_or_else(|| panic!("remap_class_id: ClassId({}) not in class_map", id.0))
    }

    /// Remap a BlockId through the block_map. Panics if not found.
    pub fn remap_block_id(&self, id: BlockId) -> BlockId {
        *self
            .block_map
            .get(&id)
            .unwrap_or_else(|| panic!("remap_block_id: BlockId({}) not in block_map", id))
    }

    /// Remap all ClassId and BlockId references in an EffectfulOp.
    pub fn remap_effectful_op(&self, op: &EffectfulOp) -> EffectfulOp {
        match op {
            EffectfulOp::Load { addr, ty, result } => EffectfulOp::Load {
                addr: self.remap_class_id(*addr),
                ty: ty.clone(),
                result: self.remap_class_id(*result),
            },
            EffectfulOp::Store { addr, val, ty } => EffectfulOp::Store {
                addr: self.remap_class_id(*addr),
                val: self.remap_class_id(*val),
                ty: ty.clone(),
            },
            EffectfulOp::Call {
                func,
                args,
                arg_tys,
                ret_tys,
                results,
            } => EffectfulOp::Call {
                func: func.clone(),
                args: args.iter().map(|&a| self.remap_class_id(a)).collect(),
                arg_tys: arg_tys.clone(),
                ret_tys: ret_tys.clone(),
                results: results.iter().map(|&r| self.remap_class_id(r)).collect(),
            },
            EffectfulOp::Branch {
                cond,
                cc,
                bb_true,
                bb_false,
                true_args,
                false_args,
            } => EffectfulOp::Branch {
                cond: self.remap_class_id(*cond),
                cc: *cc,
                bb_true: self.remap_block_id(*bb_true),
                bb_false: self.remap_block_id(*bb_false),
                true_args: true_args.iter().map(|&a| self.remap_class_id(a)).collect(),
                false_args: false_args.iter().map(|&a| self.remap_class_id(a)).collect(),
            },
            EffectfulOp::Jump { target, args } => EffectfulOp::Jump {
                target: self.remap_block_id(*target),
                args: args.iter().map(|&a| self.remap_class_id(a)).collect(),
            },
            EffectfulOp::Ret { val } => EffectfulOp::Ret {
                val: val.map(|v| self.remap_class_id(v)),
            },
        }
    }

    /// Remap callee blocks, producing new BasicBlocks with remapped IDs and ops.
    pub fn remap_blocks(&self, callee: &Function) -> Vec<BasicBlock> {
        callee
            .blocks
            .iter()
            .map(|block| {
                let new_id = self.remap_block_id(block.id);
                let new_ops: Vec<EffectfulOp> = block
                    .ops
                    .iter()
                    .map(|op| self.remap_effectful_op(op))
                    .collect();
                let mut new_block = BasicBlock::new(new_id, block.param_types.clone());
                new_block.ops = new_ops;
                new_block
            })
            .collect()
    }
}

/// Find the maximum UID used in LoadResult/CallResult ops across a function's e-graph
/// and effectful ops (for merged-away classes whose UIDs may not appear in live e-nodes).
fn max_uid_in_function(func: &Function) -> u32 {
    let mut max_uid = 0u32;

    // Scan e-graph nodes.
    if let Some(egraph) = &func.egraph {
        for eclass in &egraph.classes {
            for enode in &eclass.nodes {
                match &enode.op {
                    Op::LoadResult(uid, _) | Op::CallResult(uid, _) => {
                        max_uid = max_uid.max(*uid);
                    }
                    _ => {}
                }
            }
        }

        // Scan effectful ops for result ClassIds, then look up their UIDs in the egraph.
        // This catches UIDs from merged-away classes that no longer have live e-nodes.
        for block in &func.blocks {
            for op in &block.ops {
                let result_ids: Vec<ClassId> = match op {
                    EffectfulOp::Load { result, .. } => vec![*result],
                    EffectfulOp::Call { results, .. } => results.clone(),
                    _ => continue,
                };
                for cid in result_ids {
                    let canon = egraph.unionfind.find_immutable(cid);
                    if (canon.0 as usize) < egraph.classes.len() {
                        for enode in &egraph.classes[canon.0 as usize].nodes {
                            match &enode.op {
                                Op::LoadResult(uid, _) | Op::CallResult(uid, _) => {
                                    max_uid = max_uid.max(*uid);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    max_uid
}
