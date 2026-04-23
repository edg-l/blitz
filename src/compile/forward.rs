//! Intra-block store-to-load and load-to-load forwarding.
//!
//! Walks each block's `EffectfulOp` list forward, maintaining a `PendingMem`
//! table of address→value mappings. Forwarded loads are removed and their
//! result classes are unioned with the stored value classes.

use crate::compile::alias::AliasInfo;
use crate::egraph::egraph::EGraph;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

// ── PendingMem ────────────────────────────────────────────────────────────────

/// Per-block store/load tracking table.
///
/// Entries are `(canonical_addr, value_class, ty)`. Linear scan is acceptable
/// for per-block scale (typically <20 memory ops).
struct PendingMem {
    entries: Vec<(ClassId, ClassId, Type)>,
}

impl PendingMem {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Remove entries that may be clobbered by a store at `addr` of type `ty`.
    fn invalidate_may_alias(&mut self, addr: ClassId, ty: &Type, alias: &AliasInfo, eg: &EGraph) {
        self.entries.retain(|(entry_addr, _, entry_ty)| {
            !alias.store_clobbers_load(addr, ty.clone(), *entry_addr, entry_ty.clone(), eg)
        });
    }

    /// Record `(addr, val, ty)`, replacing an existing entry with the same
    /// canonical address and matching type.
    fn record(&mut self, addr: ClassId, val: ClassId, ty: Type, eg: &EGraph) {
        let canon = eg.unionfind.find_immutable(addr);
        // Replace existing entry with the same canonical addr + type.
        for entry in &mut self.entries {
            if entry.0 == canon && entry.2 == ty {
                entry.1 = val;
                return;
            }
        }
        self.entries.push((canon, val, ty));
    }

    /// Find a `must_alias` entry for `addr` with matching `ty`.
    fn lookup(&self, addr: ClassId, ty: &Type, alias: &AliasInfo, eg: &EGraph) -> Option<ClassId> {
        let canon = eg.unionfind.find_immutable(addr);
        for (entry_addr, val, entry_ty) in &self.entries {
            if entry_ty == ty && alias.must_alias(*entry_addr, canon, eg) {
                return Some(*val);
            }
        }
        None
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run intra-block store-to-load and load-to-load forwarding.
///
/// Returns the number of loads forwarded (and thus removed).
pub fn run_forwarding(func: &mut Function, egraph: &mut EGraph, alias: &AliasInfo) -> usize {
    // Collect (block_idx, op_idx_to_remove, load_result_class, forwarded_val_class)
    // from all blocks, then apply unions and removals after iteration.
    let mut pending_unions: Vec<(ClassId, ClassId)> = Vec::new();
    // Per-block lists of op indices to remove.
    let mut removals: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
    // `LoadResult(uid, ty)` placeholders whose backing `Load` op has been
    // removed. These must be stripped from the e-graph after unions apply so
    // extraction cannot pick a node with no producer.
    let mut dead_placeholders: Vec<Op> = Vec::new();

    for (b, block) in func.blocks.iter().enumerate() {
        let mut mem = PendingMem::new();

        for (i, op) in block.ops.iter().enumerate() {
            match op {
                EffectfulOp::Store { addr, val, ty } => {
                    let canon_addr = egraph.unionfind.find_immutable(*addr);
                    mem.invalidate_may_alias(canon_addr, ty, alias, egraph);
                    mem.record(canon_addr, *val, ty.clone(), egraph);
                }

                EffectfulOp::Load { addr, ty, result } => {
                    let canon_addr = egraph.unionfind.find_immutable(*addr);
                    if let Some(fwd_val) = mem.lookup(canon_addr, ty, alias, egraph) {
                        // Forward: union the load result with the stored value.
                        if crate::trace::is_enabled("alias") {
                            eprintln!(
                                "[alias] forward bb{b} load-at-idx={i} result={} -> val={}",
                                result.0, fwd_val.0
                            );
                        }
                        // Snapshot the LoadResult placeholder op for this load's
                        // result class before the merge makes it harder to find.
                        let canon_result = egraph.unionfind.find_immutable(*result);
                        if let Some(node) = egraph
                            .class(canon_result)
                            .nodes
                            .iter()
                            .find(|n| matches!(n.op, Op::LoadResult(_, _)))
                        {
                            dead_placeholders.push(node.op.clone());
                        }
                        pending_unions.push((*result, fwd_val));
                        removals[b].push(i);
                    } else {
                        // No forward: record load result so a subsequent same-addr
                        // load can be load-to-load forwarded.
                        mem.record(canon_addr, *result, ty.clone(), egraph);
                    }
                }

                // Call-barrier axiom: any call may read/write any memory.
                EffectfulOp::Call { .. } => {
                    if crate::trace::is_enabled("alias") {
                        eprintln!("[alias] forward bb{b} call-barrier clears PendingMem");
                    }
                    mem.clear();
                }

                // Terminators end the block; no action needed.
                EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => {
                }
            }
        }
    }

    // Apply all unions.
    for (a, b) in &pending_unions {
        egraph.merge(*a, *b);
    }

    // Rebuild the e-graph once after all merges.
    if !pending_unions.is_empty() {
        egraph.rebuild();
    }

    // Strip LoadResult placeholders of removed Load ops from their (now-merged)
    // classes. If this is skipped, extraction may pick the placeholder node for
    // the merged class — but the effectful Load op that would have written the
    // vreg was removed, so the vreg is garbage.
    for op in &dead_placeholders {
        egraph.remove_result_placeholder(op);
    }

    // Remove forwarded load ops in reverse index order per block.
    let total = pending_unions.len();
    for (b, indices) in removals.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let block_ops = &mut func.blocks[b].ops;
        // Reverse order to keep earlier indices valid.
        for &idx in indices.iter().rev() {
            block_ops.remove(idx);
        }
    }

    total
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::alias::AliasInfo;
    use crate::egraph::egraph::EGraph;
    use crate::egraph::enode::ENode;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::function::{BasicBlock, Function};
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn add_node(eg: &mut EGraph, op: Op, children: &[ClassId]) -> ClassId {
        eg.add(ENode {
            op,
            children: children.iter().copied().collect(),
        })
    }

    fn stack_addr(eg: &mut EGraph, n: u32) -> ClassId {
        add_node(eg, Op::StackAddr(n), &[])
    }

    fn iconst_i64(eg: &mut EGraph, v: i64) -> ClassId {
        add_node(eg, Op::Iconst(v, Type::I64), &[])
    }

    fn load_result_class(eg: &mut EGraph, uid: u32, ty: Type) -> ClassId {
        add_node(eg, Op::LoadResult(uid, ty), &[])
    }

    fn make_func_with_block(ops: Vec<EffectfulOp>) -> Function {
        let mut func = Function::new("test", vec![], vec![]);
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops = ops;
        func.blocks.push(bb);
        func.next_block_id = 1;
        func
    }

    /// Store(s0, 42); Load(s0) → load forwarded to 42.
    #[test]
    fn store_then_load_same_slot_forwarded() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let val42 = iconst_i64(&mut eg, 42);
        let load_res = load_result_class(&mut eg, 1, Type::I64);

        let store_op = EffectfulOp::Store {
            addr: s0,
            val: val42,
            ty: Type::I64,
        };
        let load_op = EffectfulOp::Load {
            addr: s0,
            ty: Type::I64,
            result: load_res,
        };
        let ret_op = EffectfulOp::Ret { val: None };

        let mut func = make_func_with_block(vec![store_op, load_op, ret_op]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        assert_eq!(count, 1, "one load should be forwarded");
        // The load op must be removed from the block.
        assert_eq!(func.blocks[0].ops.len(), 2, "only store and ret remain");
        // The load result should now be in the same e-class as val42.
        let canon_res = eg.unionfind.find_immutable(load_res);
        let canon_val = eg.unionfind.find_immutable(val42);
        assert_eq!(canon_res, canon_val, "load result merged with stored value");
    }

    /// Store(s0, 42); Load(s1) → no forward (different slots).
    #[test]
    fn store_then_load_different_slot_no_forward() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let s1 = stack_addr(&mut eg, 1);
        let val42 = iconst_i64(&mut eg, 42);
        let load_res = load_result_class(&mut eg, 2, Type::I64);

        let mut func = make_func_with_block(vec![
            EffectfulOp::Store {
                addr: s0,
                val: val42,
                ty: Type::I64,
            },
            EffectfulOp::Load {
                addr: s1,
                ty: Type::I64,
                result: load_res,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        assert_eq!(count, 0);
        assert_eq!(func.blocks[0].ops.len(), 3, "no ops removed");
    }

    /// Store(s0, 42); Call foo(); Load(s0) → no forward (call barrier).
    #[test]
    fn call_barrier_prevents_forward() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let val42 = iconst_i64(&mut eg, 42);
        let load_res = load_result_class(&mut eg, 3, Type::I64);

        let mut func = make_func_with_block(vec![
            EffectfulOp::Store {
                addr: s0,
                val: val42,
                ty: Type::I64,
            },
            EffectfulOp::Call {
                func: "foo".to_string(),
                args: vec![],
                arg_tys: vec![],
                ret_tys: vec![],
                results: vec![],
            },
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I64,
                result: load_res,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        assert_eq!(count, 0, "call barrier must block forwarding");
        assert_eq!(func.blocks[0].ops.len(), 4, "no ops removed");
    }

    /// Store(s0, i64_val); Load(s0) as I32 → no forward (type mismatch).
    #[test]
    fn type_mismatch_prevents_forward() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let val = iconst_i64(&mut eg, 99);
        let load_res = load_result_class(&mut eg, 4, Type::I32);

        let mut func = make_func_with_block(vec![
            EffectfulOp::Store {
                addr: s0,
                val,
                ty: Type::I64,
            },
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I32,
                result: load_res,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        // I64 store vs I32 load: store_clobbers_load returns true (mismatched
        // widths are conservative), so the store IS invalidated. But lookup
        // requires exact type match, so the load won't find a match.
        // The store was invalidated, entry removed — lookup returns None.
        assert_eq!(count, 0, "type mismatch: no forwarding");
        assert_eq!(func.blocks[0].ops.len(), 3, "no ops removed");
    }

    /// Two consecutive Load(s0) → second eliminated, unioned with first.
    #[test]
    fn load_to_load_forwarding() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let load1 = load_result_class(&mut eg, 5, Type::I64);
        let load2 = load_result_class(&mut eg, 6, Type::I64);

        let mut func = make_func_with_block(vec![
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I64,
                result: load1,
            },
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I64,
                result: load2,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        assert_eq!(count, 1, "second load forwarded to first");
        assert_eq!(func.blocks[0].ops.len(), 2, "second load op removed");
        let c1 = eg.unionfind.find_immutable(load1);
        let c2 = eg.unionfind.find_immutable(load2);
        assert_eq!(c1, c2, "load results merged");
    }

    /// Store(s0, a); Store(s0, b); Load(s0) → load forwards to b.
    #[test]
    fn second_store_overwrites_first() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let val_a = iconst_i64(&mut eg, 10);
        let val_b = iconst_i64(&mut eg, 20);
        let load_res = load_result_class(&mut eg, 7, Type::I64);

        let mut func = make_func_with_block(vec![
            EffectfulOp::Store {
                addr: s0,
                val: val_a,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: s0,
                val: val_b,
                ty: Type::I64,
            },
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I64,
                result: load_res,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let count = run_forwarding(&mut func, &mut eg, &ai);

        assert_eq!(count, 1, "load forwarded to second store");
        // load_res should be merged with val_b
        let c_res = eg.unionfind.find_immutable(load_res);
        let c_b = eg.unionfind.find_immutable(val_b);
        assert_eq!(c_res, c_b, "load result merged with second stored value");
    }
}
