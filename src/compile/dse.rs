//! Intra-block dead store elimination.
//!
//! Walks each block's `EffectfulOp` list forward, tracking stores that
//! haven't yet been observed (by a may-aliasing load or call) or killed
//! (by a later must-aliasing store that fully covers them).
//!
//! A "later covers earlier" decision requires two things:
//! 1. The two addresses must-alias (canonical e-class equality).
//! 2. The later store's byte width is at least the earlier store's width.
//!
//! Rule 2 keeps us correct when, e.g., an `I64` store is followed by an
//! `I32` store at the same address: only the low 4 bytes are overwritten,
//! so the upper 4 bytes of the earlier write are still observable.

use crate::compile::alias::AliasInfo;
use crate::egraph::egraph::EGraph;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::ClassId;
use crate::ir::types::Type;

/// Run intra-block dead store elimination. Returns the number of stores removed.
pub fn run_dse(func: &mut Function, egraph: &EGraph, alias: &AliasInfo) -> usize {
    let mut total_eliminated = 0;
    for block in func.blocks.iter_mut() {
        // Stores we've seen whose value hasn't yet been observed or killed.
        // (op_index, canonical_address, type).
        let mut pending: Vec<(usize, ClassId, Type)> = Vec::new();
        // Indices of ops to remove after the scan.
        let mut to_remove: Vec<usize> = Vec::new();

        for (i, op) in block.ops.iter().enumerate() {
            match op {
                EffectfulOp::Store { addr, ty, .. } => {
                    let canon = egraph.unionfind.find_immutable(*addr);
                    // A later store at a must-aliasing address kills earlier
                    // pending stores whose width it fully covers.
                    pending.retain(|(idx, paddr, pty)| {
                        if alias.must_alias(*paddr, canon, egraph) && later_covers_earlier(pty, ty)
                        {
                            if crate::trace::is_enabled("alias") {
                                eprintln!(
                                    "[alias] dse bb-op{idx} store killed by later store at op{i}"
                                );
                            }
                            to_remove.push(*idx);
                            false
                        } else {
                            true
                        }
                    });
                    pending.push((i, canon, ty.clone()));
                }

                EffectfulOp::Load { addr, ty, .. } => {
                    let canon = egraph.unionfind.find_immutable(*addr);
                    // Any pending store whose value this load may read is now
                    // observed — keep it.
                    pending.retain(|(_, paddr, pty)| {
                        !alias.store_clobbers_load(*paddr, pty.clone(), canon, ty.clone(), egraph)
                    });
                }

                // Calls may read any memory: all pending stores are live.
                EffectfulOp::Call { .. } => pending.clear(),

                // Terminators end the block; successors (or the caller) may
                // observe the pending stores.
                EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => {
                    pending.clear()
                }
            }
        }

        to_remove.sort_unstable();
        to_remove.dedup();
        for &idx in to_remove.iter().rev() {
            block.ops.remove(idx);
        }
        total_eliminated += to_remove.len();
    }
    total_eliminated
}

/// True if a later store of type `later` fully overwrites an earlier store of
/// type `earlier` at the same address. Requires the later width to be at least
/// the earlier width.
fn later_covers_earlier(earlier: &Type, later: &Type) -> bool {
    match (earlier.byte_size(), later.byte_size()) {
        (Some(e), Some(l)) => l >= e,
        _ => false,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

    fn iconst(eg: &mut EGraph, v: i64) -> ClassId {
        add_node(eg, Op::Iconst(v, Type::I64), &[])
    }

    fn load_result_class(eg: &mut EGraph, uid: u32, ty: Type) -> ClassId {
        add_node(eg, Op::LoadResult(uid, ty), &[])
    }

    fn make_func(ops: Vec<EffectfulOp>) -> Function {
        let mut func = Function::new("test", vec![], vec![]);
        let mut bb = BasicBlock::new(0, vec![]);
        bb.ops = ops;
        func.blocks.push(bb);
        func.next_block_id = 1;
        func
    }

    /// store(s0, a); store(s0, b); ret -> first store eliminated.
    #[test]
    fn same_width_consecutive_store_killed() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);

        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: s0,
                val: b,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 1);
        assert_eq!(func.blocks[0].ops.len(), 2);
    }

    /// store(s0, I32); store(s0, I64); ret -> I32 eliminated (covered).
    #[test]
    fn wider_store_kills_narrower() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I32,
            },
            EffectfulOp::Store {
                addr: s0,
                val: b,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 1);
    }

    /// store(s0, I64); store(s0, I32); ret -> I64 NOT killed (upper 4 bytes
    /// of the I64 are still observable).
    #[test]
    fn narrower_store_does_not_kill_wider() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: s0,
                val: b,
                ty: Type::I32,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 0);
    }

    /// store(s0, a); load(s0); store(s0, b); ret -> first store observed by
    /// the load, NOT killed.
    #[test]
    fn load_between_stores_saves_first() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);
        let r = load_result_class(&mut eg, 1, Type::I64);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I64,
            },
            EffectfulOp::Load {
                addr: s0,
                ty: Type::I64,
                result: r,
            },
            EffectfulOp::Store {
                addr: s0,
                val: b,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 0);
    }

    /// store(s0, a); call; store(s0, b); ret -> call may read memory, first
    /// store NOT killed.
    #[test]
    fn call_between_stores_saves_first() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I64,
            },
            EffectfulOp::Call {
                func: "foo".to_string(),
                args: vec![],
                arg_tys: vec![],
                ret_tys: vec![],
                results: vec![],
            },
            EffectfulOp::Store {
                addr: s0,
                val: b,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 0);
    }

    /// store(s0, a); store(s1, b); store(s0, c); ret -> first store to s0 is
    /// killed by the later s0 store; s1 store unaffected.
    #[test]
    fn non_aliasing_store_ignored() {
        let mut eg = EGraph::new();
        let s0 = stack_addr(&mut eg, 0);
        let s1 = stack_addr(&mut eg, 1);
        let a = iconst(&mut eg, 10);
        let b = iconst(&mut eg, 20);
        let c = iconst(&mut eg, 30);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: s0,
                val: a,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: s1,
                val: b,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: s0,
                val: c,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 1);
        // Order must be preserved for the surviving ops: s1 store + s0 store + ret.
        assert_eq!(func.blocks[0].ops.len(), 3);
        match &func.blocks[0].ops[0] {
            EffectfulOp::Store { addr, .. } => assert_eq!(*addr, s1),
            _ => panic!("expected s1 store"),
        }
    }

    /// store(a+0, ..); store(a+0, ..): different offsets would still must-alias
    /// only via canonical equality. Same-class case already covered; this test
    /// asserts that distinct canonical classes (different offsets) do NOT kill.
    #[test]
    fn same_base_different_offset_not_killed() {
        let mut eg = EGraph::new();
        let base = stack_addr(&mut eg, 0);
        let c0 = iconst(&mut eg, 0);
        let c8 = iconst(&mut eg, 8);
        let a0 = add_node(&mut eg, Op::Add, &[base, c0]);
        let a8 = add_node(&mut eg, Op::Add, &[base, c8]);
        let val = iconst(&mut eg, 42);
        let mut func = make_func(vec![
            EffectfulOp::Store {
                addr: a0,
                val,
                ty: Type::I64,
            },
            EffectfulOp::Store {
                addr: a8,
                val,
                ty: Type::I64,
            },
            EffectfulOp::Ret { val: None },
        ]);
        let ai = AliasInfo::new();
        let n = run_dse(&mut func, &eg, &ai);
        assert_eq!(n, 0);
    }
}
