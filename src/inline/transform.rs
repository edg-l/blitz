use std::collections::BTreeSet;

use smallvec::smallvec;

use crate::egraph::enode::ENode;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::{BasicBlock, Function};
use crate::ir::op::{ClassId, Op};

use super::remap::RemapContext;

/// Collect all ClassIds referenced as operands by effectful ops.
fn collect_referenced_ids(ops: &[EffectfulOp]) -> BTreeSet<ClassId> {
    let mut ids = BTreeSet::new();
    for op in ops {
        match op {
            EffectfulOp::Load { addr, .. } => {
                ids.insert(*addr);
            }
            EffectfulOp::Store { addr, val, .. } => {
                ids.insert(*addr);
                ids.insert(*val);
            }
            EffectfulOp::Call { args, .. } => {
                ids.extend(args);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                ids.insert(*cond);
                ids.extend(true_args);
                ids.extend(false_args);
            }
            EffectfulOp::Jump { args, .. } => {
                ids.extend(args);
            }
            EffectfulOp::Ret { val } => {
                if let Some(v) = val {
                    ids.insert(*v);
                }
            }
        }
    }
    ids.remove(&ClassId::NONE);
    ids
}

/// Collect ClassIds produced (defined) by effectful ops (Load results, Call results).
fn collect_produced_ids(ops: &[EffectfulOp]) -> BTreeSet<ClassId> {
    let mut ids = BTreeSet::new();
    for op in ops {
        match op {
            EffectfulOp::Load { result, .. } => {
                ids.insert(*result);
            }
            EffectfulOp::Call { results, .. } => {
                ids.extend(results);
            }
            _ => {}
        }
    }
    ids
}

/// Inline a single call site in the caller function.
///
/// `block_idx` and `op_idx` identify the Call op within the caller's blocks.
/// `callee` is a cloned copy of the function being inlined.
pub fn inline_call_site(caller: &mut Function, block_idx: usize, op_idx: usize, callee: &Function) {
    // Extract the Call op details.
    let call_op = &caller.blocks[block_idx].ops[op_idx];
    let (call_args, ret_tys, call_results) = match call_op {
        EffectfulOp::Call {
            args,
            ret_tys,
            results,
            ..
        } => (args.clone(), ret_tys.clone(), results.clone()),
        _ => panic!("inline_call_site: op at [{block_idx}][{op_idx}] is not a Call"),
    };

    // Build remap context BEFORE appending callee stack slots, so that
    // slot_offset = caller's original slot count (where callee slots will start).
    let mut remap = RemapContext::new(caller, callee, &call_args);

    // Append callee stack slots to caller.
    caller.stack_slots.extend_from_slice(&callee.stack_slots);

    let caller_egraph = caller.egraph.as_mut().expect("caller must have an egraph");
    let callee_egraph = callee.egraph.as_ref().expect("callee must have an egraph");
    remap.import_egraph(caller_egraph, callee_egraph);

    // Remap callee blocks.
    let mut remapped_blocks = remap.remap_blocks(callee);

    // Determine the callee entry block (first block after remapping).
    let callee_entry_id = remapped_blocks[0].id;

    // Create continuation block ID.
    let max_existing = caller.blocks.iter().map(|b| b.id).max().unwrap_or(0);
    let max_remapped = remapped_blocks.iter().map(|b| b.id).max().unwrap_or(0);
    let cont_id = max_existing.max(max_remapped) + 1;

    // Split caller block at call site.
    let original_ops = &caller.blocks[block_idx].ops;
    let ops_before: Vec<_> = original_ops[..op_idx].to_vec();
    let ops_after: Vec<_> = original_ops[op_idx + 1..].to_vec();

    // Find ClassIds referenced in ops_after that need to survive across inlined blocks.
    // These are values available before the call but used after it, excluding call results
    // (which get their own block params via the return value mechanism).
    let referenced_in_after = collect_referenced_ids(&ops_after);
    let produced_in_after = collect_produced_ids(&ops_after);
    let available_before: BTreeSet<ClassId> = {
        let mut avail = BTreeSet::new();
        avail.extend(collect_referenced_ids(&ops_before));
        avail.extend(collect_produced_ids(&ops_before));
        avail.extend(call_args.iter().copied());
        avail.extend(call_results.iter().copied());
        avail
    };

    let call_result_set: BTreeSet<ClassId> = call_results.iter().copied().collect();
    let external_ids: Vec<ClassId> = referenced_in_after
        .difference(&produced_in_after)
        .filter(|id| available_before.contains(id) && !call_result_set.contains(id))
        .copied()
        .collect();

    // Build continuation block param types: return value (if any) + external live-through values.
    let mut cont_param_types = ret_tys.clone();
    let caller_egraph = caller.egraph.as_ref().expect("caller must have egraph");
    for &id in &external_ids {
        let canonical = caller_egraph.unionfind.find_immutable(id);
        let ty = caller_egraph.classes[canonical.0 as usize].ty.clone();
        cont_param_types.push(ty);
    }

    let mut cont_block = BasicBlock::new(cont_id, cont_param_types);

    // Create BlockParam e-nodes for external params and build substitution map.
    let caller_egraph = caller.egraph.as_mut().expect("caller must have egraph");
    let ret_param_count = ret_tys.len();
    let mut subst_map: Vec<(ClassId, ClassId)> = Vec::new();

    for (i, &ext_id) in external_ids.iter().enumerate() {
        let param_idx = (ret_param_count + i) as u32;
        let canonical = caller_egraph.find(ext_id);
        let ty = caller_egraph.classes[canonical.0 as usize].ty.clone();
        let bp_enode = ENode {
            op: Op::BlockParam(cont_id, param_idx, ty),
            children: smallvec![],
        };
        let bp_class = caller_egraph.add(bp_enode);
        subst_map.push((ext_id, bp_class));
    }

    // Substitute external ClassIds in ops_after with their BlockParam equivalents.
    let substituted_ops: Vec<EffectfulOp> = ops_after
        .iter()
        .map(|op| substitute_class_ids(op, &subst_map))
        .collect();
    cont_block.ops = substituted_ops;

    // Original block: ops before + jump to callee entry.
    caller.blocks[block_idx].ops = ops_before;
    caller.blocks[block_idx].ops.push(EffectfulOp::Jump {
        target: callee_entry_id,
        args: vec![],
    });

    // Rewrite callee Ret ops to Jump to continuation block,
    // passing return value (if any) + external live-through values.
    for block in &mut remapped_blocks {
        let last_idx = block.ops.len().saturating_sub(1);
        if let Some(op) = block.ops.get_mut(last_idx) {
            match op {
                EffectfulOp::Ret { val: Some(v) } => {
                    let val = *v;
                    let mut args = vec![val];
                    args.extend(external_ids.iter().copied());
                    *op = EffectfulOp::Jump {
                        target: cont_id,
                        args,
                    };
                }
                EffectfulOp::Ret { val: None } => {
                    let args: Vec<ClassId> = external_ids.to_vec();
                    *op = EffectfulOp::Jump {
                        target: cont_id,
                        args,
                    };
                }
                _ => {}
            }
        }
    }

    // Merge CallResult with continuation block param 0.
    if ret_tys.len() == 1 {
        let caller_egraph = caller.egraph.as_mut().expect("caller must have egraph");
        let block_param_enode = ENode {
            op: Op::BlockParam(cont_id, 0, ret_tys[0].clone()),
            children: smallvec![],
        };
        let block_param_class = caller_egraph.add(block_param_enode);
        let call_result_class = call_results[0];
        caller_egraph.merge(call_result_class, block_param_class);
        caller_egraph.rebuild();
    }

    // Append remapped blocks and continuation block to caller.
    caller.blocks.extend(remapped_blocks);
    caller.blocks.push(cont_block);
}

/// Replace ClassIds in an effectful op according to a substitution map.
fn substitute_class_ids(op: &EffectfulOp, subst: &[(ClassId, ClassId)]) -> EffectfulOp {
    let sub = |id: ClassId| -> ClassId {
        for &(from, to) in subst {
            if id == from {
                return to;
            }
        }
        id
    };

    match op {
        EffectfulOp::Load { addr, ty, result } => EffectfulOp::Load {
            addr: sub(*addr),
            ty: ty.clone(),
            result: *result, // Don't substitute result ClassIds
        },
        EffectfulOp::Store { addr, val, ty } => EffectfulOp::Store {
            addr: sub(*addr),
            val: sub(*val),
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
            args: args.iter().map(|&a| sub(a)).collect(),
            arg_tys: arg_tys.clone(),
            ret_tys: ret_tys.clone(),
            results: results.clone(), // Don't substitute result ClassIds
        },
        EffectfulOp::Branch {
            cond,
            cc,
            bb_true,
            bb_false,
            true_args,
            false_args,
        } => EffectfulOp::Branch {
            cond: sub(*cond),
            cc: *cc,
            bb_true: *bb_true,
            bb_false: *bb_false,
            true_args: true_args.iter().map(|&a| sub(a)).collect(),
            false_args: false_args.iter().map(|&a| sub(a)).collect(),
        },
        EffectfulOp::Jump { target, args } => EffectfulOp::Jump {
            target: *target,
            args: args.iter().map(|&a| sub(a)).collect(),
        },
        EffectfulOp::Ret { val } => EffectfulOp::Ret { val: val.map(sub) },
    }
}
