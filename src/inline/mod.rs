pub mod callgraph;
pub mod remap;
pub mod transform;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::compile::CompileOptions;
use crate::ir::function::Function;

use callgraph::{build_call_graph, count_callers, is_recursive, should_inline, topological_order};
use transform::inline_call_site;

/// Inline function calls across a module, then eliminate dead functions.
///
/// When `is_executable` is true, only functions reachable from `main` are kept.
/// When false (library/object-only compilation), all functions are kept.
pub fn inline_module(functions: &mut Vec<Function>, opts: &CompileOptions, is_executable: bool) {
    if !opts.enable_inlining {
        return;
    }

    let call_graph = build_call_graph(functions);
    let all_names: Vec<String> = functions.iter().map(|f| f.name.clone()).collect();
    let order = topological_order(&call_graph, &all_names);

    // Build a name->index map for callee lookup.
    let func_names: BTreeMap<String, usize> = functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    // Process functions bottom-up: callees before callers.
    for func_name in &order {
        let Some(&caller_idx) = func_names.get(func_name) else {
            continue; // external, not defined
        };

        // Safety cap on inner rescan iterations.
        let max_rescans = opts.max_inline_nodes * 2;
        let mut rescan = 0;

        loop {
            if rescan >= max_rescans {
                break;
            }
            rescan += 1;

            // Scan blocks for a Call op to inline.
            let mut found = None;
            for (block_idx, block) in functions[caller_idx].blocks.iter().enumerate() {
                for (op_idx, op) in block.ops.iter().enumerate() {
                    if let crate::ir::effectful::EffectfulOp::Call { func, .. } = op {
                        if let Some(&callee_idx) = func_names.get(func) {
                            if callee_idx == caller_idx {
                                continue; // self-recursion
                            }
                            let callee = &functions[callee_idx];
                            if is_recursive(&callee.name, &call_graph) {
                                continue;
                            }
                            let caller_count = count_callers(&callee.name, &call_graph);
                            if should_inline(callee, caller_count, opts) {
                                found = Some((block_idx, op_idx, callee_idx));
                                break;
                            }
                        }
                    }
                }
                if found.is_some() {
                    break;
                }
            }

            let Some((block_idx, op_idx, callee_idx)) = found else {
                break; // no more inlining opportunities
            };

            let callee_clone = clone_function(&functions[callee_idx]);
            inline_call_site(&mut functions[caller_idx], block_idx, op_idx, &callee_clone);
        }
    }

    // Dead function elimination: only for executable mode (has main).
    if is_executable {
        eliminate_dead_functions(functions);
    }
}

/// Clone a Function (egraph and all).
fn clone_function(f: &Function) -> Function {
    Function {
        name: f.name.clone(),
        param_types: f.param_types.clone(),
        return_types: f.return_types.clone(),
        blocks: f.blocks.clone(),
        param_class_ids: f.param_class_ids.clone(),
        egraph: f.egraph.clone(),
        stack_slots: f.stack_slots.clone(),
        noinline: f.noinline,
        next_block_id: f.next_block_id,
    }
}

/// Remove functions that are not reachable from "main" via the call graph.
fn eliminate_dead_functions(functions: &mut Vec<Function>) {
    use std::collections::BTreeSet;

    let call_graph = build_call_graph(functions);
    let func_names: BTreeSet<String> = functions.iter().map(|f| f.name.clone()).collect();

    // BFS from "main".
    let mut reachable = BTreeSet::new();
    let mut worklist = vec!["main".to_string()];

    while let Some(name) = worklist.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(callees) = call_graph.get(&name) {
            for callee in callees {
                if !reachable.contains(callee) {
                    worklist.push(callee.clone());
                }
            }
        }
    }

    // Keep functions that are reachable or not defined in this module (externs).
    functions.retain(|f| reachable.contains(&f.name) || !func_names.contains(&f.name));
}
