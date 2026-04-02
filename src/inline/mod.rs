pub mod callgraph;
pub mod remap;
pub mod transform;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::compile::CompileOptions;
use crate::ir::function::Function;

use callgraph::{build_call_graph, is_recursive, should_inline};
use transform::inline_call_site;

/// Inline function calls across a module, then eliminate dead functions.
pub fn inline_module(functions: &mut Vec<Function>, opts: &CompileOptions) {
    if !opts.enable_inlining {
        return;
    }

    let max_iterations = opts.max_inline_depth as usize * 10;

    // Build a name->index map for callee lookup.
    let func_names: BTreeMap<String, usize> = functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    let call_graph = build_call_graph(functions);

    // Process each function for inlining opportunities.
    // We iterate by index because we need to clone callees while mutating callers.
    for caller_idx in 0..functions.len() {
        let caller_name = functions[caller_idx].name.clone();
        let mut iteration = 0;

        'rescan: loop {
            if iteration >= max_iterations {
                break;
            }
            iteration += 1;

            // Scan blocks for Call ops to inline.
            let mut found = None;
            'search: for (block_idx, block) in functions[caller_idx].blocks.iter().enumerate() {
                for (op_idx, op) in block.ops.iter().enumerate() {
                    if let crate::ir::effectful::EffectfulOp::Call { func, .. } = op
                        && let Some(&callee_idx) = func_names.get(func)
                    {
                        if callee_idx == caller_idx {
                            continue;
                        }
                        let callee = &functions[callee_idx];
                        if !is_recursive(&callee.name, &call_graph)
                            && should_inline(callee, iteration as u32, opts)
                        {
                            found = Some((block_idx, op_idx, callee_idx));
                            break 'search;
                        }
                    }
                }
            }

            let Some((block_idx, op_idx, callee_idx)) = found else {
                break 'rescan;
            };

            // Clone the callee so we can mutate the caller.
            let callee_clone = clone_function(&functions[callee_idx]);
            inline_call_site(&mut functions[caller_idx], block_idx, op_idx, &callee_clone);

            // After inlining, rescan from the beginning for more opportunities.
        }

        let _ = &caller_name; // suppress unused warning
    }

    // Dead function elimination.
    eliminate_dead_functions(functions);
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
    }
}

/// Remove functions that are not reachable from "main" via the call graph.
fn eliminate_dead_functions(functions: &mut Vec<Function>) {
    use std::collections::BTreeSet;

    let call_graph = build_call_graph(functions);
    let func_names: BTreeSet<String> = functions.iter().map(|f| f.name.clone()).collect();

    // BFS from "main" (and any extern-visible entry points).
    let mut reachable = BTreeSet::new();
    let mut worklist = vec!["main".to_string()];

    // Also keep any function whose name suggests it's an entry point
    // (for now, just "main").
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
