use std::collections::{BTreeMap, BTreeSet};

use crate::compile::CompileOptions;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;

/// Build a call graph: caller name -> set of callee names.
pub fn build_call_graph(functions: &[Function]) -> BTreeMap<String, BTreeSet<String>> {
    let mut graph = BTreeMap::new();
    for func in functions {
        let callees = graph.entry(func.name.clone()).or_insert_with(BTreeSet::new);
        for block in &func.blocks {
            for op in &block.ops {
                if let EffectfulOp::Call { func: callee, .. } = op {
                    callees.insert(callee.clone());
                }
            }
        }
    }
    graph
}

/// Returns true if `name` is reachable from itself in the call graph (direct or mutual recursion).
pub fn is_recursive(name: &str, call_graph: &BTreeMap<String, BTreeSet<String>>) -> bool {
    let mut visited = BTreeSet::new();
    let mut stack = Vec::new();

    // Start from callees of `name`.
    if let Some(callees) = call_graph.get(name) {
        for c in callees {
            stack.push(c.as_str());
        }
    }

    while let Some(current) = stack.pop() {
        if current == name {
            return true;
        }
        if !visited.insert(current) {
            continue;
        }
        if let Some(callees) = call_graph.get(current) {
            for c in callees {
                stack.push(c.as_str());
            }
        }
    }

    false
}

/// Decide whether a callee should be inlined.
pub fn should_inline(callee: &Function, depth: u32, opts: &CompileOptions) -> bool {
    if depth >= opts.max_inline_depth {
        return false;
    }
    let egraph = match &callee.egraph {
        Some(eg) => eg,
        None => return false,
    };
    if callee.return_types.len() > 1 {
        return false;
    }
    if egraph.node_count() > opts.max_inline_nodes {
        return false;
    }
    if callee.param_class_ids.is_empty() && !callee.param_types.is_empty() {
        return false;
    }
    if callee.blocks.is_empty() {
        return false;
    }
    true
}
