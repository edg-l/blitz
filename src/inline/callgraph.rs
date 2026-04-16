use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compile::CompileOptions;
use crate::ir::effectful::EffectfulOp;
use crate::ir::function::Function;
use crate::ir::op::Op;

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

/// Compute a weighted cost for inlining a callee function.
/// Sums per-op weights across egraph nodes and effectful ops.
pub fn inline_cost(callee: &Function) -> u32 {
    let mut cost = 0u32;

    // E-graph node costs (pure ops)
    if let Some(egraph) = &callee.egraph {
        for eclass in &egraph.classes {
            if eclass.nodes.is_empty() {
                continue; // merged away
            }
            for enode in &eclass.nodes {
                cost += op_cost(&enode.op);
                break; // only count one representative node per class
            }
        }
    }

    // Effectful op costs
    for block in &callee.blocks {
        for op in &block.ops {
            cost += effectful_op_cost(op);
        }
    }

    cost
}

fn op_cost(op: &Op) -> u32 {
    match op {
        // Free: constants, params, projections, addresses
        Op::Iconst(..)
        | Op::Fconst(..)
        | Op::Param(..)
        | Op::BlockParam(..)
        | Op::LoadResult(..)
        | Op::CallResult(..)
        | Op::StoreBarrier
        | Op::VoidCallBarrier
        | Op::StackAddr(..)
        | Op::GlobalAddr(..)
        | Op::Proj0
        | Op::Proj1 => 0,

        // Cheap: simple ALU, comparisons, conversions
        Op::Add
        | Op::Sub
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Shl
        | Op::Shr
        | Op::Sar
        | Op::Sext(..)
        | Op::Zext(..)
        | Op::Trunc(..)
        | Op::Bitcast(..)
        | Op::Icmp(..)
        | Op::Fcmp(..)
        | Op::Select
        | Op::IntToFloat(..)
        | Op::FloatToInt(..)
        | Op::FloatExt
        | Op::FloatTrunc => 1,

        // Medium: multiplies, float ALU
        Op::Mul | Op::Fadd | Op::Fsub | Op::Fmul => 2,

        // Expensive: division, remainder, sqrt
        Op::UDiv | Op::SDiv | Op::URem | Op::SRem | Op::Fdiv | Op::Fsqrt => 10,

        // x86 machine ops: same weight as their generic counterpart
        Op::X86Add | Op::X86Sub | Op::X86And | Op::X86Or | Op::X86Xor => 1,
        Op::X86Shl | Op::X86Shr | Op::X86Sar => 1,
        Op::X86Imul3 => 2,

        // Anything else (address modes, lea, etc.): 1
        _ => 1,
    }
}

fn effectful_op_cost(op: &EffectfulOp) -> u32 {
    match op {
        EffectfulOp::Call { .. } => 20,
        EffectfulOp::Load { .. } => 3,
        EffectfulOp::Store { .. } => 3,
        EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. } => 0,
    }
}

/// Count how many distinct functions call `callee_name` (excluding self-calls).
pub fn count_callers(callee_name: &str, call_graph: &BTreeMap<String, BTreeSet<String>>) -> usize {
    call_graph
        .iter()
        .filter(|(caller, callees)| *caller != callee_name && callees.contains(callee_name))
        .count()
}

/// Compute a bottom-up order for inlining: callees before callers.
/// Functions in recursive SCCs are appended at the end.
pub fn topological_order(
    call_graph: &BTreeMap<String, BTreeSet<String>>,
    all_functions: &[String],
) -> Vec<String> {
    // In-degree = number of callees each function has (out-degree in call graph).
    // We process functions with no callees first (leaves).
    let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
    // Reverse graph: callee -> set of callers
    let mut reverse: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

    for name in all_functions {
        in_degree.entry(name.as_str()).or_insert(0);
    }

    for (caller, callees) in call_graph {
        // Only count callees that are in our function list (defined, not external)
        let defined_callees: usize = callees
            .iter()
            .filter(|c| in_degree.contains_key(c.as_str()))
            .count();
        *in_degree.entry(caller.as_str()).or_insert(0) = defined_callees;

        for callee in callees {
            if in_degree.contains_key(callee.as_str()) {
                reverse
                    .entry(callee.as_str())
                    .or_default()
                    .push(caller.as_str());
            }
        }
    }

    // Seed queue with zero in-degree functions (leaves that call nothing defined).
    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(&name, _)| name)
        .collect();

    let mut result: Vec<String> = Vec::new();

    while let Some(node) = queue.pop_front() {
        result.push(node.to_string());
        // For each caller of this node, decrement their in-degree
        if let Some(callers) = reverse.get(node) {
            for &caller in callers {
                if let Some(deg) = in_degree.get_mut(caller) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(caller);
                    }
                }
            }
        }
    }

    // Append any remaining functions (in SCCs / recursive)
    for name in all_functions {
        if !result.contains(name) {
            result.push(name.clone());
        }
    }

    result
}

/// Decide whether a callee should be inlined.
pub fn should_inline(callee: &Function, caller_count: usize, opts: &CompileOptions) -> bool {
    if callee.noinline {
        return false;
    }
    if callee.egraph.is_none() {
        return false;
    }
    if callee.return_types.len() > 1 {
        return false;
    }
    if callee.param_class_ids.is_empty() && !callee.param_types.is_empty() {
        return false;
    }
    if callee.blocks.is_empty() {
        return false;
    }

    let node_count = callee.egraph.as_ref().map_or(0, |eg| eg.node_count());

    // Single-caller functions are always inlined (unless extremely large).
    if caller_count == 1 {
        return node_count <= opts.max_inline_nodes * 4;
    }

    // Hard cap on node count (safety valve).
    if node_count > opts.max_inline_nodes {
        return false;
    }

    // Cost-based check.
    inline_cost(callee) <= opts.inline_cost_threshold
}
