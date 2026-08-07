//! A readable dump of the e-graph, under `BLITZ_DEBUG=egraph`.
//!
//! What this is for: the e-graph is the instruction selector, so the question
//! worth asking of it is *what alternatives does a class hold and which did the
//! cost model pick*. A class with one node offered no choice; a class with
//! several is where isel and extraction are doing the work.

use std::collections::BTreeMap;

use crate::egraph::cost::CostModel;
use crate::egraph::egraph::EGraph;
use crate::egraph::extract::ExtractedNode;
use crate::ir::op::ClassId;
use crate::ir::print::fmt_op;

/// Print every live e-class with its nodes, their own costs, and the extracted
/// winner where there is one.
///
/// `choices` is extraction's answer, and is `None` before instruction selection
/// has run: every node is generic IR priced at infinity there, so extraction
/// fails rather than choosing. That is not a defect of the dump -- it is the
/// reason `saturate_isel` is a correctness step.
///
/// Costs shown are each node's *own* cost, not the subtree cost extraction
/// compares. A node reading a cheap child can lose to a dearer one reading a
/// dearer child, so a winner that is not the cheapest line is expected.
pub fn dump(
    egraph: &EGraph,
    func_name: &str,
    label: &str,
    cost_model: &CostModel,
    choices: Option<&BTreeMap<ClassId, ExtractedNode>>,
) {
    if !crate::trace::is_enabled("egraph") || !crate::trace::fn_matches(func_name) {
        return;
    }

    let live: Vec<ClassId> = (0..egraph.arena_len() as u32)
        .map(ClassId)
        .filter(|&c| egraph.find_immutable(c) == c)
        .collect();
    let nodes: usize = live.iter().map(|&c| egraph.class(c).nodes.len()).sum();
    let avg = if live.is_empty() {
        0.0
    } else {
        nodes as f64 / live.len() as f64
    };
    eprintln!(
        "[egraph] {func_name} {label}: {} classes, {nodes} nodes, avg {avg:.3} nodes/class",
        live.len()
    );

    for id in live {
        let class = egraph.class(id);
        let chosen = choices.and_then(|c| c.get(&id));
        let winner = match chosen {
            Some(node) => fmt_op(&node.op),
            None => "-".into(),
        };
        eprintln!(
            "[egraph] c{} ({:?}) size={} winner={winner}",
            id.0,
            class.ty,
            class.nodes.len()
        );

        for node in &class.nodes {
            let children: Vec<String> = node
                .children
                .iter()
                .filter(|c| **c != ClassId::NONE)
                .map(|c| format!("c{}", egraph.find_immutable(*c).0))
                .collect();
            let cost = cost_model.cost(&node.op);
            let cost = if cost.is_infinite() {
                // Generic IR with no x86-64 encoding. Extraction must never
                // pick one, which is what the infinity is for.
                "inf".to_string()
            } else {
                format!("{cost}")
            };
            // The winner is matched on the node, not the op: one class can hold
            // two nodes of the same op over different children.
            let mark = match chosen {
                Some(c) if c.op == node.op && c.children == node.children.as_slice() => {
                    "   <- extracted"
                }
                _ => "",
            };
            // A leaf carries its operands in the op itself, so the empty
            // parentheses of `iconst(7, I32)()` would say nothing.
            let text = if children.is_empty() {
                fmt_op(&node.op)
            } else {
                format!("{}({})", fmt_op(&node.op), children.join(", "))
            };
            eprintln!("[egraph]   {text:<38} cost {cost}{mark}");
        }
    }
}
