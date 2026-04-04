use std::collections::BTreeMap;

use crate::egraph::extract::{VReg, vreg_insts_for_block};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::print::{PrintableBlock, PrintableGroup, print_function_ir};
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};

use super::barrier::{assign_barrier_groups, build_barrier_context};
use super::cfg::{collect_block_roots, compute_rpo};
use super::{CompileError, CompileOptions, run_egraph_and_extract};

pub fn compile_to_ir_string(
    mut func: Function,
    opts: &CompileOptions,
) -> Result<String, CompileError> {
    crate::trace::init_tracing();

    let mut egraph = func
        .egraph
        .take()
        .expect("Function must contain an EGraph; use FunctionBuilder::finalize()");
    let func = &func;

    // Phases 1-2: E-graph rewrites and cost-based extraction.
    let (block_param_map, extraction) = run_egraph_and_extract(func, &mut egraph, opts)?;

    // Phase 3: Build per-block VRegInst lists.
    let mut class_to_vreg: BTreeMap<ClassId, VReg> = BTreeMap::new();
    let mut next_vreg: u32 = 0;
    let rpo_order = compute_rpo(func);

    let mut block_vreg_insts: Vec<Vec<crate::egraph::extract::VRegInst>> =
        vec![Vec::new(); func.blocks.len()];
    for &block_idx in &rpo_order {
        let block = &func.blocks[block_idx];
        let roots = collect_block_roots(block, &egraph);
        let block_id = block.id;
        let mut all_roots = roots;
        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                all_roots.push(cid);
            }
        }
        all_roots.sort_by_key(|c| c.0);
        all_roots.dedup();
        let mut insts =
            vreg_insts_for_block(&extraction, &all_roots, &mut class_to_vreg, &mut next_vreg);

        for pidx in 0..block.param_types.len() as u32 {
            if let Some(&cid) = block_param_map.get(&(block_id, pidx)) {
                let canon = egraph.unionfind.find_immutable(cid);
                if let Some(&vreg) = class_to_vreg.get(&canon) {
                    if let Some(inst) = insts.iter_mut().find(|i| i.dst == vreg) {
                        inst.op = Op::BlockParam(
                            block_id,
                            pidx,
                            block.param_types[pidx as usize].clone(),
                        );
                        inst.operands.clear();
                    } else {
                        // VReg emitted by a prior block -- allocate a fresh one.
                        let fresh_vreg = VReg(next_vreg);
                        next_vreg += 1;
                        for inst in insts.iter_mut() {
                            for operand in inst.operands.iter_mut() {
                                if *operand == Some(vreg) {
                                    *operand = Some(fresh_vreg);
                                }
                            }
                        }
                        insts.push(crate::egraph::extract::VRegInst {
                            dst: fresh_vreg,
                            op: Op::BlockParam(
                                block_id,
                                pidx,
                                block.param_types[pidx as usize].clone(),
                            ),
                            operands: vec![],
                        });
                    }
                }
            }
        }

        block_vreg_insts[block_idx] = insts;
    }

    // Phase 4: Schedule per block.
    let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![Vec::new(); func.blocks.len()];
    for (block_idx, insts) in block_vreg_insts.iter().enumerate() {
        let dag = ScheduleDag::build(insts);
        let sched = schedule(&dag);
        block_schedules[block_idx] = sched;
    }

    // Phase 4b: Reorder each block's schedule to respect effectful op barriers.
    for (block_idx, block) in func.blocks.iter().enumerate() {
        if block.non_term_count() == 0 {
            continue;
        }

        let (vreg_to_result_of_barrier, vreg_to_arg_of_barrier) =
            build_barrier_context(block, &egraph, &class_to_vreg);

        let sched = &block_schedules[block_idx];
        let vreg_group =
            assign_barrier_groups(sched, &vreg_to_result_of_barrier, &vreg_to_arg_of_barrier);

        let mut indexed: Vec<(usize, &ScheduledInst)> = sched.iter().enumerate().collect();
        indexed.sort_by_key(|(orig_idx, inst)| {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            let param_order: u8 = match inst.op {
                Op::Param(_, _) => 0,
                Op::LoadResult(_, _) | Op::CallResult(_, _) => 1,
                _ => 2,
            };
            (g, param_order, *orig_idx)
        });
        let reordered: Vec<ScheduledInst> =
            indexed.into_iter().map(|(_, inst)| inst.clone()).collect();
        block_schedules[block_idx] = reordered;
    }

    // Build PrintableBlocks for the printer.
    let mut printable_blocks: Vec<PrintableBlock> = Vec::new();
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let non_term_count = block.non_term_count();
        let non_term_ops = &block.ops[..non_term_count];
        let num_barriers = non_term_ops.len();

        let (vreg_to_result_of_barrier, vreg_to_arg_of_barrier) =
            build_barrier_context(block, &egraph, &class_to_vreg);
        let vreg_group = assign_barrier_groups(
            &block_schedules[block_idx],
            &vreg_to_result_of_barrier,
            &vreg_to_arg_of_barrier,
        );

        // Partition scheduled insts into groups.
        let mut groups_insts: Vec<Vec<ScheduledInst>> = vec![Vec::new(); num_barriers + 1];
        for inst in &block_schedules[block_idx] {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            groups_insts[g].push(inst.clone());
        }

        // Build PrintableGroups: barrier k has pure ops from groups_insts[k] + barrier op k.
        let mut groups: Vec<PrintableGroup> = Vec::new();
        for k in 0..num_barriers {
            groups.push(PrintableGroup {
                pure_ops: groups_insts[k].clone(),
                barrier: Some(non_term_ops[k].clone()),
            });
        }
        // Trailing group (after last barrier).
        groups.push(PrintableGroup {
            pure_ops: groups_insts[num_barriers].clone(),
            barrier: None,
        });

        let terminator = block
            .ops
            .last()
            .expect("block must have terminator")
            .clone();

        printable_blocks.push(PrintableBlock {
            id: block.id,
            param_types: block.param_types.clone(),
            groups,
            terminator,
        });
    }

    Ok(print_function_ir(
        func,
        &printable_blocks,
        &class_to_vreg,
        &egraph.unionfind,
    ))
}

/// Compile multiple functions to IR strings.
pub fn compile_module_to_ir(
    mut functions: Vec<Function>,
    opts: &CompileOptions,
) -> Result<String, CompileError> {
    crate::inline::inline_module(&mut functions, opts);
    let mut results = Vec::new();
    for func in functions {
        results.push(compile_to_ir_string(func, opts)?);
    }
    Ok(results.join("\n"))
}
