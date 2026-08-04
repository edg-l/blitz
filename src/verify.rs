//! Structural IR verification, run at pipeline pass boundaries.
//!
//! Off by default, so a normal compile pays nothing beyond one `OnceLock` read
//! per boundary. When a check fails the compiler panics naming the stage that
//! produced the broken IR, which is what makes a fuzzed miscompile attributable
//! to a single pass instead of to the pipeline as a whole.
//!
//! # Levels
//!
//! - `BLITZ_VERIFY=1` (also `on`, `normal`): structural invariants.
//! - `BLITZ_VERIFY=strict`: additionally requires every `ClassId` the CFG holds
//!   to be canonical.
//!
//! Both must stay green; a failure at either level is a bug.
//!
//! # Scope
//!
//! CFG structure and the CFG/e-graph interface:
//!
//! - blocks are well-formed (exactly one terminator, in final position)
//! - block ids are unique and below `next_block_id`
//! - every branch/jump target exists
//! - every edge passes exactly as many arguments as the target declares, with
//!   matching types
//! - the entry block takes no parameters
//! - `Ret` arity agrees with the function signature
//! - `Call` operand/result vectors agree with their type vectors
//! - every `ClassId` an effectful op references resolves to a real class
//! - (strict) those `ClassId`s are canonical
//!
//! # Why canonicality is a separate level
//!
//! Passes that merge e-classes used to leave pre-merge `ClassId`s in the CFG,
//! with every consumer compensating via `find_immutable` on read. That is
//! soundness-neutral until one consumer forgets, which is what `ca2e400` (a
//! LICM miscompile) was. `compile::canon::canonicalize_class_refs` now runs
//! after each merging pass, so strict mode passes; keeping it a distinct level
//! means a pass that reintroduces stale ids is caught immediately rather than
//! waiting for a consumer to mishandle one.
//!
//! # CFG versus schedule
//!
//! [`verify_cfg_schedule_agreement`] compares the two answers the pipeline holds
//! for "which register carries this value here": the `ClassId` the CFG names,
//! resolved through the class map, and the VReg the schedule carries as an
//! operand. Two answers to one question is the defect behind most wrong-code bugs
//! here, and this is the invariant that says they agree.
//!
//! # Machine level
//!
//! [`verify_machinsts`] checks the emitted instruction stream: no virtual
//! register survives, and nothing reads a physical register that was never
//! written. Every wrong-code bug found by the random generator was an instance
//! of the second one -- a folded addressing mode using the array base as its
//! own index, a load reading a register whose value had been spilled away, a
//! store through a register the following `lea` had not written yet.
//!
//! [`verify_register_sharing`] adds the check that two VRegs live at once must
//! not hold the same register, which it does against liveness recomputed from
//! the schedules as emitted rather than against the allocator's own graph.
//!
//! Not covered yet: callee-saved preservation, and a value that is wrong rather
//! than absent -- the register is written on every path to the read, just with
//! somebody else's value. Emission-order divergences of that shape stay the
//! differential harness's job.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::egraph::EGraph;
use crate::ir::effectful::{EffectfulOp, TermArgs};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op};
use crate::ir::types::Type;

static LEVEL: OnceLock<VerifyLevel> = OnceLock::new();

/// How much the verifier checks. Parsed once from `BLITZ_VERIFY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyLevel {
    /// No checking (default).
    Off,
    /// Invariants the pipeline guarantees today.
    Normal,
    /// Also requires canonical `ClassId`s in the CFG.
    Strict,
}

/// The configured verification level.
pub fn level() -> VerifyLevel {
    *LEVEL.get_or_init(|| match std::env::var("BLITZ_VERIFY") {
        Ok(val) => match val.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" => VerifyLevel::Off,
            "1" | "on" | "normal" => VerifyLevel::Normal,
            "2" | "strict" => VerifyLevel::Strict,
            other => {
                eprintln!(
                    "warning: unknown BLITZ_VERIFY value '{other}', valid: 0, 1, strict; \
                     treating as 1"
                );
                VerifyLevel::Normal
            }
        },
        Err(_) => VerifyLevel::Off,
    })
}

/// Returns true if verification is on at any level.
pub fn is_enabled() -> bool {
    level() != VerifyLevel::Off
}

/// Verify `func` and panic if any invariant is broken. No-op unless
/// `BLITZ_VERIFY` is set.
///
/// `stage` names the pass that just ran, and is what the panic reports.
pub fn verify_stage(stage: &str, func: &Function, egraph: &EGraph) {
    let level = level();
    if level == VerifyLevel::Off {
        return;
    }
    let errors = verify_function_at(func, egraph, level);
    if !errors.is_empty() {
        panic!(
            "BLITZ_VERIFY: {} invariant violation(s) in function '{}' after stage '{}':\n  - {}",
            errors.len(),
            func.name,
            stage,
            errors.join("\n  - ")
        );
    }
}

/// Check every `Normal`-level invariant and return one message per violation.
///
/// Always runs regardless of `BLITZ_VERIFY`; the env var only gates the
/// automatic pass-boundary calls. Tests call this directly.
pub fn verify_function(func: &Function, egraph: &EGraph) -> Vec<String> {
    verify_function_at(func, egraph, VerifyLevel::Normal)
}

/// Check invariants at an explicit level and return one message per violation.
pub fn verify_function_at(func: &Function, egraph: &EGraph, level: VerifyLevel) -> Vec<String> {
    let mut v = Verifier {
        func,
        egraph,
        strict: level == VerifyLevel::Strict,
        errors: Vec::new(),
    };
    v.run();
    v.errors
}

struct Verifier<'a> {
    func: &'a Function,
    egraph: &'a EGraph,
    strict: bool,
    errors: Vec<String>,
}

impl Verifier<'_> {
    fn error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    fn run(&mut self) {
        if self.func.blocks.is_empty() {
            self.error("function has no blocks".to_string());
            return;
        }

        self.check_block_ids();
        self.check_entry_block();

        for (idx, block) in self.func.blocks.iter().enumerate() {
            self.check_block_shape(idx);
            for (op_idx, op) in block.ops.iter().enumerate() {
                self.check_op(idx, op_idx, op);
            }
        }
    }

    /// Block ids must be unique, and `next_block_id` must not hand out an id
    /// that is already taken (LICM and the inliner both allocate through it).
    fn check_block_ids(&mut self) {
        let mut seen = std::collections::BTreeSet::new();
        for block in &self.func.blocks {
            if !seen.insert(block.id) {
                self.error(format!("duplicate block id {}", block.id));
            }
            if block.id >= self.func.next_block_id {
                self.error(format!(
                    "block {} is at or above next_block_id {} (fresh_block_id would collide)",
                    block.id, self.func.next_block_id
                ));
            }
        }
    }

    /// The entry block takes no parameters: nothing jumps to it, so nothing
    /// could supply arguments.
    fn check_entry_block(&mut self) {
        let entry = &self.func.blocks[0];
        if !entry.param_types.is_empty() {
            self.error(format!(
                "entry block {} declares {} parameter(s); the entry block takes none",
                entry.id,
                entry.param_types.len()
            ));
        }
    }

    /// Exactly one terminator, in final position.
    fn check_block_shape(&mut self, idx: usize) {
        let block = &self.func.blocks[idx];
        if block.ops.is_empty() {
            self.error(format!("block {} has no operations", block.id));
            return;
        }
        if !block.ops.last().unwrap().is_terminator() {
            self.error(format!("block {} does not end with a terminator", block.id));
        }
        let last = block.ops.len() - 1;
        for (op_idx, op) in block.ops[..last].iter().enumerate() {
            if op.is_terminator() {
                self.error(format!(
                    "block {} has a terminator at op {} (before the final position {})",
                    block.id, op_idx, last
                ));
            }
        }
    }

    fn check_op(&mut self, block_idx: usize, op_idx: usize, op: &EffectfulOp) {
        let block_id = self.func.blocks[block_idx].id;
        let at = format!("block {block_id} op {op_idx}");

        match op {
            EffectfulOp::Load { addr, result, .. } => {
                self.check_class(*addr, &format!("{at}: Load addr"));
                self.check_class(*result, &format!("{at}: Load result"));
            }
            EffectfulOp::Store { addr, val, .. } => {
                self.check_class(*addr, &format!("{at}: Store addr"));
                self.check_class(*val, &format!("{at}: Store val"));
            }
            EffectfulOp::Call {
                func,
                args,
                arg_tys,
                ret_tys,
                results,
            } => {
                if args.len() != arg_tys.len() {
                    self.error(format!(
                        "{at}: Call '{func}' has {} args but {} arg types",
                        args.len(),
                        arg_tys.len()
                    ));
                }
                if results.len() != ret_tys.len() {
                    self.error(format!(
                        "{at}: Call '{func}' has {} results but {} return types",
                        results.len(),
                        ret_tys.len()
                    ));
                }
                for (i, arg) in args.iter().enumerate() {
                    self.check_class(*arg, &format!("{at}: Call '{func}' arg {i}"));
                }
                for (i, res) in results.iter().enumerate() {
                    self.check_class(*res, &format!("{at}: Call '{func}' result {i}"));
                }
            }
            EffectfulOp::Branch {
                cond,
                bb_true,
                bb_false,
                true_args,
                false_args,
                ..
            } => {
                self.check_class(*cond, &format!("{at}: Branch cond"));
                self.check_edge(&at, "true", *bb_true, true_args);
                self.check_edge(&at, "false", *bb_false, false_args);
            }
            EffectfulOp::Jump { target, args } => {
                self.check_edge(&at, "jump", *target, args);
            }
            EffectfulOp::Ret { val } => {
                let expected = self.func.return_types.len();
                let actual = usize::from(val.is_some());
                if actual != expected {
                    self.error(format!(
                        "{at}: Ret returns {actual} value(s) but function '{}' declares {expected}",
                        self.func.name
                    ));
                }
                if let Some(v) = val {
                    self.check_class(*v, &format!("{at}: Ret val"));
                }
            }
        }
    }

    /// A CFG edge must target an existing block and pass exactly the arguments
    /// that block declares, with matching types.
    fn check_edge(&mut self, at: &str, which: &str, target: u32, args: &TermArgs) {
        let Some(target_block) = self.func.blocks.iter().find(|b| b.id == target) else {
            self.error(format!(
                "{at}: {which} target block {target} does not exist"
            ));
            return;
        };

        let param_types = target_block.param_types.clone();
        if args.len() != param_types.len() {
            self.error(format!(
                "{at}: {which} edge to block {target} passes {} arg(s) but the block declares {} param(s)",
                args.len(),
                param_types.len()
            ));
        }

        for (i, arg) in args.class_ids().enumerate() {
            let arg_ty = self.check_class(arg, &format!("{at}: {which} edge arg {i}"));
            if let (Some(arg_ty), Some(param_ty)) = (arg_ty, param_types.get(i))
                && arg_ty != *param_ty
            {
                self.error(format!(
                    "{at}: {which} edge arg {i} to block {target} has type {arg_ty:?} but the block parameter is {param_ty:?}",
                ));
            }
        }
    }

    /// A `ClassId` referenced by an effectful op must resolve to a real class,
    /// and under `strict` must already be canonical. Returns the resolved type
    /// when the reference is valid.
    fn check_class(&mut self, id: ClassId, what: &str) -> Option<Type> {
        if id == ClassId::NONE {
            self.error(format!("{what}: ClassId::NONE used as an operand"));
            return None;
        }
        if id.0 as usize >= self.egraph.arena_len() {
            self.error(format!(
                "{what}: ClassId({}) is out of range (e-graph has {} classes)",
                id.0,
                self.egraph.arena_len()
            ));
            return None;
        }
        let canon = self.egraph.find_immutable(id);
        if self.strict && canon != id {
            self.error(format!(
                "{what}: ClassId({}) is not canonical (canonical form is ClassId({}))",
                id.0, canon.0
            ));
        }
        Some(self.egraph.class(canon).ty.clone())
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::function::BasicBlock;
    use crate::ir::op::Op;

    /// `fn f(a: I64) -> I64 { return a + 1; }`
    fn simple_function() -> (Function, EGraph) {
        let mut b = FunctionBuilder::new("f", &[Type::I64], &[Type::I64]);
        let p = b.params().to_vec();
        let one = b.iconst(1, Type::I64);
        let sum = b.add(p[0], one);
        b.ret(Some(sum));
        let mut func = b.finalize().expect("finalize");
        let egraph = func.egraph.take().expect("builder attaches an e-graph");
        (func, egraph)
    }

    #[test]
    fn clean_function_verifies() {
        let (func, egraph) = simple_function();
        assert_eq!(verify_function(&func, &egraph), Vec::<String>::new());
    }

    #[test]
    fn detects_missing_terminator() {
        let (mut func, egraph) = simple_function();
        func.blocks[0].ops.pop();
        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("does not end with a terminator")
                    || e.contains("has no operations")),
            "{errors:?}"
        );
    }

    /// Two blocks: block 0 defines VReg 1 and VReg 2 and jumps to block 1, whose
    /// parameters are `block1_params`. Block 0's terminator passes `passed`, and
    /// block 1 reads `block1_uses`.
    fn two_values_across_an_edge(
        assignment: &[(u32, Reg)],
        passed: &[u32],
        block1_params: &[u32],
        block1_uses: &[u32],
    ) -> Vec<String> {
        let inst = |dst: u32, operands: Vec<u32>| ScheduledInst {
            op: Op::Iconst(0, Type::I32),
            dst: VReg(dst),
            operands: operands.into_iter().map(VReg).collect(),
        };
        let schedules = vec![
            vec![inst(1, vec![]), inst(2, vec![])],
            vec![inst(3, block1_uses.to_vec())],
        ];
        let phi_uses = vec![passed.iter().copied().map(VReg).collect(), BTreeSet::new()];
        let block_params = vec![
            BTreeSet::new(),
            block1_params.iter().copied().map(VReg).collect(),
        ];
        verify_register_sharing(
            &schedules,
            &phi_uses,
            &block_params,
            &[vec![1], vec![]],
            &assignment.iter().map(|&(v, r)| (VReg(v), r)).collect(),
            &BTreeMap::new(),
            &[],
        )
    }

    #[test]
    fn detects_two_live_values_in_one_register() {
        let errors =
            two_values_across_an_edge(&[(1, Reg::RBX), (2, Reg::RBX)], &[1, 2], &[], &[1, 2]);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("VReg 1 and VReg 2 are both live and both hold RBX")),
            "{errors:?}"
        );
    }

    #[test]
    fn distinct_registers_verify() {
        assert_eq!(
            two_values_across_an_edge(&[(1, Reg::RBX), (2, Reg::RCX)], &[1, 2], &[], &[1, 2]),
            Vec::<String>::new()
        );
    }

    /// A parameter of the successor is written by the phi copy on the edge, so a
    /// def of that same VReg in the predecessor is dead and its register is free
    /// for anything else there. Reporting it is what kept two correct programs
    /// out of `tests/lit`.
    #[test]
    fn a_successors_parameter_is_not_live_in_its_predecessor() {
        assert_eq!(
            two_values_across_an_edge(&[(1, Reg::RBX), (2, Reg::RBX)], &[2], &[1], &[1]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn detects_terminator_before_final_position() {
        let (mut func, egraph) = simple_function();
        let term = func.blocks[0].ops.last().unwrap().clone();
        func.blocks[0].ops.insert(0, term);
        let errors = verify_function(&func, &egraph);
        assert!(
            errors.iter().any(|e| e.contains("terminator at op 0")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_missing_jump_target() {
        let (mut func, egraph) = simple_function();
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Jump {
            target: 99,
            args: TermArgs::default(),
        };
        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("target block 99 does not exist")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_block_param_arity_mismatch() {
        let (mut func, egraph) = simple_function();
        let ret_val = match func.blocks[0].ops.last().unwrap() {
            EffectfulOp::Ret { val } => val.expect("has return value"),
            other => panic!("expected Ret, got {other:?}"),
        };
        // Add a block taking one param, then jump to it passing none.
        let target_id = func.fresh_block_id();
        let mut target = BasicBlock::new(target_id, vec![Type::I64]);
        target.ops.push(EffectfulOp::Ret { val: Some(ret_val) });
        func.blocks.push(target);
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Jump {
            target: target_id,
            args: TermArgs::default(),
        };

        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("passes 0 arg(s) but the block declares 1 param(s)")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_out_of_range_class() {
        let (mut func, egraph) = simple_function();
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Ret {
            val: Some(ClassId(9999)),
        };
        let errors = verify_function(&func, &egraph);
        assert!(
            errors.iter().any(|e| e.contains("is out of range")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_ret_arity_mismatch() {
        let (mut func, egraph) = simple_function();
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Ret { val: None };
        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("Ret returns 0 value(s) but function 'f' declares 1")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_duplicate_block_id() {
        let (mut func, egraph) = simple_function();
        let dup = func.blocks[0].clone();
        func.blocks.push(dup);
        let errors = verify_function(&func, &egraph);
        assert!(
            errors.iter().any(|e| e.contains("duplicate block id")),
            "{errors:?}"
        );
    }

    #[test]
    fn detects_entry_block_with_params() {
        let (mut func, egraph) = simple_function();
        func.blocks[0].param_types.push(Type::I64);
        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("the entry block takes none")),
            "{errors:?}"
        );
    }

    /// Merging classes leaves the CFG holding a pre-merge id. Normal accepts
    /// it (that is how the pipeline runs today); strict reports it.
    #[test]
    fn stale_class_id_is_strict_only() {
        let (mut func, mut egraph) = simple_function();
        let returned = match func.blocks[0].ops.last().unwrap() {
            EffectfulOp::Ret { val } => val.expect("has return value"),
            other => panic!("expected Ret, got {other:?}"),
        };
        // Merge the returned class with a fresh one; whichever of the two loses
        // union-find is now a stale id that still resolves. Point the CFG at it.
        let other = egraph.add(ENode {
            op: Op::Iconst(4242, Type::I64),
            children: smallvec![],
        });
        egraph.merge(returned, other);
        egraph.rebuild();
        let stale = if egraph.find_immutable(returned) != returned {
            returned
        } else {
            other
        };
        assert_ne!(
            egraph.find_immutable(stale),
            stale,
            "merge should have left one of the two ids non-canonical"
        );
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Ret { val: Some(stale) };

        assert_eq!(
            verify_function_at(&func, &egraph, VerifyLevel::Normal),
            Vec::<String>::new(),
            "normal level must tolerate stale ids"
        );

        let strict = verify_function_at(&func, &egraph, VerifyLevel::Strict);
        assert!(
            strict.iter().any(|e| e.contains("is not canonical")),
            "{strict:?}"
        );
    }

    #[test]
    fn detects_call_arity_mismatch() {
        let (mut func, egraph) = simple_function();
        let call = EffectfulOp::Call {
            func: "g".to_string(),
            args: vec![ClassId(0), ClassId(0)],
            arg_tys: vec![Type::I64],
            ret_tys: vec![],
            results: vec![],
        };
        func.blocks[0].ops.insert(0, call);
        let errors = verify_function(&func, &egraph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("has 2 args but 1 arg types")),
            "{errors:?}"
        );
    }
}

// ── CFG versus schedule ──────────────────────────────────────────────────────

use crate::compile::barrier;
use crate::compile::program_point::ProgramPoint;
use crate::egraph::extract::{ClassVRegMap, VReg};
use crate::schedule::scheduler::ScheduledInst;
use crate::x86::inst::MachInst;
use crate::x86::reg::Reg;

/// Compare the two answers the pipeline holds for "which register carries this
/// value here": the `ClassId` the CFG names, resolved through the class map, and
/// the VReg the schedule carries as an operand.
///
/// The CFG names pure values by `ClassId`, which is expression identity and has
/// no position; every consumer after extraction needs occurrence identity. So
/// each consumer resolves a class through a map keyed by program point, while the
/// schedule already records the VReg that pass rewrote. Where the two disagree
/// the emitted code reads a register holding some other value, and nothing else
/// can see it: the register is written, so def-before-use holds, and no two
/// modelled live ranges overlap, so the sharing check holds.
///
/// Checked where both representations exist: an effectful op's role operands (a
/// `Load`'s address, a `Store`'s address and value, a `Call`'s arguments in ABI
/// order) against the leading operands of the barrier instruction consuming that
/// op, by index.
///
/// Block arguments are not checked and cannot disagree: the CFG names them by
/// VReg, and `Op::TerminatorArgs` is a copy of that list rather than a second
/// resolution of it.
///
/// Only positions where *both* sides have an answer are compared. A class the map
/// cannot resolve at that point has no second answer to disagree with.
pub fn verify_cfg_schedule_agreement(
    func: &Function,
    egraph: &EGraph,
    block_schedules: &[Vec<ScheduledInst>],
    block_snapshots: &[ClassVRegMap],
) -> Vec<String> {
    let mut errors = Vec::new();
    if block_schedules.len() != func.blocks.len() || block_snapshots.len() != func.blocks.len() {
        return errors;
    }

    for (block_idx, block) in func.blocks.iter().enumerate() {
        let schedule = &block_schedules[block_idx];
        let snapshot = &block_snapshots[block_idx];
        // An ambiguity is a finding in its own right: two segments of one class
        // covering a point without nesting means the map cannot say which VReg
        // holds the value there, and `lookup` asserts rather than choosing. It
        // stays latent while no pass asks at that point.
        let resolve = |cid: ClassId, point: ProgramPoint| -> Result<Option<VReg>, String> {
            let canon = egraph.unionfind.find_immutable(cid);
            match snapshot.lookup_ambiguity(canon, point) {
                Some(why) => Err(why),
                None => Ok(snapshot.lookup(canon, point)),
            }
        };

        // The schedule position of each barrier instruction, in effectful-op
        // order. `lower_effectful_op` finds its barrier the same way.
        let barrier_positions: Vec<usize> = schedule
            .iter()
            .enumerate()
            .filter(|(_, inst)| {
                matches!(
                    inst.op,
                    Op::LoadResult(..)
                        | Op::CallResult(..)
                        | Op::VoidCallBarrier
                        | Op::StoreBarrier
                )
            })
            .map(|(i, _)| i)
            .collect();

        for (barrier_k, op) in block.ops[..block.non_term_count()].iter().enumerate() {
            let roles: Vec<(&str, ClassId)> = match op {
                EffectfulOp::Load { addr, .. } => vec![("Load address", *addr)],
                EffectfulOp::Store { addr, val, .. } => {
                    vec![("Store address", *addr), ("Store value", *val)]
                }
                EffectfulOp::Call { args, .. } => {
                    args.iter().map(|&cid| ("Call argument", cid)).collect()
                }
                _ => continue,
            };
            let Some(&barrier_pos) = barrier_positions.get(barrier_k) else {
                continue;
            };
            let Some(barrier) = schedule.get(barrier_pos) else {
                continue;
            };
            // The point `lower_effectful_op` asks at, which is the barrier's own
            // instruction index. `ProgramPoint::barrier_point` would give one
            // more, and the splitter's segments run `[def .. last reference]` over
            // raw indices -- so a reload consumed by this barrier has a segment
            // ending exactly here, and asking one point later falls through to the
            // pre-spill VReg's full-range entry instead.
            let point = if barrier_pos > 0 {
                ProgramPoint::inst_point(block_idx, barrier_pos)
            } else {
                ProgramPoint::block_entry(block_idx)
            };
            let role_count = barrier::role_operand_count(op);
            for (i, (what, cid)) in roles.iter().enumerate() {
                if i >= role_count {
                    break;
                }
                let Some(&from_schedule) = barrier.operands.get(i) else {
                    continue;
                };
                let from_map = match resolve(*cid, point) {
                    Err(why) => {
                        errors.push(format!("block {block_idx}: {what} {i}: {why}"));
                        continue;
                    }
                    Ok(None) => continue,
                    Ok(Some(v)) => v,
                };
                if from_map != from_schedule {
                    errors.push(format!(
                        "block {block_idx}: {what} {i} (class {:?}) resolves to VReg {} \
                         through the class map at {point:?}, but the {:?} operand at that \
                         index is VReg {}",
                        egraph.unionfind.find_immutable(*cid),
                        from_map.0,
                        barrier.op,
                        from_schedule.0,
                    ));
                }
            }
        }
    }

    errors
}

/// Run [`verify_cfg_schedule_agreement`] and report the count on stderr. No-op
/// unless `BLITZ_VERIFY` is set, or when the two representations agree.
///
/// For the stages where they do *not* agree today, so the number is visible and
/// a regression in it is visible, without failing the suite on a defect that is
/// latent. It is latent because the consumer does not trust the map first: an
/// effectful op reads the barrier's own operand and falls back to the map only
/// where there is no operand at that index. Committing the per-block VReg choice
/// into the CFG is what removes the second answer, as it already has for block
/// arguments; until the remaining `EffectfulOp` operands follow, a reload whose
/// segment does not reach the barrier consuming it leaves the map answering with
/// the pre-spill VReg.
pub fn report_cfg_schedule_agreement(
    stage: &str,
    func: &Function,
    egraph: &EGraph,
    block_schedules: &[Vec<ScheduledInst>],
    block_snapshots: &[ClassVRegMap],
) {
    if !is_enabled() {
        return;
    }
    let errors = verify_cfg_schedule_agreement(func, egraph, block_schedules, block_snapshots);
    if errors.is_empty() {
        return;
    }
    eprintln!(
        "BLITZ_VERIFY: {} CFG/schedule disagreement(s) in function '{}' after stage '{}':\n  - {}",
        errors.len(),
        func.name,
        stage,
        errors.join("\n  - ")
    );
}

/// Run [`verify_cfg_schedule_agreement`] and panic if the two representations
/// disagree. No-op unless `BLITZ_VERIFY` is set.
pub fn verify_cfg_schedule_agreement_stage(
    stage: &str,
    func: &Function,
    egraph: &EGraph,
    block_schedules: &[Vec<ScheduledInst>],
    block_snapshots: &[ClassVRegMap],
) {
    if !is_enabled() {
        return;
    }
    let errors = verify_cfg_schedule_agreement(func, egraph, block_schedules, block_snapshots);
    if !errors.is_empty() {
        panic!(
            "BLITZ_VERIFY: {} CFG/schedule disagreement(s) in function '{}' after stage \
             '{}':\n  - {}",
            errors.len(),
            func.name,
            stage,
            errors.join("\n  - ")
        );
    }
}

// ── Machine level ────────────────────────────────────────────────────────────

/// Registers that already hold something on entry to a function: the stack and
/// frame pointers, and the SysV argument registers. Reading one of these before
/// the function writes it is normal.
fn entry_live(uses_frame_pointer: bool) -> BTreeSet<Reg> {
    use crate::x86::abi::{FP_ARG_REGS, GPR_ARG_REGS};
    let mut set: BTreeSet<Reg> = GPR_ARG_REGS.iter().copied().collect();
    set.extend(FP_ARG_REGS);
    set.insert(Reg::RSP);
    if uses_frame_pointer {
        set.insert(Reg::RBP);
    }
    set
}

/// Check the emitted instruction stream for a function.
///
/// Returns one message per violation. `insts` is the final, fully lowered
/// stream in emission order.
///
/// The use-before-def check is deliberately conservative about control flow.
/// Reading a register that *no* instruction in the function writes is always a
/// bug. Reading one before its first write in emission order is only reported
/// when the function has no backward branch, because a loop can legitimately
/// read on iteration two what the bottom of the body wrote on iteration one.
pub fn verify_machinsts(
    insts: &[MachInst],
    labels: &BTreeMap<u32, usize>,
    uses_frame_pointer: bool,
) -> Vec<String> {
    let mut errors = Vec::new();

    for (i, inst) in insts.iter().enumerate() {
        if has_vreg(inst) {
            errors.push(format!(
                "inst {i}: a virtual register survived register allocation: {inst:?}"
            ));
        }
    }

    // "Written on every path from entry" per instruction, by forward dataflow
    // over the control-flow graph recovered from labels and branches. Meeting
    // predecessors with intersection is what makes this sound: a register
    // written on only one arm of a branch is not written at the join.
    let live_in = entry_live(uses_frame_pointer);
    let leaders = block_leaders(insts, labels);
    let (blocks, succs) = build_cfg(insts, labels, &leaders);

    let mut entry_state: Vec<Option<BTreeSet<Reg>>> = vec![None; blocks.len()];
    if !blocks.is_empty() {
        entry_state[0] = Some(live_in.clone());
    }
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..blocks.len() {
            let Some(state) = entry_state[b].clone() else {
                continue;
            };
            let mut out = state;
            for i in blocks[b].clone() {
                out.extend(insts[i].defs());
            }
            for &s in &succs[b] {
                let merged = match &entry_state[s] {
                    // Intersection: only registers written on *both* paths.
                    Some(existing) => existing.intersection(&out).copied().collect(),
                    None => out.clone(),
                };
                if entry_state[s].as_ref() != Some(&merged) {
                    entry_state[s] = Some(merged);
                    changed = true;
                }
            }
        }
    }

    for (b, range) in blocks.iter().enumerate() {
        // A block the dataflow never reached is unreachable code; nothing it
        // reads can be executed, so there is nothing to check.
        let Some(state) = entry_state[b].clone() else {
            continue;
        };
        let mut written = state;
        for i in range.clone() {
            for r in insts[i].uses() {
                if !written.contains(&r) {
                    errors.push(format!(
                        "inst {i}: reads {r:?} on a path where nothing writes it: {:?}",
                        insts[i]
                    ));
                }
            }
            written.extend(insts[i].defs());
        }
    }

    errors
}

/// Report every read of a spill slot that no store on some path to it has
/// written.
///
/// Two VRegs live at the same program point must not hold the same physical
/// register. One of them would be reading the other's value.
///
/// This is the invariant a graph colorer exists to maintain, so checking it
/// against the allocator's own interference graph would be circular and always
/// pass. Liveness is therefore recomputed here from the schedules **as emitted**
/// -- after splitting, after coalescing, in final order -- and compared against
/// the final assignment. What it catches is precisely a divergence between the
/// liveness the allocator colored and the code that came out:
///
/// - a value emitted somewhere its interference was never measured, which is one
///   of the two shapes nearly every wrong-code bug in this backend has had;
/// - a VReg whose live range has a *hole* -- the splitter truncates a segment at
///   its SpillStore -- coalesced with a later VReg, so the merged value is valid
///   over only part of its own range while the allocator fills the hole with
///   something else.
///
/// The machine-level checks cannot see either: the register is written on every
/// path to the read, just with the wrong value.
///
/// VRegs absent from `vreg_to_reg` are skipped. They hold no register to share:
/// spill pseudo-op destinations are dummies, and a coalesced-away VReg no longer
/// appears in the emitted operands at all.
pub fn verify_register_sharing(
    block_schedules: &[Vec<ScheduledInst>],
    phi_uses: &[std::collections::BTreeSet<VReg>],
    block_params: &[std::collections::BTreeSet<VReg>],
    cfg_succs: &[Vec<usize>],
    vreg_to_reg: &BTreeMap<VReg, Reg>,
    coalesce_aliases: &BTreeMap<VReg, VReg>,
    copy_pairs: &[(VReg, VReg)],
) -> Vec<String> {
    use crate::regalloc::global_liveness::compute_global_liveness_with_block_params;

    let mut errors = Vec::new();
    if block_schedules.len() != phi_uses.len()
        || block_schedules.len() != block_params.len()
        || block_schedules.len() != cfg_succs.len()
    {
        return errors;
    }

    // Everything is named by its coalesce representative, which is what the
    // emitted code uses. Two things need it. A schedule operand naming a VReg
    // coalescing merged away has no register of its own, and skipping it would
    // skip the interesting case, since the emitted code reads the *alias's*
    // register. And `phi_uses` was built before coalescing, so a renamed VReg
    // there is a use with no def in any schedule -- liveness would carry it back
    // to the function entry and report a clash at every point on the way.
    let canon = |mut v: VReg| -> VReg {
        for _ in 0..coalesce_aliases.len() + 1 {
            match coalesce_aliases.get(&v) {
                Some(&next) if next != v => v = next,
                _ => break,
            }
        }
        v
    };
    let phi_uses_canon: Vec<std::collections::BTreeSet<VReg>> = phi_uses
        .iter()
        .map(|set| set.iter().map(|&v| canon(v)).collect())
        .collect();
    let schedules_canon: Vec<Vec<ScheduledInst>> = block_schedules
        .iter()
        .map(|sched| {
            sched
                .iter()
                .map(|inst| ScheduledInst {
                    op: inst.op.clone(),
                    dst: canon(inst.dst),
                    operands: inst.operands.iter().map(|&v| canon(v)).collect(),
                })
                .collect()
        })
        .collect();
    // A block parameter is written by the phi copy on the edge, so it is not live
    // at a predecessor's exit; what is live there is whatever the predecessor's
    // terminator passes, which `phi_uses` carries. Without this, a VReg that is
    // both a parameter of some block and the destination of a dead def in a
    // predecessor reads as live across that predecessor, and every value holding
    // its register in between reads as a clash.
    let block_params_canon: Vec<std::collections::BTreeSet<VReg>> = block_params
        .iter()
        .map(|set| set.iter().map(|&v| canon(v)).collect())
        .collect();
    let liveness = compute_global_liveness_with_block_params(
        &schedules_canon,
        cfg_succs,
        &phi_uses_canon,
        &block_params_canon,
    );

    // A phi copy's source and destination hold the SAME value, so sharing a
    // register is not a clash -- it is what coalescing the pair achieves, and
    // both are live at the copy point by construction. Only directly related
    // pairs are exempt: two different arguments of one target block are two
    // distinct values, and a source coalesced transitively onto some later
    // value is the bug this check exists to find.
    let phi_related: BTreeSet<(VReg, VReg)> = copy_pairs
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();

    // One report per (register, lower VReg, higher VReg): the same clash is live
    // over a range of points and would otherwise be reported at each of them.
    let mut seen: BTreeSet<(Reg, VReg, VReg)> = BTreeSet::new();

    let check = |live: &std::collections::BTreeSet<VReg>,
                 where_: &str,
                 errors: &mut Vec<String>,
                 seen: &mut BTreeSet<(Reg, VReg, VReg)>| {
        let mut by_reg: BTreeMap<Reg, VReg> = BTreeMap::new();
        for &v in live {
            let Some(&reg) = vreg_to_reg.get(&v) else {
                continue;
            };
            match by_reg.get(&reg) {
                None => {
                    by_reg.insert(reg, v);
                }
                Some(&other) => {
                    let key = (reg, other.min(v), other.max(v));
                    if phi_related.contains(&(key.1, key.2)) {
                        continue;
                    }
                    if seen.insert(key) {
                        errors.push(format!(
                            "{where_}: VReg {} and VReg {} are both live and both hold {:?}",
                            key.1.0, key.2.0, reg,
                        ));
                    }
                }
            }
        }
    };

    for (b, schedule) in schedules_canon.iter().enumerate() {
        // Backward walk, so `live` is the live-before set of each instruction.
        let mut live = liveness.live_out[b].clone();
        check(&live, &format!("block {b} exit"), &mut errors, &mut seen);
        for (i, inst) in schedule.iter().enumerate().rev() {
            live.remove(&inst.dst);
            for &op in &inst.operands {
                live.insert(op);
            }
            check(
                &live,
                &format!("block {b} before [{i}] {:?}", inst.op),
                &mut errors,
                &mut seen,
            );
        }
    }

    errors
}

/// Same forward dataflow as the register check, keyed by frame displacement
/// instead: meet predecessors with intersection, so a slot written on one arm of
/// a branch is not written at the join. A reload from a slot nothing wrote is a
/// value the allocator believes it saved and did not, which no register-level
/// check can see -- the register the reload writes is written, just with
/// whatever the caller left on the stack.
///
/// Scoped to `[spill_lo, spill_hi)`, the slot region of the frame. The rest of
/// the frame is off limits for this: user stack slots hold arrays and structs
/// written through computed addresses this pass does not track, and the outgoing
/// argument area and anything above the frame belong to the caller, which
/// legitimately wrote them before the call.
fn verify_spill_slots(
    insts: &[MachInst],
    labels: &BTreeMap<u32, usize>,
    spill_base: Reg,
    spill_lo: i32,
    spill_hi: i32,
) -> Vec<String> {
    if spill_hi <= spill_lo {
        return Vec::new();
    }
    // The frame displacement a memory operand names, when it is a plain
    // `[spill_base + disp]` inside the slot region. An indexed address is not a
    // slot reference: nothing addresses a spill slot that way.
    let slot_of = |addr: &crate::x86::addr::Addr| -> Option<i32> {
        if addr.base != Some(spill_base) || addr.index.is_some() {
            return None;
        }
        (addr.disp >= spill_lo && addr.disp < spill_hi).then_some(addr.disp)
    };

    let leaders = block_leaders(insts, labels);
    let (blocks, succs) = build_cfg(insts, labels, &leaders);
    let mut entry_state: Vec<Option<BTreeSet<i32>>> = vec![None; blocks.len()];
    if !blocks.is_empty() {
        entry_state[0] = Some(BTreeSet::new());
    }
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..blocks.len() {
            let Some(state) = entry_state[b].clone() else {
                continue;
            };
            let mut out = state;
            for i in blocks[b].clone() {
                if let Some(addr) = insts[i].mem_store_addr()
                    && let Some(slot) = slot_of(addr)
                {
                    out.insert(slot);
                }
            }
            for &s in &succs[b] {
                let merged = match &entry_state[s] {
                    Some(existing) => existing.intersection(&out).copied().collect(),
                    None => out.clone(),
                };
                if entry_state[s].as_ref() != Some(&merged) {
                    entry_state[s] = Some(merged);
                    changed = true;
                }
            }
        }
    }

    let mut errors = Vec::new();
    for (b, range) in blocks.iter().enumerate() {
        let Some(state) = entry_state[b].clone() else {
            continue;
        };
        let mut written = state;
        for i in range.clone() {
            if let Some(addr) = insts[i].mem_load_addr()
                && let Some(slot) = slot_of(addr)
                && !written.contains(&slot)
            {
                errors.push(format!(
                    "inst {i}: reloads spill slot at {:?}{:+} on a path where nothing stored it: {:?}",
                    spill_base, slot, insts[i]
                ));
            }
            if let Some(addr) = insts[i].mem_store_addr()
                && let Some(slot) = slot_of(addr)
            {
                written.insert(slot);
            }
        }
    }
    errors
}

/// Instruction indices that begin a basic block: the entry, every label, and
/// whatever follows a branch or return.
fn block_leaders(insts: &[MachInst], labels: &BTreeMap<u32, usize>) -> Vec<usize> {
    let mut set: BTreeSet<usize> = BTreeSet::new();
    if !insts.is_empty() {
        set.insert(0);
    }
    for &pos in labels.values() {
        if pos < insts.len() {
            set.insert(pos);
        }
    }
    for (i, inst) in insts.iter().enumerate() {
        if matches!(
            inst,
            MachInst::Jmp { .. } | MachInst::Jcc { .. } | MachInst::Ret
        ) && i + 1 < insts.len()
        {
            set.insert(i + 1);
        }
    }
    set.into_iter().collect()
}

type Cfg = (Vec<std::ops::Range<usize>>, Vec<Vec<usize>>);

/// Basic block ranges and their successor lists.
fn build_cfg(insts: &[MachInst], labels: &BTreeMap<u32, usize>, leaders: &[usize]) -> Cfg {
    let mut blocks: Vec<std::ops::Range<usize>> = Vec::new();
    for (n, &start) in leaders.iter().enumerate() {
        let end = leaders.get(n + 1).copied().unwrap_or(insts.len());
        blocks.push(start..end);
    }
    let block_of = |i: usize| blocks.iter().position(|r| r.contains(&i));
    let mut succs = vec![Vec::new(); blocks.len()];
    for (b, range) in blocks.iter().enumerate() {
        let Some(last) = range.clone().last() else {
            continue;
        };
        match &insts[last] {
            MachInst::Jmp { target } => {
                if let Some(t) = labels.get(target).and_then(|&p| block_of(p)) {
                    succs[b].push(t);
                }
            }
            MachInst::Jcc { target, .. } => {
                if let Some(t) = labels.get(target).and_then(|&p| block_of(p)) {
                    succs[b].push(t);
                }
                if b + 1 < blocks.len() {
                    succs[b].push(b + 1);
                }
            }
            MachInst::Ret => {}
            _ => {
                if b + 1 < blocks.len() {
                    succs[b].push(b + 1);
                }
            }
        }
    }
    (blocks, succs)
}

fn has_vreg(inst: &MachInst) -> bool {
    // Operands are not enumerable without another exhaustive match, and the
    // Debug form names the variant, so match on the rendered text. This runs
    // only under BLITZ_VERIFY.
    format!("{inst:?}").contains("VReg(")
}

/// Verify an emitted function and panic on any violation. No-op unless
/// `BLITZ_VERIFY` is set.
pub fn verify_machinsts_stage(
    stage: &str,
    func_name: &str,
    insts: &[MachInst],
    labels: &BTreeMap<u32, usize>,
    frame: &crate::x86::abi::FrameLayout,
    spill_slots: u32,
) {
    if level() == VerifyLevel::Off {
        return;
    }
    let mut errors = verify_machinsts(insts, labels, frame.uses_frame_pointer);
    errors.extend(verify_spill_slots(
        insts,
        labels,
        frame.spill_base,
        frame.spill_offset,
        frame.spill_offset + (spill_slots as i32) * 8,
    ));
    if !errors.is_empty() {
        panic!(
            "BLITZ_VERIFY: {} machine-level violation(s) in function '{}' after stage '{}':\n  - {}",
            errors.len(),
            func_name,
            stage,
            errors.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod machine_tests {
    use super::*;
    use crate::x86::addr::Addr;
    use crate::x86::inst::{MachInst, OpSize, Operand};
    use crate::x86::reg::Reg;

    fn reg(r: Reg) -> Operand {
        Operand::Reg(r)
    }

    fn at(base: Reg) -> Addr {
        Addr {
            base: Some(base),
            index: None,
            scale: 1,
            disp: 0,
        }
    }

    fn verify(insts: &[MachInst]) -> Vec<String> {
        verify_machinsts(insts, &BTreeMap::new(), false)
    }

    #[test]
    fn clean_stream_verifies() {
        // rdi is an argument register, so reading it needs no local write.
        let insts = vec![
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RDI),
            },
            MachInst::Ret,
        ];
        assert_eq!(verify(&insts), Vec::<String>::new());
    }

    /// The shape of the reload-before-store bug: a store through a register
    /// nothing ever writes.
    #[test]
    fn detects_read_of_never_written_register() {
        let insts = vec![
            MachInst::MovMR {
                size: OpSize::S32,
                addr: at(Reg::R15),
                src: reg(Reg::RDI),
            },
            MachInst::Ret,
        ];
        let errors = verify(&insts);
        assert!(errors.iter().any(|e| e.contains("R15")), "{errors:?}");
    }

    /// The shape of the barrier-ordering bug: the address computation emitted
    /// after the store that uses it.
    #[test]
    fn detects_use_before_def_in_order() {
        let insts = vec![
            MachInst::MovMR {
                size: OpSize::S32,
                addr: at(Reg::RBX),
                src: reg(Reg::RDI),
            },
            MachInst::Lea {
                size: OpSize::S64,
                dst: reg(Reg::RBX),
                addr: at(Reg::RDI),
            },
            MachInst::Ret,
        ];
        let errors = verify(&insts);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("inst 0") && e.contains("RBX")),
            "{errors:?}"
        );
    }

    /// Written on one arm of a branch only: not written at the join, so the
    /// meet has to be intersection rather than union.
    #[test]
    fn detects_register_written_on_only_one_path() {
        let mut labels = BTreeMap::new();
        labels.insert(1u32, 3usize); // join block starts at inst 3
        let insts = vec![
            MachInst::Jcc {
                cc: crate::ir::condcode::CondCode::Eq,
                target: 1,
            },
            MachInst::MovRI {
                size: OpSize::S64,
                dst: reg(Reg::RBX),
                imm: 7,
            },
            MachInst::Jmp { target: 1 },
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RBX),
            },
            MachInst::Ret,
        ];
        let errors = verify_machinsts(&insts, &labels, false);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("inst 3") && e.contains("RBX")),
            "{errors:?}"
        );
    }

    /// The same shape, but written on both arms of a real diamond: no
    /// violation, which is what keeps the check from crying wolf on every
    /// if/else.
    #[test]
    fn accepts_register_written_on_all_paths() {
        let mut labels = BTreeMap::new();
        labels.insert(1u32, 3usize); // taken arm
        labels.insert(2u32, 4usize); // join
        let insts = vec![
            MachInst::Jcc {
                cc: crate::ir::condcode::CondCode::Eq,
                target: 1,
            },
            MachInst::MovRI {
                size: OpSize::S64,
                dst: reg(Reg::RBX),
                imm: 7,
            },
            MachInst::Jmp { target: 2 },
            MachInst::MovRI {
                size: OpSize::S64,
                dst: reg(Reg::RBX),
                imm: 9,
            },
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RBX),
            },
            MachInst::Ret,
        ];
        assert_eq!(
            verify_machinsts(&insts, &labels, false),
            Vec::<String>::new()
        );
    }

    #[test]
    fn zeroing_idiom_is_not_a_read() {
        let insts = vec![
            MachInst::XorRR {
                size: OpSize::S64,
                dst: reg(Reg::RAX),
                src: reg(Reg::RAX),
            },
            MachInst::Ret,
        ];
        assert_eq!(verify(&insts), Vec::<String>::new());
    }

    #[test]
    fn detects_surviving_vreg() {
        let insts = vec![
            MachInst::MovRR {
                size: OpSize::S64,
                dst: Operand::VReg(crate::egraph::extract::VReg(7)),
                src: reg(Reg::RDI),
            },
            MachInst::Ret,
        ];
        let errors = verify(&insts);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("virtual register survived")),
            "{errors:?}"
        );
    }

    /// A call leaves its result in RAX, so reading RAX afterwards is fine.
    #[test]
    fn call_defines_caller_saved_registers() {
        let insts = vec![
            MachInst::CallDirect {
                target: "f".to_string(),
            },
            MachInst::MovRR {
                size: OpSize::S64,
                dst: reg(Reg::RBX),
                src: reg(Reg::RAX),
            },
            MachInst::Ret,
        ];
        assert_eq!(verify(&insts), Vec::<String>::new());
    }
}
