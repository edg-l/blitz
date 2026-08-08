//! End-to-end compilation pipeline for Blitz.
//!
//! Phases:
//!  1. E-graph rewrite rules (algebraic, strength reduction, isel)
//!  2. Cost-based extraction
//!  3. VRegInst linearization, then trivial block-parameter removal, which
//!     re-extracts and runs phase 3 again over the reduced CFG
//!  4. DAG scheduling
//!  5. Register allocation
//!  6. VReg-to-phys rewrite
//!  7. Op -> MachInst lowering
//!  8. Peephole optimization
//!  9. NOP alignment (optional)
//! 10. Encoding
//! 11. ELF emission

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use crate::egraph::cost::{CostModel, OptGoal};
use crate::egraph::extract::{VReg, extract};
use crate::egraph::phases::{CompileOptions as EGraphOptions, run_phases};
use crate::emit::object::{FunctionInfo, ObjectFile};
use crate::emit::peephole::peephole;
use crate::ir::effectful::{BlockId, EffectfulOp};
use crate::ir::function::Function;
use crate::ir::op::{ClassId, Op, PseudoOp, PureOp};
use crate::regalloc::allocate_global;
use crate::regalloc::allocator::RegAllocResult;
use crate::regalloc::slots::SlotAllocator;
use crate::schedule::scheduler::{ScheduleDag, ScheduledInst, schedule};
use crate::x86::abi::{
    SCRATCH_GPR, compute_frame_layout, emit_epilogue, emit_prologue, sequentialize_copies,
};
use crate::x86::encode::{Encoder, inst_size};
use crate::x86::inst::{LabelId, MachInst};
use crate::x86::reg::Reg;

pub(crate) mod barrier;
pub mod program_point;
use barrier::{
    assign_barrier_groups, build_barrier_context, insert_early_barrier_spills,
    populate_effectful_operands,
};
mod canon;
use canon::canonicalize_class_refs;
pub(crate) mod cfg;
mod linearize;
mod phi_removal;
mod sccp;
use cfg::{
    collect_alloc_block_params, collect_externals, collect_roots, commit_block_param_vregs,
    commit_effectful_vregs, commit_terminator_arg_vregs, compute_copy_pairs_from_schedules,
    compute_loop_depths, projection_copy_pairs,
};
mod effectful;
use effectful::lower_effectful_op;
mod dce;
mod flags_remat;
pub(crate) use flags_remat::writes_flags as flags_writer;
mod licm;
pub(crate) mod lower;
use lower::lower_block_pure_ops;
mod precolor;
pub(crate) mod pressure;
use precolor::{add_call_precolors_for_block, assign_param_vregs_from_map};
mod rotate;
mod terminator;
use terminator::{lower_terminator, remove_fallthrough_jumps, thread_branches};
pub mod alias;
pub(crate) mod split;
pub use alias::{AddrBase, AliasInfo};
pub mod dse;
pub mod forward;

// ── Public options / error types ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations: fast compilation, minimal rewrites.
    O0,
    /// Full optimizations: inlining, egraph saturation, peephole.
    O1,
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub opt_level: OptLevel,
    pub opt_goal: OptGoal,
    /// Maximum e-graph saturation iterations. The loop exits early when no rules fire,
    /// so this is a safety cap. Typical programs converge in 2-4 iterations; 16 is generous.
    /// O0 uses 1 (minimum for isel; the lowerer requires x86 ops).
    pub saturation_limit: u32,
    pub enable_peephole: bool,
    /// Pad loop headers onto 16-byte boundaries with multi-byte NOPs.
    ///
    /// **On at `-O1`, off at `-O0`**: it buys speed with bytes, and it is the
    /// only thing that makes a loop's distance from a fetch boundary a property
    /// of the loop rather than of everything emitted before it.
    /// `BLITZ_PASSES=-loop-align` turns it off.
    pub enable_nop_alignment: bool,
    /// Copy a loop's test onto its back edge, so the back edge is the
    /// conditional and the loop closes on no unconditional `jmp`.
    ///
    /// **On at `-O1`, off at `-O0`.** It is a code-quality transform and `-O0`
    /// buys nothing with it, and keeping it off there leaves `run_diff.sh`'s
    /// `-O0`-vs-`-O1` leg able to see a bug in it. `BLITZ_PASSES=-loop-rotation`
    /// turns it off.
    pub enable_loop_rotation: bool,
    pub verbosity: Verbosity,
    /// Force the frame pointer (push rbp / mov rbp, rsp / pop rbp) to always be emitted.
    /// Defaults to `false`: the frame pointer is omitted when not needed, freeing RBP as a
    /// general-purpose register. Set to `true` for debuggability or when a frame pointer is
    /// required (e.g. kernel code).
    pub force_frame_pointer: bool,
    /// Enable Loop-Invariant Code Motion (LICM) before e-graph optimization.
    pub enable_licm: bool,
    /// Eliminate dead loads. The CFG half of DCE -- constant branch folding and
    /// unreachable-block removal -- is not gated on this and runs at every level.
    pub enable_dce: bool,
    /// Put every value in a frame slot and borrow registers one instruction at
    /// a time (`regalloc::fast`), instead of the Chaitin-Briggs colouring
    /// allocator.
    ///
    /// **On at `-O0`, off at `-O1`.** That is the point: the two levels stop
    /// sharing an allocator, so an allocation bug becomes a disagreement
    /// `run_diff.sh` can see rather than an answer both levels give, and the
    /// bug priors put regalloc first by a wide margin. It also skips the
    /// pressure splitter, which an allocator holding nothing across an
    /// instruction has no use for -- worth 1.41x on a 6048-line input.
    ///
    /// `BLITZ_PASSES=-fast-regalloc` puts `-O0` back on the colouring path.
    pub enable_fast_regalloc: bool,
    /// Enable intra-block store-to-load and load-to-load forwarding.
    pub enable_store_forwarding: bool,
    /// Enable intra-block dead store elimination.
    pub enable_dse: bool,
    /// Remove block parameters whose predecessors all pass one VReg, and
    /// linearize again over the reduced CFG.
    ///
    /// A canonicalization rather than an optimization: SSA construction creates a
    /// parameter for every variable live across an edge and never revisits it, and
    /// 85-94% of them turn out to name a single value.
    pub enable_phi_removal: bool,
    /// Remove block parameters whose predecessors all pass the same constant,
    /// before saturation, so the rules fold through them.
    ///
    /// **Off at `-O0`, for the same reason the dead-load half of DCE is.** A
    /// parameter carrying a merged constant is a source variable at the merge,
    /// and removing it takes away the storage a debugger would read it from;
    /// `-O0` is the level that maps to source. Being an `-O1`-only pass also
    /// puts it inside what `run_diff.sh`'s `-O0`-vs-`-O1` leg can see, where a
    /// transform running at both levels is visible only to the `cc` oracle.
    pub enable_sccp: bool,
    /// Enable function inlining before optimization.
    pub enable_inlining: bool,
    /// This module is the entire program being linked, so a function no path
    /// from `main` reaches is dead and can be dropped.
    ///
    /// **Defaults to `false`, and the default is the safe one.** A module is a
    /// translation unit; nothing inside the compiler can tell whether a
    /// definition it holds is the one another object needs at link time. Only
    /// the driver knows, and only when it is producing an executable from this
    /// module alone -- with `-c`, or with a second input file, an unreferenced
    /// non-`static` definition must survive or the object is unlinkable.
    ///
    /// Note that C linkage is not modelled: `ir::Function` carries no
    /// visibility, so a `static` helper is indistinguishable from an
    /// externally-visible one and is kept too.
    pub whole_program: bool,
    /// Turn a tail call to this same function into a jump to the top of its own
    /// body. Off at `-O0`: the recursion's frames are what a debugger walks.
    pub enable_tail_calls: bool,
    /// Drop instructions from the final stream whose result nothing reads.
    /// Off at `-O0`, where a value nobody reads is still one somebody may inspect.
    pub enable_dead_insts: bool,
    /// Maximum inlining rescan iterations per caller function. Each rescan inlines one level
    /// of calls; a depth-3 chain A->B->C->D needs 3 rescans. Note: this limits rescans,
    /// not true nesting depth; a function with many independent leaf calls also consumes
    /// iterations. Default 3 handles most practical transitive inlining.
    pub max_inline_depth: u32,
    /// Maximum callee e-graph node count (pre-saturation) for inlining eligibility.
    /// Measures raw e-nodes from IR construction, not post-optimization size. Rough proxy
    /// for code complexity. Default 50 corresponds to roughly 20-30 IR instructions.
    pub max_inline_nodes: usize,
    /// Maximum weighted cost for inlining a callee. Each IR operation has a weight
    /// (e.g. Add=1, SDiv=10, Call=20). If the total cost exceeds this threshold,
    /// the callee is not inlined (unless it has a single caller). Default 100.
    pub inline_cost_threshold: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Silent,
    Normal,
    Verbose,
}

impl CompileOptions {
    /// Apply `BLITZ_PASSES` to the pipeline this level configured.
    ///
    /// The optimization level is the only pipeline *configuration* this compiler
    /// offers: `-O0` and `-O1` are what it claims to compile correctly, and every
    /// gate runs those two. Deviating from a level is a debugging facility --
    /// `CLAUDE.md`'s fifth technique, bisecting the pass set to attribute a
    /// miscompile -- so it lives in the environment beside `BLITZ_DEBUG` and
    /// `BLITZ_VERIFY` rather than in the argument list beside `-O1`.
    ///
    /// The syntax is signed deltas against the level, comma separated:
    ///
    /// ```text
    /// BLITZ_PASSES=-licm            # -O1 without LICM
    /// BLITZ_PASSES=+inlining        # -O0 with inlining
    /// BLITZ_PASSES=-dse,-dce
    /// ```
    ///
    /// Both directions are needed: `-O0 +inlining` is as much a bisection step as
    /// `-O1 -licm`, so a disable-only list could not express half of them.
    ///
    /// An unknown pass name is a hard error rather than a silent no-op. A typo in
    /// a bisection step that quietly changes nothing would attribute a bug to the
    /// wrong pass, which is worse than not bisecting.
    pub fn apply_pass_overrides(&mut self) -> Result<(), String> {
        let Ok(spec) = std::env::var("BLITZ_PASSES") else {
            return Ok(());
        };
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (on, name) = match item.split_at(1) {
                ("+", rest) => (true, rest),
                ("-", rest) => (false, rest),
                _ => {
                    return Err(format!(
                        "BLITZ_PASSES: `{item}` must start with `+` or `-` (e.g. `-licm`)"
                    ));
                }
            };
            match name {
                "licm" => self.enable_licm = on,
                "dce" => self.enable_dce = on,
                "store-forwarding" => self.enable_store_forwarding = on,
                "dse" => self.enable_dse = on,
                "phi-removal" => self.enable_phi_removal = on,
                "sccp" => self.enable_sccp = on,
                "inlining" => self.enable_inlining = on,
                "tail-calls" => self.enable_tail_calls = on,
                "dead-insts" => self.enable_dead_insts = on,
                "peephole" => self.enable_peephole = on,
                "fast-regalloc" => self.enable_fast_regalloc = on,
                "loop-align" => self.enable_nop_alignment = on,
                "loop-rotation" => self.enable_loop_rotation = on,
                other => {
                    return Err(format!(
                        "BLITZ_PASSES: unknown pass `{other}`; known: licm, dce, \
                         store-forwarding, dse, phi-removal, sccp, inlining, \
                         peephole, fast-regalloc, tail-calls, dead-insts, loop-align, \
                         loop-rotation"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn o0() -> Self {
        CompileOptions {
            opt_level: OptLevel::O0,
            opt_goal: OptGoal::Balanced,
            saturation_limit: 1,
            enable_peephole: false,
            enable_nop_alignment: false,
            enable_loop_rotation: false,
            verbosity: Verbosity::Silent,
            force_frame_pointer: false,
            enable_licm: false,
            enable_dce: false,
            enable_fast_regalloc: true,
            enable_store_forwarding: false,
            enable_dse: false,
            enable_phi_removal: true,
            enable_sccp: false,
            enable_inlining: false,
            whole_program: false,
            enable_tail_calls: false,
            enable_dead_insts: false,
            max_inline_depth: 3,
            max_inline_nodes: 50,
            inline_cost_threshold: 0,
        }
    }

    pub fn o1() -> Self {
        CompileOptions {
            opt_level: OptLevel::O1,
            opt_goal: OptGoal::Balanced,
            saturation_limit: 16,
            enable_peephole: true,
            enable_nop_alignment: true,
            enable_loop_rotation: true,
            verbosity: Verbosity::Silent,
            force_frame_pointer: false,
            enable_licm: true,
            enable_dce: true,
            enable_fast_regalloc: false,
            enable_store_forwarding: true,
            enable_dse: true,
            enable_phi_removal: true,
            enable_sccp: true,
            enable_inlining: true,
            whole_program: false,
            enable_tail_calls: true,
            enable_dead_insts: true,
            max_inline_depth: 3,
            max_inline_nodes: 50,
            inline_cost_threshold: 100,
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions::o1()
    }
}

#[derive(Debug)]
pub struct CompileError {
    pub phase: String,
    pub message: String,
    pub location: Option<IrLocation>,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "phase '{}': {}", self.phase, self.message)?;
        if let Some(loc) = &self.location {
            write!(f, " (in function '{}'", loc.function)?;
            if let Some(b) = loc.block {
                write!(f, ", block {b}")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug)]
pub struct IrLocation {
    pub function: String,
    pub block: Option<u32>,
    pub inst: Option<usize>,
}

pub trait DiagnosticSink {
    fn phase_stats(&mut self, phase: &str, stats: &str);
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Returns true if any block in `func` contains a Call effectful operation.
fn func_has_calls(func: &Function) -> bool {
    func.blocks.iter().any(|b| {
        b.ops
            .iter()
            .any(|op| matches!(op, EffectfulOp::Call { .. }))
    })
}

// ── Shared egraph + extraction phases ────────────────────────────────────────

use crate::egraph::egraph::EGraph;
use crate::egraph::extract::ExtractionResult;

/// Which class names each block parameter, beside the chosen node per class.
type ExtractionOutput = (BTreeMap<(BlockId, u32), ClassId>, ExtractionResult);

/// Run the e-graph rewrite rules and cost-based extraction (phases 1-2).
///
/// Shared between `compile()` and `compile_to_ir_string()`.
pub(super) fn run_egraph_and_extract(
    func: &mut Function,
    egraph: &mut EGraph,
    opts: &CompileOptions,
) -> Result<ExtractionOutput, CompileError> {
    let egraph_opts = EGraphOptions {
        iteration_limit: opts.saturation_limit,
        max_classes: 500_000,
    };
    // Before the rules run, so a parameter the meet proves constant is a
    // constant the rules can fold through rather than an opaque value.
    if opts.enable_sccp && sccp::run_sccp(func, egraph) {
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("sccp", func, egraph);
    }
    // Before the rules run, so what they added is readable off the pair. No
    // extraction choices exist yet: nothing has a machine form to win with.
    crate::egraph::dump::dump(
        egraph,
        &func.name,
        "pre-saturation",
        &CostModel::new(opts.opt_goal),
        None,
    );
    run_phases(egraph, &egraph_opts).map_err(|e| CompileError {
        phase: "egraph".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;
    // Again, because saturation is what proves most arguments constant: an
    // argument that was `x * 0` on the way in is a constant now.
    if opts.enable_sccp && sccp::run_sccp(func, egraph) {
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("sccp", func, egraph);
    }
    crate::egraph::algebraic::apply_algebraic_rules(egraph);
    egraph.rebuild();

    // Close instruction selection over everything the rules above produced.
    // Extraction has no fallback for a class without a machine op, so this is
    // a correctness step, not part of the optimization budget.
    crate::egraph::phases::saturate_isel(egraph).map_err(|e| CompileError {
        phase: "isel".into(),
        message: e,
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    extract_from_egraph(func, egraph, opts)
}

/// Choose one node per class, and say which class names each block parameter.
///
/// Separate from the rewriting above because removing a block parameter changes
/// both answers -- the numbering the `BlockParam` nodes carry and the roots the
/// CFG names -- without changing anything the rules would rewrite. Step 2 runs
/// this again rather than re-saturating.
pub(super) fn extract_from_egraph(
    func: &Function,
    egraph: &EGraph,
    opts: &CompileOptions,
) -> Result<ExtractionOutput, CompileError> {
    let block_param_map = egraph.block_param_classes();

    let mut all_roots = collect_roots(func, egraph);
    all_roots.extend(block_param_map.values().copied());
    all_roots.sort_by_key(|c| c.0);
    all_roots.dedup();

    let cost_model = CostModel::new(opts.opt_goal);
    let extraction = extract(egraph, &all_roots, &cost_model).map_err(|e| CompileError {
        phase: "extraction".into(),
        message: e.to_string(),
        location: Some(IrLocation {
            function: func.name.clone(),
            block: None,
            inst: None,
        }),
    })?;

    // After saturation *and* isel closure, which is the one point where every
    // alternative exists side by side and extraction has said which it wants.
    crate::egraph::dump::dump(
        egraph,
        &func.name,
        "post-saturation",
        &cost_model,
        Some(&extraction.choices),
    );

    Ok((block_param_map, extraction))
}

/// Write linearization's choice of VReg for every block argument and every
/// effectful-op operand into the CFG.
///
/// From here a block argument is a VReg the terminator names, not a class every
/// consumer resolves for itself, and the same holds for the address a `Load`
/// reads. Before scheduling, so the schedules are built from what this wrote,
/// and before the splitter, whose operand rewriting is what makes a pressure
/// decision stick.
fn commit_args(
    func: &mut Function,
    egraph: &EGraph,
    lin: &linearize::Linearized,
    block_param_map: &BTreeMap<(BlockId, u32), ClassId>,
) -> Result<(), CompileError> {
    let mut rpo_pos = vec![0usize; func.blocks.len()];
    for (pos, &idx) in lin.rpo_order.iter().enumerate() {
        rpo_pos[idx] = pos;
    }
    let name = func.name.clone();
    let fail = |phase: &str| {
        let (phase, name) = (phase.to_string(), name.clone());
        move |message: String| CompileError {
            phase,
            message,
            location: Some(IrLocation {
                function: name,
                block: None,
                inst: None,
            }),
        }
    };
    // The parameters first: a terminator argument that *is* a parameter of its
    // target resolves through the VReg that block states, so the destination end
    // has to be committed before the argument end reads it.
    commit_block_param_vregs(func, &lin.block_param_vregs);
    commit_terminator_arg_vregs(
        func,
        egraph,
        &lin.class_to_vreg,
        &lin.block_snapshots,
        block_param_map,
        &rpo_pos,
    )
    .map_err(fail("terminator-args"))?;
    commit_effectful_vregs(func, egraph, &lin.block_snapshots).map_err(fail("effectful-operands"))
}

/// What the IR-level passes hand to the machine-level half of the pipeline.
pub(super) struct IrPasses {
    pub extra_roots: licm::ExtraRoots,
    pub block_param_map: BTreeMap<(BlockId, u32), ClassId>,
    pub extraction: ExtractionResult,
}

/// Run every IR-level pass: forwarding, DSE, LICM, saturation + extraction, DCE2.
///
/// Shared by `compile()` and `compile_to_ir_string()` so an `--emit-ir` dump
/// always reflects the IR a real compile would produce.
///
/// Each merging pass is followed by [`canonicalize_class_refs`], and each pass
/// boundary by a `BLITZ_VERIFY` check naming that stage.
pub(super) fn run_ir_passes(
    func: &mut Function,
    egraph: &mut EGraph,
    opts: &CompileOptions,
    sink: &mut Option<&mut dyn DiagnosticSink>,
) -> Result<IrPasses, CompileError> {
    crate::verify::verify_stage("ir-entry", func, egraph);

    // Store-to-load / load-to-load forwarding: intra-block, pre-LICM so
    // hoisting and saturation both benefit from fewer memory ops.
    if opts.enable_store_forwarding {
        let alias = AliasInfo::new();
        forward::run_forwarding(func, egraph, &alias);
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("forwarding", func, egraph);
    }

    // Dead store elimination: runs after forwarding (more forwarded loads =
    // fewer pending-store "observed by load" cancellations = more kills).
    if opts.enable_dse {
        let alias = AliasInfo::new();
        dse::run_dse(func, egraph, &alias);
        crate::verify::verify_stage("dse", func, egraph);
    }

    // LICM: detect loops, insert preheaders, identify invariant classes.
    let extra_roots = if opts.enable_licm {
        let roots = licm::run_licm(func, egraph);
        canonicalize_class_refs(func, egraph);
        crate::verify::verify_stage("licm", func, egraph);
        roots
    } else {
        Default::default()
    };

    // Phases 1-2: E-graph rewrites and cost-based extraction.
    // Extraction only reads `func`; DCE2 below needs it mutable again.
    let (block_param_map, extraction) = run_egraph_and_extract(func, egraph, opts)?;
    canonicalize_class_refs(func, egraph);
    crate::verify::verify_stage("saturation+extraction", func, egraph);

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "egraph",
            &format!(
                "classes={}, nodes={}",
                egraph.class_count(),
                egraph.node_count()
            ),
        );
        s.phase_stats(
            "extraction",
            &format!("classes_extracted={}", extraction.choices.len()),
        );
    }

    // DCE2: constant branch folding, unreachable block elimination, dead loads.
    // Must run before the caller freezes `func` and builds index-keyed structures.
    //
    // The CFG half runs at every level. It is not an optimization the level
    // buys: a block no path reaches costs the scheduler, the splitter and the
    // allocator their full price for code that cannot run, and compile time
    // here is `~(blocks * classes)^0.86`. On a 6048-line input it is 2519 of
    // 3763 blocks, which is the whole of why `-O0` used to cost 2.6x `-O1`.
    // Only the dead-load half is gated, because only that one takes a read
    // away from a value someone might want to look at.
    let extra_roots = {
        let roots =
            dce::run_dce2_with_extra_roots(func, egraph, &extraction, extra_roots, opts.enable_dce);
        crate::verify::verify_stage("dce2", func, egraph);
        roots
    };

    Ok(IrPasses {
        extra_roots,
        block_param_map,
        extraction,
    })
}

// ── compile() ────────────────────────────────────────────────────────────────

/// Order the moves that relocate incoming arguments to the registers the
/// allocator chose for them.
///
/// These are one *parallel* copy, not a sequence. Every parameter is already in
/// its ABI register before the function's first instruction runs, so a move
/// writing one parameter's destination can be reading another's source: with
/// arguments in RDI..R9, `mov rcx, rdi` destroys the fourth argument before the
/// move that reads it, and the parameter silently takes the first one's value.
/// `sequentialize_copies` orders them and breaks cycles through `SCRATCH_GPR`,
/// which is the same treatment `setup_call_args` gives the caller's side of the
/// same problem. It drops self-copies, so a parameter already in its register
/// costs nothing here.
///
/// `S64` for the same reason the call side uses it: SysV makes the caller
/// extend a sub-word argument to the full register, so a narrower move would
/// leave the high bits of a value the callee may widen.
fn emit_entry_param_copies(copies: &[(Reg, Reg)]) -> Vec<MachInst> {
    // The two lists that feed this overlap -- a parameter reached by the
    // schedule walk can also be in `unprecolored_params` -- and naming one
    // copy twice is still one copy. Removing the duplicates here is what keeps
    // `sequentialize_copies`'s repeated-destination assertion meaning what it
    // says: two *different* values would land in one register.
    let mut copies = copies.to_vec();
    copies.sort_unstable();
    copies.dedup();
    sequentialize_copies(&copies, SCRATCH_GPR)
        .into_iter()
        .map(|(src, dst)| crate::emit::phi_elim::copy_inst(src, dst, crate::x86::inst::OpSize::S64))
        .collect()
}

/// What `allocate_global` needs derived from the current schedules: the block
/// parameters it must treat as written at block entry, and coalescing's copy
/// pairs. The probe computes both, and the round that fits hands them on rather
/// than having them derived a second time over the same schedules.
type AllocInputs = (
    Vec<crate::regalloc::interference::VRegSet>,
    Vec<(VReg, VReg)>,
);

/// Compile a single function to an object file.
///
/// Consumes the `Function` (including its embedded e-graph).
pub fn compile(
    mut func: Function,
    opts: &CompileOptions,
    mut sink: Option<&mut dyn DiagnosticSink>,
) -> Result<ObjectFile, CompileError> {
    crate::trace::init_tracing();

    // Compute user stack space in 8-byte units. Each slot may be larger than
    // 8 bytes (e.g. string literal buffers), so sum actual sizes rounded up.
    let user_stack_slots: u32 = func.stack_slots.iter().map(|s| s.size.div_ceil(8)).sum();
    let mut egraph = func
        .egraph
        .take()
        .expect("Function must contain an EGraph; use FunctionBuilder::finalize()");

    let IrPasses {
        extra_roots,
        block_param_map,
        extraction,
    } = run_ir_passes(&mut func, &mut egraph, opts, &mut sink)?;

    // `func` stays mutable until linearization has chosen a VReg for every block
    // argument and `commit_terminator_arg_vregs` has written that choice into the
    // terminators; from there the CFG is read-only.

    // Must be after DCE2, which changes what the indices are.
    let block_id_to_idx = cfg::block_id_to_idx(&func);

    // Detect whether this function contains any call instructions (must be after DCE2).
    let has_calls = func_has_calls(&func);

    // Phase 3: linearize. One e-class can become several VRegs here, and every
    // later question about which register carries a class is a question about
    // what this returned.
    //
    // Runs twice when step 2 removes a block parameter: the block that loses one
    // has to name the class directly, and whether that needs the class re-emitted
    // there is linearization's decision and nobody else's.
    let mut block_param_map = block_param_map;
    let mut extraction = extraction;
    let mut lin = linearize::linearize(&func, &egraph, &extraction, &block_param_map, &extra_roots);
    commit_args(&mut func, &egraph, &lin, &block_param_map)?;

    if opts.enable_phi_removal {
        // Tier 2 first, tier 1 as the fallback, and within each a few rounds of
        // dropping whichever block the acceptance test names. The test is a
        // property of the whole removal, so without the retry one bad parameter
        // costs the function every removal the rest of it had earned.
        const RETRIES: usize = 3;
        'tiers: for tier2 in [true, false] {
            let mut removal = phi_removal::find_trivial_params(
                &func,
                &egraph,
                &lin,
                &extraction,
                &block_param_map,
                tier2,
            );
            for round in 0..=RETRIES {
                if removal.is_empty() {
                    continue 'tiers;
                }
                // Kept so the removal can be undone: whether it was legal is a
                // question only the re-linearization answers.
                let saved_func = func.clone();
                let saved_egraph = egraph.clone();

                phi_removal::apply(&mut func, &mut egraph, &removal);
                canonicalize_class_refs(&mut func, &egraph);
                crate::verify::verify_stage("phi-removal", &func, &egraph);
                // The chosen node for a parameter's class carried its old
                // position, and a removed parameter has no node at all now.
                let (new_map, new_extraction) = extract_from_egraph(&func, &egraph, opts)?;
                let new_lin =
                    linearize::linearize(&func, &egraph, &new_extraction, &new_map, &extra_roots);

                let offenders = phi_removal::barrier_offenders(&func, &new_lin);
                if offenders.is_empty() {
                    block_param_map = new_map;
                    extraction = new_extraction;
                    lin = new_lin;
                    commit_args(&mut func, &egraph, &lin, &block_param_map)?;
                    if crate::trace::is_enabled("phi") && crate::trace::fn_matches(&func.name) {
                        tracing::debug!(
                            target: "blitz::phi",
                            "[{}] tier{}: removed {} parameter(s) across {} block(s) in round {round}",
                            func.name,
                            if tier2 { "1+2" } else { "1" },
                            removal.removed,
                            removal.keep.len(),
                        );
                    }
                    break 'tiers;
                }
                func = saved_func;
                egraph = saved_egraph;
                if round == RETRIES {
                    break;
                }
                removal = removal.without(&func, &offenders);
            }
        }
    }

    let linearize::Linearized {
        mut class_to_vreg,
        mut next_vreg,
        rpo_order,
        block_vreg_insts,
        block_snapshots: block_class_to_vreg_snapshot,
        mut vreg_types,
        ..
    } = lin;

    let mut rpo_pos = vec![0usize; func.blocks.len()];
    for (pos, &idx) in rpo_order.iter().enumerate() {
        rpo_pos[idx] = pos;
    }

    // NOW freeze func for the rest of the pipeline.
    let func = &func;

    // Phase 4: Schedule per block (indexed by block index, same as block_vreg_insts).
    let mut block_schedules: Vec<Vec<ScheduledInst>> = vec![Vec::new(); func.blocks.len()];
    let mut total_insts = 0usize;
    for (block_idx, insts) in block_vreg_insts.iter().enumerate() {
        let dag = ScheduleDag::build(insts);
        let sched = schedule(&dag);
        total_insts += sched.len();
        block_schedules[block_idx] = sched;
    }

    if let Some(s) = sink.as_mut() {
        s.phase_stats("schedule", &format!("insts={total_insts}"));
    }
    // Every VReg a schedule names must have a type.
    //
    // A missing entry is not a pessimism: `lower.rs`'s `result_size` falls back
    // to `OpSize::S64`, which turned a flags-only 32-bit compare into
    // `cmp r8, rdi` against a zero-extended `mov edi, -2`, so `14 < -2` was true
    // (`9207141`). The cause was a class re-emitted in a later block, whose VReg
    // the function-wide map does not keep. A `debug_assert` rather than a gate:
    // the `checked` profile carries it through every harness already, and the
    // rule is that the gate set stays fixed and invariants go inside it.
    #[cfg(debug_assertions)]
    for (block_idx, sched) in block_schedules.iter().enumerate() {
        for inst in sched {
            if !inst.op.has_no_result() {
                debug_assert!(
                    vreg_types.contains_key(&inst.dst),
                    "{}: block {} defines {:?} with op {:?} and no type; \
                     lowering would size it S64 by default",
                    func.name,
                    block_idx,
                    inst.dst,
                    inst.op,
                );
            }
            for operand in &inst.operands {
                debug_assert!(
                    vreg_types.contains_key(operand),
                    "{}: block {} reads {:?} with no type (op {:?})",
                    func.name,
                    block_idx,
                    operand,
                    inst.op,
                );
            }
        }
    }

    // Phase 4b: Reorder each block's schedule to respect effectful op barriers.
    //
    // Effectful ops (loads, stores, calls) impose ordering constraints on pure ops:
    // pure ops that consume a LoadResult must come after the corresponding Load,
    // and pure ops that are inputs to a Call must come before the Call. The
    // scheduler doesn't know about effectful ops, so we reorder the schedule
    // here so the regalloc sees correct liveness.
    for (block_idx, block) in func.blocks.iter().enumerate() {
        let has_branch = block
            .ops
            .last()
            .is_some_and(|op| matches!(op, EffectfulOp::Branch { .. }));

        if block.non_term_count() == 0 && !has_branch {
            continue; // No effectful ops and no branch to constrain ordering.
        }

        let (vreg_to_result_of_barrier, vreg_to_arg_of_barrier) = if block.non_term_count() > 0 {
            build_barrier_context(
                block,
                &egraph,
                &block_class_to_vreg_snapshot[block_idx],
                &block_schedules[block_idx],
            )
        } else {
            (BTreeMap::new(), BTreeMap::new())
        };

        let sched = &block_schedules[block_idx];
        let vreg_group =
            assign_barrier_groups(sched, &vreg_to_result_of_barrier, &vreg_to_arg_of_barrier);

        // Identify the branch condition's flags-producing instruction so it
        // sorts to the end of its barrier group. On x86, any ALU instruction
        // clobbers EFLAGS, so the flags-producing instruction must be the
        // last ALU op before the terminator. We only move the immediate
        // flags chain (proj1 + its parent ALU op), not the full transitive
        // operand tree, to avoid disrupting scheduling of shared operands.
        let mut branch_cond_chain: BTreeSet<VReg> = BTreeSet::new();
        if let Some(EffectfulOp::Branch { cond, .. }) = block.ops.last() {
            // The VReg the CFG states. This asked the function-wide map at the
            // block's exit while `mark_branch_cond_barrier` asked the block's own
            // snapshot at its entry -- two derivations of one answer, and a
            // flags-typed class is re-emitted in every block that names it, so
            // the function-wide one holds whichever block emitted it first.
            if let Some(vreg) = cond.vreg() {
                // Add the flags VReg (proj1).
                branch_cond_chain.insert(vreg);
                // Find the instruction that produces it and add its parent
                // (the ALU op that sets EFLAGS, e.g. x86_sub).
                for inst in sched {
                    if inst.dst == vreg {
                        if matches!(inst.op, Op::Pure(PureOp::Proj1)) {
                            for &op in &inst.operands {
                                branch_cond_chain.insert(op);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Stable-sort by group to reorder while preserving within-group order.
        // Barrier results (LoadResult/CallResult) sort to the FRONT of their
        // group: their values are produced by effectful ops at the group
        // boundary, so the register is occupied from the start of the group.
        // Placing them after pure ops would let the regalloc think the register
        // is free, causing incorrect reuse and clobbering.
        // Branch condition chain sorts to the END of its group to prevent
        // other ALU ops from clobbering EFLAGS between the flags-producing
        // instruction and the branch terminator.
        let mut indexed: Vec<(usize, &ScheduledInst)> = sched.iter().enumerate().collect();
        indexed.sort_by_key(|(orig_idx, inst)| {
            let g = *vreg_group.get(&inst.dst).unwrap_or(&0);
            let param_order: u8 = match inst.op {
                Op::Pure(PureOp::Param(_, _)) => 0,
                Op::Pseudo(PseudoOp::LoadResult(_, _)) | Op::Pseudo(PseudoOp::CallResult(_, _)) => {
                    1
                }
                // Spill reloads must happen early in their consumer group,
                // BEFORE any op that uses the reloaded value. Pushing the
                // SpillLoad's orig_idx to the end of the block (via barrier.rs)
                // would otherwise place it after its consumer under the
                // default param_order=2 tier. param_order=1 places it right
                // after the group's barrier result and before pure ops.
                Op::Pseudo(PseudoOp::SpillLoad(_)) | Op::Pseudo(PseudoOp::XmmSpillLoad(_)) => 1,
                _ if branch_cond_chain.contains(&inst.dst) => 3,
                _ => 2,
            };
            (g, param_order, *orig_idx)
        });
        let reordered: Vec<ScheduledInst> =
            indexed.into_iter().map(|(_, inst)| inst.clone()).collect();
        block_schedules[block_idx] = reordered;

        if crate::trace::is_enabled("sched") && crate::trace::fn_matches(&func.name) {
            tracing::debug!(
                target: "blitz::sched",
                "[{}] block {block_idx} after barrier sort:\n{}",
                func.name,
                crate::trace::format_schedule(&block_schedules[block_idx], Some(&vreg_group)),
            );
        }
    }

    // Hoist Param and BlockParam ops to the front of their block's schedule.
    //
    // Neither computes anything: the value is already in its register when the
    // block starts, put there by the ABI or by a predecessor's phi copy. Their
    // pseudo-ops are still where scheduling left them, though, and liveness reads
    // a def position as the start of a live range -- so a block param whose
    // pseudo-op the backward pass pulled down next to its use looked dead over
    // the earlier part of its own block, and the allocator handed its register to
    // something defined there:
    //
    //   movsd xmm3,xmm4     ; x + y, into the register holding param 4
    //   ...
    //   addsd xmm1,xmm3     ; wanted param 4, reads x + y
    //
    // Hoisting makes the def position match where the value actually arrives.
    for sched in block_schedules.iter_mut() {
        sched.sort_by_key(|inst| {
            if matches!(
                inst.op,
                Op::Pure(PureOp::Param(_, _)) | Op::Pure(PureOp::BlockParam(_, _, _))
            ) {
                0u8
            } else {
                1
            }
        });
    }

    // A Flags value cannot be spilled, so it is re-emitted wherever something
    // has written EFLAGS since it was computed. Before allocation, so the
    // operands the re-emitted compare reads have their live ranges extended in
    // the graph the allocator colours.
    {
        let flags_classes = crate::regalloc::build_vreg_classes_from_all_blocks(&block_schedules);
        let mut next = block_schedules
            .iter()
            .flatten()
            .flat_map(|i| std::iter::once(i.dst.0).chain(i.operands.iter().map(|v| v.0)))
            .max()
            .map_or(0, |m| m + 1);
        let n = flags_remat::remat_flags(&mut block_schedules, &flags_classes, &mut next);
        if n > 0 && crate::trace::is_enabled("sched") && crate::trace::fn_matches(&func.name) {
            eprintln!(
                "[sched] {}: re-emitted {n} comparison(s) for stale flags",
                func.name
            );
        }
        // The pass proving its own result. EFLAGS is in no register file, so
        // nothing else in the verifier can see a consumer reading the wrong
        // flags -- they are written on every path to it, just by the wrong
        // instruction.
        if crate::verify::level() != crate::verify::VerifyLevel::Off {
            let errs = crate::verify::verify_flags_liveness(&block_schedules, &flags_classes);
            assert!(
                errs.is_empty(),
                "flags liveness broken in '{}' after flags_remat:\n  - {}",
                func.name,
                errs.join("\n  - ")
            );
        }
    }

    // Phase 5: Register allocation -- per-block with cross-block live range splitting.
    //
    // Single-block fast path: skip global liveness and run allocate() directly.
    // Multi-block path: compute global liveness, assign cross-block spill slots,
    // rewrite each block to insert spill/reload code at boundaries, then run
    // allocate() per block and merge results.
    let entry_has_calls = func.blocks[0]
        .ops
        .iter()
        .any(|op| matches!(op, EffectfulOp::Call { .. }));
    let func_arg_locs = crate::x86::abi::assign_args(&func.param_types);
    // Call arguments that travel on the stack: `setup_call_args` pushes them, and
    // a push can read memory, so neither allocator has to find a register for one.
    let stack_arg_vregs = precolor::stack_arg_vregs(func);
    let param_vregs = assign_param_vregs_from_map(
        func,
        &class_to_vreg,
        &egraph,
        entry_has_calls,
        &func_arg_locs,
    );

    // Compute loop depths from the CFG for spill selection.
    let loop_depths = compute_loop_depths(func, &block_schedules);

    // Block params the splitter routed through a stack slot.
    // Passed to lower_terminator so predecessor terminators emit slot stores.
    let mut slot_spilled_params: crate::compile::split::BlockParamSlotMap =
        std::collections::BTreeMap::new();
    // End-of-block SpillLoad VRegs inserted by cross-block spills. The lowering
    // pass forces these into the trailing barrier group so they execute after all
    // calls in their block (preventing loads from uninitialized stack slots).

    // The block parameters, kept for `verify::verify_register_sharing` after
    // allocation: a parameter is written by the phi copy at the edge, so it is
    // not live at a predecessor's exit unless the predecessor's terminator passes
    // it. The spill loop never renames one, so this copy stays current where a
    // copy of the terminator uses would not.
    let verify_block_params: Vec<crate::regalloc::interference::VRegSet>;
    // And the copy pairs, so the check exempts the same phi-related VRegs
    // coalescing was allowed to merge.
    let verify_copy_pairs: Vec<(VReg, VReg)>;

    // Every pass that spills draws its slot numbers from here, so no two name the
    // same 8-byte cell of the frame and each slot can say which pass owns it.
    let mut slots = SlotAllocator::new();

    // One allocator for every function: a single-block function is a special
    // case of the general one, not a separate algorithm.
    let (regalloc_result, block_rewritten, coalesce_aliases) = {
        // Step 1: Compute CFG successors. Terminator uses come from the
        // `Op::TerminatorArgs` operands once those exist, below.
        let cfg_succs = cfg::successor_indices(func);

        // `add_call_precolors_for_block` reads `EffectfulOp::Call`'s args
        // positionally to pin each to its ABI register, so it needs the arg list
        // as the CFG states it.
        let mut call_arg_precolors: Vec<(VReg, Reg)> = Vec::new();
        for block in func.blocks.iter() {
            let mut dummy_live_out: std::collections::BTreeSet<VReg> =
                std::collections::BTreeSet::new();
            add_call_precolors_for_block(block, &mut call_arg_precolors, &mut dummy_live_out);
        }

        // Shorten a barrier result's live range where its consumer is two or more
        // barrier groups away, by spilling it at the def and reloading it at the
        // use. First of the three passes that spill in this function, all drawing
        // their slots from `slots`.
        {
            let mut early_next_vreg = next_vreg;
            for (block_idx, block) in func.blocks.iter().enumerate() {
                let non_term_count = block.non_term_count();
                if non_term_count > 0 {
                    let (result_map, arg_map) = build_barrier_context(
                        block,
                        &egraph,
                        &block_class_to_vreg_snapshot[block_idx],
                        &block_schedules[block_idx],
                    );
                    let mut vreg_group =
                        assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                    insert_early_barrier_spills(
                        &mut block_schedules[block_idx],
                        &result_map,
                        &arg_map,
                        &mut vreg_group,
                        &vreg_types,
                        &mut early_next_vreg,
                        &mut slots,
                    );
                }
            }
            next_vreg = early_next_vreg;
        }

        // Populate effectful-op operands onto barrier instructions (LoadResult,
        // CallResult, StoreBarrier, VoidCallBarrier) in each block's schedule
        // BEFORE global liveness, so compute_global_liveness sees them as regular
        // instruction operands and includes them in cross-block liveness.
        // This MUST happen AFTER call_arg_precolors collection.
        //
        // Resolved through each block's own snapshot, not the global map. A class
        // re-emitted per block has one VReg per block and the global map holds
        // whichever was restored last, so an `if`/`else` pair that both call
        // `printf("%d\n", ...)` would record the other arm's VReg for the format
        // string -- and since Phase 7 now trusts these operands as the record of
        // what each op reads, that names a register holding something else.
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let non_term_count = block.non_term_count();
            if non_term_count > 0 {
                let non_term_ops = &block.ops[..non_term_count];
                let block_map = &block_class_to_vreg_snapshot[block_idx];
                let (result_map, arg_map) =
                    build_barrier_context(block, &egraph, block_map, &block_schedules[block_idx]);
                let mut vreg_group =
                    assign_barrier_groups(&block_schedules[block_idx], &result_map, &arg_map);
                populate_effectful_operands(
                    &mut block_schedules[block_idx],
                    non_term_ops,
                    &mut vreg_group,
                    &mut next_vreg,
                );
            }

            // The terminator's arguments, as operands, for the same reason the
            // barrier ops carry theirs: so the splitter and coalescing rewrite
            // them instead of three passes re-deriving them from a map those
            // passes mutate. A copy of what the CFG names, not a second
            // resolution of it.
            if let Some(term) = block.ops.last() {
                crate::compile::barrier::append_terminator_args(
                    &mut block_schedules[block_idx],
                    term,
                    &mut next_vreg,
                )
                .map_err(|message| CompileError {
                    phase: "terminator-args".into(),
                    message,
                    location: Some(IrLocation {
                        function: func.name.clone(),
                        block: Some(block.id),
                        inst: None,
                    }),
                })?;
            }
        }

        // Terminator uses, read straight off the schedules. `Op::TerminatorArgs`
        // is the record of what each terminator consumes: the splitter rewrites
        // its operands and coalescing renames them, so recomputing this after
        // either pass gives that pass's answer rather than a second, independent
        // derivation that can disagree with it.
        let mut phi_uses = barrier::terminator_uses(&block_schedules);

        // The dominator tree of the schedules' own CFG. Every round of the
        // splitter asks dominance questions, and so does the phi-copy routing
        // below, but neither changes `func.blocks` -- the splitter rewrites
        // schedules. One derivation serves all of them.
        let split_dom = {
            let rpo = cfg::compute_rpo(func);
            cfg::DomOrder::new(&cfg::compute_idom(func, &rpo))
        };

        // Pressure-driven splitter.
        // CRITICAL ORDER: apply_plan_to must run BEFORE collect_block_param_vregs_per_block.
        //
        // The loop stops when the *allocator* fits, not when the splitter runs
        // out of ideas.
        //
        // From the second round on, each iteration colours first with the
        // allocator's own spill loop switched off. A round that fits ends the
        // loop and its result is the allocation. This is what stops the splitter
        // relieving pressure the colouring did not need relieved: one round
        // lowers pressure at the points it looked at, and the reloads it inserts
        // are live somewhere themselves, so the next round finds more to do --
        // for as long as anyone keeps asking it.
        //
        // The probe starts at round 1 because round 0 cannot tell anyone
        // anything. On a chordal interference graph the chromatic number is the
        // maximum clique, so a plan that is non-empty means pressure already
        // exceeds the budget and the colouring cannot fit; a plan that is empty
        // ends the loop without splitting anything. Either way round 0's
        // colouring is a full allocation spent on a foregone conclusion, and
        // skipping it is worth more than the rounds it would have saved.
        //
        // The probe keeps the allocator's spill loop as the last resort rather
        // than the first: the splitter places a value better, so relief is asked
        // of it while it still has something to offer, and only a loop that ends
        // with no fitting round falls through to spilling inside the allocator.
        //
        // Commit an empty plan so the class map records that the splitter has
        // been given its chance. `collect_block_param_vregs_per_block` asserts
        // on `split_generation > 0` to catch a caller reading block parameters
        // between planning and committing, where the plan's truncations are
        // decided but not yet in the map. Reading them before anything is
        // planned is a different thing and is what the probe does.
        split::apply_plan_to(
            &mut block_schedules,
            &mut class_to_vreg,
            &mut next_vreg,
            split::SplitPlan::default(),
        );

        let mut probed: Option<crate::regalloc::GlobalRegAllocResult> = None;
        let mut probed_inputs: Option<AllocInputs> = None;
        // An allocator that puts every value in a slot has no pressure for the
        // splitter to relieve: nothing is live across an instruction, so the
        // only shape it cannot place is a single instruction reading more
        // operands than the machine has registers, which no split can help.
        let split_rounds = if opts.enable_fast_regalloc {
            0
        } else {
            split::MAX_SPLIT_ROUNDS
        };
        for _round in 0..split_rounds {
            use crate::regalloc::coloring::{AVAILABLE_XMM_COLORS, available_gpr_colors};

            if _round > 0 && !opts.enable_fast_regalloc {
                let bp = collect_alloc_block_params(
                    func,
                    &egraph,
                    &block_param_map,
                    &class_to_vreg,
                    &slot_spilled_params,
                    &block_id_to_idx,
                );
                let mut cp = compute_copy_pairs_from_schedules(
                    func,
                    &block_schedules,
                    &egraph,
                    &class_to_vreg,
                    &block_param_map,
                );
                cp.extend(projection_copy_pairs(&block_schedules));
                if let Ok(result) = allocate_global(
                    &block_schedules,
                    &param_vregs,
                    call_arg_precolors.clone(),
                    &cp,
                    &loop_depths,
                    &cfg_succs,
                    &bp,
                    &func.name,
                    opts.force_frame_pointer,
                    &func_arg_locs,
                    &stack_arg_vregs,
                    &mut slots,
                    0,
                ) {
                    probed = Some(result);
                    probed_inputs = Some((bp, cp));
                    break;
                }
            }

            let gpr_budget = available_gpr_colors(opts.force_frame_pointer);
            let xmm_budget = AVAILABLE_XMM_COLORS;

            // Compute global liveness to seed the splitter's per-block backward scans.
            let split_global_liveness = crate::regalloc::global_liveness::compute_global_liveness(
                &block_schedules,
                &cfg_succs,
                &phi_uses,
            );

            let split_cost_model = CostModel::new(opts.opt_goal);
            let plan = split::plan_splits(
                &block_schedules,
                &class_to_vreg,
                &extraction,
                &egraph,
                &split_cost_model,
                &split_global_liveness,
                gpr_budget,
                xmm_budget,
                next_vreg,
                &mut slots,
                &loop_depths,
                func,
                &split_dom,
                &block_param_map,
                &slot_spilled_params,
                &func_arg_locs,
            );
            // An empty plan still goes through `apply_plan_to`: that is what
            // bumps `split_generation`, and `collect_block_param_vregs_per_block`
            // asserts the splitter has committed at least once.
            let converged = plan.is_empty();
            // Accumulate the maps across rounds. The plan keeps its
            // `slot_spilled_params`: `apply_plan_to` reads them to truncate each
            // param's segment past block entry, which is what stops a register
            // being allocated to a value that lives in a slot.
            slot_spilled_params.extend(plan.slot_spilled_params.clone());

            // Build old→new VReg remap restricted to CALL-ARG positions so we
            // can update call_arg_precolors after apply_plan_to. The precolors
            // were collected before the splitter ran; if the splitter rewrites a
            // call-arg VReg to a reload VReg, the precolor must follow the reload.
            //
            // Restriction to call-arg positions avoids using the wrong reload VReg
            // when the same old VReg is rewritten in multiple blocks (non-call uses
            // create reload VRegs that should NOT inherit the ABI precolor).
            let call_arg_vreg_set: std::collections::BTreeSet<VReg> =
                call_arg_precolors.iter().map(|(v, _)| *v).collect();
            let mut vreg_remap: BTreeMap<VReg, VReg> = BTreeMap::new();
            for &(bi, ii, oi, new_vreg) in &plan.operand_rewrites {
                // Only update if the old VReg is a call-arg precolor candidate.
                if let Some(old_vreg) = block_schedules
                    .get(bi)
                    .and_then(|s| s.get(ii))
                    .and_then(|inst| inst.operands.get(oi))
                    .copied()
                    && call_arg_vreg_set.contains(&old_vreg)
                {
                    // Only keep first entry (the call-site reload VReg; later
                    // entries for the same old VReg are non-call-site reloads).
                    vreg_remap.entry(old_vreg).or_insert(new_vreg);
                }
            }

            split::apply_plan_to(
                &mut block_schedules,
                &mut class_to_vreg,
                &mut next_vreg,
                plan,
            );

            if crate::trace::is_enabled("split") && crate::trace::fn_matches(&func.name) {
                for (block_idx, sched) in block_schedules.iter().enumerate() {
                    tracing::debug!(
                        target: "blitz::split",
                        "[{}] block {block_idx} after split plan:\n{}",
                        func.name,
                        crate::trace::format_schedule(sched, None),
                    );
                }
            }

            // Update call_arg_precolors: transfer each precolor to its reload VReg.
            if !vreg_remap.is_empty() {
                for (precolor_vreg, _reg) in call_arg_precolors.iter_mut() {
                    if let Some(&new_vreg) = vreg_remap.get(precolor_vreg) {
                        *precolor_vreg = new_vreg;
                    }
                }
            }

            // An argument whose destination parameter lives in a slot leaves the
            // terminator's operand list, and becomes its own store unless the slot
            // already holds its value.
            //
            // Leaving it as an operand defeats the routing: the operand list is
            // the parallel copy, which needs every argument readable at one point,
            // so sixteen slot-bound arguments still ask for sixteen registers at
            // once and the clique comes back one instruction later. A store to a
            // slot clobbers no other argument's register, so these go one at a
            // time, and as ordinary instructions with one operand each that is
            // exactly what liveness then sees.
            //
            // Decided per argument, by the destination it feeds -- never by
            // "this VReg belongs to some routed parameter". An edge can pass one
            // routed parameter's VReg to a different parameter, and dropping that
            // operand for the wrong reason left the destination's slot unwritten
            // while every use reloaded from it.
            //
            // `build_phi_copies` needs no matching change: an argument index with
            // no operand is already the signal that the value travels through a
            // slot and the store is already in this block.
            // A back edge is one whose target dominates its source, which is what
            // says the slot was filled before the loop began.
            let compile_dominates = |a: usize, b: usize| split_dom.dominates(a, b);
            for (block_idx, block) in func.blocks.iter().enumerate() {
                let terminator = block.ops.last().expect("block must have terminator");
                let dests = barrier::terminator_arg_destinations(terminator);
                // `(arg VReg, destination info)`, and whether a store is needed:
                // an argument that IS its destination parameter has nothing to
                // store, the slot holds that value already.
                let routed: Vec<(u32, VReg, (BlockId, u32), split::SlotSpilledParamInfo)> =
                    barrier::terminator_arg_operands(&block_schedules[block_idx])
                        .into_iter()
                        .filter_map(|(arg_idx, vreg)| {
                            let &(target, pidx) = dests.get(arg_idx as usize)?;
                            let info = slot_spilled_params.get(&(target, pidx))?.clone();
                            Some((arg_idx, vreg, (target, pidx), info))
                        })
                        .collect();
                if routed.is_empty() {
                    continue;
                }
                // Which edge needs no store: a BACK edge whose argument is the
                // parameter itself, because the forward edge into that block
                // filled the slot before the loop began. Identity alone is not
                // the question -- an argument equal to its parameter on a forward
                // edge is the one edge that must store, and reading identity as
                // "already filled" leaves the slot unwritten while every use
                // reloads it. Nor is identity by VReg: two values share a VReg
                // whenever linearization had no reason to give them separate
                // storage, so the class is what says whether this argument IS the
                // parameter.
                let arg_classes = barrier::terminator_arg_classes(terminator);
                let stores: Vec<(VReg, split::SlotSpilledParamInfo)> = routed
                    .iter()
                    .filter(|(arg_idx, vreg, (dest_bid, dest_pidx), info)| {
                        let dest_idx = block_id_to_idx[dest_bid];
                        let back_edge = compile_dominates(dest_idx, block_idx);
                        let is_param = arg_classes
                            .get(*arg_idx as usize)
                            .copied()
                            .zip(block_param_map.get(&(*dest_bid, *dest_pidx)).copied())
                            .is_some_and(|(arg_class, param_class)| {
                                egraph.unionfind.find_immutable(arg_class)
                                    == egraph.unionfind.find_immutable(param_class)
                            })
                            || *vreg == info.vreg
                            || func.blocks[dest_idx]
                                .param_vreg(*dest_pidx)
                                .is_some_and(|own| *vreg == own);
                        !(back_edge && is_param)
                    })
                    .map(|(_, vreg, _, info)| (*vreg, info.clone()))
                    .collect();
                let schedule = &mut block_schedules[block_idx];
                let drop_args: BTreeSet<u32> = routed.iter().map(|(idx, _, _, _)| *idx).collect();
                barrier::remove_terminator_arg_operands(schedule, &drop_args);
                if stores.is_empty() {
                    continue;
                }

                // Each store goes immediately after the last point its value is
                // needed for anything else -- its definition, or a later use.
                // Putting them all at the end instead relieves nothing: the value
                // is live until the instruction that reads it, so sixteen stores
                // at the end keep sixteen registers occupied until the end,
                // exactly as the terminator operands did.
                let last_needed = |vreg: VReg, slot: i64, sched: &[ScheduledInst]| -> usize {
                    let mut at = sched
                        .iter()
                        .enumerate()
                        .filter(|(_, inst)| {
                            !matches!(inst.op, Op::Pseudo(PseudoOp::TerminatorArgs(_)))
                                && (inst.dst == vreg || inst.operands.contains(&vreg))
                        })
                        .map(|(i, _)| i + 1)
                        .max()
                        .unwrap_or(0);
                    // And after the block's own reads of that slot. The store
                    // carries the value the NEXT iteration receives; a reload
                    // standing after it reads this iteration's parameter, and
                    // overwriting the cell first hands it the new value one
                    // iteration early.
                    at = at.max(
                        sched
                            .iter()
                            .enumerate()
                            .filter(|(_, inst)| {
                                matches!(inst.op, Op::Pseudo(PseudoOp::SpillLoad(s)) | Op::Pseudo(PseudoOp::XmmSpillLoad(s)) if s == slot)
                            })
                            .map(|(i, _)| i + 1)
                            .max()
                            .unwrap_or(0),
                    );
                    // A projection reads a register its producer still owns, so
                    // nothing goes between a division and the projections taking
                    // its quotient and remainder out of RAX and RDX.
                    while sched.get(at).is_some_and(|inst| {
                        matches!(inst.op, Op::Pure(PureOp::Proj0) | Op::Pure(PureOp::Proj1))
                    }) {
                        at += 1;
                    }
                    at
                };
                let mut planned: Vec<(usize, ScheduledInst)> = stores
                    .into_iter()
                    .map(|(vreg, info)| {
                        let store = ScheduledInst {
                            op: match info.reg_class {
                                crate::x86::reg::RegClass::XMM => {
                                    Op::Pseudo(PseudoOp::XmmSpillStore(info.slot))
                                }
                                crate::x86::reg::RegClass::GPR => {
                                    Op::Pseudo(PseudoOp::SpillStore(info.slot))
                                }
                                crate::x86::reg::RegClass::Flags => {
                                    unreachable!("flags cannot be spilled to a slot")
                                }
                            },
                            dst: VReg(next_vreg),
                            operands: vec![vreg],
                        };
                        next_vreg += 1;
                        (last_needed(vreg, info.slot, schedule), store)
                    })
                    .collect();
                // Descending, so an insertion never moves a position not yet used.
                planned.sort_by_key(|p| Reverse(p.0));
                for (at, store) in planned {
                    schedule.insert(at.min(schedule.len()), store);
                }
            }

            // The plan rewrote terminator operands to its reload VRegs, and the
            // slot-routed ones are gone, so re-reading the schedules is what
            // makes a spilled value stop being live out.
            phi_uses = barrier::terminator_uses(&block_schedules);

            if converged {
                break;
            }
        }

        // Step 3: Determine block params per block (passed to allocate_global).
        // CRITICAL ORDER: must run AFTER apply_plan_to (splitter output committed).
        //
        // Reuse what the probe already computed when it fitted: recomputing them
        // over the same schedules answers the same question twice.
        let (probe_bp, probe_cp) = match probed_inputs {
            Some((bp, cp)) => (Some(bp), Some(cp)),
            None => (None, None),
        };
        let block_param_vregs_per_block = probe_bp.unwrap_or_else(|| {
            collect_alloc_block_params(
                func,
                &egraph,
                &block_param_map,
                &class_to_vreg,
                &slot_spilled_params,
                &block_id_to_idx,
            )
        });

        // Coalescing's copy pairs, read off the final schedules. Resolving each
        // argument's class through the function-wide map answers a per-block
        // question with whichever VReg the class was last given: a pure class is
        // re-emitted in every block that needs it, so the answer can be a VReg
        // defined in a block the edge never reaches, and merging the destination
        // parameter onto that VReg hands the parameter a register chosen for an
        // unrelated value.
        //
        // CRITICAL ORDER: after the splitter, so an argument it routed through a
        // stack slot -- which has no operand and no copy -- contributes no pair.
        let copy_pairs = probe_cp.unwrap_or_else(|| {
            let mut pairs = compute_copy_pairs_from_schedules(
                func,
                &block_schedules,
                &egraph,
                &class_to_vreg,
                &block_param_map,
            );
            // A `Proj0` is a copy out of the pair VReg, and it is the largest
            // group of copies phi pairs alone do not reach.
            pairs.extend(projection_copy_pairs(&block_schedules));
            pairs
        });

        verify_block_params = block_param_vregs_per_block.clone();
        verify_copy_pairs = copy_pairs.clone();
        let global_result = if let Some(result) = probed {
            Ok(result)
        } else if opts.enable_fast_regalloc {
            crate::regalloc::fast::allocate_fast(
                &block_schedules,
                &param_vregs,
                call_arg_precolors,
                &cfg_succs,
                &block_param_vregs_per_block,
                &func.name,
                opts.force_frame_pointer,
                &func_arg_locs,
                &stack_arg_vregs,
                &mut slots,
                &mut vreg_types,
            )
        } else {
            allocate_global(
                &block_schedules,
                &param_vregs,
                call_arg_precolors,
                &copy_pairs,
                &loop_depths,
                &cfg_succs,
                &block_param_vregs_per_block,
                &func.name,
                opts.force_frame_pointer,
                &func_arg_locs,
                &stack_arg_vregs,
                &mut slots,
                crate::regalloc::MAX_GLOBAL_SPILL_ROUNDS,
            )
        }
        .map_err(|e| CompileError {
            phase: "regalloc".into(),
            message: e,
            location: Some(IrLocation {
                function: func.name.clone(),
                block: None,
                inst: None,
            }),
        })?;

        let block_rewritten_storage = global_result.per_block_insts;
        let merged_assignment = global_result.assignment;
        let mut merged_callee_saved = global_result.callee_saved_used;
        let global_unprecolored_params = global_result.unprecolored_params;
        let coalesce_aliases: BTreeMap<VReg, VReg> = global_result.coalesce_aliases;

        // Every slot in these schedules came from `slots`, whichever pass spilled
        // to it, so the frame reserves exactly what it handed out.
        let spill_slot_counter = slots.count();

        merged_callee_saved.sort_by_key(|r| *r as u8);
        merged_callee_saved.dedup();

        let merged_result = RegAllocResult {
            assignment: merged_assignment,
            spill_slots: spill_slot_counter,
            callee_saved_used: merged_callee_saved,
            insts: vec![],
            unprecolored_params: global_unprecolored_params,
        };

        (merged_result, block_rewritten_storage, coalesce_aliases)
    };

    // A spill op naming a slot no pass allocated addresses a frame cell the
    // prologue does not reserve, or one another value owns. Nothing downstream can
    // see it: the displacement is well-formed and the store writes it, so both the
    // machine verifier and the slot-traffic dump read it as an ordinary slot.
    if cfg!(debug_assertions) {
        let unowned: BTreeSet<i64> = block_rewritten
            .iter()
            .flatten()
            .filter_map(|inst| match inst.op {
                Op::Pseudo(PseudoOp::SpillStore(slot))
                | Op::Pseudo(PseudoOp::SpillLoad(slot))
                | Op::Pseudo(PseudoOp::XmmSpillStore(slot))
                | Op::Pseudo(PseudoOp::XmmSpillLoad(slot)) => {
                    slots.owner(slot as u32).is_none().then_some(slot)
                }
                _ => None,
            })
            .collect();
        assert!(
            unowned.is_empty(),
            "[{}] spill ops name slots {unowned:?}, outside the {} this function allocated",
            func.name,
            slots.count(),
        );
    }

    if let Some(s) = sink.as_mut() {
        s.phase_stats(
            "regalloc",
            &format!(
                "regs_used={}, spill_slots={}",
                regalloc_result.assignment.len(),
                regalloc_result.spill_slots
            ),
        );
    }

    if crate::trace::is_enabled("regalloc") && crate::trace::fn_matches(&func.name) {
        tracing::debug!(
            target: "blitz::regalloc",
            "[{}] final assignment ({} vregs, {} spill slots):\n{}",
            func.name,
            regalloc_result.assignment.len(),
            regalloc_result.spill_slots,
            crate::trace::format_assignment(&regalloc_result.assignment),
        );
    }

    // Two VRegs live at once must not share a register. Checked here rather than
    // at machine level because it needs the VRegs, which the rewrite ahead
    // erases, and liveness recomputed from the schedules as emitted -- asking the
    // allocator's own interference graph would be circular.
    if crate::verify::is_enabled() && func.blocks.len() > 1 {
        let succs = cfg::successor_indices(func);
        let errors = crate::verify::verify_register_sharing(
            &block_rewritten,
            &verify_block_params,
            &succs,
            &regalloc_result.assignment,
            &coalesce_aliases,
            &verify_copy_pairs,
        );
        if !errors.is_empty() {
            panic!(
                "BLITZ_VERIFY: {} register-sharing violation(s) in function '{}' after \
                 register allocation:\n  - {}",
                errors.len(),
                func.name,
                errors.join("\n  - ")
            );
        }
    }

    // Build the set of param VRegs so lowering can skip their Iconst sentinels.
    let param_vreg_set: BTreeSet<VReg> = param_vregs.iter().map(|(v, _)| *v).collect();

    // Compute frame layout early so spill lowering can use it during Phase 7.
    let frame_layout = compute_frame_layout(
        regalloc_result.spill_slots,
        &regalloc_result.callee_saved_used,
        0,
        has_calls,
        opts.force_frame_pointer,
        user_stack_slots,
    );

    // Phase 7: Per-block MachInst lowering + phi elimination + terminator emission.
    //
    // Emitted in layout order, not RPO: RPO answers which block dominates
    // which, and what should *follow* a block in memory is a different
    // question -- see `cfg::block_layout_order`. Everything before this point
    // reads the RPO, and nothing here needs one: every per-block structure is
    // keyed by block index, and the only thing a position means here is which
    // block a fallthrough reaches.
    //
    // LabelIds are block IDs (block.id), which are stable across reordering.
    let layout_order = cfg::block_layout_order(func);
    let n_blocks = func.blocks.len();
    // Extra labels for trampoline code start after the maximum block id + 1.
    let max_block_id = func.blocks.iter().map(|b| b.id).max().unwrap_or(0);
    let mut next_label: LabelId = max_block_id + 1;
    // block_items[i] holds the items for the block at layout_order[i].
    let mut block_items: Vec<Vec<BlockItem>> = Vec::with_capacity(n_blocks);

    for (layout_pos, &block_idx) in layout_order.iter().enumerate() {
        let block = &func.blocks[block_idx];
        // Strip barrier pseudo-ops before Phase 7 grouping: their dummy dst VRegs
        // are not in barrier maps and would be misrouted to group 0.
        let rewritten: Vec<ScheduledInst> = block_rewritten[block_idx]
            .iter()
            .filter(|inst| {
                !matches!(
                    inst.op,
                    Op::Pseudo(PseudoOp::StoreBarrier) | Op::Pseudo(PseudoOp::VoidCallBarrier)
                )
            })
            .cloned()
            .collect();
        let rewritten = &rewritten;
        // Retain the un-stripped schedule for effectful-op lowering: the
        // StoreBarrier/VoidCallBarrier pseudo-ops carry the (post-spill)
        // operand renames that resolve_*_regs_after_spilling uses to find
        // reload/remat VRegs for Store val and Call args.
        let full_schedule_for_barriers = &block_rewritten[block_idx];

        // The block that follows this one in emission order (for fallthrough).
        let next_block_id: Option<BlockId> = layout_order
            .get(layout_pos + 1)
            .map(|&next_idx| func.blocks[next_idx].id);

        // Handle non-terminator effectful ops (loads, stores, calls).
        let non_term_count = block.non_term_count();
        let non_term_ops = &block.ops[..non_term_count];

        let mut all_insts: Vec<MachInst> = Vec::new();

        // Move register parameters from their ABI registers to the registers
        // the allocator chose for them. Both sources feed one list: the params
        // whose precoloring lowering itself declined, and the ones
        // `merge_precolorings_global` dropped because they are live across a
        // call that clobbers their ABI register. Must be at the very start of
        // the function, before any call arg setup.
        let arg_locs = &func_arg_locs;
        let mut entry_copies: Vec<(Reg, Reg)> = Vec::new();
        for inst in rewritten.iter() {
            if let Op::Pure(PureOp::Param(param_idx, _)) = &inst.op
                && !param_vreg_set.contains(&inst.dst)
                && let Some(crate::x86::abi::ArgLoc::Reg(abi_reg)) =
                    arg_locs.get(*param_idx as usize)
                && let Some(dst_reg) = regalloc_result
                    .assignment
                    .get(&inst.dst)
                    .copied()
                    .and_then(crate::regalloc::Assignment::reg)
            {
                entry_copies.push((*abi_reg, dst_reg));
            }
        }
        if block_idx == layout_order[0] {
            for &(param_vreg, abi_reg) in &regalloc_result.unprecolored_params {
                if let Some(dst_reg) = regalloc_result
                    .assignment
                    .get(&param_vreg)
                    .copied()
                    .and_then(crate::regalloc::Assignment::reg)
                {
                    entry_copies.push((abi_reg, dst_reg));
                }
            }
        }
        all_insts.extend(emit_entry_param_copies(&entry_copies));

        // Emit in schedule order.
        //
        // The schedule was ordered by barrier group during linearization, before
        // register allocation, and the splitter inserts at positions inside it.
        // So this order is both the intended emission order and -- the part that
        // matters -- the order the allocator measured liveness on.
        //
        // Assigning barrier groups a second time here and emitting in *that*
        // answer is what let a register be clobbered between a def and its use:
        // wherever the two groupings disagreed, a value was emitted somewhere its
        // interference had never been measured, and no amount of pinning
        // individual VRegs back into place fixed the general case.
        //
        // A barrier instruction stands in for its effectful op. Emit the load,
        // store or call at the barrier's own position and no code for the
        // pseudo-op itself; everything else lowers as a pure op.
        // Computed over the whole block: lowering below runs on each run of pure
        // ops between barriers, and a projection is often in a different run from
        // the op it projects.
        let (div_dst_vregs, has_proj0_consumer) =
            crate::compile::lower::division_and_proj0_sets(full_schedule_for_barriers);

        let lower_pending = |pending: &mut Vec<ScheduledInst>,
                             out: &mut Vec<MachInst>|
         -> Result<(), CompileError> {
            if pending.is_empty() {
                return Ok(());
            }
            out.extend(lower_block_pure_ops(
                pending,
                &regalloc_result,
                func,
                &param_vreg_set,
                &frame_layout,
                &vreg_types,
                arg_locs,
                &div_dst_vregs,
                &has_proj0_consumer,
            )?);
            pending.clear();
            Ok(())
        };

        // Which effectful op of this block, if any, is a tail call to this very
        // function -- the last one before a `Ret` that returns exactly its result.
        //
        // Only self-calls, and only with every argument in a register. A call to
        // another function would jump to a body whose frame is not this one's, and
        // a stack argument would have to be written into this function's own
        // incoming argument area, which is the caller's memory; both are sound to
        // do and neither is done here.
        let mut tail_is_self = false;
        let tail_call_at: Option<usize> = (|| {
            if !opts.enable_tail_calls {
                return None;
            }
            let EffectfulOp::Ret { val } = block.ops.last()? else {
                return None;
            };
            let last = non_term_ops.len().checked_sub(1)?;
            let EffectfulOp::Call {
                func: callee,
                args,
                arg_tys,
                ret_tys,
                results,
                variadic,
            } = &non_term_ops[last]
            else {
                return None;
            };
            if *variadic {
                return None;
            }
            // A tail call to another function is the same transform with the frame
            // coming down first, but it needs one thing a self-call gets for free:
            // the callee's return has to *be* this function's return, so the
            // signatures have to agree on it. A `long` returned where an `int` is
            // declared would leave the caller reading a register nobody narrowed.
            if *callee != func.name && ret_tys != &func.return_types {
                return None;
            }
            if crate::x86::abi::assign_args(arg_tys)
                .iter()
                .any(|l| !matches!(l, crate::x86::abi::ArgLoc::Reg(_)))
            {
                return None;
            }
            // The returned value has to be the call's result and nothing else,
            // or the call is not in tail position however it looks.
            let same = match (val, results.first()) {
                (None, None) => true,
                (Some(v), Some(r)) => {
                    egraph.unionfind.find_immutable(v.class())
                        == egraph.unionfind.find_immutable(r.class())
                }
                _ => false,
            };
            if !same || args.len() != arg_tys.len() {
                return None;
            }
            tail_is_self = *callee == func.name;
            Some(last)
        })();

        let mut pending: Vec<ScheduledInst> = Vec::new();
        let mut barrier_k = 0usize;
        for inst in full_schedule_for_barriers.iter() {
            if !matches!(
                inst.op,
                Op::Pseudo(PseudoOp::CallResult(_, _))
                    | Op::Pseudo(PseudoOp::LoadResult(_, _))
                    | Op::Pseudo(PseudoOp::VoidCallBarrier)
                    | Op::Pseudo(PseudoOp::StoreBarrier)
            ) {
                pending.push(inst.clone());
                continue;
            }
            lower_pending(&mut pending, &mut all_insts)?;
            // Barrier instructions appear in effectful-op order, which is what
            // makes counting them the barrier index.
            if let Some(op) = non_term_ops.get(barrier_k) {
                all_insts.extend(lower_effectful_op(
                    op,
                    block_idx,
                    barrier_k,
                    &regalloc_result,
                    func,
                    &egraph.unionfind,
                    full_schedule_for_barriers,
                    &frame_layout,
                    (tail_call_at == Some(barrier_k)).then(|| {
                        if tail_is_self {
                            effectful::TailCall::SelfEntry(
                                func.blocks[0].id as crate::x86::inst::LabelId,
                            )
                        } else {
                            effectful::TailCall::Other
                        }
                    }),
                )?);
            }
            barrier_k += 1;
        }
        lower_pending(&mut pending, &mut all_insts)?;
        if barrier_k != non_term_ops.len() && std::env::var("BLITZ_PROBE").is_ok() {
            eprintln!(
                "[probe] block {block_idx} schedule: {:?}",
                rewritten
                    .iter()
                    .map(|i| format!("v{}={:?}", i.dst.0, i.op))
                    .collect::<Vec<_>>()
            );
        }
        debug_assert_eq!(
            barrier_k,
            non_term_ops.len(),
            "block {} of '{}' has {} barrier instructions for {} effectful ops",
            block_idx,
            func.name,
            barrier_k,
            non_term_ops.len(),
        );

        // After the global allocator, every VReg in the schedule must have a
        // physical register. Only active for multi-block functions (single-block
        // uses the old allocator path with its own guarantees).
        //
        // A Ret's value needs no check here: `lower_terminator` fails outright
        // when it cannot name a register for one, which is the same property
        // stated where the register is actually needed, and unconditionally
        // rather than only in a debug build.
        if func.blocks.len() > 1 {
            // A flags operand is exempt, and is the one operand that must be:
            // it names EFLAGS, which no instruction addresses and no allocation
            // can hand out. Its consumer reads the flags the comparison left.
            let flags_vregs = crate::regalloc::build_vreg_classes_from_all_blocks(
                std::slice::from_ref(rewritten),
            );
            for inst in rewritten.iter() {
                for &op in &inst.operands {
                    if flags_vregs.get(&op) == Some(&crate::x86::reg::RegClass::Flags) {
                        continue;
                    }
                    debug_assert!(
                        regalloc_result.assignment.contains_key(&op),
                        "8a-effectful safety net fired after global regalloc: \
                         operand VReg {:?} in block {} of function '{}' has no register assignment",
                        op,
                        block_idx,
                        func.name,
                    );
                }
            }
        }

        // Handle the terminator. A tail self-call has already jumped, so the
        // `Ret` it stood in front of is unreachable and its epilogue would be
        // dead code after an unconditional jump.
        let terminator = block.ops.last().expect("block must have terminator");
        let term_items = if tail_call_at.is_some() {
            Vec::new()
        } else {
            lower_terminator(
                terminator,
                block_idx,
                next_block_id,
                &egraph,
                &class_to_vreg,
                &block_param_map,
                // Straight off the final schedule: post-split, post-coalesce, the
                // VRegs the allocator actually assigned registers to.
                &barrier::terminator_arg_operands(&block_rewritten[block_idx])
                    .into_iter()
                    .collect::<BTreeMap<u32, VReg>>(),
                &coalesce_aliases,
                &regalloc_result,
                func,
                &mut next_label,
                &slot_spilled_params,
                &frame_layout,
                &block_rewritten,
            )?
        };

        // Phase 8: Peephole on this block's pure/effectful instructions.
        //
        // The terminator's own instructions, up to its first label, are handed
        // over as context: the phi copies on the outgoing edge end a register's
        // live range and the branch reads the flags, and a scan that stops at
        // the last instruction of the body can see neither. Nothing past a label
        // is reachable only from here, so the run stops there.
        let final_insts = if opts.enable_peephole {
            let tail: Vec<MachInst> = term_items
                .iter()
                .map_while(|item| match item {
                    BlockItem::Inst(inst) => Some(inst.clone()),
                    BlockItem::BindLabel(_) => None,
                })
                .collect();
            peephole(all_insts, &tail)
        } else {
            all_insts
        };

        // Reassemble into BlockItems.
        let mut items: Vec<BlockItem> = final_insts.into_iter().map(BlockItem::Inst).collect();
        items.extend(term_items);
        block_items.push(items);
    }

    // Branch threading: rewrite Jcc/Jmp targets that point to empty blocks
    // containing only a single Jmp. Repeat until fixed point.
    thread_branches(&mut block_items, func, &layout_order);

    // Threading can leave a jump over nothing: its new target may be the block
    // that follows, which `lower_terminator` could not see when it chose to
    // emit the jump at all.
    remove_fallthrough_jumps(&mut block_items, func, &layout_order);

    // Copy each loop's test onto its back edge, so the back edge is the
    // conditional and the unconditional `jmp` that closed the loop is gone.
    // After the fallthrough jump above, since that is what leaves a header as a
    // bare test ending in one conditional, and before it again, since the jump
    // to the exit the rotation appends is a fallthrough whenever the trace put
    // the exit next.
    let rotated_headers = if opts.enable_loop_rotation {
        let rotated = rotate::rotate_loops(&mut block_items, func, &layout_order);
        if !rotated.is_empty() {
            remove_fallthrough_jumps(&mut block_items, func, &layout_order);
        }
        rotated
    } else {
        BTreeMap::new()
    };

    // Phase 10: Encoding with branch relaxation.
    //
    // Step 10a: Flatten all BlockItems into a linear instruction sequence,
    // recording label positions (label -> instruction index immediately after
    // the label binding point).
    //
    // Block labels are bound at the start of each block; trampoline labels
    // (BlockItem::BindLabel) are bound at whatever position they appear.
    // We represent label bindings as a sentinel NOP(0) to anchor their
    // position in the flat list, paired with a side table of label->inst_idx.

    // flat_insts: the instruction sequence passed to relax_branches.
    // flat_labels: for each instruction index, any labels bound just before it.
    // label_positions: label -> instruction index (for relax_branches).
    // Block labels use block.id (not block_idx) so Jump targets resolve correctly.
    // Instructions whose result nothing reads, dropped before the flat list is
    // built so that labels are never renumbered: the items are what carries them,
    // and removing an `Inst` item leaves every `BindLabel` where it was.
    //
    // Address folding is what makes this pay. Lowering puts `[base + idx*4]` in
    // the addressing mode of the load that uses it and decides that per consumer;
    // when every consumer folds, the `lea` is left behind with nothing reading it.
    // No earlier pass can see that -- DCE runs on the CFG before scheduling and
    // the e-graph never sees an effectful op.
    if opts.enable_dead_insts {
        let mut scratch: Vec<MachInst> = Vec::new();
        let mut scratch_labels: BTreeMap<LabelId, usize> = BTreeMap::new();
        let mut origin: Vec<(usize, usize)> = Vec::new();
        for (layout_pos, items) in block_items.iter().enumerate() {
            let block_id = func.blocks[layout_order[layout_pos]].id;
            scratch_labels.insert(block_id as LabelId, scratch.len());
            for (item_idx, item) in items.iter().enumerate() {
                match item {
                    BlockItem::Inst(inst) => {
                        origin.push((layout_pos, item_idx));
                        scratch.push(inst.clone());
                    }
                    BlockItem::BindLabel(label_id) => {
                        scratch_labels.insert(*label_id, scratch.len());
                    }
                }
            }
        }
        let dead = crate::emit::dead_inst::dead_value_moves(
            &scratch,
            &scratch_labels,
            frame_layout.uses_frame_pointer,
        );
        if !dead.is_empty() {
            let mut drop_at: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); block_items.len()];
            for &flat in &dead {
                let (b, item) = origin[flat];
                drop_at[b].insert(item);
            }
            for (b, items) in block_items.iter_mut().enumerate() {
                let mut idx = 0;
                items.retain(|_| {
                    idx += 1;
                    !drop_at[b].contains(&(idx - 1))
                });
            }
        }
    }

    // The flat stream is a projection of `block_items`: one entry per `Inst`
    // item, in emission order, with every label bound at the index of the
    // instruction that follows it. The encoder walks `block_items` against this
    // by index, so the two must be derived the same way each time one is
    // rebuilt.
    let flatten = |block_items: &[Vec<BlockItem>]| {
        let mut insts: Vec<MachInst> = Vec::new();
        let mut labels: BTreeMap<LabelId, usize> = BTreeMap::new();
        for (layout_pos, items) in block_items.iter().enumerate() {
            let block_id = func.blocks[layout_order[layout_pos]].id;
            // The block label is bound before the first instruction of this block.
            labels.insert(block_id as LabelId, insts.len());

            for item in items {
                match item {
                    BlockItem::Inst(inst) => insts.push(inst.clone()),
                    BlockItem::BindLabel(label_id) => {
                        // Trampoline label: bound at the position of the next instruction.
                        labels.insert(*label_id, insts.len());
                    }
                }
            }
        }
        (insts, labels)
    };

    let (mut flat_insts, mut label_positions) = flatten(&block_items);

    // MachInst::Ret is lowered by `emit_epilogue`, not by `encode_inst`. The
    // default `inst_size` routes Ret through `encode_inst` (a single c3 byte)
    // and therefore underestimates the expansion to epilogue size (frame
    // teardown + callee-saved pops + ret). Byte offsets would drift,
    // potentially leaving a short jump whose real displacement is out of rel8
    // range (panics at fixup time). Provide a size oracle that substitutes the
    // actual epilogue byte count for each Ret.
    let epilogue_size = {
        let mut scratch = Encoder::new();
        emit_epilogue(&mut scratch, &frame_layout);
        scratch.buf.len()
    };
    let inst_size_for_relax = |inst: &MachInst| -> usize {
        if matches!(inst, MachInst::Ret) {
            epilogue_size
        } else {
            inst_size(inst)
        }
    };

    // Step 10a2: pad loop headers onto 16-byte boundaries.
    //
    // Where a hot loop falls relative to a fetch boundary is otherwise whatever
    // the instruction stream happens to produce, and it is worth up to 20% --
    // removing four dead instructions from `live/matmul_rt.c` cost `+20.7%`
    // cycles, and three unrelated instructions before the same loop took the
    // comparison back the other way.
    //
    // The padding goes in before branch relaxation because relaxation only ever
    // widens a jump: bytes inserted here are counted by the relaxation that
    // follows, so no jump can be left short and out of range. What relaxation
    // can do is move a header off the boundary it was just padded to, which
    // `align::loop_header_pads` iterates against.
    if opts.enable_nop_alignment {
        // A rotated loop is entered at its body: the header runs once as a
        // guard, so padding it pads a block outside the loop.
        let headers: std::collections::BTreeSet<LabelId> = cfg::loop_header_blocks(func)
            .into_iter()
            .map(|id| id as LabelId)
            .map(|id| rotated_headers.get(&id).copied().unwrap_or(id))
            .collect();
        let prologue_size = {
            let mut scratch = Encoder::new();
            emit_prologue(&mut scratch, &frame_layout);
            scratch.buf.len()
        };
        let pads = crate::emit::loop_header_pads(
            &flat_insts,
            &label_positions,
            &headers,
            prologue_size,
            &inst_size_for_relax,
        );
        if !pads.is_empty() {
            insert_alignment_nops(&mut block_items, func, &layout_order, &pads);
            let rebuilt = flatten(&block_items);
            flat_insts = rebuilt.0;
            label_positions = rebuilt.1;
        }
    }

    // Step 10b: Branch relaxation -- determine which jumps use short (rel8) form.
    let (flat_insts, is_short) =
        crate::emit::relax::relax_branches(&flat_insts, &label_positions, &inst_size_for_relax);

    // Machine-level verification of the final stream: no surviving virtual
    // register, nothing reads a physical register the function never writes, and
    // no reload reads a spill slot nothing stored. This is the check that turns a
    // wrong-code bug from "segfault somewhere in a large program" into a named
    // instruction.
    if crate::trace::is_enabled("slots") && crate::trace::fn_matches(&func.name) {
        tracing::debug!(
            target: "blitz::slots",
            "[{}] spill slot traffic:\n{}",
            func.name,
            crate::trace::format_slot_traffic(
                &flat_insts,
                frame_layout.spill_base,
                frame_layout.spill_offset,
                &slots,
            ),
        );
    }

    crate::verify::verify_machinsts_stage(
        "encode",
        &func.name,
        &flat_insts,
        &label_positions,
        &frame_layout,
        regalloc_result.spill_slots,
    );

    // Step 10c: Encode.
    let mut encoder = Encoder::new();
    let func_start = encoder.buf.len();
    emit_prologue(&mut encoder, &frame_layout);

    // Bind block labels and trampoline labels in RPO order.
    // Labels use block.id so that jump targets encoded in lower_terminator resolve.
    let mut flat_idx = 0usize;
    for (layout_pos, items) in block_items.iter().enumerate() {
        let block_id = func.blocks[layout_order[layout_pos]].id;
        encoder.bind_label(block_id as LabelId);

        for item in items {
            match item {
                BlockItem::Inst(inst) => {
                    let short = is_short[flat_idx];
                    flat_idx += 1;
                    if *inst == MachInst::Ret {
                        emit_epilogue(&mut encoder, &frame_layout);
                    } else {
                        // A tail call to another function tears this frame down
                        // first: the callee's prologue builds its own, and RSP has
                        // to be back on the return address the original `call`
                        // pushed so the callee returns past this function.
                        if matches!(inst, MachInst::TailCallDirect { .. }) {
                            crate::x86::abi::emit_frame_teardown(&mut encoder, &frame_layout);
                        }
                        encoder.encode_inst_with_form(&flat_insts[flat_idx - 1], short);
                    }
                }
                BlockItem::BindLabel(label_id) => {
                    encoder.bind_label(*label_id);
                }
            }
        }
    }

    encoder.resolve_fixups();

    let func_size = encoder.buf.len() - func_start;

    if let Some(s) = sink.as_mut() {
        s.phase_stats("encoding", &format!("bytes={func_size}"));
    }

    // One line per function, in a shape `tests/run_codesize.sh` parses into the
    // checked-in baselines. `insts` counts the body: the prologue and epilogue
    // are encoded from the frame layout rather than emitted as instructions, and
    // they are in `bytes`.
    if crate::trace::is_enabled("stats") && crate::trace::fn_matches(&func.name) {
        let (spills, reloads) = crate::trace::count_slot_traffic(
            &flat_insts,
            frame_layout.spill_base,
            frame_layout.spill_offset,
            &slots,
        );
        let k = crate::trace::classify_copies(&flat_insts, &label_positions);
        let (copies, two_address) = (k.total, k.two_address);
        let (cp_edge, cp_arg, cp_entry, cp_other) = (k.edge, k.call_arg, k.entry, k.other);
        tracing::debug!(
            target: "blitz::stats",
            "name={} insts={} bytes={func_size} spills={spills} reloads={reloads} \
             copies={copies} two_addr={two_address} cp_edge={cp_edge} cp_arg={cp_arg} \
             cp_entry={cp_entry} cp_other={cp_other} frame={} slots={}",
            func.name,
            flat_insts.len(),
            frame_layout.frame_size,
            slots.count(),
        );
    }

    if crate::trace::is_enabled("asm") && crate::trace::fn_matches(&func.name) {
        let code_bytes = &encoder.buf[func_start..];
        if let Some(disasm) = crate::test_utils::objdump_disasm(code_bytes) {
            tracing::debug!(
                target: "blitz::asm",
                "[{}] disassembly ({func_size} bytes):\n{disasm}",
                func.name,
            );
        } else {
            tracing::debug!(
                target: "blitz::asm",
                "[{}] disassembly unavailable (objdump not found), {func_size} bytes",
                func.name,
            );
        }
    }

    // Collect externals (symbols referenced by call instructions).
    let externals: Vec<String> = collect_externals(func);

    Ok(ObjectFile {
        code: encoder.buf,
        relocations: encoder.relocations,
        functions: vec![FunctionInfo {
            name: func.name.clone(),
            offset: func_start,
            size: func_size,
        }],
        externals,
        globals: vec![],
        rodata: vec![],
    })
}

mod ir_print;
pub use ir_print::{compile_module_to_ir, compile_to_ir_string};

mod module;
pub use module::{compile_module, compile_module_with_globals};

/// A flat item emitted for a block: either a MachInst or a label binding.
pub(crate) enum BlockItem {
    Inst(MachInst),
    BindLabel(LabelId),
}

/// Put each loop-header pad in the item list, immediately *before* the point
/// its label binds.
///
/// Which side of the label the NOPs land on is the whole question: padding
/// after the binding puts them inside the loop, where every iteration pays for
/// them. A block's label is bound by the encoder before the block's first item,
/// so the pad for one belongs at the end of the block that precedes it in
/// emission order; a trampoline label is bound where its `BindLabel` item sits,
/// so the pad goes just before that item.
///
/// The first emitted block cannot be padded this way -- nothing precedes it but
/// the prologue -- and does not need to be: it starts at the function's own
/// 16-byte-aligned first byte plus the prologue, and a back edge to the entry
/// block is not a loop header any code generated here produces.
fn insert_alignment_nops(
    block_items: &mut [Vec<BlockItem>],
    func: &Function,
    layout_order: &[usize],
    pads: &BTreeMap<LabelId, u8>,
) {
    // (block position, item index, pad). An item index equal to the list length
    // means "append".
    let mut inserts: Vec<Vec<(usize, u8)>> = vec![Vec::new(); block_items.len()];

    for (layout_pos, items) in block_items.iter().enumerate() {
        let block_id = func.blocks[layout_order[layout_pos]].id as LabelId;
        if let Some(&pad) = pads.get(&block_id)
            && layout_pos > 0
        {
            let prev = layout_pos - 1;
            let at = block_items[prev].len();
            inserts[prev].push((at, pad));
        }
        for (item_idx, item) in items.iter().enumerate() {
            if let BlockItem::BindLabel(label) = item
                && let Some(&pad) = pads.get(label)
            {
                inserts[layout_pos].push((item_idx, pad));
            }
        }
    }

    for (layout_pos, mut sites) in inserts.into_iter().enumerate() {
        // Descending, so an insertion never shifts one still to be made.
        sites.sort_by_key(|&(at, _)| std::cmp::Reverse(at));
        for (at, pad) in sites {
            block_items[layout_pos].insert(at, BlockItem::Inst(MachInst::Nop { size: pad }));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
