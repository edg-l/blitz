//! An optimizing compiler backend for x86-64, and only x86-64.
//!
//! Blitz takes SSA-style IR and produces linkable ELF object files. Targeting a
//! single ISA is the point: the optimizer reasons about addressing modes, flags,
//! `lea` and multi-output instructions directly, instead of going through a
//! target-neutral IR and hoping a peephole pass recovers them afterwards.
//!
//! Optimization and instruction selection share one e-graph. Algebraic
//! rewrites, strength reduction, constant folding and x86 instruction selection
//! all compete in the same equality-saturation pass, and a cost model picks the
//! winner; there is no later selection phase to undo the optimizer's work.
//!
//! # Building a function
//!
//! [`ir::builder::FunctionBuilder`] is the entry point. Give it a name, the
//! parameter types and the result types, build the body from the values it hands
//! back, and finish with [`compile`](compile::compile):
//!
//! ```
//! use blitz::compile::{CompileOptions, compile};
//! use blitz::ir::builder::FunctionBuilder;
//! use blitz::ir::types::Type;
//!
//! // fn add(a: i64, b: i64) -> i64 { a + b }
//! let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], &[Type::I64]);
//! let p = b.params().to_vec();
//! let sum = b.add(p[0], p[1]);
//! b.ret(Some(sum));
//!
//! let obj = compile(b.finalize()?, &CompileOptions::default(), None)?;
//!
//! // The addition folded into the address unit: `lea rax,[rdi+rsi]` then `ret`.
//! assert_eq!(obj.code, [0x48, 0x8d, 0x04, 0x37, 0xc3]);
//!
//! obj.write_to(std::path::Path::new("add.o"))?;
//! # std::fs::remove_file("add.o").ok();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The resulting `.o` links against C like any other object file:
//!
//! ```sh
//! cc main.c add.o -o demo
//! ```
//!
//! Use [`compile_module`](compile::compile_module) to put several functions in
//! one object; inlining works across whatever you hand it.
//!
//! # The IR in one minute
//!
//! * **Pure values are SSA and unnamed.** [`b.add(x, y)`](ir::builder::FunctionBuilder::add)
//!   returns a handle to a value, not a register. Where it ends up is the
//!   allocator's decision.
//! * **Control flow uses block parameters, not phi nodes.** A block declares the
//!   values it receives and each predecessor passes them on the edge, so there
//!   is no phi placement to get wrong.
//! * **Memory and calls are effectful** and stay ordered relative to each other;
//!   pure arithmetic is free to move, and the scheduler places it.
//! * **`finalize` validates.** It returns an error rather than a half-built
//!   function, so a malformed body is caught before any pass runs.
//!
//! There is no aggregate type either: a struct is memory plus a constant offset
//! per field, which keeps layout policy where it belongs, in the frontend.
//! [`ir::layout::Layout`] computes the offsets from the rules you pick, and
//! [`FunctionBuilder::load_field`](ir::builder::FunctionBuilder::load_field) and
//! friends address them.
//!
//! `docs/ir.md` covers all of this properly, with worked examples of branches,
//! loops, memory, structs and tagged enums.
//!
//! # Module map
//!
//! Most users need [`ir::builder`] and [`compile`]; the rest is the pipeline.
//!
//! | Module | What it does |
//! |---|---|
//! | [`ir`] | IR types, [`ir::builder::FunctionBuilder`], the CFG |
//! | [`compile`] | The pipeline entry points and its passes |
//! | [`egraph`] | Equality saturation, rewrite rules, cost-based extraction |
//! | [`schedule`] | List scheduling with a register-pressure heuristic |
//! | [`regalloc`] | Function-scope Chaitin-Briggs allocation |
//! | [`x86`] | Instruction encoding, registers, the SysV ABI |
//! | [`emit`] | ELF64 object emission |
//! | [`verify`] | IR and machine-level checks, see below |
//!
//! # Debugging
//!
//! `BLITZ_VERIFY=1` checks the IR at every pass boundary and the machine code
//! after branch relaxation, panicking with the name of the stage that produced
//! bad IR. `BLITZ_VERIFY=strict` also requires every e-class reference in the
//! CFG to be canonical.
//!
//! `BLITZ_DEBUG` traces a phase: `sched`, `liveness`, `regalloc`, `asm`, `licm`,
//! `egraph`, `split`, `stats`, or `all`. `BLITZ_DEBUG_FN=name` narrows it to one
//! function.

pub mod compile;
pub mod egraph;
pub mod emit;
pub mod inline;
pub mod ir;
pub mod regalloc;
pub mod schedule;
pub mod test_utils;
pub mod trace;
pub mod verify;
pub mod x86;
