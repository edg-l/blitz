<div align="center">

<img src="logo.svg" width="132" alt="Blitz">

# Blitz

**A compiler backend that targets x86-64, and nothing else, on purpose.**

<sub>
  <img src="https://img.shields.io/badge/target-x86--64%20SysV-3B82F6?style=flat-square" alt="target: x86-64 SysV">
  <img src="https://img.shields.io/badge/output-ELF%20objects-3B82F6?style=flat-square" alt="output: ELF objects">
  <img src="https://img.shields.io/badge/rust-2024%20edition-7C3AED?style=flat-square" alt="rust 2024 edition">
  <img src="https://img.shields.io/badge/tests-1033%20unit%20%C2%B7%20592%20lit%20%C2%B7%201352%20fuzz-22C55E?style=flat-square" alt="tests">
</sub>

</div>

---

Blitz turns SSA-style IR into linkable ELF object files. It targets a single ISA
so the optimizer can reason about addressing modes, flags, `lea`, and
multi-output instructions directly, instead of laundering them through a
target-neutral IR and hoping the peephole pass finds them again.

Optimization and instruction selection happen in **one e-graph**: algebraic
rewrites, strength reduction, constant folding, and x86 instruction selection all
compete in the same equality-saturation pass, and a cost model picks the winner.
There is no separate "isel" phase to undo the optimizer's work.

## Quick start

```rust
use blitz::compile::{CompileOptions, compile};
use blitz::ir::builder::FunctionBuilder;
use blitz::ir::types::Type;

// fn add(a: i64, b: i64) -> i64 { a + b }
let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], &[Type::I64]);
let p = b.params().to_vec();
let sum = b.add(p[0], p[1]);
b.ret(Some(sum));

let obj = compile(b.finalize()?, &CompileOptions::default(), None)?;
obj.write_to(std::path::Path::new("add.o"))?;
```

Link it against C like any other object file:

```console
$ cc main.c add.o -o demo && ./demo
42
```

The emitted `add` is what you would hope for: the addition folded into the
address unit, no frame, no moves.

```asm
add:
    lea    (%rdi,%rsi,1),%rax
    ret
```

Use `compile_module` to put several functions in one object; inlining works
across whatever you hand it. [`examples/basic.rs`](examples/basic.rs) builds a
handful of functions and [`examples/main.c`](examples/main.c) drives them:

```console
$ cargo run --example basic
$ cc examples/main.c output.o -o demo && ./demo
```

[`docs/ir.md`](docs/ir.md) is the guide to the IR itself: block parameters
instead of phi nodes, pure versus effectful ops, branches, loops, memory, structs
and tagged enums, with worked examples.

## Code quality

The point of the project is the output, so it is measured rather than asserted,
on 24 loop kernels that seed their data from `argc` — no reference compiler can
fold one away and print the answer.

<table>
<tr><th align="left">geomean over 24 kernels</th><th>vs <code>gcc -O2</code></th><th>vs <code>clang -O2</code></th></tr>
<tr><td><b>cycles</b> &nbsp;<code>tests/run_perf.sh</code></td><td align="center"><b>x2.21</b></td><td align="center"><b>x2.83</b></td></tr>
<tr><td>instructions &nbsp;<code>run_codesize.sh --gap</code></td><td align="center">x1.05</td><td align="center">x0.72</td></tr>
</table>

**Cycles is the ranking; instructions is a diagnostic.** They disagree often
enough that keeping both is the point. Laying a loop body after its header moved
cycles 7.9% with the instruction count *flat*, because what changed is which
branches are taken and no static count models that. Folding a stack slot into
the addressing mode that reads it took 3% of the instructions off the loop
corpus and moved cycles by nothing, so it was measured and shelved.

The `clang` instruction figure is not a victory lap: clang unrolls and vectorizes
where blitz does not, which inflates its count on these kernels. Blitz has no
vectorizer, and that is worth about 12% of the cycles gap — the rest is scalar
codegen.

## What's in it

- **E-graph optimizer** — union-find, hashcons, typed e-classes, phased
  rewrites, and cost-based extraction with DAG-sharing awareness. Optimization
  goals: latency, throughput, size, balanced.
- **Function-scope register allocator** — Chaitin-Briggs coloring with MCS
  ordering, optimistic coalescing, rematerialization, loop-aware spill choice,
  and pressure-driven live-range splitting across blocks.
- **A second allocator, on purpose** — `-O0` puts every value in a frame slot
  and borrows registers one instruction at a time. Two implementations of one
  contract is what makes an allocation bug a disagreement the differential
  harness can see, rather than an answer both levels give.
- **Hand-written encoder** — 93 x86-64 instruction forms with correct REX,
  ModRM, SIB and displacement encoding, plus branch relaxation.
- **SysV AMD64 ABI** — register and stack arguments, callee-saved preservation,
  16-byte frames, parallel-copy sequentialization for phi elimination.
- **Optimization passes** — inlining, SCCP, LICM, store-to-load and load-to-load
  forwarding, dead store elimination with offset-aware alias analysis, DCE,
  block layout, peephole, dead-instruction removal, loop-header alignment.

## Pipeline

```
  IR  →  inline  →  DCE  →  memory  →  LICM  →  SCCP  →  e-graph  →  extract
                                                                       ↓
  ELF  ←  encode  ←  layout  ←  post-RA  ←  regalloc  ←  split  ←  schedule
```

The IR is dual: the e-graph holds pure values, the CFG holds effectful ops and
control flow, and effectful ops reference pure values by e-class.

Two orders matter and they are not the same order. **RPO is the dominance
order** and everything up to register allocation reads it; **block layout is a
greedy trace** that puts a loop body directly after its header, so the header's
conditional leaves the loop and the fallthrough enters it.

[`ROADMAP.md`](ROADMAP.md) has the priorities, the non-goals, and — the part
worth reading — what previous attempts measured, including the ones that lost.

## Testing

```console
$ cargo test --all-targets --workspace   # 1033 unit and codegen tests
$ bash tests/lit/run_tests.sh            # 592 FileCheck-style tests
$ bash tests/lit/run_diff.sh             # 356 comparisons: -O0 vs -O1 vs cc
$ bash tests/fuzz/run_fuzz.sh            # 1352 random UB-free programs
$ bash tests/fuzz/run_corpus.sh          # the saved failures, in seconds
$ bash tests/run_codesize.sh --check     # 1008 rows of code quality, vs baselines
```

Two harnesses check *values* rather than patterns, and they fail differently.
`run_diff.sh` compiles every runnable test at both optimization levels and
against a reference compiler, so it catches a pass that changes behavior *and* a
bug that is equally wrong at both levels. `run_fuzz.sh` generates programs that
are free of undefined behavior by construction and interprets them as it
generates, so it knows the expected output before any compiler runs.

`BLITZ_VERIFY=1` checks the IR at every pass boundary and the machine code after
branch relaxation; `BLITZ_VERIFY=strict` also requires every e-class reference in
the CFG to be canonical. `BLITZ_PASSES=-licm` and friends bisect the pass set
against a miscompile.

End-to-end tests need `cc` on `PATH` and skip gracefully without it.

## Status

Correct code for integer and floating-point arithmetic (F32/F64 via SSE2),
branches, loops with block parameters, calls with register and stack arguments,
memory access with addressing-mode fusion, and programs that need spilling.

`crates/tinyc` is a small C frontend that exists to feed the backend realistic
input in tests. It is not a product.
