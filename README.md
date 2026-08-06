<p align="center">
  <img src="logo.svg" width="160" alt="Blitz logo">
</p>

<h1 align="center">Blitz</h1>

<p align="center">
  <em>A compiler backend that targets x86-64, and nothing else, on purpose.</em>
</p>

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

## Code quality

The point of the project is the output, so it is measured rather than asserted.
`bash tests/run_codesize.sh --gap` compares instruction counts against system
compilers on eight loop kernels that seed their data from `argc`, so no
reference compiler can fold the program away and print the answer:

| | instructions, geomean |
|---|---|
| vs `gcc -O2` | **x1.36** |
| vs `clang -O2` | **x0.74** |

The `clang` figure is not a victory lap; clang unrolls where blitz does not,
which inflates its count on these kernels. `gcc -O2` is the number to watch. The
worst kernel is `call_hot` at `x2.44`, the ABI cost of a loop around a `noinline`
callee.

## What's in it

- **E-graph optimizer**: union-find, hashcons, typed e-classes, phased rewrites,
  and cost-based extraction with DAG-sharing awareness. Optimization goals:
  latency, throughput, size, balanced.
- **Function-scope register allocator**: Chaitin-Briggs coloring with MCS
  ordering, conservative coalescing, rematerialization, loop-aware spill choice,
  and pressure-driven live-range splitting across blocks.
- **Hand-written encoder**: 70+ x86-64 instruction forms with correct REX,
  ModRM, SIB and displacement encoding, plus branch relaxation.
- **SysV AMD64 ABI**: register and stack arguments, callee-saved preservation,
  16-byte frames, parallel-copy sequentialization for phi elimination.
- **Optimization passes**: inlining, LICM, store-to-load and load-to-load
  forwarding, dead store elimination with offset-aware alias analysis, DCE,
  peephole, loop-header alignment.

## Pipeline

```
  IR  →  inline  →  DCE  →  memory  →  LICM  →  e-graph  →  extract
                                                               ↓
  ELF  ←  encode  ←  post-RA  ←  regalloc  ←  split  ←  schedule
```

The IR is dual: the e-graph holds pure values, the CFG holds effectful ops and
control flow, and effectful ops reference pure values by e-class. Priorities and
non-goals are in [`ROADMAP.md`](ROADMAP.md).

## Testing

```console
$ cargo test --all-targets --workspace   # 925 unit and codegen tests
$ bash tests/lit/run_tests.sh            # 498 FileCheck-style tests
$ bash tests/lit/run_diff.sh             # 311 comparisons: -O0 vs -O1 vs cc
$ bash tests/fuzz/run_fuzz.sh 40 mixed   # random UB-free programs
```

Two harnesses check *values* rather than patterns, and they fail differently.
`run_diff.sh` compiles every runnable test at both optimization levels and
against a reference compiler, so it catches a pass that changes behavior *and* a
bug that is equally wrong at both levels. `run_fuzz.sh` generates programs that
are free of undefined behavior by construction and interprets them as it
generates, so it knows the expected output before any compiler runs.

`BLITZ_VERIFY=1` checks the IR at every pass boundary and the machine code after
branch relaxation; `BLITZ_VERIFY=strict` also requires every e-class reference in
the CFG to be canonical.

End-to-end tests need `cc` on `PATH` and skip gracefully without it.

## Status

Correct code for integer and floating-point arithmetic (F32/F64 via SSE2),
branches, loops with block parameters, calls with register and stack arguments,
memory access with addressing-mode fusion, and programs that need spilling.

`crates/tinyc` is a small C frontend that exists to feed the backend realistic
input in tests. It is not a product.
