# Blitz

A pure-Rust compiler backend targeting x86-64, optimizing for generated code quality.

Blitz uses a custom e-graph for unified optimization and instruction selection, SSA-based chordal register allocation, and emits ELF64 relocatable object files. It targets only x86-64, which lets the optimizer natively understand addressing modes, flags, LEA tricks, and multi-output instructions without the abstraction penalty of multi-target backends.

## Features

- **Custom e-graph engine** with union-find, hashcons, typed e-classes, and phased rewrite rules (algebraic simplification, strength reduction, constant folding, x86-64 instruction selection, addressing mode formation, LEA/flag fusion)
- **Cost-based extraction** with DAG sharing awareness and configurable optimization goals (latency, throughput, code size, balanced)
- **SSA chordal register allocation** with MCS ordering, optimal greedy coloring, aggressive coalescing, loop-aware spill selection, rematerialization, and per-block allocation with cross-block live range splitting
- **Hand-written x86-64 encoder** covering 70+ instruction forms (integer ALU, shifts, multiply/divide, MOV, LEA, branches, CALL/RET, CMOV, SETCC, SSE2 scalar FP, MOVQ, PUSH/POP, NOP) with correct REX, ModRM, SIB, and displacement encoding
- **SystemV AMD64 ABI** with register/stack argument passing, callee-saved preservation, 16-byte aligned frames, parallel copy sequentialization for phi elimination, and caller-saved clobber tracking across calls
- **ELF64 object emission** with .text, .symtab, .strtab, .shstrtab, .rela.text, and .note.GNU-stack sections
- **Post-RA passes**: peephole optimization (redundant MOV elimination, XOR-zero idiom, TEST-for-CMP, INC/DEC substitution, branch threading), NOP alignment for loop headers, branch relaxation (short/near form selection)
- **Multi-block support** with RPO ordering, fallthrough optimization, block parameter passing (SSA phi equivalent), and per-block register allocation with global liveness dataflow

## Quick Start

```rust
use blitz::compile::{CompileOptions, compile};
use blitz::ir::builder::FunctionBuilder;
use blitz::ir::types::Type;

// Build: fn add(a: i64, b: i64) -> i64 { a + b }
let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], &[Type::I64]);
let p = b.params().to_vec();
let sum = b.add(p[0], p[1]);
b.ret(Some(sum));
let (func, egraph) = b.finalize().expect("finalize");

// Compile to object file
let obj = compile(&func, egraph, &CompileOptions::default(), None).expect("compile");
obj.write_to(std::path::Path::new("add.o")).expect("write");

// Link with C: cc main.c add.o -o test
```

See [`examples/basic.rs`](examples/basic.rs) for a complete example with add, max, and sum-to-N functions, and [`examples/main.c`](examples/main.c) for the C driver.

```sh
cargo run --example basic
cc examples/main.c output.o -o test && ./test
```

## Architecture

```
Source IR (FunctionBuilder API)
       |
       v
  [ E-graph ]  -- algebraic simplification, strength reduction,
       |           constant folding, commutativity
       v
  [ E-graph ]  -- x86-64 instruction selection, addressing modes,
       |           LEA formation, flag fusion
       v
  [ Extraction ]  -- cost-based bottom-up DAG extraction
       |
       v
  [ Scheduling ]  -- list scheduler with register pressure heuristic
       |
       v
  [ Register Allocation ]  -- per-block chordal coloring with
       |                       cross-block live range splitting
       v
  [ Post-RA ]  -- phi elimination, peephole, NOP alignment,
       |           branch relaxation
       v
  [ Encoding ]  -- x86-64 binary encoder with label fixups
       |
       v
  [ ELF Emission ]  -- relocatable .o file
```

## Testing

```sh
cargo test
```

315 tests covering:
- 64 instruction encoding tests (byte-level verification)
- 22+ end-to-end tests (compile -> link with C -> run -> verify results)
- 10 miscompile regression tests (signed overflow, spill correctness, phi permutations, nested control flow)
- Unit tests for every pipeline phase (e-graph, extraction, scheduling, regalloc, peephole, ELF)

End-to-end tests require `cc` (gcc/clang) on PATH. They skip gracefully if unavailable.

## Status

The compiler produces correct code for integer arithmetic, floating-point (F32/F64 via SSE2), conditional branches, loops with block parameters, function calls with up to 6+ register args and stack args, memory loads/stores with addressing mode fusion, and programs requiring register spilling.

## License

MIT
