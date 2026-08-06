# The Blitz IR

Everything you need to drive [`FunctionBuilder`](../src/ir/builder.rs). Four
ideas, then worked examples.

## The four ideas

**Values are SSA handles, not registers.** `b.add(x, y)` gives you back a
`Value`. It has no name, no register, and no location; where it lives is decided
much later by the allocator. You can use a `Value` as many times as you like,
from any block that the defining block reaches.

**Blocks take parameters instead of phi nodes.** A block declares the values it
receives, and every predecessor supplies them on the edge. There is no phi
placement to get right, and no "which incoming edge was this again" bookkeeping.

**Pure and effectful ops are separated.** Arithmetic is pure: it lives in the
e-graph, gets rewritten and reordered freely, and disappears if nothing uses it.
Loads, stores and calls are effectful: they stay ordered with respect to each
other and keep their place in the block.

**`finalize` validates.** It returns `Result<Function, BuildError>`, so a block
with no terminator, an edge passing the wrong number of arguments, or a type
mismatch is caught before any pass runs, not as a panic somewhere in the middle
of the pipeline.

## Types

`I8`, `I16`, `I32`, `I64`, `F32`, `F64`. Both operands of a binary op must have
the same type; use `sext`, `zext` or `trunc` to change width and `bitcast` to
reinterpret bits at the same width.

Two types exist that you do not construct directly. `Flags` is what `icmp` and
`fcmp` produce, and it only feeds `select` and `branch`. `Pair` is how
multi-output machine instructions carry their results internally.

## Straight-line code

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
```

`params()` returns the incoming values in declaration order. `ret(None)` returns
void.

That compiles to two instructions, with the addition folded into the address
unit:

```asm
add:
    lea    (%rdi,%rsi,1),%rax
    ret
```

## Branches and loops

A block is created with `create_block()`, or `create_block_with_params(&[types])`
when it needs to receive values; the second form hands back a `Value` per
parameter. `set_block(id)` chooses where subsequent ops are appended. Each block
ends with exactly one terminator: `jump`, `branch` or `ret`.

Both `jump` and `branch` carry the arguments for their targets, which is where
a phi would otherwise go. A loop is then just a block that jumps back to itself
with updated arguments:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::condcode::CondCode;
# use blitz::ir::types::Type;
// fn sum_to(n: i64) -> i64 {
//     let (mut acc, mut i) = (0, 1);
//     while i <= n { acc += i; i += 1; }
//     acc
// }
let mut b = FunctionBuilder::new("sum_to", &[Type::I64], &[Type::I64]);
let n = b.params()[0];

// The header carries the loop-carried state: (acc, i).
let (header, hdr) = b.create_block_with_params(&[Type::I64, Type::I64]);
let body = b.create_block();
let (exit, exit_p) = b.create_block_with_params(&[Type::I64]);

let zero = b.iconst(0, Type::I64);
let one = b.iconst(1, Type::I64);
b.jump(header, &[zero, one]);

b.set_block(header);
let (acc, i) = (hdr[0], hdr[1]);
let cond = b.icmp(CondCode::Sle, i, n);
b.branch(cond, body, exit, &[], &[acc]);   // args for the taken / not-taken edge

b.set_block(body);
let acc2 = b.add(acc, i);
let one2 = b.iconst(1, Type::I64);
let i2 = b.add(i, one2);
b.jump(header, &[acc2, i2]);               // back edge updates the state

b.set_block(exit);
b.ret(Some(exit_p[0]));

let func = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`sum_to(100)` returns 5050. Note that nothing here mentions a register or a
stack slot, and the loop-carried values are ordinary parameters.

## Memory

`store(addr, val)` and `load(addr, ty)` take an address as a `Value`, so pointer
arithmetic is just arithmetic; the optimizer folds what it can into x86
addressing modes:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::types::Type;
// fn bump(p: *mut i64) -> i64 { *p = 7; return *p + 1; }
let mut b = FunctionBuilder::new("bump", &[Type::I64], &[Type::I64]);
let p = b.params()[0];
let seven = b.iconst(7, Type::I64);
b.store(p, seven);
let v = b.load(p, Type::I64);
let one = b.iconst(1, Type::I64);
let r = b.add(v, one);
b.ret(Some(r));
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Store-to-load forwarding replaces that `load` with the 7 that was just stored,
and the addition then folds. `bump` never reads memory back; the store stays
because it is observable, and the return value is a constant:

```asm
bump:
    mov    $0x7,%rcx
    mov    $0x8,%rax
    mov    %rcx,(%rdi)
    ret
```

For locals, `create_stack_slot(size, align)` reserves frame space and
`stack_addr(slot)` gives you its address as an `I64` value. `global_addr(name)`
does the same for a named global.

## Calls

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::types::Type;
# let mut b = FunctionBuilder::new("f", &[Type::I64], &[Type::I64]);
# let x = b.params()[0];
let results = b.call("callee", &[x], &[Type::I64]);
# b.ret(Some(results[0]));
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Arguments and results follow the SysV AMD64 ABI: the first six integers in
registers, the first eight floats in XMM, the rest on the stack, all handled for
you. A callee that is not in the module becomes an undefined symbol for the
linker to resolve.

Pass several functions to `compile_module` to get one object file, and inlining
will consider calls between them.

## Optimization level

`CompileOptions::default()` is `-O1`: inlining, LICM, memory forwarding, dead
store elimination, full equality saturation. `CompileOptions` also selects the
cost model's goal, which decides what extraction optimizes for when two forms
are equivalent.

## When something is wrong

`finalize` reports `NoTerminator`, `BlockAlreadyTerminated`, `TypeMismatch`,
`ArgCountMismatch`, `NoBlocks`, `UndefinedValue` or `UnsealedBlock`, each naming
the block involved.

Past that point, set `BLITZ_VERIFY=1` to check the IR at every pass boundary and
the machine code after branch relaxation; it panics naming the stage that
produced the bad IR. `BLITZ_DEBUG=regalloc` (or `sched`, `licm`, `egraph`,
`split`, `asm`, `stats`, `all`) traces one phase, and `BLITZ_DEBUG_FN=name`
narrows it to a single function.
