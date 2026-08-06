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
`stack_addr(slot)` gives you its address as an `I64` value; `alloc_layout` below
does both from a struct layout. `global_addr(name)` does the same for a named
global.

## Structs

There is no struct type, and you do not need one. A struct is a block of memory
plus a constant offset per field, so the layout is the frontend's decision —
padding rules belong to your source language, not to the backend — and blitz only
ever sees the resulting integers.

`Layout` works those integers out for you. `Layout::c` applies C and SysV rules,
`Layout::packed` adds no padding, `Layout::array` strides by the element's size,
and `Layout::explicit` takes offsets you state outright:

```rust
# use blitz::ir::layout::Layout;
# use blitz::ir::types::Type;
// struct Point { int x; int y; long tag; }
let point = Layout::c(&[Type::I32, Type::I32, Type::I64]);
assert_eq!(point.offsets(), [0, 4, 8]);
assert_eq!(point.size(), 16);
assert_eq!(point.align(), 8);
```

`field_addr`, `load_field` and `store_field` then address a field by index, using
the field's own type as the access width:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::layout::Layout;
# use blitz::ir::types::Type;
# let point = Layout::c(&[Type::I32, Type::I32, Type::I64]);
// fn dot(p: *const Point) -> i64 { p->x * p->y + p->tag }
let mut b = FunctionBuilder::new("dot", &[Type::I64], &[Type::I64]);
let p = b.params()[0];

let x = b.load_field(p, &point, 0);      // I32, at offset 0
let y = b.load_field(p, &point, 1);      // I32, at offset 4
let tag = b.load_field(p, &point, 2);    // I64, at offset 8

let prod = b.mul(x, y);
let prod = b.sext(prod, Type::I64);      // widen before adding to the I64 tag
let r = b.add(prod, tag);
b.ret(Some(r));
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

A *local* struct is a stack slot sized and aligned from the same layout:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::layout::Layout;
# use blitz::ir::types::Type;
# let point = Layout::c(&[Type::I32, Type::I32, Type::I64]);
let mut b = FunctionBuilder::new("local", &[], &[Type::I64]);
let base = b.alloc_layout(&point);       // stack slot + its address

let six = b.iconst(6, Type::I32);
b.store_field(base, &point, 0, six);     // p.x = 6
let tag = b.load_field(base, &point, 2);
b.ret(Some(tag));
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Arrays use `elem_addr(base, index, elem_size)`, which is `base + index *
elem_size` written the way the addressing-mode rules expect, so an element size
of 1, 2, 4 or 8 folds into the scale:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::types::Type;
// fn nth(a: *const i64, i: i64) -> i64 { a[i] }
let mut b = FunctionBuilder::new("nth", &[Type::I64, Type::I64], &[Type::I64]);
let p = b.params().to_vec();
let addr = b.elem_addr(p[0], p[1], 8);
let v = b.load(addr, Type::I64);
b.ret(Some(v));
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

That is one instruction, the index scaled inside the addressing mode:

```asm
nth:
    mov    (%rdi,%rsi,8),%rax
    ret
```

Prefer these over hand-written `add`/`iconst`. The rules that fold arithmetic
into an addressing mode match on *structure*, so an address built a different way
can silently miss the fold and cost an instruction with nothing to report it.
Going through `offset`, `elem_addr` and the field helpers is the supported way
onto that path. Nothing stops you doing the arithmetic yourself when you need a
shape they do not cover.

Layouts nest. `Layout::compose` takes members rather than types, so a struct can
contain a struct or an array, and a nested field's `ty` is `None` because it has
no single access width — take its `field_addr` and read its own fields.

Two fields of one struct are two locations, and the alias analysis knows it: a
store to `p->x` does not stop a load of `p->tag` being forwarded, because their
`[offset, offset + width)` ranges do not overlap.

## Tagged enums

A tagged enum is a struct whose first field is the tag and whose second is a
payload sized for the widest variant. Declare the payload at that width and read
it with the variant's own type:

```rust
# use blitz::ir::builder::FunctionBuilder;
# use blitz::ir::condcode::CondCode;
# use blitz::ir::layout::Layout;
# use blitz::ir::types::Type;
// enum Value { Int(i64), Float(f64) }
// as { i64 tag; union { i64; f64 } payload; }
let value = Layout::c(&[Type::I64, Type::I64]);
const TAG: usize = 0;
const PAYLOAD: usize = 1;
const TAG_INT: i64 = 0;

// fn as_i64(v: *const Value) -> i64 {
//     match v { Int(i) => i, Float(f) => f as i64 }
// }
let mut b = FunctionBuilder::new("as_i64", &[Type::I64], &[Type::I64]);
let v = b.params()[0];

let int_arm = b.create_block();
let float_arm = b.create_block();
let (done, done_p) = b.create_block_with_params(&[Type::I64]);

let tag = b.load_field(v, &value, TAG);
let want = b.iconst(TAG_INT, Type::I64);
let is_int = b.icmp(CondCode::Eq, tag, want);
b.branch(is_int, int_arm, float_arm, &[], &[]);

b.set_block(int_arm);
let i = b.load_field(v, &value, PAYLOAD);     // the declared type, I64
b.jump(done, &[i]);

b.set_block(float_arm);
let addr = b.field_addr(v, &value, PAYLOAD);  // same address...
let f = b.load(addr, Type::F64);              // ...read as F64
let as_int = b.float_to_int(f, Type::I64);
b.jump(done, &[as_int]);

b.set_block(done);
b.ret(Some(done_p[0]));                       // whichever arm ran passed its value
# let _ = b.finalize()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The union is the two reads: one address, two types. `load_field` uses the type
the layout declares, so the variant that disagrees takes `field_addr` and names
its own — which is the reason the two are separate calls. Because the payload is
read with the variant's type, the `F64` arm lands in an XMM register and the
`I64` arm in a general-purpose one without you saying so.

The merge block's parameter is the `match` expression's value. That is the
pattern for any expression whose value depends on which branch ran; with more
variants, chain the comparisons and have every arm jump to the same `done`.

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
