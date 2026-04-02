# tinyc Struct Support Plan (Revised)

## Overview

Add struct type support to the TinyC frontend of the Blitz compiler, covering type system extensions, lexer/parser changes, codegen for stack-allocated structs with field access, sizeof, struct copy, and by-pointer function parameters.

## Requirements

- **Explicit**: Named struct definitions with typed fields. Field access via `.` and `->`. Struct assignment (whole-struct copy). `sizeof(struct Foo)`. Struct function parameters passed by hidden pointer. Struct return is out of scope.
- **Inferred**: Structs are always stack-allocated (no SSA for aggregates). Struct pointers work like any other pointer. Recursive structs via pointer indirection only. Struct definitions must precede usage. No anonymous structs, no nested struct definitions, no bitfields.
- **Assumptions**: Natural alignment (each field aligned to its own size, struct padded to max field alignment). No struct return values. No unions. No initializer lists.

## Architecture Decision

Structs are represented as `CType::Struct(String)`. A `StructRegistry` (`HashMap<String, StructLayout>`) is computed once at codegen time and threaded into `FnCtx`. All struct variables live in stack slots; field access computes `stack_addr + offset` and loads/stores the field's scalar IR type. Whole-struct assignment emits field-by-field copy. By-value params lowered to hidden I64 pointer.

## Implementation Plan

### Phase 1: Type System Foundation (4 tasks)

- [ ] Task 1.1: Add `Struct(String)` variant to `CType` enum in `crates/tinyc/src/ast.rs` (Complexity: Low)
- [ ] Task 1.2: Update `CType` methods for `Struct` variant (Complexity: Medium)
  - `to_ir_type()`: `CType::Struct(_) => None`
  - `bit_width()`: `CType::Struct(_) => panic!("use StructRegistry::byte_size()")`
  - `is_integer()`: change to `!matches!(self, CType::Void | CType::Ptr(_) | CType::Struct(_))`
  - `rank()`: `CType::Struct(_) => panic!("rank() called on struct type")`
  - `promoted()`: `CType::Struct(_) => panic!("promoted() called on struct type")`
- [ ] Task 1.3: Add `is_struct()` method returning `matches!(self, CType::Struct(_))` (Complexity: Low)
- [ ] Task 1.4: Update `pointee_size()` to handle `Ptr(Struct(_))` -- panic directing callers to use StructRegistry. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 2: Lexer and Parser -- Struct Definitions and Variables (7 tasks)

- [ ] Task 2.1: Add `Struct`, `Dot`, `Arrow` tokens to `Token` enum in `lexer.rs` (Complexity: Low)
- [ ] Task 2.2: Add lexer rules: `"struct"` -> Struct, `'.'` -> Dot, `"->"` -> Arrow (check chars[pos+1] == '>' before Minus) (Complexity: Low)
- [ ] Task 2.3: Add `struct_defs: Vec<(String, Vec<(String, CType)>)>` to `Program` struct in `ast.rs`. Initialize to empty in Parser::parse. (Complexity: Low)
- [ ] Task 2.4: Implement `parse_struct_def()`: consume Struct, expect Ident(name), expect LBrace, parse fields (parse_type + Ident + Semi), expect RBrace, expect Semi. (Complexity: Medium)
- [ ] Task 2.5: Add dispatch in `Parser::parse` top-level loop: when `Token::Struct` and tokens[pos+2] is LBrace, call parse_struct_def and push to struct_defs. Otherwise fall through to function parsing. (Complexity: Medium)
- [ ] Task 2.6: Update `peek_is_type()` to include `Token::Struct`. Update `parse_type()`: Token::Struct -> advance, expect Ident(name), return CType::Struct(name). Trailing `*` loop handles pointers. (Complexity: Medium)
- [ ] Task 2.7: Make `VarDecl.init` optional -- atomic change across 3 files: (a) `init: Expr` -> `init: Option<Expr>` in ast.rs, (b) parser: if Token::Semi after name, return init: None; else expect Assign, parse as Some(expr), (c) codegen VarDecl arm: `if let Some(init) = init { ... } else { /* just allocate */ }`, (d) addr_analysis walk_stmt: `if let Some(init) = init { walk_expr(init, set); }` (Complexity: Medium)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 3: AST and Parser -- Field Access and Assignment (6 tasks)

- [ ] Task 3.1: Add `Expr::FieldAccess { expr: Box<Expr>, field: String }` to Expr enum. (Complexity: Low)
- [ ] Task 3.2: Parse `.field` as postfix at bp 25: advance, expect Ident(field), wrap in FieldAccess. (Complexity: Medium)
- [ ] Task 3.3: Parse `->field` as postfix at bp 25: advance, expect Ident(field), desugar to `FieldAccess { expr: Deref(lhs), field }`. (Complexity: Medium)
- [ ] Task 3.4: Add `Stmt::FieldAssign { expr: Expr, field: String, value: Expr }` to Stmt enum. (Complexity: Low)
- [ ] Task 3.5: In parse_stmt, after parsing LHS: if LHS is `FieldAccess { expr, field }` and next is Assign, produce FieldAssign. (Complexity: Medium)
- [ ] Task 3.6: Update cast disambiguation in parse_primary: extend next_is_type check to match `Token::Struct`. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 4a: Codegen -- Struct Registry and Variable Allocation (6 tasks)

- [ ] Task 4a.1: Create `StructLayout` struct: `fields: Vec<(String, CType, u32)>` (name, type, offset), `byte_size: u32`, `alignment: u32`. (Complexity: Low)
- [ ] Task 4a.2: Create `StructRegistry` as `HashMap<String, StructLayout>`. Implement `build_struct_registry()` with natural alignment layout: for each field, align to field size, compute offset. Struct size padded to max alignment. Scalars: bit_width()/8, pointers: 8, nested structs: lookup in registry (error if not found = forward reference). All sizes u32. (Complexity: High)
- [ ] Task 4a.3: Cycle detection in build_struct_registry: direct self-reference -> error. `Ptr(_)` always 8 bytes, no recursion. Empty structs -> error. Duplicate field names -> error. (Complexity: Medium)
- [ ] Task 4a.4: Add `struct_registry: &StructRegistry` to FnCtx. Thread through Codegen::generate -> compile_fn -> FnCtx::new. (Complexity: Medium)
- [ ] Task 4a.5: VarDecl codegen for structs: create_stack_slot(byte_size, alignment), store in stack_slots. Init None: no store. Init Some: call emit_struct_copy (stub as todo!()). (Complexity: Medium)
- [ ] Task 4a.6: Expr::Var for struct types: return stack_addr as value (do NOT load). (Complexity: Medium)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 4b: Codegen -- Field Access and Assignment (6 tasks)

- [ ] Task 4b.1: Helper `resolve_field(registry, struct_name, field) -> (u32, CType)`. Error on unknown struct/field. (Complexity: Low)
- [ ] Task 4b.2: FieldAccess codegen: compile inner expr (struct addr), resolve field offset+type, compute addr+offset, load scalar field. (Complexity: Medium)
- [ ] Task 4b.3: Nested struct field: if field type is Struct, return address without loading. (Complexity: Medium)
- [ ] Task 4b.4: FieldAssign codegen: compute field address, compile value, emit_convert, store. (Complexity: Medium)
- [ ] Task 4b.5: FieldAssign with struct value: call emit_struct_copy (todo!() until Phase 5). (Complexity: Low)
- [ ] Task 4b.6: Audit all `to_ir_type().unwrap()` callsites in codegen: guard Expr::Var (done in 4a.6), compile_fn params (Phase 6), emit_convert (panic guard), val_to_flags (panic guard), Expr::Index with struct pointee (return address). (Complexity: Medium)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 5: Sizeof, Struct Copy, Address-of (6 tasks)

- [ ] Task 5.1: Sizeof codegen for structs: lookup registry byte_size, emit iconst. (Complexity: Low)
- [ ] Task 5.2: Implement `emit_struct_copy(ctx, dst_addr, src_addr, struct_name)`: for each field, compute src/dst addrs, load+store scalars, recurse for nested structs. (Complexity: High)
- [ ] Task 5.3: Replace todo!() stubs in 4a.5 and 4b.5 with emit_struct_copy calls. (Complexity: Low)
- [ ] Task 5.4: Stmt::Assign for struct types: skip emit_convert, use emit_struct_copy. (Complexity: Medium)
- [ ] Task 5.5: AddrOf on FieldAccess: compute field address, return as Ptr(field_type). (Complexity: Medium)
- [ ] Task 5.6: Update addr_analysis: (a) FieldAccess in walk_expr: recurse into inner, (b) FieldAssign in walk_stmt: walk expr and value, (c) AddrOf(FieldAccess(Var(name))): mark name as addressed. (Complexity: Medium)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 6: Struct Function Parameters (5 tasks)

- [ ] Task 6.1: Error on struct params in extern declarations. (Complexity: Low)
- [ ] Task 6.2: fn_signatures pre-scan: keep CType::Struct as-is (IR lowering at call/def sites only). (Complexity: Low)
- [ ] Task 6.3: compile_fn: struct params -> Type::I64 in IR signature. Create local stack slot, copy from pointer param to local via emit_struct_copy. (Complexity: High)
- [ ] Task 6.4: Expr::Call: struct args -> allocate temp slot, copy struct into it, pass temp address as I64. Skip emit_convert for struct args. (Complexity: High)
- [ ] Task 6.5: Error on struct return types. (Complexity: Low)

- [ ] **Checkpoint**: `cargo check -p tinyc` and `cargo test -p tinyc` pass.

### Phase 7: Tests (7 tasks)

- [ ] Task 7.1: Struct declaration + field access: `p.x = 10; p.y = 20; return p.x + p.y;` -> 30
- [ ] Task 7.2: Struct pointer + arrow: `set_x(&p, 42); return p.x;` -> 42
- [ ] Task 7.3: Whole-struct copy: `b = a; return b.x + b.y;` -> 3
- [ ] Task 7.4: Sizeof: `sizeof(struct Pair)` with int+long -> 16
- [ ] Task 7.5: Nested struct: `o.i.v = 7; o.w = 3; return o.i.v + o.w;` -> 10
- [ ] Task 7.6: By-value param: `sum(p)` where sum reads p.x + p.y -> 42
- [ ] Task 7.7: Error cases: recursive struct, undefined struct, struct param in extern

- [ ] **Checkpoint**: All tests pass.

- [ ] **Final Audit** -- Verify each task has implementation, resolve gaps.

## Edge Cases & Risks

- **Recursive struct detection**: Direct self-reference errors. Pointer-to-self is valid (8 bytes).
- **Forward references**: Definitions must appear before use (processed in order).
- **Alignment padding**: `{ char c; long l; }` = 16 bytes (1 + 7 padding + 8).
- **to_ir_type().unwrap() panics**: Audited in 4b.6 with panic guards.
- **Struct in boolean context**: val_to_flags panic guard catches it.
- **C ABI incompatibility**: Hidden-pointer convention for by-value. Extern decl error prevents silent miscompilation.
- **Empty structs**: Error in build_struct_registry.
