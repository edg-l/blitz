//! Basic Blitz usage: compile simple functions to .o files.
//!
//! Run with: cargo run --example basic
//!
//! This creates `output.o` which can be linked with a C main:
//!   cc main.c output.o -o test && ./test

use blitz::compile::{CompileOptions, compile_module};
use blitz::ir::builder::FunctionBuilder;
use blitz::ir::condcode::CondCode;
use blitz::ir::types::Type;

/// Build: fn add(a: i64, b: i64) -> i64 { a + b }
fn build_add() -> blitz::ir::function::Function {
    let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], &[Type::I64]);
    let p = b.params().to_vec();
    let sum = b.add(p[0], p[1]);
    b.ret(Some(sum));
    b.finalize().expect("add")
}

/// Build: fn max(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
fn build_max() -> blitz::ir::function::Function {
    let mut b = FunctionBuilder::new("max", &[Type::I64, Type::I64], &[Type::I64]);
    let p = b.params().to_vec();
    let cond = b.icmp(CondCode::Sgt, p[0], p[1]);
    let result = b.select(cond, p[0], p[1]);
    b.ret(Some(result));
    b.finalize().expect("max")
}

/// Build: fn sum_to(n: i64) -> i64 { 1 + 2 + ... + n }
/// Uses a loop with block parameters (SSA phi equivalent).
fn build_sum_to() -> blitz::ir::function::Function {
    let mut b = FunctionBuilder::new("sum_to", &[Type::I64], &[Type::I64]);
    let n = b.params()[0];

    // BB1: loop header with params (acc: i64, i: i64)
    let (bb1, bb1_params) = b.create_block_with_params(&[Type::I64, Type::I64]);
    let acc = bb1_params[0];
    let i = bb1_params[1];

    // BB2: exit with param (result: i64)
    let (bb2, bb2_params) = b.create_block_with_params(&[Type::I64]);
    let result = bb2_params[0];

    // BB0 (entry): jump to loop with acc=0, i=1
    let zero = b.iconst(0, Type::I64);
    let one = b.iconst(1, Type::I64);
    b.jump(bb1, &[zero, one]);

    // BB1 (loop body):
    b.set_block(bb1);
    let new_acc = b.add(acc, i);
    let new_i = b.add(i, one);
    let cond = b.icmp(CondCode::Sle, new_i, n);
    b.branch(cond, bb1, bb2, &[new_acc, new_i], &[new_acc]);

    // BB2 (exit):
    b.set_block(bb2);
    b.ret(Some(result));

    b.finalize().expect("sum_to")
}

/// Build: fn optimized(x: i64) -> i64 { (x / 1) | -1 }
/// Demonstrates egraph algebraic optimizations:
///   - sdiv(x, 1) is eliminated (identity)
///   - or(x, -1) folds to -1 (annihilation)
///   - The whole function becomes: return -1
fn build_optimized() -> blitz::ir::function::Function {
    let mut b = FunctionBuilder::new("optimized", &[Type::I64], &[Type::I64]);
    let x = b.params().to_vec()[0];
    let one = b.iconst(1, Type::I64);
    let neg1 = b.iconst(-1, Type::I64);
    let div = b.sdiv(x, one); // sdiv(x, 1) = x
    let result = b.or(div, neg1); // or(x, -1) = -1
    b.ret(Some(result));
    b.finalize().expect("optimized")
}

/// Build: fn array_idx(base: i64, i: i64) -> i64 { base + i * 8 + 16 }
/// Demonstrates cross-category egraph optimization:
///   - strength reduction: mul(i, 8) -> shl(i, 3)
///   - addr mode fusion: add(base, shl(i, 3)) -> addr(scale=8)(base, i)
///   - LEA formation: add(addr, 16) -> x86_lea4(scale=8, disp=16)(base, i)
fn build_array_idx() -> blitz::ir::function::Function {
    let mut b = FunctionBuilder::new("array_idx", &[Type::I64, Type::I64], &[Type::I64]);
    let p = b.params().to_vec();
    let c8 = b.iconst(8, Type::I64);
    let c16 = b.iconst(16, Type::I64);
    let offset = b.mul(p[1], c8); // strength -> shl(i, 3)
    let addr = b.add(p[0], offset); // addr_mode -> scaled addressing
    let result = b.add(addr, c16); // lea4 fusion
    b.ret(Some(result));
    b.finalize().expect("array_idx")
}

fn main() {
    let opts = CompileOptions::default();

    // Compile all functions into one .o file
    let functions = vec![
        build_add(),
        build_max(),
        build_sum_to(),
        build_optimized(),
        build_array_idx(),
    ];
    let obj = compile_module(functions, &opts).expect("compilation failed");

    let path = std::path::Path::new("output.o");
    obj.write_to(path).expect("failed to write output.o");

    println!("Wrote output.o with {} functions:", obj.functions.len());
    for f in &obj.functions {
        println!(
            "  {} (offset={:#x}, size={} bytes)",
            f.name, f.offset, f.size
        );
    }
    println!();
    println!("Link with C:");
    println!("  cc main.c output.o -o test && ./test");
}
