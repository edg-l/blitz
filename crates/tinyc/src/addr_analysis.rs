use std::collections::HashSet;

use crate::ast::{Expr, Stmt, UnaryOp};

/// Recursively walk all expressions and statements to find variables whose
/// address is taken via `&var_name`.
pub(crate) fn find_addressed_vars(stmts: &[Stmt]) -> HashSet<String> {
    let mut set = HashSet::new();
    for stmt in stmts {
        walk_stmt(stmt, &mut set);
    }
    set
}

fn walk_expr(expr: &Expr, set: &mut HashSet<String>) {
    match expr {
        Expr::UnaryOp {
            op: UnaryOp::AddrOf,
            expr: inner,
        } => {
            if let Expr::Var(name) = inner.as_ref() {
                set.insert(name.clone());
            }
            walk_expr(inner, set);
        }
        Expr::IntLit(_) => {}
        Expr::StringLit(_) => {}
        Expr::Var(_) => {}
        Expr::BinOp { lhs, rhs, .. } => {
            walk_expr(lhs, set);
            walk_expr(rhs, set);
        }
        Expr::UnaryOp { expr: inner, .. } => {
            walk_expr(inner, set);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, set);
            }
        }
        Expr::Cast { expr: inner, .. } => {
            walk_expr(inner, set);
        }
        Expr::Sizeof(_) => {}
        Expr::Index { base, index } => {
            walk_expr(base, set);
            walk_expr(index, set);
        }
    }
}

fn walk_stmt(stmt: &Stmt, set: &mut HashSet<String>) {
    match stmt {
        Stmt::Return(Some(expr)) => walk_expr(expr, set),
        Stmt::Return(None) => {}
        Stmt::ExprStmt(expr) => walk_expr(expr, set),
        Stmt::VarDecl { init, .. } => walk_expr(init, set),
        Stmt::Assign { expr, .. } => walk_expr(expr, set),
        Stmt::DerefAssign { addr_expr, value } => {
            walk_expr(addr_expr, set);
            walk_expr(value, set);
        }
        Stmt::IndexAssign { base, index, value } => {
            walk_expr(base, set);
            walk_expr(index, set);
            walk_expr(value, set);
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            walk_expr(cond, set);
            for s in then_body {
                walk_stmt(s, set);
            }
            if let Some(else_stmts) = else_body {
                for s in else_stmts {
                    walk_stmt(s, set);
                }
            }
        }
        Stmt::While { cond, body } => {
            walk_expr(cond, set);
            for s in body {
                walk_stmt(s, set);
            }
        }
    }
}
