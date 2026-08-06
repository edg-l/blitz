//! Canonicalization of the class references stored in the CFG.
//!
//! Passes that merge e-classes (forwarding, saturation) leave effectful ops
//! holding pre-merge `ClassId`s. Those ids still resolve, so every consumer
//! compensates by calling `find_immutable` on read. That works until one
//! forgets: `ca2e400` was a LICM miscompile caused by exactly that omission.
//!
//! Running this sweep after each merging pass removes the hazard at the source
//! instead of relying on ~30 scattered call sites to remember. It is not a
//! codegen change -- the memory passes already canonicalize at the point of
//! comparison, so nothing downstream sees different values -- and it is what
//! makes `BLITZ_VERIFY=strict` enforceable.

use crate::egraph::egraph::EGraph;
use crate::ir::function::Function;
use crate::ir::op::ClassId;

/// Rewrite every `ClassId` in `func`'s effectful ops to its canonical form.
///
/// Returns the number of references that were stale, which is what the tests
/// and the `alias` trace category report.
pub(super) fn canonicalize_class_refs(func: &mut Function, egraph: &EGraph) -> usize {
    let mut rewritten = 0;
    for block in &mut func.blocks {
        for op in &mut block.ops {
            op.for_each_class_id_mut(|id| {
                // NONE is a sentinel for an absent operand, not a class.
                if *id == ClassId::NONE {
                    return;
                }
                let canon = egraph.find_immutable(*id);
                if canon != *id {
                    *id = canon;
                    rewritten += 1;
                }
            });
        }
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::effectful::{EffOperand, EffectfulOp};
    use crate::ir::op::{Op, PureOp};
    use crate::ir::types::Type;
    use crate::verify::{VerifyLevel, verify_function_at};

    /// `fn f(a: I64) -> I64 { return a + 1; }` plus a merge that leaves the
    /// returned id stale.
    fn function_with_stale_ref() -> (Function, EGraph) {
        let mut b = FunctionBuilder::new("f", &[Type::I64], &[Type::I64]);
        let p = b.params().to_vec();
        let one = b.iconst(1, Type::I64);
        let sum = b.add(p[0], one);
        b.ret(Some(sum));
        let mut func = b.finalize().expect("finalize");
        let mut egraph = func.egraph.take().expect("builder attaches an e-graph");

        // The builder speaks `Value`; the class the CFG actually holds is on
        // the Ret it just emitted.
        let returned = match func.blocks[0].ops.last().unwrap() {
            EffectfulOp::Ret { val } => val.expect("has return value"),
            other => panic!("expected Ret, got {other:?}"),
        };

        let other = egraph.add(ENode {
            op: Op::Pure(PureOp::Iconst(4242, Type::I64)),
            children: smallvec![],
        });
        egraph.merge(returned.class(), other);
        egraph.rebuild();

        let stale = if egraph.find_immutable(returned.class()) != returned.class() {
            returned
        } else {
            EffOperand::Class(other)
        };
        *func.blocks[0].ops.last_mut().unwrap() = EffectfulOp::Ret { val: Some(stale) };
        (func, egraph)
    }

    #[test]
    fn rewrites_stale_reference() {
        let (mut func, egraph) = function_with_stale_ref();
        assert_eq!(canonicalize_class_refs(&mut func, &egraph), 1);

        let val = match func.blocks[0].ops.last().unwrap() {
            EffectfulOp::Ret { val } => val.expect("has return value"),
            other => panic!("expected Ret, got {other:?}"),
        };
        assert_eq!(egraph.find_immutable(val.class()), val.class());
    }

    #[test]
    fn makes_strict_verification_pass() {
        let (mut func, egraph) = function_with_stale_ref();
        assert!(
            !verify_function_at(&func, &egraph, VerifyLevel::Strict).is_empty(),
            "expected a strict violation before the sweep"
        );

        canonicalize_class_refs(&mut func, &egraph);

        assert_eq!(
            verify_function_at(&func, &egraph, VerifyLevel::Strict),
            Vec::<String>::new()
        );
    }

    #[test]
    fn is_idempotent() {
        let (mut func, egraph) = function_with_stale_ref();
        assert_eq!(canonicalize_class_refs(&mut func, &egraph), 1);
        assert_eq!(
            canonicalize_class_refs(&mut func, &egraph),
            0,
            "a second sweep must find nothing left to rewrite"
        );
    }
}
