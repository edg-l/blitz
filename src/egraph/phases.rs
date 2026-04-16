use crate::egraph::addr_mode::apply_addr_mode_rules;
use crate::egraph::algebraic::apply_algebraic_rules;
use crate::egraph::distributive::apply_distributive_rules;
use crate::egraph::egraph::EGraph;
use crate::egraph::isel::apply_isel_rules;
use crate::egraph::known_bits::{apply_known_bits_rules, propagate_known_bits};
use crate::egraph::strength::apply_strength_reduction;

/// Options controlling equality saturation.
pub struct CompileOptions {
    /// Maximum iterations for the unified saturation loop.
    pub iteration_limit: u32,
    /// Abort saturation if e-class count exceeds this value. Default: 500_000.
    pub max_classes: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            iteration_limit: 16,
            max_classes: 500_000,
        }
    }
}

/// Run all rewrite rules in a single unified saturation loop.
///
/// Returns `Err` with a diagnostic message if the e-class count exceeds
/// `opts.max_classes` during any iteration.
pub fn run_phases(egraph: &mut EGraph, opts: &CompileOptions) -> Result<(), String> {
    let mut hit_limit = false;
    for iter in 0..opts.iteration_limit {
        let mut changed = false;
        changed |= apply_algebraic_rules(egraph);
        changed |= apply_strength_reduction(egraph);
        changed |= apply_distributive_rules(egraph, opts.max_classes);
        changed |= apply_isel_rules(egraph);
        changed |= apply_addr_mode_rules(egraph);
        changed |= propagate_known_bits(egraph);
        changed |= apply_known_bits_rules(egraph);
        egraph.rebuild();
        check_blowup(egraph, opts, iter)?;
        if !changed {
            break;
        }
        if iter == opts.iteration_limit - 1 {
            hit_limit = true;
        }
    }
    if hit_limit && crate::trace::is_enabled("egraph") {
        eprintln!(
            "[egraph] warning: saturation did not converge after {} iterations",
            opts.iteration_limit
        );
    }
    Ok(())
}

fn check_blowup(egraph: &mut EGraph, opts: &CompileOptions, iter: u32) -> Result<(), String> {
    let count = egraph.class_count();
    if count > opts.max_classes {
        Err(format!(
            "e-graph blowup: {count} classes after iteration {iter} (limit: {}); saturation aborted",
            opts.max_classes
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::egraph::enode::ENode;
    use crate::ir::op::{ClassId, Op};
    use crate::ir::types::Type;

    fn iconst(g: &mut EGraph, v: i64) -> ClassId {
        g.add(ENode {
            op: Op::Iconst(v, Type::I64),
            children: smallvec![],
        })
    }

    // 4.19: Integration test — a[i] = a[i] + 1 style IR
    // Build: Add(base_ptr, Mul(i, 8)) + Iconst(1) chain, run all phases,
    // verify fused addressing form exists in the e-graph.
    #[test]
    fn integration_array_index_plus_one() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        // Represent: base + i * 8 + 1
        // (a[i] address = base + i*8, value = *(base+i*8) + 1)
        let base = iconst(&mut g, 0x1000); // stand-in for a base pointer
        let i = iconst(&mut g, 3); // stand-in for index i=3

        let eight = iconst(&mut g, 8);
        let one = iconst(&mut g, 1);

        // i * 8
        let i_times_8 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![i, eight],
        });

        // base + i*8
        let addr = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, i_times_8],
        });

        // (base + i*8) + 1
        let val_plus_one = g.add(ENode {
            op: Op::Add,
            children: smallvec![addr, one],
        });

        run_phases(&mut g, &opts).expect("phases should complete without blowup");

        // After phases, i*8 should be equivalent to Shl(i, 3)
        let three = iconst(&mut g, 3);
        let shl3 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![i, three],
        });
        assert_eq!(
            g.find(i_times_8),
            g.find(shl3),
            "Mul(i, 8) should be equivalent to Shl(i, 3) after strength reduction"
        );

        // val_plus_one should contain X86Lea4{scale:1, disp:1} or similar LEA form
        let val_canon = g.find(val_plus_one);
        let val_class = g.class(val_canon);
        let has_lea = val_class
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::X86Lea4 { disp: 1, .. } | Op::X86Lea2));
        assert!(
            has_lea,
            "val+1 class should contain an X86Lea node after addr mode rules"
        );
    }

    // 4.18: Blowup protection test
    #[test]
    fn blowup_protection() {
        let mut g = EGraph::new();
        let opts = CompileOptions {
            max_classes: 5,
            ..Default::default()
        };

        // Insert enough classes to exceed the limit
        for i in 0..10i64 {
            let _ = g.add(ENode {
                op: Op::Iconst(i, Type::I64),
                children: smallvec![],
            });
        }

        let result = run_phases(&mut g, &opts);
        assert!(result.is_err(), "should error due to blowup");
    }

    // 4.18: Unified saturation — all rules fire every iteration
    #[test]
    fn unified_saturation() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let a = iconst(&mut g, 7);
        let zero = iconst(&mut g, 0);
        let two = iconst(&mut g, 2);

        // Add(a, 0) => a (algebraic)
        let add_zero = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, zero],
        });

        // Mul(a, 4) => Shl(a, 2) (strength reduction)
        let four = iconst(&mut g, 4);
        let mul4 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, four],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Algebraic: Add(a,0) = a
        assert_eq!(g.find(add_zero), g.find(a));

        // Strength: Mul(a,4) = Shl(a,2)
        let shl2 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, two],
        });
        assert_eq!(g.find(mul4), g.find(shl2));
    }

    // Cross-category interaction: Add(Mul(a, 4), 0) in one unified iteration.
    // Algebraic removes +0, strength reduces *4 to Shl, isel lowers to X86Shl.
    // The old phased approach would handle these sequentially; the unified loop
    // handles them in a single pass.
    #[test]
    fn cross_category_algebraic_strength_isel() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let four = iconst(&mut g, 4);
        let zero = iconst(&mut g, 0);
        let mul = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, four],
        });
        let add_zero = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul, zero],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // add_zero should equal mul (identity removal)
        assert_eq!(g.find(add_zero), g.find(mul));

        // mul class should contain Shl(a, 2) from strength reduction
        let two = iconst(&mut g, 2);
        let shl = g.add(ENode {
            op: Op::Shl,
            children: smallvec![a, two],
        });
        assert_eq!(g.find(mul), g.find(shl));

        // And isel should have produced X86Shl somewhere reachable
        // (the Proj0/X86Shl class is merged with the mul class)
        let mul_canon = g.find(mul);
        let class = g.class(mul_canon);
        let has_x86 = class
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::Proj0 | Op::X86Lea3 { .. }));
        assert!(has_x86, "unified saturation should produce x86 lowering");
    }

    // ── Unified saturation proofs ─────────────────────────────────────────────
    //
    // These tests verify that the unified saturation loop finds optimizations
    // that a sequential phased approach (algebraic to fixpoint, then strength
    // to fixpoint, then isel, etc.) would miss due to the phase-ordering problem.
    //
    // The key pattern: a later-phase rule (e.g. strength reduction) produces
    // output that an earlier-phase rule (e.g. algebraic shift combining) needs.
    // In phased execution, the earlier phase has already completed by the time
    // the later phase fires, so the cross-category optimization is never found.

    // Proof: Shl(Mul(param, 2), 3) should equal Shl(param, 4).
    //
    // Path: strength reduces Mul(param,2) -> Shl(param,1), then algebraic
    // shift combining fires on Shl(Shl(param,1), 3) -> Shl(param, 1+3=4).
    //
    // In phased execution, algebraic sees Shl(Mul_class, 3) during its phase.
    // Shift combining scans Mul_class and finds only Mul(param,2), not a Shl.
    // No match. Strength runs later and creates Shl(param,1), but algebraic
    // already finished. Shl(param, 4) is never discovered.
    #[test]
    fn unified_proof_strength_feeds_shift_combining() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let two = iconst(&mut g, 2);
        let three = iconst(&mut g, 3);

        let mul2 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![param, two],
        });
        let shl_outer = g.add(ENode {
            op: Op::Shl,
            children: smallvec![mul2, three],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        let four = iconst(&mut g, 4);
        let expected = g.add(ENode {
            op: Op::Shl,
            children: smallvec![param, four],
        });
        assert_eq!(
            g.find(shl_outer),
            g.find(expected),
            "Shl(Mul(param, 2), 3) should equal Shl(param, 4) via strength + shift combining"
        );
    }

    // Direct comparison: phased approach misses what unified finds.
    //
    // Runs algebraic to fixpoint, then strength to fixpoint (simulating the
    // old phased pipeline), and verifies the optimization is NOT found.
    // Then runs unified saturation on a fresh graph and verifies it IS found.
    #[test]
    fn phased_misses_unified_finds() {
        // --- Phased approach: algebraic to fixpoint, then strength to fixpoint ---
        let mut g = EGraph::new();
        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let two = iconst(&mut g, 2);
        let three = iconst(&mut g, 3);
        let mul2 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![param, two],
        });
        let shl_outer = g.add(ENode {
            op: Op::Shl,
            children: smallvec![mul2, three],
        });

        // Phase 1: algebraic to fixpoint
        for _ in 0..16 {
            if !apply_algebraic_rules(&mut g) {
                break;
            }
            g.rebuild();
        }
        // Phase 2: strength to fixpoint
        for _ in 0..16 {
            if !apply_strength_reduction(&mut g) {
                break;
            }
            g.rebuild();
        }

        // Phased approach should NOT find Shl(param, 4)
        let four = iconst(&mut g, 4);
        let combined = g.add(ENode {
            op: Op::Shl,
            children: smallvec![param, four],
        });
        assert_ne!(
            g.find(shl_outer),
            g.find(combined),
            "phased approach should NOT find Shl(param, 4) -- phase ordering problem"
        );

        // --- Unified approach on a fresh graph ---
        let mut g2 = EGraph::new();
        let param2 = g2.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let two2 = iconst(&mut g2, 2);
        let three2 = iconst(&mut g2, 3);
        let mul2_2 = g2.add(ENode {
            op: Op::Mul,
            children: smallvec![param2, two2],
        });
        let shl_outer2 = g2.add(ENode {
            op: Op::Shl,
            children: smallvec![mul2_2, three2],
        });

        run_phases(&mut g2, &CompileOptions::default()).expect("no blowup");

        let four2 = iconst(&mut g2, 4);
        let combined2 = g2.add(ENode {
            op: Op::Shl,
            children: smallvec![param2, four2],
        });
        assert_eq!(
            g2.find(shl_outer2),
            g2.find(combined2),
            "unified approach SHOULD find Shl(param, 4)"
        );
    }

    // Three-category chain: algebraic identity + strength reduction + shift combining.
    //
    // Input: Shl(Mul(Add(param, 0), 2), 3)
    //   1. Algebraic identity: Add(param, 0) -> param
    //   2. Strength reduction: Mul(param, 2) -> Shl(param, 1)
    //   3. Algebraic shift combining: Shl(Shl(param, 1), 3) -> Shl(param, 4)
    //
    // Steps 2->3 require unified saturation (strength output feeds algebraic).
    // Step 1->2 works in phased too, but the full chain 1->2->3 does not.
    #[test]
    fn three_category_chain_identity_strength_shift() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let zero = iconst(&mut g, 0);
        let two = iconst(&mut g, 2);
        let three = iconst(&mut g, 3);

        // Add(param, 0)
        let add_zero = g.add(ENode {
            op: Op::Add,
            children: smallvec![param, zero],
        });
        // Mul(Add(param, 0), 2)
        let mul2 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![add_zero, two],
        });
        // Shl(Mul(...), 3)
        let outer = g.add(ENode {
            op: Op::Shl,
            children: smallvec![mul2, three],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Full chain should produce Shl(param, 4)
        let four = iconst(&mut g, 4);
        let expected = g.add(ENode {
            op: Op::Shl,
            children: smallvec![param, four],
        });
        assert_eq!(
            g.find(outer),
            g.find(expected),
            "Shl(Mul(Add(param, 0), 2), 3) should equal Shl(param, 4)"
        );
    }

    // Strength + addr_mode chain via unified saturation.
    //
    // Input: Add(base, Shl(Mul(idx, 2), 1))
    //   1. Strength: Mul(idx, 2) -> Shl(idx, 1)
    //   2. Shift combining: Shl(Shl(idx, 1), 1) -> Shl(idx, 2)
    //   3. Addr_mode: Add(base, Shl(idx, 2)) -> Addr{scale:4, disp:0}(base, idx)
    //
    // Steps 1->2 require unified saturation. Step 3 then benefits from the
    // simplified Shl(idx, 2) being in the class (scale 4 is a valid x86 scale).
    #[test]
    fn strength_shift_combine_enables_addr_mode() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let base = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let idx = g.add(ENode {
            op: Op::Param(1, Type::I64),
            children: smallvec![],
        });
        let two = iconst(&mut g, 2);
        let one = iconst(&mut g, 1);

        // Mul(idx, 2)
        let mul2 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![idx, two],
        });
        // Shl(Mul(idx, 2), 1)
        let shl1 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![mul2, one],
        });
        // Add(base, Shl(Mul(idx, 2), 1))
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, shl1],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // After unified: Shl(Mul(idx,2), 1) -> Shl(Shl(idx,1), 1) -> Shl(idx, 2)
        // This enables Addr{scale:4} formation
        let add_canon = g.find(add);
        let add_class = g.class(add_canon);
        let has_scale4_addr = add_class.nodes.iter().any(|n| {
            matches!(
                n.op,
                Op::Addr { scale: 4, disp: 0 } | Op::X86Lea3 { scale: 4 }
            )
        });
        assert!(
            has_scale4_addr,
            "unified saturation should produce scale-4 addressing from Shl(Mul(idx,2), 1)"
        );
    }

    // Reassociation enables constant folding across add chains:
    // (param + 3) + 5 should produce param + 8 via reassociation + folding.
    #[test]
    fn reassociation_enables_folding() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let param = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let c3 = iconst(&mut g, 3);
        let c5 = iconst(&mut g, 5);
        let inner = g.add(ENode {
            op: Op::Add,
            children: smallvec![param, c3],
        });
        let outer = g.add(ENode {
            op: Op::Add,
            children: smallvec![inner, c5],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Should find param + 8 in the same class
        let c8 = iconst(&mut g, 8);
        let folded = g.add(ENode {
            op: Op::Add,
            children: smallvec![param, c8],
        });
        assert_eq!(
            g.find(outer),
            g.find(folded),
            "(param + 3) + 5 should equal param + 8 via reassociation"
        );
    }

    // ── Cross-rule interaction tests ─────────────────────────────────────────
    //
    // These tests verify that the NEW rules (comparison folding, distributive
    // factoring, known-bits) interact with EXISTING rules to produce
    // optimizations that neither could achieve alone.

    // Distributive factoring → strength reduction:
    // Add(Mul(a, 4), Mul(a, 8)) → Mul(a, 12) → Add(Shl(a, 2), Shl(a, 3)) or LEA
    #[test]
    fn factoring_feeds_strength_reduction() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let c4 = iconst(&mut g, 4);
        let c8 = iconst(&mut g, 8);

        let mul4 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, c4],
        });
        let mul8 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, c8],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul4, mul8],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Factoring: Mul(a,4) + Mul(a,8) = Mul(a, 12)
        let c12 = iconst(&mut g, 12);
        let mul12 = g.add(ENode {
            op: Op::Mul,
            children: smallvec![a, c12],
        });
        assert_eq!(
            g.find(add),
            g.find(mul12),
            "factoring should produce Mul(a, 12)"
        );
    }

    // Comparison folding → algebraic simplification:
    // Select(Icmp(x == x), Add(a, 0), garbage) → Add(a, 0) → a
    #[test]
    fn comparison_fold_exposes_algebraic() {
        use crate::ir::condcode::CondCode;

        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let x = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let a = g.add(ENode {
            op: Op::Param(1, Type::I64),
            children: smallvec![],
        });
        let zero = iconst(&mut g, 0);
        let garbage = iconst(&mut g, 999);

        let cond = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![x, x],
        });
        let add_zero = g.add(ENode {
            op: Op::Add,
            children: smallvec![a, zero],
        });
        let sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![cond, add_zero, garbage],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Comparison fold: Icmp(Eq, x, x) always true → Select = Add(a, 0)
        // Algebraic identity: Add(a, 0) = a
        // Combined: Select(...) = a
        assert_eq!(
            g.find(sel),
            g.find(a),
            "comparison fold + identity should reduce Select to a"
        );
    }

    // Known-bits constant promotion → further constant folding:
    // Add(Shl(1, 3), Shl(1, 2)) → known-bits promotes to Iconst(8) + Iconst(4)
    // → constant folding: 8 + 4 = 12
    #[test]
    fn known_bits_promotion_feeds_constant_folding() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let c1 = iconst(&mut g, 1);
        let c3 = iconst(&mut g, 3);
        let c2 = iconst(&mut g, 2);

        let shl3 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![c1, c3],
        });
        let shl2 = g.add(ENode {
            op: Op::Shl,
            children: smallvec![c1, c2],
        });
        let add = g.add(ENode {
            op: Op::Add,
            children: smallvec![shl3, shl2],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // known-bits promotes Shl(1,3)=8 and Shl(1,2)=4 to constants
        // then constant folding: 8 + 4 = 12
        let c12 = iconst(&mut g, 12);
        assert_eq!(
            g.find(add),
            g.find(c12),
            "known-bits promotion + constant folding should yield 12"
        );
    }

    // Strength reduction (div→shr) + known-bits → constant promotion:
    // UDiv(Iconst(16), Iconst(4)) → Shr(16, 2) → known-bits → Iconst(4)
    #[test]
    fn div_strength_feeds_known_bits_promotion() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let c16 = iconst(&mut g, 16);
        let c4 = iconst(&mut g, 4);

        let udiv = g.add(ENode {
            op: Op::UDiv,
            children: smallvec![c16, c4],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Strength: UDiv(16, 4) → Shr(16, 2)
        // Known-bits: Shr(Iconst(16), Iconst(2)) → all bits known → Iconst(4)
        // (or direct constant folding of UDiv(16,4) = 4)
        let expected = iconst(&mut g, 4);
        assert_eq!(
            g.find(udiv),
            g.find(expected),
            "UDiv(16, 4) should reduce to 4 via strength + known-bits or constant fold"
        );
    }

    // Nested Select cascade: comparison folding on outer + inner Selects.
    // Select(Icmp(a==a), Select(Icmp(3<5), x, y), z)
    //   → Select(always-true, Select(...), z) = Select(Icmp(3<5), x, y)
    //   → Select(always-true, x, y) = x
    #[test]
    fn nested_select_cascade_fold() {
        use crate::ir::condcode::CondCode;

        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let a = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let x = iconst(&mut g, 42);
        let y = iconst(&mut g, 99);
        let z = iconst(&mut g, 0);

        let c3 = iconst(&mut g, 3);
        let c5 = iconst(&mut g, 5);

        // Inner: Select(Icmp(Slt, 3, 5), x, y) → x (since 3 < 5 is true)
        let inner_cond = g.add(ENode {
            op: Op::Icmp(CondCode::Slt),
            children: smallvec![c3, c5],
        });
        let inner_sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![inner_cond, x, y],
        });

        // Outer: Select(Icmp(Eq, a, a), inner_sel, z) → inner_sel (since a==a)
        let outer_cond = g.add(ENode {
            op: Op::Icmp(CondCode::Eq),
            children: smallvec![a, a],
        });
        let outer_sel = g.add(ENode {
            op: Op::Select,
            children: smallvec![outer_cond, inner_sel, z],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Outer fold: Select(true, inner, z) = inner
        // Inner fold: Select(true, x, y) = x = Iconst(42)
        assert_eq!(
            g.find(outer_sel),
            g.find(x),
            "nested Select cascade should fold to x=42"
        );
    }

    // Factoring → strength reduction → addr mode:
    // Add(base, Add(Mul(idx, 2), Mul(idx, 2))) → Mul(idx, 4) → addr scale 4
    #[test]
    fn factoring_to_strength_to_addr_mode() {
        let mut g = EGraph::new();
        let opts = CompileOptions::default();

        let base = g.add(ENode {
            op: Op::Param(0, Type::I64),
            children: smallvec![],
        });
        let idx = g.add(ENode {
            op: Op::Param(1, Type::I64),
            children: smallvec![],
        });
        let c2 = iconst(&mut g, 2);

        let mul_a = g.add(ENode {
            op: Op::Mul,
            children: smallvec![idx, c2],
        });
        let mul_b = g.add(ENode {
            op: Op::Mul,
            children: smallvec![idx, c2],
        });
        let inner_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![mul_a, mul_b],
        });
        let outer_add = g.add(ENode {
            op: Op::Add,
            children: smallvec![base, inner_add],
        });

        run_phases(&mut g, &opts).expect("no blowup");

        // Factoring: Mul(idx,2) + Mul(idx,2) = Mul(idx, 4)
        // Strength: Mul(idx, 4) = Shl(idx, 2)
        // Addr mode: Add(base, Shl(idx, 2)) → Addr{scale:4} or LEA3{scale:4}
        let outer_canon = g.find(outer_add);
        let class = g.class(outer_canon);
        let has_scale4 = class
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::Addr { scale: 4, .. } | Op::X86Lea3 { scale: 4 }));
        assert!(
            has_scale4,
            "factoring + strength + addr mode should produce scale-4 addressing"
        );
    }
}
