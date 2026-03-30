use crate::egraph::addr_mode::apply_addr_mode_rules;
use crate::egraph::algebraic::apply_algebraic_rules;
use crate::egraph::egraph::EGraph;
use crate::egraph::isel::apply_isel_rules;
use crate::egraph::strength::apply_strength_reduction;

/// Options controlling how many rule-application iterations each phase runs.
pub struct CompileOptions {
    /// Maximum iterations for phase 1 (algebraic + constant folding).
    pub phase1_limit: u32,
    /// Maximum iterations for phase 2 (strength reduction).
    pub phase2_limit: u32,
    /// Maximum iterations for phase 3 (isel + addressing modes + LEA).
    pub phase3_limit: u32,
    /// Abort saturation if e-class count exceeds this value. Default: 500_000.
    pub max_classes: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            phase1_limit: 16,
            phase2_limit: 8,
            phase3_limit: 8,
            max_classes: 500_000,
        }
    }
}

/// Run all three compilation phases on the e-graph.
///
/// Returns `Err` with a diagnostic message if the e-class count exceeds
/// `opts.max_classes` during any phase.
pub fn run_phases(egraph: &mut EGraph, opts: &CompileOptions) -> Result<(), String> {
    // Phase 1: algebraic simplification + constant folding
    for iter in 0..opts.phase1_limit {
        let changed = apply_algebraic_rules(egraph);
        egraph.rebuild();
        check_blowup(egraph, opts, 1, iter)?;
        if !changed {
            break;
        }
    }

    // Phase 2: strength reduction
    for iter in 0..opts.phase2_limit {
        let changed = apply_strength_reduction(egraph);
        egraph.rebuild();
        check_blowup(egraph, opts, 2, iter)?;
        if !changed {
            break;
        }
    }

    // Phase 3: x86-64 isel + addressing modes + LEA formation
    for iter in 0..opts.phase3_limit {
        let mut changed = false;
        changed |= apply_isel_rules(egraph);
        changed |= apply_addr_mode_rules(egraph);
        egraph.rebuild();
        check_blowup(egraph, opts, 3, iter)?;
        if !changed {
            break;
        }
    }

    Ok(())
}

fn check_blowup(
    egraph: &mut EGraph,
    opts: &CompileOptions,
    phase: u32,
    iter: u32,
) -> Result<(), String> {
    let count = egraph.class_count();
    if count > opts.max_classes {
        Err(format!(
            "e-graph blowup: {count} classes after phase {phase} iteration {iter} \
             (limit: {}); saturation aborted",
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

    // 4.18: Phase ordering — algebraic before strength before isel
    #[test]
    fn phase_ordering() {
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
}
