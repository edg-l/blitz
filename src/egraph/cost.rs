use crate::ir::op::Op;

/// Optimization objective for the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptGoal {
    Latency,
    Throughput,
    CodeSize,
    Balanced,
}

/// Per-operation cost tuple (latency cycles, reciprocal throughput, code size in bytes).
struct CostTuple {
    latency: f64,
    throughput: f64,
    size: f64,
}

impl CostTuple {
    fn weighted(&self, goal: OptGoal) -> f64 {
        match goal {
            OptGoal::Latency => self.latency,
            OptGoal::Throughput => self.throughput,
            OptGoal::CodeSize => self.size,
            OptGoal::Balanced => self.latency + self.throughput + self.size * 0.1,
        }
    }
}

/// Cost model that assigns a scalar cost to each e-node operation.
///
/// Generic IR operations that have no x86-64 encoding return `f64::INFINITY`.
/// Costs are based on Agner Fog's instruction tables for modern x86-64.
pub struct CostModel {
    pub goal: OptGoal,
}

impl CostModel {
    pub fn new(goal: OptGoal) -> Self {
        Self { goal }
    }

    /// Cost of a single node (not including children).
    ///
    /// Returns `f64::INFINITY` for generic IR ops that have no x86-64 encoding.
    pub fn cost(&self, op: &Op) -> f64 {
        match op {
            // ── Constants: free (materialized as immediate or folded into insn) ──
            Op::Iconst(..) | Op::Fconst(_) => 0.0,

            // ── Function parameters: free (value lives in an ABI register on entry) ──
            Op::Param(..) => 0.0,

            // ── Stack slot address: free (LEA emitted during lowering) ───────────
            Op::StackAddr(..) => 0.0,

            // ── Global variable address: free (LEA [RIP+disp32] emitted during lowering) ──
            Op::GlobalAddr(_) => 0.0,

            // ── Block parameters: free (value comes from predecessor block) ──────────
            Op::BlockParam(..) => 0.0,

            // ── Load result placeholder: free (instruction emitted by effectful lowering) ──
            Op::LoadResult(_, _) => 0.0,

            // ── Call result placeholder: free (result captured after CallDirect) ──
            Op::CallResult(_, _) => 0.0,

            // ── Addr: inlined into load/store, no separate instruction ───────────
            Op::Addr { .. } => 0.0,

            // ── Projections: no separate instruction ─────────────────────────────
            Op::Proj0 | Op::Proj1 => 0.0,

            // ── x86-64 ALU: latency=1, throughput=0.25, size=3 ──────────────────
            Op::X86Add | Op::X86Sub | Op::X86And | Op::X86Or | Op::X86Xor => CostTuple {
                latency: 1.0,
                throughput: 0.25,
                size: 3.0,
            }
            .weighted(self.goal),

            // ── x86-64 shifts (variable count via CL): latency=1, throughput=0.5, size=3 ──
            Op::X86Shl | Op::X86Sar | Op::X86Shr => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 3.0,
            }
            .weighted(self.goal),

            // ── x86-64 immediate-form shifts: slightly cheaper (no CL constraint) ─
            Op::X86ShlImm(_) | Op::X86ShrImm(_) | Op::X86SarImm(_) => {
                CostTuple {
                    latency: 1.0,
                    throughput: 0.5,
                    size: 3.0, // same encoding size as CL form
                }
                .weighted(self.goal)
                    * 0.9
            } // small discount to prefer imm form when available

            // ── LEA variants ─────────────────────────────────────────────────────
            Op::X86Lea2 => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::X86Lea3 { .. } => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 5.0,
            }
            .weighted(self.goal),
            Op::X86Lea4 { .. } => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 7.0,
            }
            .weighted(self.goal),

            // ── X86Idiv / X86Div: latency=35, throughput=21, size=5 (64-bit div) ──
            Op::X86Idiv | Op::X86Div => CostTuple {
                latency: 35.0,
                throughput: 21.0,
                size: 5.0,
            }
            .weighted(self.goal),

            // ── X86Imul3: latency=3, throughput=1.0, size=4 ──────────────────────
            Op::X86Imul3 => CostTuple {
                latency: 3.0,
                throughput: 1.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Cmov: latency=1, throughput=0.5, size=4 ───────────────────────
            Op::X86Cmov(_) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Setcc: latency=1, throughput=0.5, size=3 ──────────────────────
            Op::X86Setcc(_) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 3.0,
            }
            .weighted(self.goal),

            // ── x86 FP ops SSE2 double (sd) ───────────────────────────────────────
            Op::X86Addsd | Op::X86Subsd => CostTuple {
                latency: 3.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::X86Mulsd => CostTuple {
                latency: 5.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::X86Divsd | Op::X86Sqrtsd => CostTuple {
                latency: 13.0,
                throughput: 4.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── x86 FP ops SSE single (ss) ────────────────────────────────────────
            Op::X86Addss | Op::X86Subss => CostTuple {
                latency: 3.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::X86Mulss => CostTuple {
                latency: 5.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::X86Divss | Op::X86Sqrtss => CostTuple {
                latency: 13.0,
                throughput: 4.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Movsx/X86Movzx: latency=1, throughput=0.25, size=4 ────────────
            Op::X86Movsx { .. } | Op::X86Movzx { .. } => CostTuple {
                latency: 1.0,
                throughput: 0.25,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Trunc: free — upper bits are simply ignored on x86-64 ──────────
            Op::X86Trunc { .. } => 0.0,

            // ── X86Bitcast: one MOVQ instruction for cross-class, or free for same ─
            Op::X86Bitcast { from, to } => {
                if from.is_integer() == to.is_integer() {
                    // Same register class (int->int or float->float same size): just a copy.
                    0.0
                } else {
                    // Cross-class (int<->float): MOVQ instruction.
                    CostTuple {
                        latency: 1.0,
                        throughput: 0.33,
                        size: 5.0,
                    }
                    .weighted(self.goal)
                }
            }

            // ── Generic IR ops: must be lowered before extraction ─────────────────
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::UDiv
            | Op::SDiv
            | Op::URem
            | Op::SRem
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Shl
            | Op::Shr
            | Op::Sar
            | Op::Sext(_)
            | Op::Zext(_)
            | Op::Trunc(_)
            | Op::Bitcast(_)
            | Op::Icmp(_)
            | Op::Fadd
            | Op::Fsub
            | Op::Fmul
            | Op::Fdiv
            | Op::Fsqrt
            | Op::Select => f64::INFINITY,

            // Spill pseudo-ops are never costed by the e-graph.
            Op::SpillStore(_) | Op::SpillLoad(_) | Op::XmmSpillStore(_) | Op::XmmSpillLoad(_) => {
                unreachable!("spill pseudo-ops are not part of the e-graph")
            }

            Op::StoreBarrier | Op::VoidCallBarrier => {
                unreachable!("barrier pseudo-ops are not part of the e-graph")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x86add_balanced_is_finite() {
        let cm = CostModel::new(OptGoal::Balanced);
        let c = cm.cost(&Op::X86Add);
        assert!(c.is_finite(), "X86Add should have finite balanced cost");
    }

    #[test]
    fn add_is_infinite() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Add), f64::INFINITY);
    }

    #[test]
    fn iconst_is_free() {
        let cm = CostModel::new(OptGoal::Latency);
        use crate::ir::types::Type;
        assert_eq!(cm.cost(&Op::Iconst(42, Type::I64)), 0.0);
    }

    #[test]
    fn addr_is_free() {
        let cm = CostModel::new(OptGoal::CodeSize);
        assert_eq!(cm.cost(&Op::Addr { scale: 4, disp: 0 }), 0.0);
    }

    #[test]
    fn lea_vs_add_cost_size() {
        let cm = CostModel::new(OptGoal::CodeSize);
        let add_cost = cm.cost(&Op::X86Add);
        let lea2_cost = cm.cost(&Op::X86Lea2);
        // X86Add size=3, X86Lea2 size=4 — add is cheaper by code size
        assert!(add_cost < lea2_cost);
    }

    #[test]
    fn x86imul3_higher_latency_than_add() {
        let cm = CostModel::new(OptGoal::Latency);
        let add_cost = cm.cost(&Op::X86Add);
        let imul_cost = cm.cost(&Op::X86Imul3);
        assert!(
            imul_cost > add_cost,
            "imul3 latency=3 should exceed add latency=1"
        );
    }

    #[test]
    fn proj0_proj1_free() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Proj0), 0.0);
        assert_eq!(cm.cost(&Op::Proj1), 0.0);
    }

    #[test]
    fn select_is_infinite() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Select), f64::INFINITY);
    }

    #[test]
    fn x86movsx_has_finite_cost() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::X86Movsx {
            from: Type::I32,
            to: Type::I64,
        });
        assert!(cost.is_finite(), "X86Movsx should have finite cost");
    }

    #[test]
    fn x86movzx_has_finite_cost() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::X86Movzx {
            from: Type::I8,
            to: Type::I64,
        });
        assert!(cost.is_finite(), "X86Movzx should have finite cost");
    }

    #[test]
    fn x86trunc_is_free() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::X86Trunc {
            from: Type::I64,
            to: Type::I32,
        });
        assert_eq!(cost, 0.0, "X86Trunc should be free");
    }

    #[test]
    fn sext_is_infinite() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(
            cm.cost(&Op::Sext(Type::I64)),
            f64::INFINITY,
            "generic Sext should have infinite cost"
        );
    }
}
