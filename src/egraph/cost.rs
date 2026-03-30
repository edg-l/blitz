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

            // ── Block parameters: free (value comes from predecessor block) ──────────
            Op::BlockParam(..) => 0.0,

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

            // ── x86-64 shifts: latency=1, throughput=0.5, size=3 ────────────────
            Op::X86Shl | Op::X86Sar | Op::X86Shr => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 3.0,
            }
            .weighted(self.goal),

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
}
