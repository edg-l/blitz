use crate::ir::op::{MachOp, Op, PseudoOp, PureOp};

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

    /// Cost of getting a constant into a register with `mov r, imm32`.
    ///
    /// `Iconst` is priced at 0.0 because a constant is normally an immediate
    /// field of the instruction that reads it, and so costs nothing of its own.
    /// An operand position with no immediate encoding is the exception: there
    /// the constant needs a register and a `mov` to fill it, and only the
    /// consumer knows that. See [`Self::operand_needs_register`].
    pub fn const_materialization(&self) -> f64 {
        CostTuple {
            latency: 1.0,
            throughput: 0.25,
            size: 7.0,
        }
        .weighted(self.goal)
    }

    /// Whether an operand of `op` has to be a register, so a constant there is
    /// paid for at [`Self::const_materialization`] rather than being free.
    ///
    /// x86-64 addressing modes encode a displacement, never an immediate base
    /// or index, so every operand of an address computation is a register.
    pub fn operand_needs_register(op: &Op) -> bool {
        matches!(
            op,
            Op::Pure(PureOp::Addr { .. })
                | Op::Mach(MachOp::X86Lea2)
                | Op::Mach(MachOp::X86Lea3 { .. })
                | Op::Mach(MachOp::X86Lea4 { .. })
        )
    }

    /// Cost of a single node (not including children).
    ///
    /// Returns `f64::INFINITY` for generic IR ops that have no x86-64 encoding.
    pub fn cost(&self, op: &Op) -> f64 {
        match op {
            // ── Constants: free (materialized as immediate or folded into insn) ──
            Op::Pure(PureOp::Iconst(..)) | Op::Pure(PureOp::Fconst(_, _)) => 0.0,

            // ── Function parameters: free (value lives in an ABI register on entry) ──
            Op::Pure(PureOp::Param(..)) => 0.0,

            // ── Stack slot address: free (LEA emitted during lowering) ───────────
            Op::Pseudo(PseudoOp::StackAddr(..)) => 0.0,

            // ── Global variable address: free (LEA [RIP+disp32] emitted during lowering) ──
            Op::Pseudo(PseudoOp::GlobalAddr(_)) => 0.0,

            // ── Block parameters: free (value comes from predecessor block) ──────────
            Op::Pure(PureOp::BlockParam(..)) => 0.0,

            // ── Load result placeholder: free (instruction emitted by effectful lowering) ──
            Op::Pseudo(PseudoOp::LoadResult(_, _)) => 0.0,

            // ── Call result placeholder: free (result captured after CallDirect) ──
            Op::Pseudo(PseudoOp::CallResult(_, _)) => 0.0,

            // ── Addr: inlined into load/store, no separate instruction ───────────
            Op::Pure(PureOp::Addr { .. }) => 0.0,

            // ── Projections: no separate instruction ─────────────────────────────
            Op::Pure(PureOp::Proj0) | Op::Pure(PureOp::Proj1) => 0.0,

            // ── x86-64 ALU: latency=1, throughput=0.25, size=3 ──────────────────
            Op::Mach(MachOp::X86Add)
            | Op::Mach(MachOp::X86Sub)
            | Op::Mach(MachOp::X86And)
            | Op::Mach(MachOp::X86Or)
            | Op::Mach(MachOp::X86Xor) => CostTuple {
                latency: 1.0,
                throughput: 0.25,
                size: 3.0,
            }
            .weighted(self.goal),

            // ── x86-64 immediate-form ALU ────────────────────────────────────────
            //
            // Priced against the register form it replaces, not on its own: the
            // node that form needs in addition is an `Iconst`, which costs 0.0
            // here, so the `mov r, imm32` materializing it is invisible. The
            // real comparison is against `mov` + the register op, seven bytes.
            //
            // An `imm8` form is three bytes against those seven, so it wins by a
            // margin no tie-break has to be invented for. An `imm32` form is six
            // against seven -- one byte -- and pricing it to win costs
            // `tests/lit/control/main_falls_off_end.c` its compile: it changes
            // the schedule at a point the allocator already cannot colour.
            // Measured, the wide case is worth -1.4pp of instructions on `lit`
            // and `fuzz` and 0.13pp on `bench`, which is not a capacity failure's
            // worth. It comes back when the credit can be made honest -- it is
            // only owed where the constant has a single use, and the e-graph has
            // no parents map to ask.
            Op::Mach(MachOp::X86AddI(imm))
            | Op::Mach(MachOp::X86SubI(imm))
            | Op::Mach(MachOp::X86AndI(imm))
            | Op::Mach(MachOp::X86OrI(imm))
            | Op::Mach(MachOp::X86XorI(imm)) => CostTuple {
                latency: 1.0,
                throughput: 0.25,
                size: if (-128..=127).contains(imm) { 1.0 } else { 4.0 },
            }
            .weighted(self.goal),

            // ── x86-64 shifts (variable count via CL): latency=1, throughput=0.5, size=3 ──
            Op::Mach(MachOp::X86Shl) | Op::Mach(MachOp::X86Sar) | Op::Mach(MachOp::X86Shr) => {
                CostTuple {
                    latency: 1.0,
                    throughput: 0.5,
                    size: 3.0,
                }
                .weighted(self.goal)
            }

            // ── x86-64 immediate-form shifts: slightly cheaper (no CL constraint) ─
            Op::Mach(MachOp::X86ShlImm(_))
            | Op::Mach(MachOp::X86ShrImm(_))
            | Op::Mach(MachOp::X86SarImm(_))
            | Op::Mach(MachOp::X86RolImm(_)) => {
                CostTuple {
                    latency: 1.0,
                    throughput: 0.5,
                    size: 3.0, // same encoding size as CL form
                }
                .weighted(self.goal)
                    * 0.9
            } // small discount to prefer imm form when available

            // ── x86 flag-only compare with immediate: no register output; slightly
            //    cheaper than Proj1(X86Sub) since we don't pay for the sub's
            //    dst register write. imm=0 lowers to `test r, r` (2 bytes, even
            //    cheaper). imm!=0 lowers to `cmp r, imm8/imm32`.
            Op::Mach(MachOp::X86CmpI { imm, .. }) => {
                let size = if *imm == 0 {
                    2.0
                } else if (-128..=127).contains(imm) {
                    3.0
                } else {
                    6.0
                };
                CostTuple {
                    latency: 1.0,
                    throughput: 0.25,
                    size,
                }
                .weighted(self.goal)
                    * 0.9
            } // discount so extraction prefers it over Proj1(X86Sub) when possible

            // ── LEA variants ─────────────────────────────────────────────────────
            Op::Mach(MachOp::X86Lea2) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Lea3 { .. }) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 5.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Lea4 { .. }) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 7.0,
            }
            .weighted(self.goal),

            // ── X86Idiv / X86Div: latency=35, throughput=21, size=5 (64-bit div) ──
            Op::Mach(MachOp::X86Idiv(..)) | Op::Mach(MachOp::X86Div(..)) => CostTuple {
                latency: 35.0,
                throughput: 21.0,
                size: 5.0,
            }
            .weighted(self.goal),

            // ── X86Imul3: latency=3, throughput=1.0, size=4 ──────────────────────
            Op::Mach(MachOp::X86Imul3) => CostTuple {
                latency: 3.0,
                throughput: 1.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Cmov: latency=1, throughput=0.5, size=4 ───────────────────────
            Op::Mach(MachOp::X86Cmov(_)) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Setcc: latency=1, throughput=0.5, size=3 ──────────────────────
            Op::Mach(MachOp::X86Setcc(_)) => CostTuple {
                latency: 1.0,
                throughput: 0.5,
                size: 3.0,
            }
            .weighted(self.goal),

            // ── x86 FP ops SSE2 double (sd) ───────────────────────────────────────
            Op::Mach(MachOp::X86Addsd) | Op::Mach(MachOp::X86Subsd) => CostTuple {
                latency: 3.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Mulsd) => CostTuple {
                latency: 5.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Divsd) | Op::Mach(MachOp::X86Sqrtsd) => CostTuple {
                latency: 13.0,
                throughput: 4.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── x86 FP ops SSE single (ss) ────────────────────────────────────────
            Op::Mach(MachOp::X86Addss) | Op::Mach(MachOp::X86Subss) => CostTuple {
                latency: 3.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Mulss) => CostTuple {
                latency: 5.0,
                throughput: 0.5,
                size: 4.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Divss) | Op::Mach(MachOp::X86Sqrtss) => CostTuple {
                latency: 13.0,
                throughput: 4.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── x86 FP conversion ops ─────────────────────────────────────────────
            Op::Mach(MachOp::X86Cvtsi2sd(_)) | Op::Mach(MachOp::X86Cvtsi2ss(_)) => CostTuple {
                latency: 4.0,
                throughput: 1.0,
                size: 5.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Cvttsd2si(_)) | Op::Mach(MachOp::X86Cvttss2si(_)) => CostTuple {
                latency: 4.0,
                throughput: 1.0,
                size: 5.0,
            }
            .weighted(self.goal),
            Op::Mach(MachOp::X86Cvtsd2ss) | Op::Mach(MachOp::X86Cvtss2sd) => CostTuple {
                latency: 3.0,
                throughput: 1.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── x86 FP comparison ops ─────────────────────────────────────────────
            Op::Mach(MachOp::X86Ucomisd)
            | Op::Mach(MachOp::X86Ucomiss)
            | Op::Mach(MachOp::X86UcomisdCc(_))
            | Op::Mach(MachOp::X86UcomissCc(_)) => CostTuple {
                latency: 3.0,
                throughput: 1.0,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Movsx/X86Movzx: latency=1, throughput=0.25, size=4 ────────────
            Op::Mach(MachOp::X86Movsx { .. }) | Op::Mach(MachOp::X86Movzx { .. }) => CostTuple {
                latency: 1.0,
                throughput: 0.25,
                size: 4.0,
            }
            .weighted(self.goal),

            // ── X86Trunc: free — upper bits are simply ignored on x86-64 ──────────
            Op::Mach(MachOp::X86Trunc { .. }) => 0.0,

            // ── X86Bitcast: one MOVQ instruction for cross-class, or free for same ─
            Op::Mach(MachOp::X86Bitcast { from, to }) => {
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
            //
            // One arm for the whole type, with no exceptions. This and `lower.rs`'s
            // rejection used to be two hand-written lists of the same 30 variants,
            // and they had drifted: this one priced `Fcmp(OrdEq)` and
            // `Fcmp(UnordNe)` finitely because those two skipped isel and were
            // lowered by a path of their own. They no longer skip it -- isel gives
            // them `MachOp::X86UcomisdCc`/`X86UcomissCc` -- so every pure op is
            // unlowered here and the two sites cannot disagree again.
            Op::Pure(_) => f64::INFINITY,
            // Spill pseudo-ops are never costed by the e-graph.
            Op::Pseudo(PseudoOp::SpillStore(_))
            | Op::Pseudo(PseudoOp::SpillLoad(_))
            | Op::Pseudo(PseudoOp::XmmSpillStore(_))
            | Op::Pseudo(PseudoOp::XmmSpillLoad(_)) => {
                unreachable!("spill pseudo-ops are not part of the e-graph")
            }

            Op::Pseudo(PseudoOp::StoreBarrier)
            | Op::Pseudo(PseudoOp::VoidCallBarrier)
            | Op::Pseudo(PseudoOp::TerminatorArgs(_)) => {
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
        let c = cm.cost(&Op::Mach(MachOp::X86Add));
        assert!(c.is_finite(), "X86Add should have finite balanced cost");
    }

    #[test]
    fn add_is_infinite() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Pure(PureOp::Add)), f64::INFINITY);
    }

    #[test]
    fn iconst_is_free() {
        let cm = CostModel::new(OptGoal::Latency);
        use crate::ir::types::Type;
        assert_eq!(cm.cost(&Op::Pure(PureOp::Iconst(42, Type::I64))), 0.0);
    }

    #[test]
    fn addr_is_free() {
        let cm = CostModel::new(OptGoal::CodeSize);
        assert_eq!(cm.cost(&Op::Pure(PureOp::Addr { scale: 4, disp: 0 })), 0.0);
    }

    #[test]
    fn lea_vs_add_cost_size() {
        let cm = CostModel::new(OptGoal::CodeSize);
        let add_cost = cm.cost(&Op::Mach(MachOp::X86Add));
        let lea2_cost = cm.cost(&Op::Mach(MachOp::X86Lea2));
        // X86Add size=3, X86Lea2 size=4 — add is cheaper by code size
        assert!(add_cost < lea2_cost);
    }

    #[test]
    fn x86imul3_higher_latency_than_add() {
        let cm = CostModel::new(OptGoal::Latency);
        let add_cost = cm.cost(&Op::Mach(MachOp::X86Add));
        let imul_cost = cm.cost(&Op::Mach(MachOp::X86Imul3));
        assert!(
            imul_cost > add_cost,
            "imul3 latency=3 should exceed add latency=1"
        );
    }

    #[test]
    fn proj0_proj1_free() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Pure(PureOp::Proj0)), 0.0);
        assert_eq!(cm.cost(&Op::Pure(PureOp::Proj1)), 0.0);
    }

    #[test]
    fn select_is_infinite() {
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(cm.cost(&Op::Pure(PureOp::Select)), f64::INFINITY);
    }

    #[test]
    fn x86movsx_has_finite_cost() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::Mach(MachOp::X86Movsx {
            from: Type::I32,
            to: Type::I64,
        }));
        assert!(cost.is_finite(), "X86Movsx should have finite cost");
    }

    #[test]
    fn x86movzx_has_finite_cost() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::Mach(MachOp::X86Movzx {
            from: Type::I8,
            to: Type::I64,
        }));
        assert!(cost.is_finite(), "X86Movzx should have finite cost");
    }

    #[test]
    fn x86trunc_is_free() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        let cost = cm.cost(&Op::Mach(MachOp::X86Trunc {
            from: Type::I64,
            to: Type::I32,
        }));
        assert_eq!(cost, 0.0, "X86Trunc should be free");
    }

    #[test]
    fn sext_is_infinite() {
        use crate::ir::types::Type;
        let cm = CostModel::new(OptGoal::Balanced);
        assert_eq!(
            cm.cost(&Op::Pure(PureOp::Sext(Type::I64))),
            f64::INFINITY,
            "generic Sext should have infinite cost"
        );
    }
}
