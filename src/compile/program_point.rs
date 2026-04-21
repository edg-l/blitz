use crate::ir::op::Op;
use crate::schedule::scheduler::ScheduledInst;

/// A stable, total-ordered program point within a function.
///
/// Points are lexicographically ordered: first by `block` index (in the
/// function-scope schedule), then by `inst` index within the block.
///
/// Special sentinel values:
/// - `BLOCK_ENTRY(b)`: `inst = 0` -- before any instruction in block `b`.
/// - `BLOCK_EXIT(b)`:  `inst = u32::MAX` -- after all instructions in block `b`.
///
/// Regular instruction points use `inst` values in `1..u32::MAX - 1` so they
/// sort after `BLOCK_ENTRY` and before `BLOCK_EXIT`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ProgramPoint {
    pub block: u32,
    pub inst: u32,
}

impl ProgramPoint {
    /// The point immediately before any scheduled instruction in `block_idx`.
    pub fn block_entry(block_idx: usize) -> Self {
        ProgramPoint {
            block: block_idx as u32,
            inst: 0,
        }
    }

    /// The point immediately after all scheduled instructions in `block_idx`.
    pub fn block_exit(block_idx: usize) -> Self {
        ProgramPoint {
            block: block_idx as u32,
            inst: u32::MAX,
        }
    }

    /// The point for a regular instruction at `inst_idx` within `block_idx`.
    ///
    /// `inst_idx` must be in the range `1..u32::MAX - 1` (i.e. not 0 or
    /// `u32::MAX`), which is guaranteed as long as schedules remain under
    /// ~4 billion instructions.
    pub fn inst_point(block_idx: usize, inst_idx: usize) -> Self {
        debug_assert!(
            inst_idx > 0 && inst_idx < (u32::MAX - 1) as usize,
            "inst_idx {inst_idx} must be in 1..u32::MAX-1"
        );
        ProgramPoint {
            block: block_idx as u32,
            inst: inst_idx as u32,
        }
    }

    /// Map a barrier op (by its 0-based index among non-terminator effectful ops
    /// in the block) to its program point in the scheduled instruction list.
    ///
    /// Barrier result instructions (`LoadResult`, `CallResult`, `VoidCallBarrier`,
    /// `StoreBarrier`) appear in the schedule in the same order as the
    /// corresponding effectful ops in `block.ops`. This function finds the
    /// schedule position of the `barrier_idx`-th such instruction and returns
    /// it as a `ProgramPoint`.
    ///
    /// If the schedule has no barrier-result instruction at `barrier_idx`
    /// (e.g. barrier_idx is out of range), the function falls back to
    /// `(barrier_idx + 1)` as the `inst` value.
    pub fn barrier_point(block_idx: usize, barrier_idx: usize, schedule: &[ScheduledInst]) -> Self {
        // Collect schedule indices of all barrier-result instructions, in order.
        let barrier_positions: Vec<usize> = schedule
            .iter()
            .enumerate()
            .filter(|(_, inst)| {
                matches!(
                    &inst.op,
                    Op::LoadResult(..)
                        | Op::CallResult(..)
                        | Op::VoidCallBarrier
                        | Op::StoreBarrier
                )
            })
            .map(|(i, _)| i)
            .collect();

        // inst values for real instructions start at 1 (0 = BLOCK_ENTRY).
        let inst = barrier_positions
            .get(barrier_idx)
            .map(|&p| p + 1)
            .unwrap_or(barrier_idx + 1) as u32;

        debug_assert!(
            inst < u32::MAX - 1,
            "barrier_point inst {inst} exceeds maximum (schedule too large)"
        );
        ProgramPoint {
            block: block_idx as u32,
            inst,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering() {
        // Same block: smaller inst comes first.
        let p0 = ProgramPoint::block_entry(0);
        let p1 = ProgramPoint::inst_point(0, 1);
        let p2 = ProgramPoint::inst_point(0, 5);
        let px = ProgramPoint::block_exit(0);
        assert!(p0 < p1);
        assert!(p1 < p2);
        assert!(p2 < px);

        // Different blocks: block 0 < block 1 regardless of inst.
        let q0 = ProgramPoint::block_exit(0);
        let q1 = ProgramPoint::block_entry(1);
        assert!(q0 < q1);

        // Reflexivity.
        assert_eq!(p1, p1);
        assert!(p1 <= p1);
    }

    #[test]
    fn entry_exit_distinct() {
        let entry = ProgramPoint::block_entry(0);
        let exit = ProgramPoint::block_exit(0);
        assert_ne!(entry, exit);
        assert!(entry < exit);

        // Entry of block 1 comes after exit of block 0 in total order.
        let entry1 = ProgramPoint::block_entry(1);
        assert!(exit < entry1);

        // Different blocks have different entry/exit.
        assert_ne!(ProgramPoint::block_entry(0), ProgramPoint::block_entry(1));
        assert_ne!(ProgramPoint::block_exit(0), ProgramPoint::block_exit(1));
    }

    #[test]
    fn barrier_point_finds_barrier_result() {
        use crate::egraph::extract::VReg;
        use crate::ir::types::Type;

        // Build a minimal schedule with a CallResult at schedule index 1.
        // Barrier index 0 corresponds to the 0th barrier-result instruction
        // in the schedule, which is the CallResult at position 1.
        let v0 = VReg(0);
        let v1 = VReg(1);
        let v2 = VReg(2);

        let schedule = vec![
            ScheduledInst {
                op: Op::Iconst(1, Type::I64),
                dst: v0,
                operands: vec![],
            },
            ScheduledInst {
                op: Op::CallResult(0, Type::I64),
                dst: v1,
                operands: vec![],
            },
            ScheduledInst {
                op: Op::Iconst(2, Type::I64),
                dst: v2,
                operands: vec![],
            },
        ];

        // The CallResult is at schedule index 1; inst = 1+1 = 2.
        let pp = ProgramPoint::barrier_point(0, 0, &schedule);
        assert_eq!(pp.block, 0);
        assert_eq!(pp.inst, 2);

        // It should be strictly between BLOCK_ENTRY and BLOCK_EXIT.
        assert!(ProgramPoint::block_entry(0) < pp);
        assert!(pp < ProgramPoint::block_exit(0));

        // Barrier index 1 with no second barrier falls back to barrier_idx+1 = 2.
        let pp2 = ProgramPoint::barrier_point(0, 1, &schedule);
        assert_eq!(pp2.block, 0);
        assert_eq!(pp2.inst, 2); // fallback: barrier_idx+1 = 2
    }
}
