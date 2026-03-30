use crate::x86::abi::sequentialize_copies;
use crate::x86::inst::{MachInst, Operand};
use crate::x86::reg::Reg;

/// Insert parallel copies for block parameter passing.
///
/// `copies` is a list of `(src_reg, dst_reg)` pairs from the jump/branch
/// arguments to block parameters. `temp` is a free register used to break
/// copy cycles.
///
/// Returns the `MachInst` sequence to insert before the jump/branch.
pub fn phi_copies(copies: &[(Reg, Reg)], temp: Reg) -> Vec<MachInst> {
    // Filter out self-copies (src == dst, already coalesced by the register allocator).
    let filtered: Vec<(Reg, Reg)> = copies
        .iter()
        .copied()
        .filter(|&(src, dst)| src != dst)
        .collect();

    if filtered.is_empty() {
        return Vec::new();
    }

    // Use sequentialize_copies to produce a safe sequential ordering,
    // including temp-register-based cycle breaking.
    let seq = sequentialize_copies(&filtered, temp);

    seq.into_iter()
        .map(|(src, dst)| MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86::reg::Reg;

    fn mov(src: Reg, dst: Reg) -> MachInst {
        MachInst::MovRR {
            dst: Operand::Reg(dst),
            src: Operand::Reg(src),
        }
    }

    #[test]
    fn simple_non_conflicting_copies() {
        // RAX -> RCX, RDX -> RSI: no conflict, both emitted.
        let copies = [(Reg::RAX, Reg::RCX), (Reg::RDX, Reg::RSI)];
        let insts = phi_copies(&copies, Reg::R11);
        assert_eq!(insts.len(), 2);
        // Both moves must appear (order may vary).
        assert!(insts.contains(&mov(Reg::RAX, Reg::RCX)));
        assert!(insts.contains(&mov(Reg::RDX, Reg::RSI)));
    }

    #[test]
    fn swap_cycle_uses_temp() {
        // RAX <-> RCX swap: requires a temporary.
        let copies = [(Reg::RAX, Reg::RCX), (Reg::RCX, Reg::RAX)];
        let insts = phi_copies(&copies, Reg::R11);
        // Simulate execution to verify correctness.
        use std::collections::HashMap;
        let mut state: HashMap<Reg, u64> = [(Reg::RAX, 10), (Reg::RCX, 20), (Reg::R11, 0)]
            .into_iter()
            .collect();
        for inst in &insts {
            if let MachInst::MovRR {
                dst: Operand::Reg(d),
                src: Operand::Reg(s),
            } = inst
            {
                let v = state[s];
                state.insert(*d, v);
            }
        }
        assert_eq!(state[&Reg::RAX], 20, "RAX should have RCX's original value");
        assert_eq!(state[&Reg::RCX], 10, "RCX should have RAX's original value");
    }

    #[test]
    fn three_way_cycle() {
        // RAX->RCX, RCX->RDX, RDX->RAX: three-way rotation.
        let copies = [
            (Reg::RAX, Reg::RCX),
            (Reg::RCX, Reg::RDX),
            (Reg::RDX, Reg::RAX),
        ];
        let insts = phi_copies(&copies, Reg::R11);
        use std::collections::HashMap;
        let mut state: HashMap<Reg, u64> =
            [(Reg::RAX, 1), (Reg::RCX, 2), (Reg::RDX, 3), (Reg::R11, 0)]
                .into_iter()
                .collect();
        for inst in &insts {
            if let MachInst::MovRR {
                dst: Operand::Reg(d),
                src: Operand::Reg(s),
            } = inst
            {
                let v = state[s];
                state.insert(*d, v);
            }
        }
        // After rotation: RCX=1, RDX=2, RAX=3
        assert_eq!(state[&Reg::RCX], 1);
        assert_eq!(state[&Reg::RDX], 2);
        assert_eq!(state[&Reg::RAX], 3);
    }

    #[test]
    fn self_copy_eliminated() {
        // RAX -> RAX should produce no instruction.
        let copies = [(Reg::RAX, Reg::RAX)];
        let insts = phi_copies(&copies, Reg::R11);
        assert!(insts.is_empty());
    }

    #[test]
    fn mixed_self_and_non_self() {
        // RAX -> RAX (eliminated), RDX -> RCX (kept).
        let copies = [(Reg::RAX, Reg::RAX), (Reg::RDX, Reg::RCX)];
        let insts = phi_copies(&copies, Reg::R11);
        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0], mov(Reg::RDX, Reg::RCX));
    }

    fn simulate(
        copies: &[(Reg, Reg)],
        initial: &[(Reg, u64)],
    ) -> std::collections::HashMap<Reg, u64> {
        let insts = phi_copies(copies, Reg::R11);
        let mut state: std::collections::HashMap<Reg, u64> = initial.iter().copied().collect();
        // Ensure R11 exists as scratch in case cycle-breaking uses it.
        state.entry(Reg::R11).or_insert(0);
        for inst in &insts {
            if let MachInst::MovRR {
                dst: Operand::Reg(d),
                src: Operand::Reg(s),
            } = inst
            {
                let v = state[s];
                state.insert(*d, v);
            }
        }
        state
    }

    #[test]
    fn four_way_cycle() {
        // A->B, B->C, C->D, D->A: four-way rotation.
        // RAX=1, RCX=2, RDX=3, RSI=4
        // After: RCX=1, RDX=2, RSI=3, RAX=4
        let copies = [
            (Reg::RAX, Reg::RCX),
            (Reg::RCX, Reg::RDX),
            (Reg::RDX, Reg::RSI),
            (Reg::RSI, Reg::RAX),
        ];
        let initial = [(Reg::RAX, 1), (Reg::RCX, 2), (Reg::RDX, 3), (Reg::RSI, 4)];
        let state = simulate(&copies, &initial);
        assert_eq!(state[&Reg::RCX], 1, "RCX should receive RAX's value");
        assert_eq!(state[&Reg::RDX], 2, "RDX should receive RCX's value");
        assert_eq!(state[&Reg::RSI], 3, "RSI should receive RDX's value");
        assert_eq!(state[&Reg::RAX], 4, "RAX should receive RSI's value");
    }

    #[test]
    fn multiple_independent_cycles() {
        // Two independent 2-cycles: RAX<->RCX and RDX<->RSI.
        let copies = [
            (Reg::RAX, Reg::RCX),
            (Reg::RCX, Reg::RAX),
            (Reg::RDX, Reg::RSI),
            (Reg::RSI, Reg::RDX),
        ];
        let initial = [
            (Reg::RAX, 10),
            (Reg::RCX, 20),
            (Reg::RDX, 30),
            (Reg::RSI, 40),
        ];
        let state = simulate(&copies, &initial);
        // First cycle: RAX<->RCX
        assert_eq!(state[&Reg::RAX], 20, "RAX should have RCX's original value");
        assert_eq!(state[&Reg::RCX], 10, "RCX should have RAX's original value");
        // Second cycle: RDX<->RSI
        assert_eq!(state[&Reg::RDX], 40, "RDX should have RSI's original value");
        assert_eq!(state[&Reg::RSI], 30, "RSI should have RDX's original value");
    }
}
