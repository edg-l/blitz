//! A set of VRegs as a bit per VReg.
//!
//! The splitter measures register pressure by holding one live set per program
//! point and counting the members of a register class in each. As a
//! `BTreeSet<VReg>` that costs a heap node per live value per point, and the
//! count costs a class lookup per member; on a function whose splitter runs many
//! rounds it dominates compile time. A bit per VReg makes the per-point copy a
//! `memcpy` of `live/64` words and the count a popcount against a mask of the
//! class's VRegs.
//!
//! Iteration is ascending by VReg index, matching `BTreeSet<VReg>`, so callers
//! that depend on a deterministic order keep the order they had.
//!
//! The set grows on insert rather than trapping an out-of-range VReg: a caller
//! sizes it from the VReg count it knows about, and a pass that mints more is
//! not a bug.

use crate::egraph::extract::VReg;

const BITS: usize = 64;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VRegSet {
    words: Vec<u64>,
}

impl VRegSet {
    /// An empty set sized for VRegs `0..vregs` without reallocating.
    pub fn with_capacity(vregs: usize) -> Self {
        Self {
            words: vec![0; vregs.div_ceil(BITS)],
        }
    }

    pub fn insert(&mut self, v: VReg) {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1u64 << bit;
    }

    pub fn remove(&mut self, v: VReg) {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        if let Some(word) = self.words.get_mut(w) {
            *word &= !(1u64 << bit);
        }
    }

    pub fn contains(&self, v: VReg) -> bool {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        self.words.get(w).is_some_and(|word| word >> bit & 1 == 1)
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    pub fn len(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Add every member of `other`.
    pub fn union_with(&mut self, other: &VRegSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (w, o) in self.words.iter_mut().zip(&other.words) {
            *w |= *o;
        }
    }

    /// How many members are also in `mask`. This is the pressure query: `mask`
    /// holds the VRegs of one register class.
    pub fn count_in(&self, mask: &VRegSet) -> u32 {
        self.words
            .iter()
            .zip(&mask.words)
            .map(|(w, m)| (w & m).count_ones())
            .sum()
    }

    /// Members in ascending VReg order.
    pub fn iter(&self) -> impl Iterator<Item = VReg> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            BitsOf(word).map(move |bit| VReg((w * BITS + bit as usize) as u32))
        })
    }
}

impl Extend<VReg> for VRegSet {
    fn extend<I: IntoIterator<Item = VReg>>(&mut self, iter: I) {
        for v in iter {
            self.insert(v);
        }
    }
}

impl FromIterator<VReg> for VRegSet {
    fn from_iter<I: IntoIterator<Item = VReg>>(iter: I) -> Self {
        let mut set = Self::default();
        set.extend(iter);
        set
    }
}

/// The set bits of one word, lowest first.
struct BitsOf(u64);

impl Iterator for BitsOf {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros();
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(vregs: &[u32]) -> VRegSet {
        vregs.iter().map(|&i| VReg(i)).collect()
    }

    #[test]
    fn membership_survives_word_boundaries() {
        let s = set(&[0, 63, 64, 65, 200]);
        for &v in &[0u32, 63, 64, 65, 200] {
            assert!(s.contains(VReg(v)), "v{v} must be a member");
        }
        for &v in &[1u32, 62, 66, 199, 201, 100_000] {
            assert!(!s.contains(VReg(v)), "v{v} must not be a member");
        }
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn iteration_is_ascending_like_a_btreeset() {
        let s = set(&[200, 64, 0, 65, 63]);
        let got: Vec<u32> = s.iter().map(|v| v.0).collect();
        assert_eq!(got, vec![0, 63, 64, 65, 200]);
    }

    #[test]
    fn a_vreg_past_the_capacity_grows_the_set() {
        let mut s = VRegSet::with_capacity(8);
        s.insert(VReg(1000));
        assert!(s.contains(VReg(1000)));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn removing_a_vreg_past_the_end_is_not_an_error() {
        let mut s = set(&[1]);
        s.remove(VReg(9999));
        assert_eq!(s.len(), 1);
        s.remove(VReg(1));
        assert!(s.is_empty());
    }

    #[test]
    fn count_in_counts_the_intersection() {
        let live = set(&[1, 2, 64, 65, 130]);
        let gpr = set(&[1, 64, 130, 500]);
        assert_eq!(live.count_in(&gpr), 3);
        assert_eq!(live.count_in(&VRegSet::default()), 0);
    }

    /// A mask shorter than the live set must not read past its own words.
    #[test]
    fn count_in_tolerates_a_shorter_mask() {
        let live = set(&[1, 500]);
        let mask = set(&[1]);
        assert_eq!(live.count_in(&mask), 1);
    }

    #[test]
    fn union_takes_the_wider_of_the_two() {
        let mut a = set(&[1]);
        a.union_with(&set(&[300]));
        assert_eq!(a.iter().map(|v| v.0).collect::<Vec<_>>(), vec![1, 300]);
    }
}
