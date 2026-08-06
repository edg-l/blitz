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

/// A set is sized for the whole function's VRegs but holds a handful of live
/// values, so every query that walked all of `words` walked a thousand empty
/// ones to reach them. `summary` marks which words are non-empty -- one bit per
/// word, so a second level 64 times smaller -- and the walks go through it. On a
/// function with 16161 values that turned the pressure query from 253 word reads
/// per program point into one or two.
#[derive(Clone, Debug, Default, Eq)]
pub struct VRegSet {
    words: Vec<u64>,
    /// Bit `w` is set exactly when `words[w] != 0`.
    summary: Vec<u64>,
}

impl PartialEq for VRegSet {
    fn eq(&self, other: &Self) -> bool {
        // `summary` is derived, and two equal sets can carry different trailing
        // capacity, so the members are the comparison.
        let common = self.words.len().min(other.words.len());
        self.words[..common] == other.words[..common]
            && self.words[common..].iter().all(|w| *w == 0)
            && other.words[common..].iter().all(|w| *w == 0)
    }
}

impl VRegSet {
    /// An empty set sized for VRegs `0..vregs` without reallocating.
    pub fn with_capacity(vregs: usize) -> Self {
        let words = vregs.div_ceil(BITS);
        Self {
            words: vec![0; words],
            summary: vec![0; words.div_ceil(BITS)],
        }
    }

    fn mark(&mut self, w: usize) {
        let (sw, bit) = (w / BITS, w % BITS);
        if sw >= self.summary.len() {
            self.summary.resize(sw + 1, 0);
        }
        self.summary[sw] |= 1u64 << bit;
    }

    fn unmark(&mut self, w: usize) {
        let (sw, bit) = (w / BITS, w % BITS);
        if let Some(word) = self.summary.get_mut(sw) {
            *word &= !(1u64 << bit);
        }
    }

    /// The indices of the non-empty words, ascending.
    fn live_words(&self) -> impl Iterator<Item = usize> + '_ {
        self.summary
            .iter()
            .enumerate()
            .flat_map(|(sw, &word)| BitsOf(word).map(move |bit| sw * BITS + bit as usize))
    }

    pub fn insert(&mut self, v: VReg) {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        if w >= self.words.len() {
            // Geometric, so a set built by inserting ascending VRegs does not
            // copy itself once per word. A set sized on demand rather than to
            // the function's VReg count is what keeps a per-program-point live
            // set the size of what is live there.
            self.words.resize((w + 1).max(self.words.len() * 2), 0);
        }
        self.words[w] |= 1u64 << bit;
        self.mark(w);
    }

    pub fn remove(&mut self, v: VReg) {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        if let Some(word) = self.words.get_mut(w) {
            *word &= !(1u64 << bit);
            if *word == 0 {
                self.unmark(w);
            }
        }
    }

    pub fn contains(&self, v: VReg) -> bool {
        let (w, bit) = (v.0 as usize / BITS, v.0 as usize % BITS);
        self.words.get(w).is_some_and(|word| word >> bit & 1 == 1)
    }

    pub fn is_empty(&self) -> bool {
        self.summary.iter().all(|w| *w == 0)
    }

    pub fn len(&self) -> u32 {
        self.live_words().map(|w| self.words[w].count_ones()).sum()
    }

    /// Add every member of `other`.
    pub fn union_with(&mut self, other: &VRegSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for w in other.live_words().collect::<Vec<_>>() {
            self.words[w] |= other.words[w];
            self.mark(w);
        }
    }

    /// How many members are also in `mask`. This is the pressure query: `mask`
    /// holds the VRegs of one register class.
    pub fn count_in(&self, mask: &VRegSet) -> u32 {
        self.live_words()
            .map(|w| match mask.words.get(w) {
                Some(m) => (self.words[w] & m).count_ones(),
                None => 0,
            })
            .sum()
    }

    /// Members in ascending VReg order.
    pub fn iter(&self) -> impl Iterator<Item = VReg> + '_ {
        self.live_words().flat_map(move |w| {
            BitsOf(self.words[w]).map(move |bit| VReg((w * BITS + bit as usize) as u32))
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
