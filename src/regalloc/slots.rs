//! Frame spill slots: one numbering for the whole function, with an owner.
//!
//! Three passes spill, at three different points in the pipeline: the
//! early-barrier live-range shortening in `compile::barrier`, the
//! pressure-driven splitter, and the register allocator's own spill loop. All
//! three write slot indices into `Op::SpillStore` / `Op::SpillLoad` in the same
//! instruction stream, and a slot index is a frame displacement in disguise --
//! `spill_offset + slot * 8`. Two passes choosing one index puts two values in
//! one 8-byte cell.
//!
//! So every slot comes from one allocator per function, which makes the indices
//! distinct by construction. The owner recorded beside each slot is what the
//! index cannot say on its own: which pass a slot belongs to is recoverable
//! from a number range only while the pass order and each pass's slot count
//! hold, and nothing enforces either.
//!
//! The owner lives here rather than in the op, because an op names a frame
//! address and two ops naming the same address must stay equal. This table is
//! the single answer to "who owns slot n".

/// The pass a spill slot belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotOwner {
    /// `compile::barrier::insert_early_barrier_spills`, which shortens a barrier
    /// result's live range when its consumer is two or more barrier groups away.
    EarlyBarrier,
    /// The pressure-driven splitter's cross-block spills and block-param slot
    /// routing.
    Splitter,
    /// The register allocator's own spill loop.
    Allocator,
}

impl SlotOwner {
    pub fn as_str(self) -> &'static str {
        match self {
            SlotOwner::EarlyBarrier => "early-barrier",
            SlotOwner::Splitter => "splitter",
            SlotOwner::Allocator => "allocator",
        }
    }
}

/// Every spill slot of one function, in allocation order.
#[derive(Clone, Debug, Default)]
pub struct SlotAllocator {
    owners: Vec<SlotOwner>,
}

impl SlotAllocator {
    pub fn new() -> Self {
        Self { owners: Vec::new() }
    }

    /// A slot index no other pass holds, recorded against `owner`.
    pub fn alloc(&mut self, owner: SlotOwner) -> u32 {
        let slot = self.owners.len() as u32;
        self.owners.push(owner);
        slot
    }

    /// How many 8-byte slots the frame must reserve.
    pub fn count(&self) -> u32 {
        self.owners.len() as u32
    }

    /// The pass that allocated `slot`, or `None` for an index no pass minted --
    /// which is a slot reference the frame does not cover.
    pub fn owner(&self, slot: u32) -> Option<SlotOwner> {
        self.owners.get(slot as usize).copied()
    }

    /// Owners by slot index, for tracing.
    pub fn owners(&self) -> &[SlotOwner] {
        &self.owners
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_of_different_passes_never_share_an_index() {
        let mut slots = SlotAllocator::new();
        let a = slots.alloc(SlotOwner::EarlyBarrier);
        let b = slots.alloc(SlotOwner::Splitter);
        let c = slots.alloc(SlotOwner::EarlyBarrier);
        let d = slots.alloc(SlotOwner::Allocator);
        assert_eq!([a, b, c, d], [0, 1, 2, 3]);
        assert_eq!(slots.count(), 4);
    }

    #[test]
    fn each_slot_remembers_its_owner() {
        let mut slots = SlotAllocator::new();
        let early = slots.alloc(SlotOwner::EarlyBarrier);
        let split = slots.alloc(SlotOwner::Splitter);
        let alloc = slots.alloc(SlotOwner::Allocator);
        assert_eq!(slots.owner(early), Some(SlotOwner::EarlyBarrier));
        assert_eq!(slots.owner(split), Some(SlotOwner::Splitter));
        assert_eq!(slots.owner(alloc), Some(SlotOwner::Allocator));
    }

    #[test]
    fn an_unallocated_index_has_no_owner() {
        let mut slots = SlotAllocator::new();
        slots.alloc(SlotOwner::Splitter);
        assert_eq!(slots.owner(1), None);
    }
}
