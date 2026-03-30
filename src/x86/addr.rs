use super::reg::Reg;

/// x86-64 memory addressing mode: [base + index*scale + disp].
///
/// Constraints:
/// - `scale` must be 1, 2, 4, or 8.
/// - `index` cannot be `RSP` (hardware constraint: SIB index field 4 = no index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Addr {
    pub base: Option<Reg>,
    pub index: Option<Reg>,
    /// Scale for index: must be 1, 2, 4, or 8.
    pub scale: u8,
    pub disp: i32,
}

impl Addr {
    /// Construct an `Addr`, panicking if constraints are violated.
    pub fn new(base: Option<Reg>, index: Option<Reg>, scale: u8, disp: i32) -> Self {
        if index.is_some() {
            assert!(
                matches!(scale, 1 | 2 | 4 | 8),
                "Addr: scale must be 1, 2, 4, or 8; got {scale}"
            );
            assert!(
                index != Some(Reg::RSP),
                "Addr: RSP cannot be an index register"
            );
        }
        Self {
            base,
            index,
            scale,
            disp,
        }
    }

    /// True if the base register requires a REX extension bit (R8-R15).
    pub fn base_needs_rex(&self) -> bool {
        self.base.map_or(false, |r| r.needs_rex_ext())
    }

    /// True if the index register requires a REX extension bit (R8-R15).
    pub fn index_needs_rex(&self) -> bool {
        self.index.map_or(false, |r| r.needs_rex_ext())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_addr() {
        let a = Addr::new(Some(Reg::RBX), Some(Reg::RSI), 4, -8);
        assert_eq!(a.base, Some(Reg::RBX));
        assert_eq!(a.index, Some(Reg::RSI));
        assert_eq!(a.scale, 4);
        assert_eq!(a.disp, -8);
    }

    #[test]
    #[should_panic(expected = "scale must be 1, 2, 4, or 8")]
    fn invalid_scale() {
        Addr::new(Some(Reg::RBX), Some(Reg::RSI), 6, 0);
    }

    #[test]
    #[should_panic(expected = "RSP cannot be an index register")]
    fn rsp_index_rejected() {
        Addr::new(Some(Reg::RBX), Some(Reg::RSP), 1, 0);
    }

    #[test]
    fn base_needs_rex() {
        let a = Addr::new(Some(Reg::R8), None, 1, 0);
        assert!(a.base_needs_rex());
        let b = Addr::new(Some(Reg::RAX), None, 1, 0);
        assert!(!b.base_needs_rex());
    }

    #[test]
    fn index_needs_rex() {
        let a = Addr::new(Some(Reg::RBX), Some(Reg::R9), 2, 0);
        assert!(a.index_needs_rex());
        let b = Addr::new(Some(Reg::RBX), Some(Reg::RSI), 2, 0);
        assert!(!b.index_needs_rex());
    }
}
