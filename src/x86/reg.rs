/// x86-64 register class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegClass {
    GPR,
    XMM,
}

/// Physical x86-64 registers (GPR and XMM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Reg {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
    XMM0 = 16,
    XMM1 = 17,
    XMM2 = 18,
    XMM3 = 19,
    XMM4 = 20,
    XMM5 = 21,
    XMM6 = 22,
    XMM7 = 23,
    XMM8 = 24,
    XMM9 = 25,
    XMM10 = 26,
    XMM11 = 27,
    XMM12 = 28,
    XMM13 = 29,
    XMM14 = 30,
    XMM15 = 31,
}

impl Reg {
    /// Hardware encoding: low 4 bits of the register number used in REX/ModRM/SIB.
    /// GPRs: RAX=0..R15=15. XMMs: XMM0=0..XMM15=15.
    pub fn hw_enc(self) -> u8 {
        (self as u8) & 0xF
    }

    /// True for R8-R15 and XMM8-XMM15 (require REX.R/B/X extension bit).
    pub fn needs_rex_ext(self) -> bool {
        let v = self as u8;
        // GPRs 8-15 and XMMs 24-31 (i.e. XMM8-XMM15)
        (8..=15).contains(&v) || (24..=31).contains(&v)
    }

    pub fn is_gpr(self) -> bool {
        (self as u8) <= 15
    }

    pub fn is_xmm(self) -> bool {
        (self as u8) >= 16
    }

    pub fn reg_class(self) -> RegClass {
        if self.is_gpr() {
            RegClass::GPR
        } else {
            RegClass::XMM
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpr_hw_enc() {
        assert_eq!(Reg::RAX.hw_enc(), 0);
        assert_eq!(Reg::RCX.hw_enc(), 1);
        assert_eq!(Reg::RDX.hw_enc(), 2);
        assert_eq!(Reg::RBX.hw_enc(), 3);
        assert_eq!(Reg::RSP.hw_enc(), 4);
        assert_eq!(Reg::RBP.hw_enc(), 5);
        assert_eq!(Reg::RSI.hw_enc(), 6);
        assert_eq!(Reg::RDI.hw_enc(), 7);
        assert_eq!(Reg::R8.hw_enc(), 8);
        assert_eq!(Reg::R9.hw_enc(), 9);
        assert_eq!(Reg::R10.hw_enc(), 10);
        assert_eq!(Reg::R11.hw_enc(), 11);
        assert_eq!(Reg::R12.hw_enc(), 12);
        assert_eq!(Reg::R13.hw_enc(), 13);
        assert_eq!(Reg::R14.hw_enc(), 14);
        assert_eq!(Reg::R15.hw_enc(), 15);
    }

    #[test]
    fn xmm_hw_enc() {
        assert_eq!(Reg::XMM0.hw_enc(), 0);
        assert_eq!(Reg::XMM1.hw_enc(), 1);
        assert_eq!(Reg::XMM7.hw_enc(), 7);
        assert_eq!(Reg::XMM8.hw_enc(), 8);
        assert_eq!(Reg::XMM15.hw_enc(), 15);
    }

    #[test]
    fn needs_rex_ext() {
        // GPRs 0-7 do not need REX extension
        for r in [
            Reg::RAX,
            Reg::RCX,
            Reg::RDX,
            Reg::RBX,
            Reg::RSP,
            Reg::RBP,
            Reg::RSI,
            Reg::RDI,
        ] {
            assert!(!r.needs_rex_ext(), "{r:?} should not need REX ext");
        }
        // R8-R15 need REX extension
        for r in [
            Reg::R8,
            Reg::R9,
            Reg::R10,
            Reg::R11,
            Reg::R12,
            Reg::R13,
            Reg::R14,
            Reg::R15,
        ] {
            assert!(r.needs_rex_ext(), "{r:?} should need REX ext");
        }
        // XMM0-XMM7 do not need REX extension
        for r in [
            Reg::XMM0,
            Reg::XMM1,
            Reg::XMM2,
            Reg::XMM3,
            Reg::XMM4,
            Reg::XMM5,
            Reg::XMM6,
            Reg::XMM7,
        ] {
            assert!(!r.needs_rex_ext(), "{r:?} should not need REX ext");
        }
        // XMM8-XMM15 need REX extension
        for r in [
            Reg::XMM8,
            Reg::XMM9,
            Reg::XMM10,
            Reg::XMM11,
            Reg::XMM12,
            Reg::XMM13,
            Reg::XMM14,
            Reg::XMM15,
        ] {
            assert!(r.needs_rex_ext(), "{r:?} should need REX ext");
        }
    }

    #[test]
    fn reg_class() {
        assert_eq!(Reg::RAX.reg_class(), RegClass::GPR);
        assert_eq!(Reg::R15.reg_class(), RegClass::GPR);
        assert_eq!(Reg::XMM0.reg_class(), RegClass::XMM);
        assert_eq!(Reg::XMM15.reg_class(), RegClass::XMM);
    }
}
