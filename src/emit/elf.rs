// ELF64 structures and helpers for emitting relocatable object files.

// ── Constants ─────────────────────────────────────────────────────────────────

pub const ELFCLASS64: u8 = 2;
pub const ELFDATA2LSB: u8 = 1;
pub const ET_REL: u16 = 1;
pub const EM_X86_64: u16 = 62;
pub const EV_CURRENT: u32 = 1;

// Section types
pub const SHT_NULL: u32 = 0;
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_RELA: u32 = 4;
pub const SHT_NOBITS: u32 = 8;

// Section flags
pub const SHF_WRITE: u64 = 0x1;
pub const SHF_ALLOC: u64 = 0x2;
pub const SHF_EXECINSTR: u64 = 0x4;

// Symbol binding
pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;

// Symbol type
pub const STT_NOTYPE: u8 = 0;
pub const STT_FUNC: u8 = 2;
pub const STT_SECTION: u8 = 3;

// Relocation types
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_PC32: u32 = 2;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_X86_64_32S: u32 = 11;

pub const SHN_UNDEF: u16 = 0;

// ── ELF header ────────────────────────────────────────────────────────────────

#[repr(C)]
pub struct Elf64Ehdr {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

impl Elf64Ehdr {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(64);
        v.extend_from_slice(&self.e_ident);
        v.extend_from_slice(&self.e_type.to_le_bytes());
        v.extend_from_slice(&self.e_machine.to_le_bytes());
        v.extend_from_slice(&self.e_version.to_le_bytes());
        v.extend_from_slice(&self.e_entry.to_le_bytes());
        v.extend_from_slice(&self.e_phoff.to_le_bytes());
        v.extend_from_slice(&self.e_shoff.to_le_bytes());
        v.extend_from_slice(&self.e_flags.to_le_bytes());
        v.extend_from_slice(&self.e_ehsize.to_le_bytes());
        v.extend_from_slice(&self.e_phentsize.to_le_bytes());
        v.extend_from_slice(&self.e_phnum.to_le_bytes());
        v.extend_from_slice(&self.e_shentsize.to_le_bytes());
        v.extend_from_slice(&self.e_shnum.to_le_bytes());
        v.extend_from_slice(&self.e_shstrndx.to_le_bytes());
        v
    }
}

// ── Section header ────────────────────────────────────────────────────────────

#[repr(C)]
pub struct Elf64Shdr {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

impl Elf64Shdr {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(64);
        v.extend_from_slice(&self.sh_name.to_le_bytes());
        v.extend_from_slice(&self.sh_type.to_le_bytes());
        v.extend_from_slice(&self.sh_flags.to_le_bytes());
        v.extend_from_slice(&self.sh_addr.to_le_bytes());
        v.extend_from_slice(&self.sh_offset.to_le_bytes());
        v.extend_from_slice(&self.sh_size.to_le_bytes());
        v.extend_from_slice(&self.sh_link.to_le_bytes());
        v.extend_from_slice(&self.sh_info.to_le_bytes());
        v.extend_from_slice(&self.sh_addralign.to_le_bytes());
        v.extend_from_slice(&self.sh_entsize.to_le_bytes());
        v
    }
}

// ── Symbol ────────────────────────────────────────────────────────────────────

#[repr(C)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

impl Elf64Sym {
    pub fn null() -> Self {
        Elf64Sym {
            st_name: 0,
            st_info: 0,
            st_other: 0,
            st_shndx: 0,
            st_value: 0,
            st_size: 0,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&self.st_name.to_le_bytes());
        v.push(self.st_info);
        v.push(self.st_other);
        v.extend_from_slice(&self.st_shndx.to_le_bytes());
        v.extend_from_slice(&self.st_value.to_le_bytes());
        v.extend_from_slice(&self.st_size.to_le_bytes());
        v
    }
}

// ── Relocation entry (RELA) ───────────────────────────────────────────────────

#[repr(C)]
pub struct Elf64Rela {
    pub r_offset: u64,
    pub r_info: u64,
    pub r_addend: i64,
}

impl Elf64Rela {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&self.r_offset.to_le_bytes());
        v.extend_from_slice(&self.r_info.to_le_bytes());
        v.extend_from_slice(&self.r_addend.to_le_bytes());
        v
    }
}

// ── StringTable ───────────────────────────────────────────────────────────────

pub struct StringTable {
    data: Vec<u8>,
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StringTable {
    pub fn new() -> Self {
        // First byte is always null (empty string at offset 0)
        StringTable { data: vec![0] }
    }

    pub fn add(&mut self, name: &str) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(name.as_bytes());
        self.data.push(0);
        offset
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ── SymbolTable ───────────────────────────────────────────────────────────────

pub struct SymbolTable {
    symbols: Vec<Elf64Sym>,
    pub first_global: u32,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            symbols: vec![Elf64Sym::null()],
            first_global: 1,
        }
    }

    /// Add a local section symbol. Must be called before any globals.
    pub fn add_section(&mut self, shndx: u16) {
        let sym = Elf64Sym {
            st_name: 0,
            st_info: (STB_LOCAL << 4) | STT_SECTION,
            st_other: 0,
            st_shndx: shndx,
            st_value: 0,
            st_size: 0,
        };
        // Insert before first_global boundary; since we call this before adding
        // globals, just push and keep first_global up to date.
        self.symbols.push(sym);
        self.first_global = self.symbols.len() as u32;
    }

    pub fn add_function(&mut self, name_idx: u32, text_section: u16, offset: u64, size: u64) {
        let sym = Elf64Sym {
            st_name: name_idx,
            st_info: (STB_GLOBAL << 4) | STT_FUNC,
            st_other: 0,
            st_shndx: text_section,
            st_value: offset,
            st_size: size,
        };
        self.symbols.push(sym);
    }

    pub fn add_external(&mut self, name_idx: u32) {
        let sym = Elf64Sym {
            st_name: name_idx,
            st_info: (STB_GLOBAL << 4) | STT_NOTYPE,
            st_other: 0,
            st_shndx: SHN_UNDEF,
            st_value: 0,
            st_size: 0,
        };
        self.symbols.push(sym);
    }

    pub fn sh_info(&self) -> u32 {
        self.first_global
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.symbols.len() * 24);
        for sym in &self.symbols {
            v.extend_from_slice(&sym.to_bytes());
        }
        v
    }
}

// ── RelocationTable ───────────────────────────────────────────────────────────

pub struct RelocationTable {
    pub entries: Vec<Elf64Rela>,
}

impl Default for RelocationTable {
    fn default() -> Self {
        Self::new()
    }
}

impl RelocationTable {
    pub fn new() -> Self {
        RelocationTable {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, r_offset: u64, sym_idx: u32, r_type: u32, r_addend: i64) {
        let r_info = ((sym_idx as u64) << 32) | (r_type as u64);
        self.entries.push(Elf64Rela {
            r_offset,
            r_info,
            r_addend,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.entries.len() * 24);
        for entry in &self.entries {
            v.extend_from_slice(&entry.to_bytes());
        }
        v
    }
}
