use std::collections::HashMap;

use crate::x86::encode::{Reloc, RelocKind};

use super::elf::{
    ELFCLASS64, ELFDATA2LSB, EM_X86_64, ET_REL, EV_CURRENT, Elf64Ehdr, Elf64Shdr, R_X86_64_32S,
    R_X86_64_64, R_X86_64_PC32, R_X86_64_PLT32, RelocationTable, SHF_ALLOC, SHF_EXECINSTR,
    SHT_NULL, SHT_PROGBITS, SHT_RELA, SHT_STRTAB, SHT_SYMTAB, StringTable, SymbolTable,
};

// ── Public types ──────────────────────────────────────────────────────────────

pub struct FunctionInfo {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

pub struct ObjectFile {
    /// .text section content
    pub code: Vec<u8>,
    /// Relocations from the encoder
    pub relocations: Vec<Reloc>,
    pub functions: Vec<FunctionInfo>,
    /// Names of external (undefined) symbols referenced
    pub externals: Vec<String>,
}

// ── Section index constants ───────────────────────────────────────────────────

// We always emit: null(0), .text(1), .rela.text(2), .symtab(3), .strtab(4), .shstrtab(5)
const SEC_TEXT: u16 = 1;
const SEC_SYMTAB: u16 = 3;
const SEC_STRTAB: u16 = 4;
const SEC_SHSTRTAB: u16 = 5;
const NUM_SECTIONS: u16 = 6;

// ── Alignment helpers ─────────────────────────────────────────────────────────

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn pad_to(buf: &mut Vec<u8>, align: usize) {
    let new_len = align_up(buf.len(), align);
    buf.resize(new_len, 0);
}

// ── ObjectFile impl ───────────────────────────────────────────────────────────

impl ObjectFile {
    /// Assemble all sections into a complete ELF64 relocatable object file.
    pub fn finalize(&self) -> Vec<u8> {
        // ── Build .shstrtab ───────────────────────────────────────────────────
        let mut shstrtab = StringTable::new();
        let name_text = shstrtab.add(".text");
        let name_rela_text = shstrtab.add(".rela.text");
        let name_symtab = shstrtab.add(".symtab");
        let name_strtab = shstrtab.add(".strtab");
        let name_shstrtab = shstrtab.add(".shstrtab");

        // ── Build .strtab and collect external symbol names ───────────────────
        let mut strtab = StringTable::new();

        // Map from external symbol name -> strtab offset
        let mut ext_strtab: HashMap<String, u32> = HashMap::new();
        for ext in &self.externals {
            let off = strtab.add(ext);
            ext_strtab.insert(ext.clone(), off);
        }
        // Function name strtab offsets
        let func_strtab: Vec<u32> = self.functions.iter().map(|f| strtab.add(&f.name)).collect();

        // ── Build .symtab ─────────────────────────────────────────────────────
        let mut symtab = SymbolTable::new();
        // One local section symbol for .text
        symtab.add_section(SEC_TEXT);
        // Global function symbols
        for (i, func) in self.functions.iter().enumerate() {
            symtab.add_function(
                func_strtab[i],
                SEC_TEXT,
                func.offset as u64,
                func.size as u64,
            );
        }
        // Global external (undefined) symbols
        // Build a map from name -> symtab index for use in relocations
        let mut ext_sym_idx: HashMap<String, u32> = HashMap::new();
        for ext in &self.externals {
            let idx = symtab.len() as u32;
            symtab.add_external(*ext_strtab.get(ext).unwrap());
            ext_sym_idx.insert(ext.clone(), idx);
        }

        // ── Build .rela.text ──────────────────────────────────────────────────
        // Section sym for .text has symtab index 1.
        let text_sec_sym: u32 = 1;
        let mut rela = RelocationTable::new();
        for reloc in &self.relocations {
            let (sym_idx, r_type) = if let Some(&idx) = ext_sym_idx.get(&reloc.symbol) {
                let r_type = match reloc.kind {
                    RelocKind::PLT32 => R_X86_64_PLT32,
                    RelocKind::PC32 => R_X86_64_PC32,
                    RelocKind::Abs64 => R_X86_64_64,
                    RelocKind::Abs32S => R_X86_64_32S,
                };
                (idx, r_type)
            } else {
                // Reference to a local function - use section symbol + addend
                let r_type = match reloc.kind {
                    RelocKind::PLT32 => R_X86_64_PLT32,
                    RelocKind::PC32 => R_X86_64_PC32,
                    RelocKind::Abs64 => R_X86_64_64,
                    RelocKind::Abs32S => R_X86_64_32S,
                };
                (text_sec_sym, r_type)
            };
            rela.add(reloc.offset as u64, sym_idx, r_type, reloc.addend);
        }

        // ── Compute section sizes ─────────────────────────────────────────────
        let text_bytes = &self.code;
        let rela_bytes = rela.to_bytes();
        let symtab_bytes = symtab.to_bytes();
        let strtab_bytes = strtab.to_bytes();
        let shstrtab_bytes = shstrtab.to_bytes();

        // ── Lay out the file ──────────────────────────────────────────────────
        // Layout:
        //   [0..64)        ELF header
        //   [64..)         .text  (align 16)
        //   (..)           .rela.text (align 8, may be empty but we always emit it)
        //   (..)           .symtab (align 8)
        //   (..)           .strtab
        //   (..)           .shstrtab
        //   (..)           Section headers (align 8)

        let mut buf: Vec<u8> = Vec::new();

        // Reserve space for ELF header (64 bytes)
        buf.resize(64, 0);

        // .text
        pad_to(&mut buf, 16);
        let off_text = buf.len();
        buf.extend_from_slice(text_bytes);

        // .rela.text
        pad_to(&mut buf, 8);
        let off_rela = buf.len();
        buf.extend_from_slice(&rela_bytes);

        // .symtab
        pad_to(&mut buf, 8);
        let off_symtab = buf.len();
        buf.extend_from_slice(&symtab_bytes);

        // .strtab
        let off_strtab = buf.len();
        buf.extend_from_slice(strtab_bytes);

        // .shstrtab
        let off_shstrtab = buf.len();
        buf.extend_from_slice(shstrtab_bytes);

        // Section headers (align 8)
        pad_to(&mut buf, 8);
        let shoff = buf.len();

        // null section header
        let shdr_null = Elf64Shdr {
            sh_name: 0,
            sh_type: SHT_NULL,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: 0,
            sh_size: 0,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 0,
            sh_entsize: 0,
        };
        buf.extend_from_slice(&shdr_null.to_bytes());

        // .text section header
        let shdr_text = Elf64Shdr {
            sh_name: name_text,
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            sh_addr: 0,
            sh_offset: off_text as u64,
            sh_size: text_bytes.len() as u64,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 16,
            sh_entsize: 0,
        };
        buf.extend_from_slice(&shdr_text.to_bytes());

        // .rela.text section header
        // sh_link = index of associated symtab; sh_info = index of section being relocated
        let shdr_rela = Elf64Shdr {
            sh_name: name_rela_text,
            sh_type: SHT_RELA,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: off_rela as u64,
            sh_size: rela_bytes.len() as u64,
            sh_link: SEC_SYMTAB as u32,
            sh_info: SEC_TEXT as u32,
            sh_addralign: 8,
            sh_entsize: 24, // sizeof(Elf64Rela)
        };
        buf.extend_from_slice(&shdr_rela.to_bytes());

        // .symtab section header
        // sh_link = index of associated strtab; sh_info = index of first global symbol
        let shdr_symtab = Elf64Shdr {
            sh_name: name_symtab,
            sh_type: SHT_SYMTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: off_symtab as u64,
            sh_size: symtab_bytes.len() as u64,
            sh_link: SEC_STRTAB as u32,
            sh_info: symtab.sh_info(),
            sh_addralign: 8,
            sh_entsize: 24, // sizeof(Elf64Sym)
        };
        buf.extend_from_slice(&shdr_symtab.to_bytes());

        // .strtab section header
        let shdr_strtab = Elf64Shdr {
            sh_name: name_strtab,
            sh_type: SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: off_strtab as u64,
            sh_size: strtab_bytes.len() as u64,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 1,
            sh_entsize: 0,
        };
        buf.extend_from_slice(&shdr_strtab.to_bytes());

        // .shstrtab section header
        let shdr_shstrtab = Elf64Shdr {
            sh_name: name_shstrtab,
            sh_type: SHT_STRTAB,
            sh_flags: 0,
            sh_addr: 0,
            sh_offset: off_shstrtab as u64,
            sh_size: shstrtab_bytes.len() as u64,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 1,
            sh_entsize: 0,
        };
        buf.extend_from_slice(&shdr_shstrtab.to_bytes());

        // ── Write ELF header ──────────────────────────────────────────────────
        let mut e_ident = [0u8; 16];
        e_ident[0] = 0x7f;
        e_ident[1] = b'E';
        e_ident[2] = b'L';
        e_ident[3] = b'F';
        e_ident[4] = ELFCLASS64;
        e_ident[5] = ELFDATA2LSB;
        e_ident[6] = EV_CURRENT as u8;
        // e_ident[7..16] = 0 (OS/ABI = ELFOSABI_NONE, padding)

        let ehdr = Elf64Ehdr {
            e_ident,
            e_type: ET_REL,
            e_machine: EM_X86_64,
            e_version: EV_CURRENT,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: shoff as u64,
            e_flags: 0,
            e_ehsize: 64,
            e_phentsize: 0,
            e_phnum: 0,
            e_shentsize: 64, // sizeof(Elf64Shdr)
            e_shnum: NUM_SECTIONS,
            e_shstrndx: SEC_SHSTRTAB,
        };

        let ehdr_bytes = ehdr.to_bytes();
        buf[..64].copy_from_slice(&ehdr_bytes);

        buf
    }

    pub fn write_to(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.finalize())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::process::Command;

    use crate::test_utils::has_tool;
    use crate::x86::encode::{Reloc, RelocKind};

    use super::*;

    fn simple_add_obj() -> ObjectFile {
        // mov rax, rdi; add rax, rsi; ret
        let code = vec![0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3];
        let size = code.len();
        ObjectFile {
            code,
            relocations: vec![],
            functions: vec![FunctionInfo {
                name: "add_two".into(),
                offset: 0,
                size,
            }],
            externals: vec![],
        }
    }

    #[test]
    fn elf_magic_bytes() {
        let obj = simple_add_obj();
        let bytes = obj.finalize();
        assert_eq!(&bytes[0..4], b"\x7fELF", "ELF magic mismatch");
        assert_eq!(bytes[4], 2, "ELFCLASS64");
        assert_eq!(bytes[5], 1, "ELFDATA2LSB");
    }

    #[test]
    fn elf_section_count() {
        let obj = simple_add_obj();
        let bytes = obj.finalize();
        // e_shnum is at offset 60 (2 bytes LE)
        let e_shnum = u16::from_le_bytes([bytes[60], bytes[61]]);
        assert_eq!(e_shnum, NUM_SECTIONS, "expected {} sections", NUM_SECTIONS);
    }

    #[test]
    fn elf_machine_and_type() {
        let obj = simple_add_obj();
        let bytes = obj.finalize();
        // e_type at offset 16 (2 bytes LE), e_machine at 18
        let e_type = u16::from_le_bytes([bytes[16], bytes[17]]);
        let e_machine = u16::from_le_bytes([bytes[18], bytes[19]]);
        assert_eq!(e_type, ET_REL, "e_type should be ET_REL");
        assert_eq!(e_machine, EM_X86_64, "e_machine should be EM_X86_64");
    }

    #[test]
    fn readelf_header() {
        if !has_tool("readelf") {
            return;
        }
        let obj = simple_add_obj();
        let tmp = std::env::temp_dir().join("blitz_test_elf.o");
        obj.write_to(&tmp).expect("write failed");

        let out = Command::new("readelf")
            .args(["-h", tmp.to_str().unwrap()])
            .output()
            .expect("readelf failed");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "readelf -h failed:\n{stdout}");
        assert!(stdout.contains("ELF64"), "expected ELF64 in readelf output");
        assert!(
            stdout.contains("X86-64"),
            "expected X86-64 in readelf output"
        );
        assert!(
            stdout.contains("REL"),
            "expected REL type in readelf output"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn readelf_sections() {
        if !has_tool("readelf") {
            return;
        }
        let obj = simple_add_obj();
        let tmp = std::env::temp_dir().join("blitz_test_elf_sections.o");
        obj.write_to(&tmp).expect("write failed");

        let out = Command::new("readelf")
            .args(["-S", tmp.to_str().unwrap()])
            .output()
            .expect("readelf -S failed");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "readelf -S failed:\n{stdout}");
        assert!(stdout.contains(".text"), "missing .text section");
        assert!(stdout.contains(".symtab"), "missing .symtab section");
        assert!(stdout.contains(".strtab"), "missing .strtab section");
        assert!(stdout.contains(".shstrtab"), "missing .shstrtab section");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn link_and_run_with_c_main() {
        if !has_tool("cc") {
            return;
        }
        // Hand-assembled: mov rax, rdi; add rax, rsi; ret
        // Implements: int64_t add_two(int64_t a, int64_t b) { return a + b; }
        let code = vec![0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3];
        let size = code.len();
        let obj = ObjectFile {
            code,
            relocations: vec![],
            functions: vec![FunctionInfo {
                name: "add_two".into(),
                offset: 0,
                size,
            }],
            externals: vec![],
        };

        let dir = std::env::temp_dir();
        let obj_path = dir.join("blitz_link_test.o");
        let main_path = dir.join("blitz_link_test_main.c");
        let bin_path = dir.join("blitz_link_test_bin");

        obj.write_to(&obj_path).expect("write .o failed");

        std::fs::write(
            &main_path,
            b"#include <stdint.h>\n\
              int64_t add_two(int64_t a, int64_t b);\n\
              int main(void) {\n\
              return (int)(add_two(3, 4) == 7 ? 0 : 1);\n\
              }\n",
        )
        .expect("write main.c failed");

        let compile = Command::new("cc")
            .args([
                main_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
                "-o",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .expect("cc failed");
        assert!(
            compile.status.success(),
            "linking failed:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let run = Command::new(&bin_path).output().expect("run failed");
        assert_eq!(run.status.code(), Some(0), "program returned non-zero");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(&main_path);
        let _ = std::fs::remove_file(&bin_path);
    }

    #[test]
    fn relocation_table_populated() {
        // Build an object with a relocation to an external symbol
        let code = vec![
            // call <placeholder>: e8 00 00 00 00
            0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3,
        ];
        let obj = ObjectFile {
            code,
            relocations: vec![Reloc {
                offset: 1,
                kind: RelocKind::PLT32,
                symbol: "printf".into(),
                addend: -4,
            }],
            functions: vec![FunctionInfo {
                name: "call_printf".into(),
                offset: 0,
                size: 6,
            }],
            externals: vec!["printf".into()],
        };
        let bytes = obj.finalize();
        // Just verify it produces valid-looking ELF bytes
        assert_eq!(&bytes[0..4], b"\x7fELF");

        if has_tool("readelf") {
            let tmp = std::env::temp_dir().join("blitz_rela_test.o");
            obj.write_to(&tmp).expect("write failed");
            let out = Command::new("readelf")
                .args(["-r", tmp.to_str().unwrap()])
                .output()
                .expect("readelf -r");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(out.status.success(), "readelf -r failed:\n{stdout}");
            assert!(
                stdout.contains("PLT32") || stdout.contains("plt32"),
                "expected PLT32 reloc"
            );
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
