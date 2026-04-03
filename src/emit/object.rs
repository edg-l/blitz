use std::collections::HashMap;

use crate::x86::encode::{Reloc, RelocKind};

use super::elf::{
    ELFCLASS64, ELFDATA2LSB, EM_X86_64, ET_REL, EV_CURRENT, Elf64Ehdr, Elf64Shdr, R_X86_64_32S,
    R_X86_64_64, R_X86_64_PC32, R_X86_64_PLT32, RelocationTable, SHF_ALLOC, SHF_EXECINSTR,
    SHF_WRITE, SHT_NOBITS, SHT_NULL, SHT_PROGBITS, SHT_RELA, SHT_STRTAB, SHT_SYMTAB, StringTable,
    SymbolTable,
};

// ── Public types ──────────────────────────────────────────────────────────────

pub struct FunctionInfo {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

pub struct GlobalInfo {
    pub name: String,
    pub size: usize,
    pub align: usize,
    /// `None` -> .bss (zero-initialized), `Some(bytes)` -> .data (initialized)
    pub init: Option<Vec<u8>>,
}

pub struct ObjectFile {
    /// .text section content
    pub code: Vec<u8>,
    /// Relocations from the encoder
    pub relocations: Vec<Reloc>,
    pub functions: Vec<FunctionInfo>,
    /// Names of external (undefined) symbols referenced
    pub externals: Vec<String>,
    /// Global variable definitions
    pub globals: Vec<GlobalInfo>,
    /// Read-only data (.rodata) entries (e.g. string literals)
    pub rodata: Vec<GlobalInfo>,
}

// ── Alignment helpers ─────────────────────────────────────────────────────────

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn pad_to(buf: &mut Vec<u8>, align: usize) {
    let new_len = align_up(buf.len(), align);
    buf.resize(new_len, 0);
}

// ── Dynamic section layout ───────────────────────────────────────────────────

/// Computed section indices that adjust based on which optional sections exist.
struct SectionLayout {
    sec_text: u16,
    sec_rodata: Option<u16>,
    sec_data: Option<u16>,
    sec_bss: Option<u16>,
    #[allow(dead_code)]
    sec_rela_text: u16,
    sec_symtab: u16,
    sec_strtab: u16,
    sec_shstrtab: u16,
    // .note.GNU-stack is the last section
    num_sections: u16,
}

impl SectionLayout {
    fn compute(has_rodata: bool, has_data: bool, has_bss: bool) -> Self {
        // Section order: null(0), .text, [.rodata], [.data], [.bss], .rela.text, .symtab, .strtab, .shstrtab, .note.GNU-stack
        let mut next: u16 = 1; // 0 is null
        let sec_text = next;
        next += 1;
        let sec_rodata = if has_rodata {
            let idx = next;
            next += 1;
            Some(idx)
        } else {
            None
        };
        let sec_data = if has_data {
            let idx = next;
            next += 1;
            Some(idx)
        } else {
            None
        };
        let sec_bss = if has_bss {
            let idx = next;
            next += 1;
            Some(idx)
        } else {
            None
        };
        let sec_rela_text = next;
        next += 1;
        let sec_symtab = next;
        next += 1;
        let sec_strtab = next;
        next += 1;
        let sec_shstrtab = next;
        next += 1;
        // .note.GNU-stack
        next += 1;
        SectionLayout {
            sec_text,
            sec_rodata,
            sec_data,
            sec_bss,
            sec_rela_text,
            sec_symtab,
            sec_strtab,
            sec_shstrtab,
            num_sections: next,
        }
    }
}

// ── ObjectFile impl ───────────────────────────────────────────────────────────

impl ObjectFile {
    /// Assemble all sections into a complete ELF64 relocatable object file.
    pub fn finalize(&self) -> Vec<u8> {
        // ── Partition globals into .data and .bss ────────────────────────────
        let data_globals: Vec<&GlobalInfo> =
            self.globals.iter().filter(|g| g.init.is_some()).collect();
        let bss_globals: Vec<&GlobalInfo> =
            self.globals.iter().filter(|g| g.init.is_none()).collect();

        let has_rodata = !self.rodata.is_empty();
        let has_data = !data_globals.is_empty();
        let has_bss = !bss_globals.is_empty();
        let layout = SectionLayout::compute(has_rodata, has_data, has_bss);

        // ── Build .rodata section content ────────────────────────────────────
        let mut rodata_content: Vec<u8> = Vec::new();
        // Map: symbol name -> (section index, offset within section, size)
        let mut rodata_section_info: HashMap<String, (u16, u64, u64)> = HashMap::new();

        for g in &self.rodata {
            let align = g.align.max(1);
            let padded_offset = align_up(rodata_content.len(), align);
            rodata_content.resize(padded_offset, 0);
            let offset = rodata_content.len();
            let init = g.init.as_ref().expect("rodata entries must have init data");
            rodata_content.extend_from_slice(init);
            if init.len() < g.size {
                rodata_content.resize(offset + g.size, 0);
            }
            rodata_section_info.insert(
                g.name.clone(),
                (layout.sec_rodata.unwrap(), offset as u64, g.size as u64),
            );
        }

        // ── Build .data section content ──────────────────────────────────────
        let mut data_content: Vec<u8> = Vec::new();
        // Map: global name -> (section index, offset within section)
        let mut global_section_info: HashMap<String, (u16, u64, u64)> = HashMap::new();

        for g in &data_globals {
            let align = g.align.max(1);
            let padded_offset = align_up(data_content.len(), align);
            data_content.resize(padded_offset, 0);
            let offset = data_content.len();
            data_content.extend_from_slice(g.init.as_ref().unwrap());
            // Pad to size if init bytes are shorter
            if g.init.as_ref().unwrap().len() < g.size {
                data_content.resize(offset + g.size, 0);
            }
            global_section_info.insert(
                g.name.clone(),
                (layout.sec_data.unwrap(), offset as u64, g.size as u64),
            );
        }

        // ── Compute .bss virtual size and offsets ────────────────────────────
        let mut bss_size: usize = 0;
        for g in &bss_globals {
            let align = g.align.max(1);
            bss_size = align_up(bss_size, align);
            let offset = bss_size;
            bss_size += g.size;
            global_section_info.insert(
                g.name.clone(),
                (layout.sec_bss.unwrap(), offset as u64, g.size as u64),
            );
        }

        // ── Build .shstrtab ───────────────────────────────────────────────────
        let mut shstrtab = StringTable::new();
        let name_text = shstrtab.add(".text");
        let name_rodata = if has_rodata {
            Some(shstrtab.add(".rodata"))
        } else {
            None
        };
        let name_data = if has_data {
            Some(shstrtab.add(".data"))
        } else {
            None
        };
        let name_bss = if has_bss {
            Some(shstrtab.add(".bss"))
        } else {
            None
        };
        let name_rela_text = shstrtab.add(".rela.text");
        let name_symtab = shstrtab.add(".symtab");
        let name_strtab = shstrtab.add(".strtab");
        let name_shstrtab = shstrtab.add(".shstrtab");
        let name_note_gnu_stack = shstrtab.add(".note.GNU-stack");

        // ── Build .strtab and collect symbol names ───────────────────────────
        let mut strtab = StringTable::new();

        // External symbol strtab offsets
        let mut ext_strtab: HashMap<String, u32> = HashMap::new();
        for ext in &self.externals {
            let off = strtab.add(ext);
            ext_strtab.insert(ext.clone(), off);
        }
        // Function name strtab offsets
        let func_strtab: Vec<u32> = self.functions.iter().map(|f| strtab.add(&f.name)).collect();
        // Global name strtab offsets
        let global_strtab: Vec<u32> = self.globals.iter().map(|g| strtab.add(&g.name)).collect();
        // Rodata name strtab offsets
        let rodata_strtab: Vec<u32> = self.rodata.iter().map(|g| strtab.add(&g.name)).collect();

        // ── Build .symtab ─────────────────────────────────────────────────────
        let mut symtab = SymbolTable::new();
        // Local section symbols
        symtab.add_section(layout.sec_text);
        if let Some(sec) = layout.sec_rodata {
            symtab.add_section(sec);
        }
        if let Some(sec) = layout.sec_data {
            symtab.add_section(sec);
        }
        if let Some(sec) = layout.sec_bss {
            symtab.add_section(sec);
        }
        // Local rodata symbols (must be added before globals)
        let mut global_sym_idx: HashMap<String, u32> = HashMap::new();
        for (i, g) in self.rodata.iter().enumerate() {
            let (section, offset, size) = rodata_section_info[&g.name];
            let idx = symtab.add_local_object(rodata_strtab[i], section, offset, size);
            global_sym_idx.insert(g.name.clone(), idx);
        }
        // Global function symbols
        for (i, func) in self.functions.iter().enumerate() {
            symtab.add_function(
                func_strtab[i],
                layout.sec_text,
                func.offset as u64,
                func.size as u64,
            );
        }
        // Global data symbols
        for (i, g) in self.globals.iter().enumerate() {
            let idx = symtab.len() as u32;
            let (section, offset, size) = global_section_info[&g.name];
            symtab.add_object(global_strtab[i], section, offset, size);
            global_sym_idx.insert(g.name.clone(), idx);
        }
        // Global external (undefined) symbols
        let mut ext_sym_idx: HashMap<String, u32> = HashMap::new();
        for ext in &self.externals {
            let idx = symtab.len() as u32;
            symtab.add_external(*ext_strtab.get(ext).unwrap());
            ext_sym_idx.insert(ext.clone(), idx);
        }

        // ── Build .rela.text ──────────────────────────────────────────────────
        // Section sym for .text is always index 1 (first section sym added).
        let text_sec_sym: u32 = 1;
        let mut rela = RelocationTable::new();
        for reloc in &self.relocations {
            let (sym_idx, r_type) = if let Some(&idx) = global_sym_idx.get(&reloc.symbol) {
                // Reference to a global/rodata variable
                let r_type = match reloc.kind {
                    RelocKind::PLT32 => R_X86_64_PLT32,
                    RelocKind::PC32 => R_X86_64_PC32,
                    RelocKind::Abs64 => R_X86_64_64,
                    RelocKind::Abs32S => R_X86_64_32S,
                };
                (idx, r_type)
            } else if let Some(&idx) = ext_sym_idx.get(&reloc.symbol) {
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

        // ── Compute section byte content ─────────────────────────────────────
        let text_bytes = &self.code;
        let rela_bytes = rela.to_bytes();
        let symtab_bytes = symtab.to_bytes();
        let strtab_bytes = strtab.to_bytes();
        let shstrtab_bytes = shstrtab.to_bytes();

        // ── Lay out the file ──────────────────────────────────────────────────
        let mut buf: Vec<u8> = vec![0; 64]; // ELF header

        // .text
        pad_to(&mut buf, 16);
        let off_text = buf.len();
        buf.extend_from_slice(text_bytes);

        // .rodata (if any)
        let off_rodata = if has_rodata {
            pad_to(&mut buf, 1);
            let off = buf.len();
            buf.extend_from_slice(&rodata_content);
            off
        } else {
            0
        };

        // .data (if any)
        let off_data = if has_data {
            pad_to(&mut buf, 8);
            let off = buf.len();
            buf.extend_from_slice(&data_content);
            off
        } else {
            0
        };

        // .bss has no file content (SHT_NOBITS), so no bytes emitted

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

        // null section header (index 0)
        buf.extend_from_slice(
            &Elf64Shdr {
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
            }
            .to_bytes(),
        );

        // .text section header
        buf.extend_from_slice(
            &Elf64Shdr {
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
            }
            .to_bytes(),
        );

        // .rodata section header (if any)
        if has_rodata {
            buf.extend_from_slice(
                &Elf64Shdr {
                    sh_name: name_rodata.unwrap(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC,
                    sh_addr: 0,
                    sh_offset: off_rodata as u64,
                    sh_size: rodata_content.len() as u64,
                    sh_link: 0,
                    sh_info: 0,
                    sh_addralign: 1,
                    sh_entsize: 0,
                }
                .to_bytes(),
            );
        }

        // .data section header (if any)
        if has_data {
            buf.extend_from_slice(
                &Elf64Shdr {
                    sh_name: name_data.unwrap(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    sh_addr: 0,
                    sh_offset: off_data as u64,
                    sh_size: data_content.len() as u64,
                    sh_link: 0,
                    sh_info: 0,
                    sh_addralign: 8,
                    sh_entsize: 0,
                }
                .to_bytes(),
            );
        }

        // .bss section header (if any)
        if has_bss {
            buf.extend_from_slice(
                &Elf64Shdr {
                    sh_name: name_bss.unwrap(),
                    sh_type: SHT_NOBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    sh_addr: 0,
                    sh_offset: 0, // SHT_NOBITS has no file offset
                    sh_size: bss_size as u64,
                    sh_link: 0,
                    sh_info: 0,
                    sh_addralign: 8,
                    sh_entsize: 0,
                }
                .to_bytes(),
            );
        }

        // .rela.text section header
        buf.extend_from_slice(
            &Elf64Shdr {
                sh_name: name_rela_text,
                sh_type: SHT_RELA,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: off_rela as u64,
                sh_size: rela_bytes.len() as u64,
                sh_link: layout.sec_symtab as u32,
                sh_info: layout.sec_text as u32,
                sh_addralign: 8,
                sh_entsize: 24,
            }
            .to_bytes(),
        );

        // .symtab section header
        buf.extend_from_slice(
            &Elf64Shdr {
                sh_name: name_symtab,
                sh_type: SHT_SYMTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: off_symtab as u64,
                sh_size: symtab_bytes.len() as u64,
                sh_link: layout.sec_strtab as u32,
                sh_info: symtab.sh_info(),
                sh_addralign: 8,
                sh_entsize: 24,
            }
            .to_bytes(),
        );

        // .strtab section header
        buf.extend_from_slice(
            &Elf64Shdr {
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
            }
            .to_bytes(),
        );

        // .shstrtab section header
        buf.extend_from_slice(
            &Elf64Shdr {
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
            }
            .to_bytes(),
        );

        // .note.GNU-stack section header (empty, flags=0 -> non-executable stack)
        buf.extend_from_slice(
            &Elf64Shdr {
                sh_name: name_note_gnu_stack,
                sh_type: SHT_PROGBITS,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: 0,
                sh_size: 0,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 1,
                sh_entsize: 0,
            }
            .to_bytes(),
        );

        // ── Write ELF header ──────────────────────────────────────────────────
        let mut e_ident = [0u8; 16];
        e_ident[0] = 0x7f;
        e_ident[1] = b'E';
        e_ident[2] = b'L';
        e_ident[3] = b'F';
        e_ident[4] = ELFCLASS64;
        e_ident[5] = ELFDATA2LSB;
        e_ident[6] = EV_CURRENT as u8;

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
            e_shentsize: 64,
            e_shnum: layout.num_sections,
            e_shstrndx: layout.sec_shstrtab,
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
            globals: vec![],
            rodata: vec![],
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
        // No globals, so no .data/.bss: 7 sections (null, .text, .rela.text, .symtab, .strtab, .shstrtab, .note.GNU-stack)
        assert_eq!(e_shnum, 7, "expected 7 sections for no-globals case");
    }

    #[test]
    fn elf_machine_and_type() {
        let obj = simple_add_obj();
        let bytes = obj.finalize();
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
            globals: vec![],
            rodata: vec![],
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
        let code = vec![0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3];
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
            globals: vec![],
            rodata: vec![],
        };
        let bytes = obj.finalize();
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
