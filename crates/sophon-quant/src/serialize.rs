//! Binary weight serialization for Sophon-1.
//!
//! File format (`.sophon` extension):
//!
//! ```text
//! [Header]
//!   magic:    [u8; 8]   = b"SOPHON01"
//!   version:  u32       = 1
//!   flags:    u32       (bit 0: ternary_quantized, bits 1-31: reserved)
//!   d_model:  u32
//!   n_blocks: u32
//!   vocab:    u32
//!   ssm_n:    u32
//!   ssm_d:    u32
//!   ssm_p:    u32
//!   ssm_r:    u32
//!   kan_knots:u32
//!   n_ctrl:   u32
//!   total_params: u64
//!   reserved: [u8; 32]  (zero-filled, for future expansion)
//!
//! [Section Table]
//!   n_sections: u32
//!   For each section:
//!     kind:     u8    (0=f32_raw, 1=ternary_packed)
//!     name_len: u16
//!     name:     [u8; name_len]  (UTF-8 section name)
//!     offset:   u64   (byte offset from file start)
//!     size:     u64   (byte size of section data)
//!     n_elems:  u64   (number of logical f32 elements)
//!
//! [Data Sections]
//!   For f32_raw:   n_elems * 4 bytes (little-endian f32)
//!   For ternary_packed:
//!     scale_count: u64  (number of TernaryBlocks)
//!     scales:      scale_count * 4 bytes (f32 per block of 64)
//!     packed:      ceil(n_elems / 4) bytes (2 bits per weight)
//! ```
//!
//! Design decisions:
//! - Self-describing: header contains all architecture constants for validation
//! - Section-based: each weight tensor is an independent section
//! - Mixed precision: small params (LN gamma/beta, SSM S, log_delta) stay f32,
//!   large matrices (KAN coeffs, SSM B/C/D, embedding, head weight) are ternary
//! - Little-endian throughout (matches x86/ARM native order)

use crate::pack::{pack_all, unpack_all};
use crate::quant::{dequantize_block, ternarize_block, TernaryBlock, BLOCK_SIZE};

use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"SOPHON01";
const FORMAT_VERSION: u32 = 1;
const FLAG_TERNARY: u32 = 1;

// ---------------------------------------------------------------------------
// Section descriptor
// ---------------------------------------------------------------------------

/// What kind of encoding a section uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// Raw f32 values, little-endian.
    F32Raw = 0,
    /// Ternary-packed: per-block scales + 2-bit packed weights.
    TernaryPacked = 1,
}

impl SectionKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32Raw),
            1 => Some(Self::TernaryPacked),
            _ => None,
        }
    }
}

/// Metadata for one section in the file.
#[derive(Debug, Clone)]
pub struct SectionMeta {
    pub kind: SectionKind,
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub n_elems: u64,
}

// ---------------------------------------------------------------------------
// File header
// ---------------------------------------------------------------------------

/// Complete file header with architecture constants.
#[derive(Debug, Clone)]
pub struct FileHeader {
    pub version: u32,
    pub flags: u32,
    pub d_model: u32,
    pub n_blocks: u32,
    pub vocab: u32,
    pub ssm_n: u32,
    pub ssm_d: u32,
    pub ssm_p: u32,
    pub ssm_r: u32,
    pub kan_knots: u32,
    pub n_ctrl: u32,
    pub total_params: u64,
    pub sections: Vec<SectionMeta>,
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Accumulates sections and writes them to a binary file.
pub struct ModelWriter {
    header: FileHeader,
    /// Buffered section data. Each entry: (name, kind, raw bytes, n_elems).
    sections_data: Vec<(String, SectionKind, Vec<u8>, u64)>,
}

impl ModelWriter {
    /// Create a writer for the given architecture constants.
    pub fn new(
        d_model: u32,
        n_blocks: u32,
        vocab: u32,
        ssm_n: u32,
        ssm_d: u32,
        ssm_p: u32,
        ssm_r: u32,
        kan_knots: u32,
        n_ctrl: u32,
    ) -> Self {
        Self {
            header: FileHeader {
                version: FORMAT_VERSION,
                flags: 0,
                d_model,
                n_blocks,
                vocab,
                ssm_n,
                ssm_d,
                ssm_p,
                ssm_r,
                kan_knots,
                n_ctrl,
                total_params: 0,
                sections: Vec::new(),
            },
            sections_data: Vec::new(),
        }
    }

    /// Add a section of raw f32 values.
    pub fn add_f32_section(&mut self, name: &str, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.header.total_params += data.len() as u64;
        self.sections_data.push((
            name.to_string(),
            SectionKind::F32Raw,
            bytes,
            data.len() as u64,
        ));
    }

    /// Add a section with ternary quantization.
    ///
    /// The input f32 data is quantized into blocks of BLOCK_SIZE,
    /// then packed at 2 bits per weight with per-block scale factors.
    pub fn add_ternary_section(&mut self, name: &str, data: &[f32]) {
        self.header.flags |= FLAG_TERNARY;
        let n = data.len();
        let n_blocks = n.div_ceil(BLOCK_SIZE);

        // Quantize into blocks
        let mut scales = Vec::with_capacity(n_blocks);
        let mut all_ternary = Vec::with_capacity(n);

        for block_idx in 0..n_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(n);
            let chunk = &data[start..end];

            // Pad to BLOCK_SIZE if needed
            let mut padded = [0.0f32; BLOCK_SIZE];
            padded[..chunk.len()].copy_from_slice(chunk);
            let tb = ternarize_block(&padded);

            scales.push(tb.scale);
            // Only keep the actual (non-padded) ternary values
            for i in 0..chunk.len() {
                all_ternary.push(tb.weights[i]);
            }
        }

        // Pack ternary values
        let packed = pack_all(&all_ternary);

        // Encode: scale_count (u64) + scales (f32 LE) + packed bytes
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n_blocks as u64).to_le_bytes());
        for &s in &scales {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes.extend_from_slice(&packed);

        self.header.total_params += n as u64;
        self.sections_data.push((
            name.to_string(),
            SectionKind::TernaryPacked,
            bytes,
            n as u64,
        ));
    }

    /// Write the complete file.
    pub fn write_to<W: Write + Seek>(&self, w: &mut W) -> io::Result<()> {
        // ---- Header ----
        w.write_all(MAGIC)?;
        w.write_all(&self.header.version.to_le_bytes())?;
        w.write_all(&self.header.flags.to_le_bytes())?;
        w.write_all(&self.header.d_model.to_le_bytes())?;
        w.write_all(&self.header.n_blocks.to_le_bytes())?;
        w.write_all(&self.header.vocab.to_le_bytes())?;
        w.write_all(&self.header.ssm_n.to_le_bytes())?;
        w.write_all(&self.header.ssm_d.to_le_bytes())?;
        w.write_all(&self.header.ssm_p.to_le_bytes())?;
        w.write_all(&self.header.ssm_r.to_le_bytes())?;
        w.write_all(&self.header.kan_knots.to_le_bytes())?;
        w.write_all(&self.header.n_ctrl.to_le_bytes())?;
        w.write_all(&self.header.total_params.to_le_bytes())?;
        w.write_all(&[0u8; 32])?; // reserved

        // ---- Section table ----
        let n_sections = self.sections_data.len() as u32;
        w.write_all(&n_sections.to_le_bytes())?;

        // We need to compute offsets. First, figure out how big the header+table is.
        // Header: 8 (magic) + 11*4 (u32 fields) + 8 (total_params u64) + 32 (reserved) = 92 bytes
        // Section table header: 4 bytes (n_sections)
        // Per section: 1 (kind) + 2 (name_len) + name_len + 8 (offset) + 8 (size) + 8 (n_elems)
        let header_size = 92u64;
        let table_header_size = 4u64;
        let mut table_entry_size = 0u64;
        for (name, _, _, _) in &self.sections_data {
            table_entry_size += 1 + 2 + name.len() as u64 + 8 + 8 + 8;
        }
        let data_start = header_size + table_header_size + table_entry_size;

        // Compute offsets for each section
        let mut offset = data_start;
        let mut offsets = Vec::with_capacity(self.sections_data.len());
        for (_, _, bytes, _) in &self.sections_data {
            offsets.push(offset);
            offset += bytes.len() as u64;
        }

        // Write section table entries
        for (i, (name, kind, bytes, n_elems)) in self.sections_data.iter().enumerate() {
            w.write_all(&[*kind as u8])?;
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
            w.write_all(name_bytes)?;
            w.write_all(&offsets[i].to_le_bytes())?;
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(&n_elems.to_le_bytes())?;
        }

        // ---- Data sections ----
        for (_, _, bytes, _) in &self.sections_data {
            w.write_all(bytes)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read and validate a `.sophon` model file.
pub struct ModelReader {
    pub header: FileHeader,
}

impl ModelReader {
    /// Parse a file from a reader. Returns the header with section metadata.
    pub fn read_header<R: Read>(r: &mut R) -> io::Result<Self> {
        // Magic
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let version = read_u32(r)?;
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {version}"),
            ));
        }

        let flags = read_u32(r)?;
        let d_model = read_u32(r)?;
        let n_blocks = read_u32(r)?;
        let vocab = read_u32(r)?;
        let ssm_n = read_u32(r)?;
        let ssm_d = read_u32(r)?;
        let ssm_p = read_u32(r)?;
        let ssm_r = read_u32(r)?;
        let kan_knots = read_u32(r)?;
        let n_ctrl = read_u32(r)?;
        let total_params = read_u64(r)?;
        let mut _reserved = [0u8; 32];
        r.read_exact(&mut _reserved)?;

        // Section table
        let n_sections = read_u32(r)?;
        let mut sections = Vec::with_capacity(n_sections as usize);
        for _ in 0..n_sections {
            let mut kind_byte = [0u8; 1];
            r.read_exact(&mut kind_byte)?;
            let kind = SectionKind::from_u8(kind_byte[0]).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unknown section kind")
            })?;

            let name_len = read_u16(r)? as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let offset = read_u64(r)?;
            let size = read_u64(r)?;
            let n_elems = read_u64(r)?;

            sections.push(SectionMeta {
                kind,
                name,
                offset,
                size,
                n_elems,
            });
        }

        Ok(Self {
            header: FileHeader {
                version,
                flags,
                d_model,
                n_blocks,
                vocab,
                ssm_n,
                ssm_d,
                ssm_p,
                ssm_r,
                kan_knots,
                n_ctrl,
                total_params,
                sections,
            },
        })
    }

    /// Read a named section as f32 values from a seekable reader.
    /// For f32_raw: reads directly.
    /// For ternary_packed: decompresses on the fly.
    pub fn read_section<R: Read + Seek>(&self, r: &mut R, name: &str) -> io::Result<Vec<f32>> {
        let sec = self
            .header
            .sections
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("section '{name}' not found"),
                )
            })?;

        r.seek(SeekFrom::Start(sec.offset))?;

        match sec.kind {
            SectionKind::F32Raw => {
                let n = sec.n_elems as usize;
                let mut buf = vec![0u8; n * 4];
                r.read_exact(&mut buf)?;
                let mut out = Vec::with_capacity(n);
                for chunk in buf.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            SectionKind::TernaryPacked => {
                let n_blocks = read_u64(r)? as usize;
                let n_elems = sec.n_elems as usize;

                // Read scales
                let mut scale_buf = vec![0u8; n_blocks * 4];
                r.read_exact(&mut scale_buf)?;
                let scales: Vec<f32> = scale_buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                // Read packed bytes
                let packed_len = n_elems.div_ceil(4);
                let mut packed = vec![0u8; packed_len];
                r.read_exact(&mut packed)?;

                // Unpack
                let ternary = unpack_all(&packed, n_elems);

                // Dequantize
                let mut out = Vec::with_capacity(n_elems);
                for (block_idx, &scale) in scales.iter().enumerate() {
                    let start = block_idx * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(n_elems);
                    for i in start..end {
                        out.push(ternary[i] as f32 * scale);
                    }
                }

                Ok(out)
            }
        }
    }

    /// Find a section by name.
    pub fn find_section(&self, name: &str) -> Option<&SectionMeta> {
        self.header.sections.iter().find(|s| s.name == name)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_writer() -> ModelWriter {
        ModelWriter::new(256, 16, 256, 128, 256, 256, 16, 8, 12)
    }

    #[test]
    fn roundtrip_f32_section() {
        let mut w = make_writer();
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        w.add_f32_section("test_f32", &data);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        assert_eq!(reader.header.sections.len(), 1);
        assert_eq!(reader.header.sections[0].name, "test_f32");
        assert_eq!(reader.header.sections[0].kind, SectionKind::F32Raw);

        let loaded = reader.read_section(&mut buf, "test_f32").unwrap();
        assert_eq!(loaded.len(), 100);
        for (a, b) in data.iter().zip(&loaded) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn roundtrip_ternary_section() {
        let mut w = make_writer();
        // Data with clear +1/-1 pattern
        let data: Vec<f32> = (0..200)
            .map(|i| {
                if i % 3 == 0 {
                    1.0
                } else if i % 3 == 1 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        w.add_ternary_section("test_ternary", &data);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        assert_eq!(reader.header.sections[0].kind, SectionKind::TernaryPacked);
        assert_eq!(reader.header.flags & FLAG_TERNARY, FLAG_TERNARY);

        let loaded = reader.read_section(&mut buf, "test_ternary").unwrap();
        assert_eq!(loaded.len(), 200);
        // Ternary quantized values should preserve sign
        for (i, (&orig, &recon)) in data.iter().zip(&loaded).enumerate() {
            if orig.abs() > 0.1 {
                assert_eq!(
                    orig.signum(),
                    recon.signum(),
                    "sign mismatch at {i}: orig={orig}, recon={recon}"
                );
            }
        }
    }

    #[test]
    fn header_validation_constants() {
        let mut w = make_writer();
        w.add_f32_section("test_section", &[1.0, 2.0]);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        assert_eq!(reader.header.d_model, 256);
        assert_eq!(reader.header.n_blocks, 16);
        assert_eq!(reader.header.vocab, 256);
        assert_eq!(reader.header.ssm_n, 128);
        assert_eq!(reader.header.ssm_r, 16);
        assert_eq!(reader.header.kan_knots, 8);
        assert_eq!(reader.header.n_ctrl, 12);
        assert_eq!(reader.header.total_params, 2);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = Cursor::new(b"BADHDR01".to_vec());
        let result = ModelReader::read_header(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn missing_section_returns_error() {
        let mut w = make_writer();
        w.add_f32_section("exists", &[1.0]);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        let result = reader.read_section(&mut buf, "does_not_exist");
        assert!(result.is_err());
    }

    #[test]
    fn multiple_sections() {
        let mut w = make_writer();
        w.add_f32_section("alpha", &[1.0, 2.0, 3.0]);
        w.add_ternary_section("beta", &vec![1.0f32; 128]);
        w.add_f32_section("gamma", &[4.0, 5.0]);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        assert_eq!(reader.header.sections.len(), 3);

        let alpha = reader.read_section(&mut buf, "alpha").unwrap();
        assert_eq!(alpha.len(), 3);
        assert!((alpha[0] - 1.0).abs() < 1e-7);

        let gamma = reader.read_section(&mut buf, "gamma").unwrap();
        assert_eq!(gamma.len(), 2);
        assert!((gamma[1] - 5.0).abs() < 1e-7);

        let beta = reader.read_section(&mut buf, "beta").unwrap();
        assert_eq!(beta.len(), 128);
    }

    #[test]
    fn ternary_section_small_data() {
        // Less than one block
        let mut w = make_writer();
        let data: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, -1.0];
        w.add_ternary_section("small", &data);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.seek(SeekFrom::Start(0)).unwrap();
        let reader = ModelReader::read_header(&mut buf).unwrap();
        let loaded = reader.read_section(&mut buf, "small").unwrap();
        assert_eq!(loaded.len(), 5);
    }
}
