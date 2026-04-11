//! GGUF weight loader — Cold-start bootstrap for distillation.
//!
//! Parses GGUF v3 binary format (llama.cpp) to extract weight tensors
//! from quantised teacher models (e.g. Phi-3-mini-4k-instruct-q4.gguf).
//!
//! Novel optimisation — SBTD (Streaming Block-Tensor Decode):
//!   Instead of loading the entire file and then decoding all tensors,
//!   SBTD reads the header+metadata in one pass, builds a tensor index,
//!   then decodes individual tensors on demand via file seeking.
//!   This keeps peak memory proportional to the largest single tensor
//!   rather than the full model size.
//!
//! Supported GGUF quantisation types (decode to f32):
//!   - F32 (type 0): direct copy
//!   - F16 (type 1): half-to-float conversion
//!   - Q4_0 (type 2): 4-bit quantised, 32 weights per block, f16 scale
//!   - Q4_1 (type 3): 4-bit quantised with min, 32 weights per block
//!   - Q8_0 (type 8): 8-bit quantised, 32 weights per block, f16 scale
//!
//! This module does NOT depend on any external crate.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// -------------------------------------------------------------------------
// GGUF constants
// -------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian
const GGUF_VERSION_3: u32 = 3;
const GGUF_VERSION_2: u32 = 2;

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// GGUF tensor types we support
const GGUF_TENSOR_F32: u32 = 0;
const GGUF_TENSOR_F16: u32 = 1;
const GGUF_TENSOR_Q4_0: u32 = 2;
const GGUF_TENSOR_Q4_1: u32 = 3;
const GGUF_TENSOR_Q8_0: u32 = 8;

// Block sizes for quantised formats
const Q4_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_SIZE: usize = 32;

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

/// Metadata value from a GGUF file.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to extract as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// Try to extract as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            Self::U32(v) => Some(*v as f32),
            Self::I32(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// Tensor descriptor from GGUF header.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape (in GGUF order: innermost first, i.e. row-major from the right).
    pub shape: Vec<u64>,
    /// Quantisation type.
    pub qtype: u32,
    /// Offset from data section start.
    pub offset: u64,
    /// Total number of elements.
    pub n_elements: u64,
}

impl GgufTensorInfo {
    /// Size in bytes of the raw (quantised) data for this tensor.
    pub fn raw_byte_size(&self) -> u64 {
        let n = self.n_elements;
        match self.qtype {
            GGUF_TENSOR_F32 => n * 4,
            GGUF_TENSOR_F16 => n * 2,
            GGUF_TENSOR_Q4_0 => {
                let n_blocks = (n + Q4_BLOCK_SIZE as u64 - 1) / Q4_BLOCK_SIZE as u64;
                // Each Q4_0 block: 2 bytes (f16 scale) + 16 bytes (32 nibbles) = 18 bytes
                n_blocks * 18
            }
            GGUF_TENSOR_Q4_1 => {
                let n_blocks = (n + Q4_BLOCK_SIZE as u64 - 1) / Q4_BLOCK_SIZE as u64;
                // Each Q4_1 block: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes = 20 bytes
                n_blocks * 20
            }
            GGUF_TENSOR_Q8_0 => {
                let n_blocks = (n + Q8_BLOCK_SIZE as u64 - 1) / Q8_BLOCK_SIZE as u64;
                // Each Q8_0 block: 2 bytes (f16 scale) + 32 bytes (32 i8) = 34 bytes
                n_blocks * 34
            }
            _ => n * 4, // fallback assumption
        }
    }

    /// Human-readable quantisation type name.
    pub fn qtype_name(&self) -> &'static str {
        match self.qtype {
            GGUF_TENSOR_F32 => "F32",
            GGUF_TENSOR_F16 => "F16",
            GGUF_TENSOR_Q4_0 => "Q4_0",
            GGUF_TENSOR_Q4_1 => "Q4_1",
            GGUF_TENSOR_Q8_0 => "Q8_0",
            _ => "UNKNOWN",
        }
    }
}

/// Parsed GGUF file header (no tensor data loaded yet).
#[derive(Debug)]
pub struct GgufHeader {
    /// GGUF version (2 or 3).
    pub version: u32,
    /// Number of tensors.
    pub n_tensors: u64,
    /// Number of metadata key-value pairs.
    pub n_metadata: u64,
    /// Metadata key-value pairs.
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor descriptors, indexed by name.
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Ordered tensor names (insertion order).
    pub tensor_names: Vec<String>,
    /// Absolute byte offset where tensor data begins.
    pub data_offset: u64,
}

impl GgufHeader {
    /// Get a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get model architecture string.
    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture")?.as_str()
    }

    /// Get total parameter count estimate.
    pub fn total_params(&self) -> u64 {
        self.tensors.values().map(|t| t.n_elements).sum()
    }

    /// List tensor names matching a prefix.
    pub fn tensors_with_prefix(&self, prefix: &str) -> Vec<&str> {
        self.tensor_names
            .iter()
            .filter(|n| n.starts_with(prefix))
            .map(|n| n.as_str())
            .collect()
    }
}

// -------------------------------------------------------------------------
// Reader
// -------------------------------------------------------------------------

/// Streaming GGUF file reader (SBTD: Streaming Block-Tensor Decode).
///
/// Opens a GGUF file, parses the header and metadata in one sequential pass,
/// then provides on-demand tensor loading via seeking.
pub struct GgufReader<R: Read + Seek> {
    reader: R,
    header: GgufHeader,
}

impl<R: Read + Seek> GgufReader<R> {
    /// Open and parse the GGUF header. Does NOT load tensor data.
    pub fn open(mut reader: R) -> Result<Self, GgufError> {
        let header = parse_header(&mut reader)?;
        Ok(Self { reader, header })
    }

    /// Access the parsed header.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Load a single tensor by name, dequantised to f32.
    ///
    /// This is the core SBTD operation: seek to the tensor's offset,
    /// read exactly its raw bytes, and dequantise in-place.
    pub fn load_tensor(&mut self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?
            .clone();

        let abs_offset = self.header.data_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;

        let raw_size = info.raw_byte_size() as usize;
        let mut raw = vec![0u8; raw_size];
        self.reader.read_exact(&mut raw)?;

        dequantise(&raw, info.qtype, info.n_elements as usize)
    }

    /// Load a tensor and reshape to 2D (rows x cols).
    pub fn load_tensor_2d(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, GgufError> {
        let data = self.load_tensor(name)?;
        if data.len() != rows * cols {
            return Err(GgufError::ShapeMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(data)
    }

    /// Iterate over all tensor names.
    pub fn tensor_names(&self) -> &[String] {
        &self.header.tensor_names
    }
}

// -------------------------------------------------------------------------
// Error type
// -------------------------------------------------------------------------

#[derive(Debug)]
pub enum GgufError {
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    UnsupportedQtype(u32),
    TensorNotFound(String),
    ShapeMismatch { expected: usize, got: usize },
    UnexpectedMetadataType(u32),
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    TruncatedData { expected: usize, got: usize },
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic(m) => write!(f, "invalid GGUF magic: {m:#010x}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            Self::UnsupportedQtype(q) => write!(f, "unsupported quantisation type: {q}"),
            Self::TensorNotFound(n) => write!(f, "tensor not found: {n}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            Self::UnexpectedMetadataType(t) => write!(f, "unexpected metadata type: {t}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Utf8(e) => write!(f, "UTF-8 error: {e}"),
            Self::TruncatedData { expected, got } => {
                write!(f, "truncated data: expected {expected} bytes, got {got}")
            }
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<std::string::FromUtf8Error> for GgufError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::Utf8(e)
    }
}

// -------------------------------------------------------------------------
// Header parsing
// -------------------------------------------------------------------------

fn read_u8(r: &mut impl Read) -> Result<u8, GgufError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> Result<i8, GgufError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> Result<u16, GgufError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> Result<i16, GgufError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> Result<f32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String, GgufError> {
    let len = read_u64(r)? as usize;
    if len > 1_000_000 {
        return Err(GgufError::TruncatedData {
            expected: len,
            got: 0,
        });
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value(r: &mut impl Read, vtype: u32) -> Result<GgufValue, GgufError> {
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::Str(read_string(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            if count > 10_000_000 {
                return Err(GgufError::TruncatedData {
                    expected: count,
                    got: 0,
                });
            }
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        _ => Err(GgufError::UnexpectedMetadataType(vtype)),
    }
}

fn parse_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, GgufError> {
    // Magic
    let magic = read_u32(r)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    // Version
    let version = read_u32(r)?;
    if version != GGUF_VERSION_3 && version != GGUF_VERSION_2 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    // Counts
    let n_tensors = read_u64(r)?;
    let n_metadata = read_u64(r)?;

    // Metadata key-value pairs
    let mut metadata = HashMap::new();
    for _ in 0..n_metadata {
        let key = read_string(r)?;
        let vtype = read_u32(r)?;
        let value = read_value(r, vtype)?;
        metadata.insert(key, value);
    }

    // Tensor descriptors
    let mut tensors = HashMap::new();
    let mut tensor_names = Vec::with_capacity(n_tensors as usize);

    for _ in 0..n_tensors {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(r)?);
        }
        let qtype = read_u32(r)?;
        let offset = read_u64(r)?;

        let n_elements: u64 = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };

        let info = GgufTensorInfo {
            name: name.clone(),
            n_dims,
            shape,
            qtype,
            offset,
            n_elements,
        };
        tensor_names.push(name.clone());
        tensors.insert(name, info);
    }

    // Data section starts at next 32-byte aligned boundary after current position
    let pos = r.stream_position()?;
    let alignment = 32u64;
    let data_offset = (pos + alignment - 1) / alignment * alignment;

    Ok(GgufHeader {
        version,
        n_tensors,
        n_metadata,
        metadata,
        tensors,
        tensor_names,
        data_offset,
    })
}

// -------------------------------------------------------------------------
// Dequantisation
// -------------------------------------------------------------------------

/// Convert f16 (IEEE 754 half-precision) to f32.
///
/// Handwritten — no external dependency.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // +/- zero
            return f32::from_bits(sign << 31);
        }
        // Denormalised: convert to normalised f32
        let mut m = mant;
        let mut e: i32 = -14; // bias diff: 127 - 15 - (10 - 23) would be wrong; correct denorm handling
                              // Normalise
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // remove leading 1
        let f32_exp = ((e + 127) as u32) & 0xFF;
        let f32_mant = m << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    } else if exp == 31 {
        // Inf / NaN
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normalised
        let f32_exp = (exp + 112) as u32; // 127 - 15 = 112
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}

/// Dequantise raw bytes to f32 vector.
fn dequantise(raw: &[u8], qtype: u32, n_elements: usize) -> Result<Vec<f32>, GgufError> {
    match qtype {
        GGUF_TENSOR_F32 => dequant_f32(raw, n_elements),
        GGUF_TENSOR_F16 => dequant_f16(raw, n_elements),
        GGUF_TENSOR_Q4_0 => dequant_q4_0(raw, n_elements),
        GGUF_TENSOR_Q4_1 => dequant_q4_1(raw, n_elements),
        GGUF_TENSOR_Q8_0 => dequant_q8_0(raw, n_elements),
        _ => Err(GgufError::UnsupportedQtype(qtype)),
    }
}

fn dequant_f32(raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if raw.len() < n * 4 {
        return Err(GgufError::TruncatedData {
            expected: n * 4,
            got: raw.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        let bytes: [u8; 4] = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

fn dequant_f16(raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if raw.len() < n * 2 {
        return Err(GgufError::TruncatedData {
            expected: n * 2,
            got: raw.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 2;
        let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
        out.push(f16_to_f32(bits));
    }
    Ok(out)
}

/// Q4_0 dequantisation: each block of 32 weights stored as:
///   [f16 scale (2 bytes)] [16 bytes of packed nibbles]
/// Total 18 bytes per block.
fn dequant_q4_0(raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let n_blocks = (n + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let expected = n_blocks * 18;
    if raw.len() < expected {
        return Err(GgufError::TruncatedData {
            expected,
            got: raw.len(),
        });
    }

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * 18;
        let scale_bits = u16::from_le_bytes([raw[base], raw[base + 1]]);
        let scale = f16_to_f32(scale_bits);

        let remaining = (n - b * Q4_BLOCK_SIZE).min(Q4_BLOCK_SIZE);
        for i in 0..remaining {
            let byte_idx = base + 2 + i / 2;
            let nibble = if i % 2 == 0 {
                (raw[byte_idx] & 0x0F) as i8 - 8
            } else {
                ((raw[byte_idx] >> 4) & 0x0F) as i8 - 8
            };
            out.push(scale * nibble as f32);
        }
    }

    out.truncate(n);
    Ok(out)
}

/// Q4_1 dequantisation: each block of 32 weights stored as:
///   [f16 scale (2 bytes)] [f16 min (2 bytes)] [16 bytes of packed nibbles]
/// Total 20 bytes per block. Values = scale * nibble + min.
fn dequant_q4_1(raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let n_blocks = (n + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let expected = n_blocks * 20;
    if raw.len() < expected {
        return Err(GgufError::TruncatedData {
            expected,
            got: raw.len(),
        });
    }

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * 20;
        let scale_bits = u16::from_le_bytes([raw[base], raw[base + 1]]);
        let min_bits = u16::from_le_bytes([raw[base + 2], raw[base + 3]]);
        let scale = f16_to_f32(scale_bits);
        let min_val = f16_to_f32(min_bits);

        let remaining = (n - b * Q4_BLOCK_SIZE).min(Q4_BLOCK_SIZE);
        for i in 0..remaining {
            let byte_idx = base + 4 + i / 2;
            let nibble = if i % 2 == 0 {
                (raw[byte_idx] & 0x0F) as f32
            } else {
                ((raw[byte_idx] >> 4) & 0x0F) as f32
            };
            out.push(scale * nibble + min_val);
        }
    }

    out.truncate(n);
    Ok(out)
}

/// Q8_0 dequantisation: each block of 32 weights stored as:
///   [f16 scale (2 bytes)] [32 int8 values]
/// Total 34 bytes per block.
fn dequant_q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let n_blocks = (n + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    let expected = n_blocks * 34;
    if raw.len() < expected {
        return Err(GgufError::TruncatedData {
            expected,
            got: raw.len(),
        });
    }

    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * 34;
        let scale_bits = u16::from_le_bytes([raw[base], raw[base + 1]]);
        let scale = f16_to_f32(scale_bits);

        let remaining = (n - b * Q8_BLOCK_SIZE).min(Q8_BLOCK_SIZE);
        for i in 0..remaining {
            let val = raw[base + 2 + i] as i8;
            out.push(scale * val as f32);
        }
    }

    out.truncate(n);
    Ok(out)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper: write a minimal valid GGUF file with f32 tensors.
    fn build_test_gguf(tensors: &[(&str, &[u64], &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // n_metadata = 1 (architecture key)
        buf.extend_from_slice(&1u64.to_le_bytes());

        // Metadata: general.architecture = "test"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let val = b"test";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // Tensor info
        let mut data_blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset: u64 = 0;
        for (name, shape, data) in tensors {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in *shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&GGUF_TENSOR_F32.to_le_bytes()); // qtype = f32
            buf.extend_from_slice(&offset.to_le_bytes());

            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            offset += raw.len() as u64;
            data_blobs.push(raw);
        }

        // Pad to 32-byte alignment
        let pos = buf.len();
        let alignment = 32;
        let aligned = (pos + alignment - 1) / alignment * alignment;
        buf.resize(aligned, 0);

        // Tensor data
        for blob in &data_blobs {
            buf.extend_from_slice(blob);
        }

        buf
    }

    #[test]
    fn parse_minimal_gguf() {
        let data =
            build_test_gguf(&[("weight", &[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])]);
        let cursor = Cursor::new(data);
        let mut reader = GgufReader::open(cursor).unwrap();

        assert_eq!(reader.header().version, 3);
        assert_eq!(reader.header().n_tensors, 1);
        assert_eq!(reader.header().architecture(), Some("test"));
        assert_eq!(reader.header().total_params(), 8);
    }

    #[test]
    fn load_f32_tensor() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data = build_test_gguf(&[("w", &[2, 3], &values)]);
        let cursor = Cursor::new(data);
        let mut reader = GgufReader::open(cursor).unwrap();

        let loaded = reader.load_tensor("w").unwrap();
        assert_eq!(loaded.len(), 6);
        for (a, b) in loaded.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn load_tensor_2d_shape_check() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let data = build_test_gguf(&[("m", &[2, 2], &values)]);
        let cursor = Cursor::new(data);
        let mut reader = GgufReader::open(cursor).unwrap();

        let ok = reader.load_tensor_2d("m", 2, 2);
        assert!(ok.is_ok());

        // Reopen for mismatch test
        let data2 = build_test_gguf(&[("m", &[2, 2], &values)]);
        let cursor2 = Cursor::new(data2);
        let mut reader2 = GgufReader::open(cursor2).unwrap();

        let bad = reader2.load_tensor_2d("m", 3, 3);
        assert!(bad.is_err());
    }

    #[test]
    fn multiple_tensors() {
        let a_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_vals = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
        let data = build_test_gguf(&[("a", &[4], &a_vals), ("b", &[6], &b_vals)]);
        let cursor = Cursor::new(data);
        let mut reader = GgufReader::open(cursor).unwrap();

        assert_eq!(reader.tensor_names().len(), 2);

        let a = reader.load_tensor("a").unwrap();
        assert_eq!(a.len(), 4);
        assert!((a[0] - 1.0).abs() < 1e-6);

        let b = reader.load_tensor("b").unwrap();
        assert_eq!(b.len(), 6);
        assert!((b[4] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_not_found() {
        let data = build_test_gguf(&[("x", &[2], &[1.0, 2.0])]);
        let cursor = Cursor::new(data);
        let mut reader = GgufReader::open(cursor).unwrap();

        let err = reader.load_tensor("nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let cursor = Cursor::new(data);
        let result = GgufReader::open(cursor);
        assert!(result.is_err());
    }

    #[test]
    fn f16_conversion_basic() {
        // f16 for 1.0: sign=0, exp=15, mant=0 -> bits = 0x3C00
        let one = f16_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-6);

        // f16 for -1.0: sign=1, exp=15, mant=0 -> bits = 0xBC00
        let neg_one = f16_to_f32(0xBC00);
        assert!((neg_one + 1.0).abs() < 1e-6);

        // f16 for 0.0: all zeros
        let zero = f16_to_f32(0x0000);
        assert!((zero).abs() < 1e-10);

        // f16 for 0.5: sign=0, exp=14, mant=0 -> bits = 0x3800
        let half = f16_to_f32(0x3800);
        assert!((half - 0.5).abs() < 1e-6);
    }

    #[test]
    fn metadata_value_accessors() {
        let v = GgufValue::U32(42);
        assert_eq!(v.as_u32(), Some(42));
        assert_eq!(v.as_str(), None);

        let v = GgufValue::Str("hello".to_string());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(v.as_u32(), None);

        let v = GgufValue::F32(3.14);
        assert!((v.as_f32().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn tensors_with_prefix_filter() {
        let a = vec![1.0f32];
        let b = vec![2.0f32];
        let c = vec![3.0f32];
        let data = build_test_gguf(&[
            ("blk.0.weight", &[1], &a),
            ("blk.1.weight", &[1], &b),
            ("output.weight", &[1], &c),
        ]);
        let cursor = Cursor::new(data);
        let reader = GgufReader::open(cursor).unwrap();

        let blk = reader.header().tensors_with_prefix("blk.");
        assert_eq!(blk.len(), 2);

        let out = reader.header().tensors_with_prefix("output.");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn qtype_name_coverage() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 1,
            shape: vec![10],
            qtype: GGUF_TENSOR_Q4_0,
            offset: 0,
            n_elements: 10,
        };
        assert_eq!(info.qtype_name(), "Q4_0");

        let info2 = GgufTensorInfo { qtype: 99, ..info };
        assert_eq!(info2.qtype_name(), "UNKNOWN");
    }
}
