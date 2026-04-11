//! Corpus reader — streaming document loading from disk.
//!
//! Supports loading from:
//!   - Single text files (UTF-8)
//!   - Directory of text files
//!   - Line-delimited format (one document per line)
//!
//! Documents are yielded as raw byte sequences, consistent with the
//! byte-level tokenization scheme (VOCAB_SIZE = 256).

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A single document in the corpus.
#[derive(Debug, Clone)]
pub struct Document {
    /// Raw byte content.
    pub bytes: Vec<u8>,
    /// Source file path (for provenance).
    pub source: String,
    /// Document index within the source file.
    pub index: usize,
}

impl Document {
    /// Byte length.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the document is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Try to interpret as UTF-8.
    pub fn as_text(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes).ok()
    }

    /// Byte-level entropy estimate (Shannon entropy over byte histogram).
    pub fn byte_entropy(&self) -> f32 {
        if self.bytes.is_empty() {
            return 0.0;
        }

        let mut counts = [0u32; 256];
        for &b in &self.bytes {
            counts[b as usize] += 1;
        }

        let n = self.bytes.len() as f32;
        let mut entropy = 0.0f32;
        for &c in &counts {
            if c > 0 {
                let p = c as f32 / n;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

// ---------------------------------------------------------------------------
// CorpusReader
// ---------------------------------------------------------------------------

/// Streaming corpus reader.
///
/// Reads documents from files on disk. Supports:
/// - Single file: each line is a document
/// - Directory: each file is a document
/// - Custom delimiter
pub struct CorpusReader {
    /// Paths to read from.
    paths: Vec<PathBuf>,
    /// Whether to treat each line as a separate document.
    line_delimited: bool,
    /// Current path index.
    path_idx: usize,
    /// Current line reader (for line-delimited mode).
    current_reader: Option<BufReader<fs::File>>,
    /// Global document counter.
    doc_idx: usize,
}

impl CorpusReader {
    /// Create a reader for a single file (line-delimited).
    pub fn from_file(path: &Path) -> io::Result<Self> {
        if !path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "file not found"));
        }
        Ok(Self {
            paths: vec![path.to_path_buf()],
            line_delimited: true,
            path_idx: 0,
            current_reader: None,
            doc_idx: 0,
        })
    }

    /// Create a reader for a directory (each file = one document).
    pub fn from_directory(dir: &Path) -> io::Result<Self> {
        if !dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "not a directory",
            ));
        }

        let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        paths.sort();

        Ok(Self {
            paths,
            line_delimited: false,
            path_idx: 0,
            current_reader: None,
            doc_idx: 0,
        })
    }

    /// Create a reader from a list of paths (each file = one document).
    pub fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            line_delimited: false,
            path_idx: 0,
            current_reader: None,
            doc_idx: 0,
        }
    }

    /// Read the next document. Returns None when all documents are exhausted.
    pub fn next_document(&mut self) -> io::Result<Option<Document>> {
        if self.line_delimited {
            self.next_line_document()
        } else {
            self.next_file_document()
        }
    }

    /// Read all remaining documents into a vector.
    pub fn read_all(&mut self) -> io::Result<Vec<Document>> {
        let mut docs = Vec::new();
        while let Some(doc) = self.next_document()? {
            docs.push(doc);
        }
        Ok(docs)
    }

    /// Reset the reader to the beginning.
    pub fn reset(&mut self) {
        self.path_idx = 0;
        self.current_reader = None;
        self.doc_idx = 0;
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn next_line_document(&mut self) -> io::Result<Option<Document>> {
        loop {
            // Ensure we have a reader open
            if self.current_reader.is_none() {
                if self.path_idx >= self.paths.len() {
                    return Ok(None);
                }
                let file = fs::File::open(&self.paths[self.path_idx])?;
                self.current_reader = Some(BufReader::new(file));
            }

            // Read next line
            let reader = self.current_reader.as_mut().unwrap();
            let mut line = String::new();
            let n = reader.read_line(&mut line)?;

            if n == 0 {
                // End of file — move to next path
                self.current_reader = None;
                self.path_idx += 1;
                continue;
            }

            // Trim trailing newline
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed.is_empty() {
                continue; // Skip empty lines
            }

            let source = self.paths[self.path_idx.min(self.paths.len() - 1)]
                .to_string_lossy()
                .to_string();
            let idx = self.doc_idx;
            self.doc_idx += 1;

            return Ok(Some(Document {
                bytes: trimmed.as_bytes().to_vec(),
                source,
                index: idx,
            }));
        }
    }

    fn next_file_document(&mut self) -> io::Result<Option<Document>> {
        if self.path_idx >= self.paths.len() {
            return Ok(None);
        }

        let path = &self.paths[self.path_idx];
        let bytes = fs::read(path)?;
        let source = path.to_string_lossy().to_string();
        let idx = self.doc_idx;
        self.doc_idx += 1;
        self.path_idx += 1;

        Ok(Some(Document {
            bytes,
            source,
            index: idx,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    fn write_temp_file(name: &str, content: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("sophon_test_{}_{}.txt", std::process::id(), name));
        fs::write(&path, content).unwrap();
        path
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn line_delimited_reading() {
        let path = write_temp_file("line", "hello\nworld\nfoo bar\n");
        let mut reader = CorpusReader::from_file(&path).unwrap();
        let docs = reader.read_all().unwrap();
        cleanup(&path);
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].bytes, b"hello");
        assert_eq!(docs[1].bytes, b"world");
        assert_eq!(docs[2].bytes, b"foo bar");
    }

    #[test]
    fn empty_lines_skipped() {
        let path = write_temp_file("empty", "hello\n\n\nworld\n");
        let mut reader = CorpusReader::from_file(&path).unwrap();
        let docs = reader.read_all().unwrap();
        cleanup(&path);
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn byte_entropy_of_uniform_is_high() {
        let bytes: Vec<u8> = (0..=255).collect();
        let doc = Document {
            bytes,
            source: "test".into(),
            index: 0,
        };
        let e = doc.byte_entropy();
        assert!(e > 7.9, "uniform byte entropy should be ~8.0, got {e}");
    }

    #[test]
    fn byte_entropy_of_constant_is_zero() {
        let doc = Document {
            bytes: vec![0u8; 100],
            source: "test".into(),
            index: 0,
        };
        assert!((doc.byte_entropy() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn document_as_text() {
        let doc = Document {
            bytes: b"hello world".to_vec(),
            source: "test".into(),
            index: 0,
        };
        assert_eq!(doc.as_text(), Some("hello world"));
    }

    #[test]
    fn reset_allows_re_reading() {
        let path = write_temp_file("reset", "a\nb\nc\n");
        let mut reader = CorpusReader::from_file(&path).unwrap();
        let docs1 = reader.read_all().unwrap();
        reader.reset();
        let docs2 = reader.read_all().unwrap();
        cleanup(&path);
        assert_eq!(docs1.len(), docs2.len());
    }

    #[test]
    fn file_not_found_error() {
        let result = CorpusReader::from_file(Path::new("/nonexistent/path.txt"));
        assert!(result.is_err());
    }
}
