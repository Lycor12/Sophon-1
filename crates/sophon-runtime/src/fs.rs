//! Filesystem primitives (spec §5.1.2).

use std::io;
use std::path::Path;

/// Read a file to bytes.
pub fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Read a file to UTF-8 string (returns error if not valid UTF-8).
pub fn read_text(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
}

/// Write bytes to a file (overwrite).
pub fn write_file(path: &Path, data: &[u8]) -> io::Result<()> {
    std::fs::write(path, data)
}

/// Append bytes to a file.
pub fn append_file(path: &Path, data: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(data)
}

/// List directory entries (file names only, not recursive).
pub fn list_dir(path: &Path) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let e = entry?;
        names.push(e.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

/// File metadata: size in bytes and whether it is a directory.
pub struct FileStat {
    pub size: u64,
    pub is_dir: bool,
}

pub fn stat(path: &Path) -> io::Result<FileStat> {
    let m = std::fs::metadata(path)?;
    Ok(FileStat {
        size: m.len(),
        is_dir: m.is_dir(),
    })
}

/// Delete a file.
pub fn delete_file(path: &Path) -> io::Result<()> {
    std::fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("sophon_fs_test.bin");
        let data = b"sophon-test-payload";
        write_file(&path, data).unwrap();
        let read = read_file(&path).unwrap();
        assert_eq!(read, data);
        delete_file(&path).unwrap();
    }

    #[test]
    fn list_dir_nonempty() {
        let dir = std::env::temp_dir();
        let entries = list_dir(&dir).unwrap();
        // temp dir always has something in it on Windows
        let _ = entries; // just verify no panic
    }
}
