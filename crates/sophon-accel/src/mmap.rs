//! Memory-mapped file I/O.
//!
//! Provides read-only memory mapping for weight files.
//! Windows: CreateFileMapping + MapViewOfFile.
//! Non-Windows: placeholder that falls back to read-into-buffer.

use std::path::Path;

/// A memory-mapped file handle providing read-only byte access.
pub struct MappedFile {
    #[cfg(target_os = "windows")]
    data: *const u8,
    #[cfg(target_os = "windows")]
    len: usize,
    #[cfg(target_os = "windows")]
    _file_handle: *mut std::ffi::c_void,
    #[cfg(target_os = "windows")]
    _mapping_handle: *mut std::ffi::c_void,

    #[cfg(not(target_os = "windows"))]
    buffer: Vec<u8>,
}

// Safety: MappedFile is read-only, the mapped region doesn't change.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

/// Error type for memory mapping operations.
#[derive(Debug)]
pub enum MmapError {
    IoError(std::io::Error),
    EmptyFile,
    MappingFailed(u32),
    ViewFailed(u32),
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::IoError(e) => write!(f, "I/O error: {}", e),
            MmapError::EmptyFile => write!(f, "file is empty"),
            MmapError::MappingFailed(code) => write!(f, "CreateFileMapping failed: {}", code),
            MmapError::ViewFailed(code) => write!(f, "MapViewOfFile failed: {}", code),
        }
    }
}

// === Windows implementation ===
#[cfg(target_os = "windows")]
mod win32 {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileW(
            file_name: *const u16,
            access: u32,
            share_mode: u32,
            security: *const std::ffi::c_void,
            creation: u32,
            flags: u32,
            template: *const std::ffi::c_void,
        ) -> *mut std::ffi::c_void;

        pub fn CreateFileMappingW(
            file: *mut std::ffi::c_void,
            security: *const std::ffi::c_void,
            protect: u32,
            size_hi: u32,
            size_lo: u32,
            name: *const u16,
        ) -> *mut std::ffi::c_void;

        pub fn MapViewOfFile(
            mapping: *mut std::ffi::c_void,
            access: u32,
            offset_hi: u32,
            offset_lo: u32,
            bytes: usize,
        ) -> *const u8;

        pub fn UnmapViewOfFile(base: *const u8) -> i32;
        pub fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
        pub fn GetLastError() -> u32;
        pub fn GetFileSizeEx(file: *mut std::ffi::c_void, size: *mut i64) -> i32;
    }

    pub const GENERIC_READ: u32 = 0x80000000;
    pub const FILE_SHARE_READ: u32 = 0x00000001;
    pub const OPEN_EXISTING: u32 = 3;
    pub const FILE_ATTRIBUTE_READONLY: u32 = 0x00000001;
    pub const PAGE_READONLY: u32 = 0x02;
    pub const FILE_MAP_READ: u32 = 0x04;
    pub const INVALID_HANDLE_VALUE: *mut std::ffi::c_void = -1isize as *mut std::ffi::c_void;
}

#[cfg(target_os = "windows")]
impl MappedFile {
    /// Open a file and memory-map it read-only.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let path_str = path.as_ref().to_string_lossy();
        let wide: Vec<u16> = path_str.encode_utf16().chain(std::iter::once(0)).collect();

        unsafe {
            // Open file
            let file_handle = win32::CreateFileW(
                wide.as_ptr(),
                win32::GENERIC_READ,
                win32::FILE_SHARE_READ,
                std::ptr::null(),
                win32::OPEN_EXISTING,
                win32::FILE_ATTRIBUTE_READONLY,
                std::ptr::null(),
            );
            if file_handle == win32::INVALID_HANDLE_VALUE {
                return Err(MmapError::IoError(std::io::Error::last_os_error()));
            }

            // Get file size
            let mut size: i64 = 0;
            if win32::GetFileSizeEx(file_handle, &mut size) == 0 {
                win32::CloseHandle(file_handle);
                return Err(MmapError::IoError(std::io::Error::last_os_error()));
            }
            if size == 0 {
                win32::CloseHandle(file_handle);
                return Err(MmapError::EmptyFile);
            }

            // Create file mapping
            let mapping_handle = win32::CreateFileMappingW(
                file_handle,
                std::ptr::null(),
                win32::PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping_handle.is_null() {
                let err = win32::GetLastError();
                win32::CloseHandle(file_handle);
                return Err(MmapError::MappingFailed(err));
            }

            // Map view
            let data = win32::MapViewOfFile(mapping_handle, win32::FILE_MAP_READ, 0, 0, 0);
            if data.is_null() {
                let err = win32::GetLastError();
                win32::CloseHandle(mapping_handle);
                win32::CloseHandle(file_handle);
                return Err(MmapError::ViewFailed(err));
            }

            Ok(MappedFile {
                data,
                len: size as usize,
                _file_handle: file_handle,
                _mapping_handle: mapping_handle,
            })
        }
    }

    /// Get a byte slice of the mapped region.
    pub fn as_slice(&self) -> &[u8] {
        // Safety: data is valid for len bytes, read-only, lifetime tied to self.
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(target_os = "windows")]
impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            win32::UnmapViewOfFile(self.data);
            win32::CloseHandle(self._mapping_handle);
            win32::CloseHandle(self._file_handle);
        }
    }
}

// === Non-Windows fallback ===
#[cfg(not(target_os = "windows"))]
impl MappedFile {
    /// Fallback: read entire file into a buffer.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let buffer = std::fs::read(path.as_ref()).map_err(MmapError::IoError)?;
        if buffer.is_empty() {
            return Err(MmapError::EmptyFile);
        }
        Ok(MappedFile { buffer })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_existing_file() {
        // Write a temp file, mmap it, verify contents
        let dir = std::env::temp_dir();
        let path = dir.join("sophon_mmap_test.bin");
        let data = b"Hello, memory-mapped world!";
        std::fs::write(&path, data).expect("write temp file");

        let mapped = MappedFile::open(&path).expect("mmap open");
        assert_eq!(mapped.len(), data.len());
        assert_eq!(mapped.as_slice(), data);
        assert!(!mapped.is_empty());

        drop(mapped);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_nonexistent_file() {
        let result = MappedFile::open("nonexistent_file_12345.bin");
        assert!(result.is_err());
    }

    #[test]
    fn open_empty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("sophon_mmap_empty.bin");
        std::fs::write(&path, b"").expect("write empty file");

        let result = MappedFile::open(&path);
        assert!(matches!(result, Err(MmapError::EmptyFile)));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn large_file_mmap() {
        let dir = std::env::temp_dir();
        let path = dir.join("sophon_mmap_large.bin");
        let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        std::fs::write(&path, &data).expect("write large file");

        let mapped = MappedFile::open(&path).expect("mmap open");
        assert_eq!(mapped.len(), 10_000);
        assert_eq!(mapped.as_slice(), data.as_slice());

        drop(mapped);
        std::fs::remove_file(&path).ok();
    }
}
