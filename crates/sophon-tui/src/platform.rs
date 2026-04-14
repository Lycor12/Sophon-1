//! Platform-specific terminal handling
//!
//! Provides ANSI support for Windows terminals and cross-platform compatibility.

use std::io::{self, Write};

/// Initialize terminal for ANSI output
///
/// On Windows, this enables virtual terminal processing for ANSI escape codes.
/// On Unix, this is a no-op.
pub fn init_terminal() -> io::Result<()> {
    #[cfg(windows)]
    {
        enable_windows_ansi()?;
    }
    Ok(())
}

/// Reset terminal to original state
#[cfg(windows)]
pub fn reset_terminal() -> io::Result<()> {
    // On Windows, we don't need to do anything special on reset
    // The console mode changes are per-process
    Ok(())
}

/// Write ANSI sequence to terminal with proper platform handling
pub fn write_ansi(data: &str) -> io::Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(data.as_bytes())?;
    handle.flush()
}

/// Clear screen
pub fn clear_screen() -> io::Result<()> {
    write_ansi("\x1b[2J\x1b[H")
}

/// Clear from cursor to end of screen
pub fn clear_from_cursor() -> io::Result<()> {
    write_ansi("\x1b[0J")
}

/// Hide cursor
pub fn hide_cursor() -> io::Result<()> {
    write_ansi("\x1b[?25l")
}

/// Show cursor
pub fn show_cursor() -> io::Result<()> {
    write_ansi("\x1b[?25h")
}

/// Move cursor to position (1-indexed)
pub fn move_cursor(x: u16, y: u16) -> io::Result<()> {
    write_ansi(&format!("\x1b[{};{}H", y + 1, x + 1))
}

/// Save cursor position
pub fn save_cursor() -> io::Result<()> {
    write_ansi("\x1b[s")
}

/// Restore cursor position
pub fn restore_cursor() -> io::Result<()> {
    write_ansi("\x1b[u")
}

/// Enter alternate screen
pub fn enter_alternate_screen() -> io::Result<()> {
    write_ansi("\x1b[?1049h")
}

/// Leave alternate screen
pub fn leave_alternate_screen() -> io::Result<()> {
    write_ansi("\x1b[?1049l")
}

/// Windows-specific implementation
#[cfg(windows)]
mod windows_impl {
    use std::io;

    /// Enable ANSI escape codes on Windows
    pub fn enable_windows_ansi() -> io::Result<()> {
        use std::os::windows::io::AsRawHandle;

        // Get stdout handle
        let stdout = std::io::stdout();
        let handle = stdout.lock().as_raw_handle();

        // Enable virtual terminal processing
        let mut mode: u32 = 0;
        unsafe {
            if GetConsoleMode(handle as _, &mut mode) != 0 {
                mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                if SetConsoleMode(handle as _, mode) != 0 {
                    return Ok(());
                }
            }
        }

        // If virtual terminal processing isn't available, we continue anyway
        // The terminal will show raw codes, but that's better than crashing
        Ok(())
    }

    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    extern "system" {
        fn GetConsoleMode(hConsoleHandle: isize, lpMode: *mut u32) -> i32;
        fn SetConsoleMode(hConsoleHandle: isize, dwMode: u32) -> i32;
    }
}

#[cfg(windows)]
use windows_impl::enable_windows_ansi;

#[cfg(not(windows))]
pub fn enable_windows_ansi() -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_terminal() {
        // Should not panic
        init_terminal().unwrap();
    }

    #[test]
    fn test_write_ansi() {
        // Should not panic (writes to stdout)
        let _ = write_ansi("\x1b[0m");
    }

    #[test]
    fn test_cursor_operations() {
        // These should not panic
        let _ = save_cursor();
        let _ = move_cursor(10, 5);
        let _ = restore_cursor();
    }
}
