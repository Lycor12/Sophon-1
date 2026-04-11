//! ANSI escape sequence handling

/// ANSI escape codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnsiCode {
    Reset,
    Bold,
    Dim,
    Italic,
    Underline,
    Blink,
    Reverse,
    Hidden,
    Strikethrough,
    Foreground(ColorCode),
    Background(ColorCode),
    CursorUp(u16),
    CursorDown(u16),
    CursorForward(u16),
    CursorBack(u16),
    CursorPos(u16, u16),
    ClearScreen,
    ClearLine,
}

/// Color codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorCode {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
    Rgb(u8, u8, u8),
    Ansi256(u8),
}

impl ColorCode {
    /// Get foreground ANSI sequence
    pub fn fg_code(&self) -> String {
        match self {
            ColorCode::Black => "30".to_string(),
            ColorCode::Red => "31".to_string(),
            ColorCode::Green => "32".to_string(),
            ColorCode::Yellow => "33".to_string(),
            ColorCode::Blue => "34".to_string(),
            ColorCode::Magenta => "35".to_string(),
            ColorCode::Cyan => "36".to_string(),
            ColorCode::White => "37".to_string(),
            ColorCode::BrightBlack => "90".to_string(),
            ColorCode::BrightRed => "91".to_string(),
            ColorCode::BrightGreen => "92".to_string(),
            ColorCode::BrightYellow => "93".to_string(),
            ColorCode::BrightBlue => "94".to_string(),
            ColorCode::BrightMagenta => "95".to_string(),
            ColorCode::BrightCyan => "96".to_string(),
            ColorCode::BrightWhite => "97".to_string(),
            ColorCode::Rgb(r, g, b) => format!("38;2;{};{};{}", r, g, b),
            ColorCode::Ansi256(n) => format!("38;5;{}", n),
        }
    }

    /// Get background ANSI sequence
    pub fn bg_code(&self) -> String {
        match self {
            ColorCode::Black => "40".to_string(),
            ColorCode::Red => "41".to_string(),
            ColorCode::Green => "42".to_string(),
            ColorCode::Yellow => "43".to_string(),
            ColorCode::Blue => "44".to_string(),
            ColorCode::Magenta => "45".to_string(),
            ColorCode::Cyan => "46".to_string(),
            ColorCode::White => "47".to_string(),
            ColorCode::BrightBlack => "100".to_string(),
            ColorCode::BrightRed => "101".to_string(),
            ColorCode::BrightGreen => "102".to_string(),
            ColorCode::BrightYellow => "103".to_string(),
            ColorCode::BrightBlue => "104".to_string(),
            ColorCode::BrightMagenta => "105".to_string(),
            ColorCode::BrightCyan => "106".to_string(),
            ColorCode::BrightWhite => "107".to_string(),
            ColorCode::Rgb(r, g, b) => format!("48;2;{};{};{}", r, g, b),
            ColorCode::Ansi256(n) => format!("48;5;{}", n),
        }
    }
}

impl AnsiCode {
    /// Convert to ANSI escape sequence string
    pub fn to_string(&self) -> String {
        match self {
            AnsiCode::Reset => "\x1b[0m".to_string(),
            AnsiCode::Bold => "\x1b[1m".to_string(),
            AnsiCode::Dim => "\x1b[2m".to_string(),
            AnsiCode::Italic => "\x1b[3m".to_string(),
            AnsiCode::Underline => "\x1b[4m".to_string(),
            AnsiCode::Blink => "\x1b[5m".to_string(),
            AnsiCode::Reverse => "\x1b[7m".to_string(),
            AnsiCode::Hidden => "\x1b[8m".to_string(),
            AnsiCode::Strikethrough => "\x1b[9m".to_string(),
            AnsiCode::Foreground(c) => format!("\x1b[{}m", c.fg_code()),
            AnsiCode::Background(c) => format!("\x1b[{}m", c.bg_code()),
            AnsiCode::CursorUp(n) => format!("\x1b[{}A", n),
            AnsiCode::CursorDown(n) => format!("\x1b[{}B", n),
            AnsiCode::CursorForward(n) => format!("\x1b[{}C", n),
            AnsiCode::CursorBack(n) => format!("\x1b[{}D", n),
            AnsiCode::CursorPos(row, col) => format!("\x1b[{};{}H", row, col),
            AnsiCode::ClearScreen => "\x1b[2J\x1b[H".to_string(),
            AnsiCode::ClearLine => "\x1b[2K".to_string(),
        }
    }
}

/// Buffered ANSI output
#[derive(Debug, Default)]
pub struct AnsiBuffer {
    content: String,
}

impl AnsiBuffer {
    /// Create new empty buffer
    pub fn new() -> Self {
        AnsiBuffer {
            content: String::new(),
        }
    }

    /// Add ANSI code to buffer
    pub fn push(&mut self, code: AnsiCode) {
        self.content.push_str(&code.to_string());
    }

    /// Add raw string to buffer
    pub fn push_str(&mut self, s: &str) {
        self.content.push_str(s);
    }

    /// Add character to buffer
    pub fn push_char(&mut self, c: char) {
        self.content.push(c);
    }

    /// Move cursor to position
    pub fn move_to(&mut self, row: u16, col: u16) {
        self.push(AnsiCode::CursorPos(row, col));
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.content.clear();
    }

    /// Get buffer content
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Take buffer content (clears buffer)
    pub fn take(&mut self) -> String {
        std::mem::take(&mut self.content)
    }

    /// Write buffer to output
    pub fn flush(&mut self, output: &mut dyn std::io::Write) -> std::io::Result<()> {
        output.write_all(self.content.as_bytes())?;
        output.flush()?;
        self.content.clear();
        Ok(())
    }
}

/// Detect terminal capabilities
pub struct TerminalCapabilities;

impl TerminalCapabilities {
    /// Check if terminal supports true color
    pub fn supports_truecolor() -> bool {
        std::env::var("COLORTERM")
            .map(|v| v == "truecolor" || v == "24bit")
            .unwrap_or(false)
    }

    /// Check if terminal supports 256 colors
    pub fn supports_256color() -> bool {
        std::env::var("TERM")
            .map(|v| v.contains("256color"))
            .unwrap_or(false)
    }

    /// Get terminal width
    pub fn width() -> Option<u16> {
        terminal_size().map(|(w, _)| w)
    }

    /// Get terminal height
    pub fn height() -> Option<u16> {
        terminal_size().map(|(_, h)| h)
    }
}

/// Get terminal size using standard escape sequences
#[cfg(unix)]
pub fn terminal_size() -> Option<(u16, u16)> {
    // On Unix, use ioctl
    None // Placeholder - would need libc::ioctl
}

#[cfg(windows)]
pub fn terminal_size() -> Option<(u16, u16)> {
    // On Windows, use GetConsoleScreenBufferInfo
    None // Placeholder - would need Win32 API
}

/// ANSI reset sequence
pub const RESET: &str = "\x1b[0m";

/// Convert a Style to ANSI escape sequences
/// Returns only the changes needed from the previous style
pub fn style_to_ansi(style: &crate::style::Style, prev: &crate::style::Style) -> String {
    use crate::style::Color;

    let mut codes = Vec::new();

    // Reset if needed
    if style.fg != prev.fg || style.bg != prev.bg {
        codes.push("0".to_string());

        // Re-apply text styles after reset
        if style.text_style.bold {
            codes.push("1".to_string());
        }
        if style.text_style.dim {
            codes.push("2".to_string());
        }
        if style.text_style.italic {
            codes.push("3".to_string());
        }
        if style.text_style.underline {
            codes.push("4".to_string());
        }
        if style.text_style.blink {
            codes.push("5".to_string());
        }
        if style.text_style.reverse {
            codes.push("7".to_string());
        }
        if style.text_style.hidden {
            codes.push("8".to_string());
        }
        if style.text_style.strikethrough {
            codes.push("9".to_string());
        }

        // Foreground
        if let Some(fg) = style.fg {
            codes.push(color_to_fg_code(&fg));
        }

        // Background
        if let Some(bg) = style.bg {
            codes.push(color_to_bg_code(&bg));
        }
    } else {
        // Text style changes only
        if style.text_style.bold != prev.text_style.bold {
            codes.push(if style.text_style.bold { "1" } else { "22" }.to_string());
        }
        if style.text_style.dim != prev.text_style.dim {
            codes.push(if style.text_style.dim { "2" } else { "22" }.to_string());
        }
        if style.text_style.italic != prev.text_style.italic {
            codes.push(if style.text_style.italic { "3" } else { "23" }.to_string());
        }
        if style.text_style.underline != prev.text_style.underline {
            codes.push(if style.text_style.underline { "4" } else { "24" }.to_string());
        }
        if style.text_style.blink != prev.text_style.blink {
            codes.push(if style.text_style.blink { "5" } else { "25" }.to_string());
        }
        if style.text_style.reverse != prev.text_style.reverse {
            codes.push(if style.text_style.reverse { "7" } else { "27" }.to_string());
        }
        if style.text_style.hidden != prev.text_style.hidden {
            codes.push(if style.text_style.hidden { "8" } else { "28" }.to_string());
        }
        if style.text_style.strikethrough != prev.text_style.strikethrough {
            codes.push(if style.text_style.strikethrough { "9" } else { "29" }.to_string());
        }
    }

    if codes.is_empty() {
        String::new()
    } else {
        format!("\x1b[{}m", codes.join(";"))
    }
}

fn color_to_fg_code(color: &Color) -> String {
    match color {
        Color::Black => "30".to_string(),
        Color::Red => "31".to_string(),
        Color::Green => "32".to_string(),
        Color::Yellow => "33".to_string(),
        Color::Blue => "34".to_string(),
        Color::Magenta => "35".to_string(),
        Color::Cyan => "36".to_string(),
        Color::White => "37".to_string(),
        Color::DarkGrey => "90".to_string(),
        Color::LightRed => "91".to_string(),
        Color::LightGreen => "92".to_string(),
        Color::LightYellow => "93".to_string(),
        Color::LightBlue => "94".to_string(),
        Color::LightMagenta => "95".to_string(),
        Color::LightCyan => "96".to_string(),
        Color::LightGrey => "97".to_string(),
        Color::Rgb(r, g, b) => format!("38;2;{};{};{}", r, g, b),
    }
}

fn color_to_bg_code(color: &Color) -> String {
    match color {
        Color::Black => "40".to_string(),
        Color::Red => "41".to_string(),
        Color::Green => "42".to_string(),
        Color::Yellow => "43".to_string(),
        Color::Blue => "44".to_string(),
        Color::Magenta => "45".to_string(),
        Color::Cyan => "46".to_string(),
        Color::White => "47".to_string(),
        Color::DarkGrey => "100".to_string(),
        Color::LightRed => "101".to_string(),
        Color::LightGreen => "102".to_string(),
        Color::LightYellow => "103".to_string(),
        Color::LightBlue => "104".to_string(),
        Color::LightMagenta => "105".to_string(),
        Color::LightCyan => "106".to_string(),
        Color::LightGrey => "107".to_string(),
        Color::Rgb(r, g, b) => format!("48;2;{};{};{}", r, g, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_code_strings() {
        assert_eq!(AnsiCode::Reset.to_string(), "\x1b[0m");
        assert_eq!(AnsiCode::Bold.to_string(), "\x1b[1m");
        assert_eq!(AnsiCode::ClearScreen.to_string(), "\x1b[2J\x1b[H");
    }

    #[test]
    fn color_codes() {
        assert_eq!(ColorCode::Red.fg_code(), "31");
        assert_eq!(ColorCode::Rgb(255, 128, 0).fg_code(), "38;2;255;128;0");
    }

    #[test]
    fn ansi_buffer() {
        let mut buf = AnsiBuffer::new();
        buf.push(AnsiCode::Bold);
        buf.push_str("Hello");
        buf.push(AnsiCode::Reset);
        assert_eq!(buf.content(), "\x1b[1mHello\x1b[0m");
    }
}
}
