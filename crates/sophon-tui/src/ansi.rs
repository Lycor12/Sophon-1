//! ANSI escape sequence handling

use crate::style::Color;

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

impl From<Color> for ColorCode {
    fn from(color: Color) -> Self {
        match color {
            Color::Black => ColorCode::Black,
            Color::Red => ColorCode::Red,
            Color::Green => ColorCode::Green,
            Color::Yellow => ColorCode::Yellow,
            Color::Blue => ColorCode::Blue,
            Color::Magenta => ColorCode::Magenta,
            Color::Cyan => ColorCode::Cyan,
            Color::White => ColorCode::White,
            Color::DarkGrey => ColorCode::BrightBlack,
            Color::LightRed => ColorCode::BrightRed,
            Color::LightGreen => ColorCode::BrightGreen,
            Color::LightYellow => ColorCode::BrightYellow,
            Color::LightBlue => ColorCode::BrightBlue,
            Color::LightMagenta => ColorCode::BrightMagenta,
            Color::LightCyan => ColorCode::BrightCyan,
            Color::LightGrey => ColorCode::BrightWhite,
            Color::Rgb(r, g, b) => ColorCode::Rgb(r, g, b),
            Color::Ansi256(n) => ColorCode::Ansi256(n),
        }
    }
}

/// ANSI reset code
pub const RESET: &str = "\x1b[0m";

/// Buffer for building ANSI strings
#[derive(Debug, Clone, Default)]
pub struct AnsiBuffer {
    codes: Vec<String>,
}

impl AnsiBuffer {
    /// Create a new empty buffer
    pub fn new() -> Self {
        AnsiBuffer { codes: Vec::new() }
    }

    /// Add a code
    pub fn push(&mut self, code: impl Into<String>) {
        self.codes.push(code.into());
    }

    /// Add foreground color
    pub fn fg(&mut self, color: ColorCode) {
        self.codes.push(color.fg_code());
    }

    /// Add background color
    pub fn bg(&mut self, color: ColorCode) {
        self.codes.push(color.bg_code());
    }

    /// Add bold
    pub fn bold(&mut self) {
        self.codes.push("1".to_string());
    }

    /// Add dim
    pub fn dim(&mut self) {
        self.codes.push("2".to_string());
    }

    /// Add italic
    pub fn italic(&mut self) {
        self.codes.push("3".to_string());
    }

    /// Add underline
    pub fn underline(&mut self) {
        self.codes.push("4".to_string());
    }

    /// Add blink
    pub fn blink(&mut self) {
        self.codes.push("5".to_string());
    }

    /// Add reverse
    pub fn reverse(&mut self) {
        self.codes.push("7".to_string());
    }

    /// Add hidden
    pub fn hidden(&mut self) {
        self.codes.push("8".to_string());
    }

    /// Add strikethrough
    pub fn strikethrough(&mut self) {
        self.codes.push("9".to_string());
    }

    /// Build the ANSI escape sequence
    pub fn build(&self) -> String {
        if self.codes.is_empty() {
            String::new()
        } else {
            format!("\x1b[{}m", self.codes.join(";"))
        }
    }

    /// Clear all codes
    pub fn clear(&mut self) {
        self.codes.clear();
    }
}

/// Cursor control
pub struct Cursor;

impl Cursor {
    /// Move cursor up
    pub fn up(n: u16) -> String {
        format!("\x1b[{}A", n)
    }

    /// Move cursor down
    pub fn down(n: u16) -> String {
        format!("\x1b[{}B", n)
    }

    /// Move cursor forward
    pub fn forward(n: u16) -> String {
        format!("\x1b[{}C", n)
    }

    /// Move cursor back
    pub fn back(n: u16) -> String {
        format!("\x1b[{}D", n)
    }

    /// Move cursor to position (1-indexed)
    pub fn goto(row: u16, col: u16) -> String {
        format!("\x1b[{};{}H", row, col)
    }

    /// Save cursor position
    pub fn save() -> &'static str {
        "\x1b7"
    }

    /// Restore cursor position
    pub fn restore() -> &'static str {
        "\x1b8"
    }

    /// Hide cursor
    pub fn hide() -> &'static str {
        "\x1b[?25l"
    }

    /// Show cursor
    pub fn show() -> &'static str {
        "\x1b[?25h"
    }
}

/// Screen control
pub struct Screen;

impl Screen {
    /// Clear entire screen
    pub fn clear() -> &'static str {
        "\x1b[2J"
    }

    /// Clear from cursor to end of screen
    pub fn clear_from_cursor() -> &'static str {
        "\x1b[0J"
    }

    /// Clear from cursor to beginning of screen
    pub fn clear_to_cursor() -> &'static str {
        "\x1b[1J"
    }

    /// Clear current line
    pub fn clear_line() -> &'static str {
        "\x1b[2K"
    }

    /// Clear from cursor to end of line
    pub fn clear_line_from_cursor() -> &'static str {
        "\x1b[0K"
    }

    /// Clear from cursor to beginning of line
    pub fn clear_line_to_cursor() -> &'static str {
        "\x1b[1K"
    }

    /// Set scroll region
    pub fn set_scroll_region(top: u16, bottom: u16) -> String {
        format!("\x1b[{};{}r", top, bottom)
    }

    /// Reset scroll region
    pub fn reset_scroll_region() -> &'static str {
        "\x1b[r"
    }

    /// Enable mouse support
    pub fn enable_mouse() -> &'static str {
        "\x1b[?1000h\x1b[?1002h\x1b[?1015h\x1b[?1006h"
    }

    /// Disable mouse support
    pub fn disable_mouse() -> &'static str {
        "\x1b[?1006l\x1b[?1015l\x1b[?1002l\x1b[?1000l"
    }

    /// Enable alternate screen buffer
    pub fn enter_alt_screen() -> &'static str {
        "\x1b[?1049h"
    }

    /// Disable alternate screen buffer
    pub fn exit_alt_screen() -> &'static str {
        "\x1b[?1049l"
    }
}

/// Convert a Style to ANSI escape sequences
pub fn style_to_ansi(style: &crate::style::Style, prev: &crate::style::Style) -> String {
    let mut codes = Vec::new();

    // Check if we need a full reset
    if style.fg != prev.fg || style.bg != prev.bg {
        codes.push("0".to_string());

        // Re-apply all text styles after reset
        if style.bold {
            codes.push("1".to_string());
        }
        if style.dim {
            codes.push("2".to_string());
        }
        if style.italic {
            codes.push("3".to_string());
        }
        if style.underline {
            codes.push("4".to_string());
        }
        if style.blink {
            codes.push("5".to_string());
        }
        if style.reverse {
            codes.push("7".to_string());
        }
        if style.hidden {
            codes.push("8".to_string());
        }
        if style.strikethrough {
            codes.push("9".to_string());
        }

        // Colors
        if let Some(fg) = style.fg {
            codes.push(color_to_fg_code(&fg));
        }
        if let Some(bg) = style.bg {
            codes.push(color_to_bg_code(&bg));
        }
    } else {
        // Just text style changes
        if style.bold != prev.bold {
            codes.push(if style.bold { "1" } else { "22" }.to_string());
        }
        if style.dim != prev.dim {
            codes.push(if style.dim { "2" } else { "22" }.to_string());
        }
        if style.italic != prev.italic {
            codes.push(if style.italic { "3" } else { "23" }.to_string());
        }
        if style.underline != prev.underline {
            codes.push(if style.underline { "4" } else { "24" }.to_string());
        }
        if style.blink != prev.blink {
            codes.push(if style.blink { "5" } else { "25" }.to_string());
        }
        if style.reverse != prev.reverse {
            codes.push(if style.reverse { "7" } else { "27" }.to_string());
        }
        if style.hidden != prev.hidden {
            codes.push(if style.hidden { "8" } else { "28" }.to_string());
        }
        if style.strikethrough != prev.strikethrough {
            codes.push(if style.strikethrough { "9" } else { "29" }.to_string());
        }

        // Color changes
        if style.fg != prev.fg {
            if let Some(fg) = style.fg {
                codes.push(color_to_fg_code(&fg));
            }
        }
        if style.bg != prev.bg {
            if let Some(bg) = style.bg {
                codes.push(color_to_bg_code(&bg));
            }
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
        Color::Ansi256(n) => format!("38;5;{}", n),
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
        Color::Ansi256(n) => format!("48;5;{}", n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_codes() {
        assert_eq!(RESET, "\x1b[0m");
        assert_eq!(Cursor::up(5), "\x1b[5A");
        assert_eq!(Cursor::goto(10, 20), "\x1b[10;20H");
    }

    #[test]
    fn color_codes() {
        let red = ColorCode::Red;
        assert_eq!(red.fg_code(), "31");
        assert_eq!(red.bg_code(), "41");
    }

    #[test]
    fn ansi_buffer() {
        let mut buf = AnsiBuffer::new();
        buf.bold();
        buf.fg(ColorCode::Red);
        assert_eq!(buf.build(), "\x1b[1;31m");
    }
}
