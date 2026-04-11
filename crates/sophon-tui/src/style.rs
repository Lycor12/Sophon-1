//! Styling for TUI elements

/// ANSI colors supported by the terminal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
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
    /// RGB color (r, g, b)
    Rgb(u8, u8, u8),
    /// ANSI 256 color index
    Ansi256(u8),
}

impl Color {
    /// Convert to ANSI escape code string
    pub fn to_ansi_fg(&self) -> String {
        match self {
            Color::Black => "\x1b[30m".to_string(),
            Color::Red => "\x1b[31m".to_string(),
            Color::Green => "\x1b[32m".to_string(),
            Color::Yellow => "\x1b[33m".to_string(),
            Color::Blue => "\x1b[34m".to_string(),
            Color::Magenta => "\x1b[35m".to_string(),
            Color::Cyan => "\x1b[36m".to_string(),
            Color::White => "\x1b[37m".to_string(),
            Color::BrightBlack => "\x1b[90m".to_string(),
            Color::BrightRed => "\x1b[91m".to_string(),
            Color::BrightGreen => "\x1b[92m".to_string(),
            Color::BrightYellow => "\x1b[93m".to_string(),
            Color::BrightBlue => "\x1b[94m".to_string(),
            Color::BrightMagenta => "\x1b[95m".to_string(),
            Color::BrightCyan => "\x1b[96m".to_string(),
            Color::BrightWhite => "\x1b[97m".to_string(),
            Color::Rgb(r, g, b) => format!("\x1b[38;2;{};{};{}m", r, g, b),
            Color::Ansi256(n) => format!("\x1b[38;5;{}m", n),
        }
    }

    pub fn to_ansi_bg(&self) -> String {
        match self {
            Color::Black => "\x1b[40m".to_string(),
            Color::Red => "\x1b[41m".to_string(),
            Color::Green => "\x1b[42m".to_string(),
            Color::Yellow => "\x1b[43m".to_string(),
            Color::Blue => "\x1b[44m".to_string(),
            Color::Magenta => "\x1b[45m".to_string(),
            Color::Cyan => "\x1b[46m".to_string(),
            Color::White => "\x1b[47m".to_string(),
            Color::BrightBlack => "\x1b[100m".to_string(),
            Color::BrightRed => "\x1b[101m".to_string(),
            Color::BrightGreen => "\x1b[102m".to_string(),
            Color::BrightYellow => "\x1b[103m".to_string(),
            Color::BrightBlue => "\x1b[104m".to_string(),
            Color::BrightMagenta => "\x1b[105m".to_string(),
            Color::BrightCyan => "\x1b[106m".to_string(),
            Color::BrightWhite => "\x1b[107m".to_string(),
            Color::Rgb(r, g, b) => format!("\x1b[48;2;{};{};{}m", r, g, b),
            Color::Ansi256(n) => format!("\x1b[48;5;{}m", n),
        }
    }
}

/// Text wrapping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextWrap {
    /// No wrapping
    #[default]
    None,
    /// Wrap at word boundaries
    Word,
    /// Wrap at character boundaries
    Char,
    /// Truncate with ellipsis
    Truncate,
}

/// Border style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderStyle {
    /// No border
    #[default]
    None,
    /// Single line border: ┌─┐│└┘
    Single,
    /// Double line border: ╔═╗║╚╝
    Double,
    /// Thick border
    Thick,
    /// Dashed border
    Dashed,
    /// Dotted border
    Dotted,
    /// Rounded corners: ╭─╮│╰─╯
    Rounded,
}

impl BorderStyle {
    /// Get border characters: [top-left, top, top-right, right, bottom-right, bottom, bottom-left, left]
    pub fn chars(&self) -> [&'static str; 8] {
        match self {
            BorderStyle::None => [" ", " ", " ", " ", " ", " ", " ", " "],
            BorderStyle::Single => ["┌", "─", "┐", "│", "┘", "─", "└", "│"],
            BorderStyle::Double => ["╔", "═", "╗", "║", "╝", "═", "╚", "║"],
            BorderStyle::Thick => ["┏", "━", "┓", "┃", "┛", "━", "┗", "┃"],
            BorderStyle::Dashed => ["┌", "╌", "┐", "┊", "┘", "╌", "└", "┊"],
            BorderStyle::Dotted => ["┌", "┈", "┐", "┆", "┘", "┈", "└", "┆"],
            BorderStyle::Rounded => ["╭", "─", "╮", "│", "╯", "─", "╰", "│"],
        }
    }
}

/// Style properties for an element
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Style {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
    pub blink: bool,
    pub reverse: bool,
    pub hidden: bool,
    pub strikethrough: bool,
}

impl Style {
    /// Reset all attributes
    pub fn reset() -> &'static str {
        "\x1b[0m"
    }

    /// Apply style to produce ANSI escape sequence prefix
    pub fn apply(&self) -> String {
        let mut codes = Vec::new();

        if self.bold {
            codes.push("1");
        }
        if self.dim {
            codes.push("2");
        }
        if self.italic {
            codes.push("3");
        }
        if self.underline {
            codes.push("4");
        }
        if self.blink {
            codes.push("5");
        }
        if self.reverse {
            codes.push("7");
        }
        if self.hidden {
            codes.push("8");
        }
        if self.strikethrough {
            codes.push("9");
        }

        let mut result = String::new();

        if !codes.is_empty() {
            result.push_str(&format!("\x1b[{}m", codes.join(";")));
        }

        if let Some(fg) = self.fg {
            result.push_str(&fg.to_ansi_fg());
        }

        if let Some(bg) = self.bg {
            result.push_str(&bg.to_ansi_bg());
        }

        result
    }

    /// Merge with another style (other takes precedence for non-None values)
    pub fn merge(&self, other: &Style) -> Style {
        Style {
            fg: other.fg.or(self.fg),
            bg: other.bg.or(self.bg),
            bold: other.bold || self.bold,
            dim: other.dim || self.dim,
            italic: other.italic || self.italic,
            underline: other.underline || self.underline,
            blink: other.blink || self.blink,
            reverse: other.reverse || self.reverse,
            hidden: other.hidden || self.hidden,
            strikethrough: other.strikethrough || self.strikethrough,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_ansi_codes() {
        assert_eq!(Color::Red.to_ansi_fg(), "\x1b[31m");
        assert_eq!(Color::Rgb(255, 128, 0).to_ansi_fg(), "\x1b[38;2;255;128;0m");
    }

    #[test]
    fn border_chars() {
        let single = BorderStyle::Single.chars();
        assert_eq!(single[0], "┌");
        assert_eq!(single[1], "─");
    }

    #[test]
    fn style_apply() {
        let style = Style {
            fg: Some(Color::Red),
            bold: true,
            ..Default::default()
        };
        let result = style.apply();
        assert!(result.contains("\x1b[1m"));
        assert!(result.contains("\x1b[31m"));
    }
}
