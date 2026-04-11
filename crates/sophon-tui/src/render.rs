//! Public rendering API
//!
//! This module provides the main entry points for rendering TUI applications.
//! Use `render_to_string` for testing/debugging or `Renderer` for full terminal
//! rendering with event handling.

use crate::element::Element;
use crate::layout::{Rect, Size};
use crate::style::Style;

/// Render an element to a string (useful for testing and debugging)
pub fn render_to_string(element: &Element, width: u16, height: u16) -> String {
    let area = Rect::new(0, 0, width, height);
    let mut buffer = RenderBuffer::new(width, height);
    render_element(element, &mut buffer, area, Style::default());
    buffer.to_string()
}

/// Render an element and return lines (for line-by-line processing)
pub fn render_to_lines(element: &Element, width: u16, height: u16) -> Vec<String> {
    let rendered = render_to_string(element, width, height);
    rendered.lines().map(|s| s.to_string()).collect()
}

/// Internal render buffer for building output
#[derive(Debug, Clone)]
pub struct RenderBuffer {
    /// Buffer cells
    pub cells: Vec<Cell>,
    /// Width
    pub width: u16,
    /// Height
    pub height: u16,
}

/// Single cell in the render buffer
#[derive(Debug, Clone)]
pub struct Cell {
    /// Character (Unicode)
    pub ch: char,
    /// Cell style
    pub style: Style,
    /// Whether this cell is part of a multi-cell character (e.g., emoji)
    pub skip: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            ch: ' ',
            style: Style::default(),
            skip: false,
        }
    }
}

impl RenderBuffer {
    /// Create a new render buffer
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        RenderBuffer {
            cells: vec![Cell::default(); size],
            width,
            height,
        }
    }

    /// Get cell index
    fn index(&self, x: u16, y: u16) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    /// Get a cell (returns None if out of bounds)
    pub fn get(&self, x: u16, y: u16) -> Option<&Cell> {
        if x < self.width && y < self.height {
            Some(&self.cells[self.index(x, y)])
        } else {
            None
        }
    }

    /// Get a mutable cell
    pub fn get_mut(&mut self, x: u16, y: u16) -> Option<&mut Cell> {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            Some(&mut self.cells[idx])
        } else {
            None
        }
    }

    /// Set a cell
    pub fn set(&mut self, x: u16, y: u16, ch: char, style: Style) {
        if let Some(cell) = self.get_mut(x, y) {
            cell.ch = ch;
            cell.style = style;
        }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    /// Resize the buffer
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        let size = (width as usize) * (height as usize);
        self.cells.resize(size, Cell::default());
    }

    /// Convert to string with ANSI escape sequences
    pub fn to_ansi_string(&self) -> String {
        use crate::ansi;

        let mut result = String::new();
        let mut current_style = Style::default();

        for y in 0..self.height {
            for x in 0..self.width {
                let cell = &self.cells[self.index(x, y)];

                // Output style changes
                if cell.style != current_style {
                    result.push_str(&ansi::style_to_ansi(&cell.style, &current_style));
                    current_style = cell.style;
                }

                result.push(cell.ch);
            }

            // Reset style at end of line
            if current_style != Style::default() {
                result.push_str(ansi::RESET);
                current_style = Style::default();
            }

            if y < self.height - 1 {
                result.push('\n');
            }
        }

        result
    }
}

impl ToString for RenderBuffer {
    fn to_string(&self) -> String {
        let mut result = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                result.push(self.cells[self.index(x, y)].ch);
            }
            if y < self.height - 1 {
                result.push('\n');
            }
        }
        result
    }
}

/// Render an element to a buffer (public for advanced use)
pub fn render_element(element: &Element, buffer: &mut RenderBuffer, area: Rect, style: Style) {
    match element {
        Element::Empty => {}
        Element::Text {
            content,
            style: text_style,
        } => {
            let combined_style = style.combine(*text_style);
            render_text(buffer, area, content, combined_style);
        }
        Element::Container {
            children,
            layout,
            style: container_style,
        } => {
            let combined_style = style.combine(*container_style);
            let child_areas = layout.calculate(area, children.len() as u16);
            for (child, child_area) in children.iter().zip(child_areas.iter()) {
                render_element(child, buffer, *child_area, combined_style);
            }
        }
        Element::Box { style: box_style } => {
            render_box(buffer, area, *box_style);
        }
    }
}

/// Render text to buffer
fn render_text(buffer: &mut RenderBuffer, area: Rect, content: &str, style: Style) {
    let mut x = area.x;
    let mut y = area.y;

    for ch in content.chars() {
        if ch == '\n' {
            x = area.x;
            y += 1;
            if y >= area.y + area.height {
                break;
            }
            continue;
        }

        if x < area.x + area.width && y < area.y + area.height {
            buffer.set(x, y, ch, style);
            x += 1;
        }
    }
}

/// Render a box border
fn render_box(buffer: &mut RenderBuffer, area: Rect, style: crate::style::BorderStyle) {
    use crate::style::BorderStyle;

    let (h, v, tl, tr, bl, br) = match style {
        BorderStyle::None => return,
        BorderStyle::Plain => ('─', '│', '┌', '┐', '└', '┘'),
        BorderStyle::Rounded => ('─', '│', '╭', '╮', '╰', '╯'),
        BorderStyle::Double => ('═', '║', '╔', '╗', '╚', '╝'),
        BorderStyle::Thick => ('━', '┃', '┏', '┓', '┗', '┛'),
    };

    // Top border
    if area.height > 0 {
        buffer.set(area.x, area.y, tl, Style::default());
        for x in 1..area.width - 1 {
            buffer.set(area.x + x, area.y, h, Style::default());
        }
        buffer.set(area.x + area.width - 1, area.y, tr, Style::default());
    }

    // Side borders
    for y in 1..area.height - 1 {
        buffer.set(area.x, area.y + y, v, Style::default());
        buffer.set(area.x + area.width - 1, area.y + y, v, Style::default());
    }

    // Bottom border
    if area.height > 1 {
        buffer.set(area.x, area.y + area.height - 1, bl, Style::default());
        for x in 1..area.width - 1 {
            buffer.set(area.x + x, area.y + area.height - 1, h, Style::default());
        }
        buffer.set(
            area.x + area.width - 1,
            area.y + area.height - 1,
            br,
            Style::default(),
        );
    }
}

/// Measure text size
pub fn measure_text(text: &str) -> Size {
    let mut width = 0u16;
    let mut height = 1u16;
    let mut current_width = 0u16;

    for ch in text.chars() {
        if ch == '\n' {
            height += 1;
            width = width.max(current_width);
            current_width = 0;
        } else {
            current_width += 1;
        }
    }

    width = width.max(current_width);
    Size { width, height }
}

/// Clamp a value between min and max
fn clamp<T: Ord>(value: T, min: T, max: T) -> T {
    value.max(min).min(max)
}
