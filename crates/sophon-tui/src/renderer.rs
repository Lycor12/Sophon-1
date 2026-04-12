//! Internal renderer implementation
//!
//! The renderer orchestrates the rendering pipeline:
//! 1. Calculate layouts
//! 2. Render elements to buffer
//! 3. Output to terminal

use crate::element::Element;
use crate::layout::Rect;
use crate::render::RenderBuffer;
use crate::terminal::Terminal;

/// Rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Full terminal output
    Terminal,
    /// Output to string (for testing)
    String,
    /// Debug output
    Debug,
}

/// Renderer state
#[derive(Debug)]
pub struct Renderer {
    /// Terminal handle
    terminal: Option<Terminal>,
    /// Render mode
    mode: RenderMode,
    /// Current buffer
    buffer: RenderBuffer,
    /// Previous buffer (for diffing)
    prev_buffer: Option<RenderBuffer>,
    /// Whether to use double buffering
    double_buffer: bool,
}

impl Renderer {
    /// Create a new terminal renderer
    pub fn new() -> std::io::Result<Self> {
        let terminal = Terminal::new()?;
        let (width, height) = terminal.size()?;

        Ok(Renderer {
            terminal: Some(terminal),
            mode: RenderMode::Terminal,
            buffer: RenderBuffer::new(width, height),
            prev_buffer: None,
            double_buffer: true,
        })
    }

    /// Create a string renderer (for testing)
    pub fn new_string(width: u16, height: u16) -> Self {
        Renderer {
            terminal: None,
            mode: RenderMode::String,
            buffer: RenderBuffer::new(width, height),
            prev_buffer: None,
            double_buffer: false,
        }
    }

    /// Create a debug renderer
    pub fn new_debug(width: u16, height: u16) -> Self {
        Renderer {
            terminal: None,
            mode: RenderMode::Debug,
            buffer: RenderBuffer::new(width, height),
            prev_buffer: None,
            double_buffer: false,
        }
    }

    /// Initialize the terminal
    pub fn init(&mut self) -> std::io::Result<()> {
        if let Some(ref mut term) = self.terminal {
            term.enable_raw_mode()?;
            term.enter_alternate_screen()?;
            term.hide_cursor()?;
            term.clear()?;
        }
        Ok(())
    }

    /// Cleanup terminal
    pub fn cleanup(&mut self) -> std::io::Result<()> {
        if let Some(ref mut term) = self.terminal {
            term.show_cursor()?;
            term.leave_alternate_screen()?;
            term.disable_raw_mode()?;
        }
        Ok(())
    }

    /// Get current size
    pub fn get_size(&self) -> (u16, u16) {
        (self.buffer.width, self.buffer.height)
    }

    /// Render an element
    pub fn render(&mut self, element: &Element) -> std::io::Result<()> {
        // Clear buffer
        self.buffer.clear();

        // Calculate layout and render
        let area = Rect::new(0, 0, self.buffer.width, self.buffer.height);
        self.render_element(element, area);

        // Output
        match self.mode {
            RenderMode::Terminal => self.output_terminal(),
            RenderMode::String => Ok(()), // Buffer is the output
            RenderMode::Debug => {
                println!("{:#?}", element);
                Ok(())
            }
        }
    }

    /// Get the rendered output as string
    pub fn to_string(&self) -> String {
        self.buffer.to_string()
    }

    /// Get the rendered output as ANSI string
    pub fn to_ansi_string(&self) -> String {
        self.buffer.to_ansi_string()
    }

    /// Update terminal size
    pub fn update_size(&mut self) -> std::io::Result<()> {
        if let Some(ref term) = self.terminal {
            let (width, height) = term.size()?;
            self.buffer.resize(width, height);
            if let Some(ref mut prev) = self.prev_buffer {
                prev.resize(width, height);
            }
        }
        Ok(())
    }

    /// Get current terminal size
    pub fn size(&self) -> (u16, u16) {
        (self.buffer.width, self.buffer.height)
    }

    /// Set double buffering
    pub fn set_double_buffer(&mut self, enabled: bool) {
        self.double_buffer = enabled;
        if enabled {
            self.prev_buffer = Some(RenderBuffer::new(self.buffer.width, self.buffer.height));
        } else {
            self.prev_buffer = None;
        }
    }

    /// Internal render method
    fn render_element(&mut self, element: &Element, area: Rect) {
        use crate::render;
        use crate::style::Style;
        render::render_element(element, &mut self.buffer, area, Style::default());
    }

    /// Output buffer to terminal
    fn output_terminal(&mut self) -> std::io::Result<()> {
        if self.terminal.is_none() {
            return Ok(());
        }

        let output = if self.double_buffer {
            self.diff_output()
        } else {
            self.full_output()
        };

        // Save previous buffer for next diff
        if self.double_buffer {
            if let Some(ref mut prev) = self.prev_buffer {
                prev.cells.clone_from(&self.buffer.cells);
            }
        }

        // Write to terminal
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        stdout.write_all(output.as_bytes())?;
        stdout.flush()
    }

    /// Generate full output
    fn full_output(&self) -> String {
        // Move cursor to top-left and clear screen
        let mut output = String::new();
        output.push_str("\x1b[H"); // Move to home position

        for y in 0..self.buffer.height {
            for x in 0..self.buffer.width {
                let cell =
                    &self.buffer.cells[(y as usize) * (self.buffer.width as usize) + (x as usize)];
                output.push(cell.ch);
            }
            if y < self.buffer.height - 1 {
                output.push('\n');
            }
        }

        output
    }

    /// Generate diff output (only changed cells)
    fn diff_output(&self) -> String {
        let prev = match self.prev_buffer {
            Some(ref p) => p,
            None => return self.full_output(),
        };

        let mut output = String::new();
        let mut prev_style = crate::style::Style::default();

        for y in 0..self.buffer.height {
            let mut line_changed = false;
            let mut first_change_x: Option<u16> = None;

            // Check if this line changed
            for x in 0..self.buffer.width {
                let idx = (y as usize) * (self.buffer.width as usize) + (x as usize);
                let cell = &self.buffer.cells[idx];
                let prev_cell = &prev.cells[idx];

                if cell.ch != prev_cell.ch || cell.style != prev_cell.style {
                    line_changed = true;
                    if first_change_x.is_none() {
                        first_change_x = Some(x);
                    }
                }
            }

            if !line_changed {
                continue;
            }

            // Move cursor to first changed position
            if let Some(fx) = first_change_x {
                output.push_str(&format!("\x1b[{};{}H", y + 1, fx + 1));
            }

            // Output changed cells
            for x in 0..self.buffer.width {
                let idx = (y as usize) * (self.buffer.width as usize) + (x as usize);
                let cell = &self.buffer.cells[idx];

                if cell.style != prev_style {
                    output.push_str(&crate::ansi::style_to_ansi(&cell.style, &prev_style));
                    prev_style = cell.style;
                }
                output.push(cell.ch);
            }
        }

        // Reset style at end
        output.push_str(crate::ansi::RESET);

        output
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Renderer::new_string(80, 24)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}
