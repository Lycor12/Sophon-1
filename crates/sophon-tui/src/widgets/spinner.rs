//! Spinner widget for loading states

use crate::element::Element;
use crate::layout::{Rect, Size};
use crate::style::{Color, Style};
use crate::widgets::Widget;

/// Spinner animation frames
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinnerType {
    /// Simple dots
    Dots,
    /// Rotating line
    Line,
    /// Rotating circle
    Circle,
    /// Square brackets
    Square,
    /// Growing/shrinking circle
    Grow,
}

impl SpinnerType {
    /// Get the frames for this spinner type
    pub fn frames(&self) -> &'static [&'static str] {
        match self {
            SpinnerType::Dots => &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            SpinnerType::Line => &["-", "\\", "|", "/"],
            SpinnerType::Circle => &["◐", "◓", "◑", "◒"],
            SpinnerType::Square => &[
                "[    ]", "[=   ]", "[==  ]", "[=== ]", "[====]", "[ ===]", "[  ==]", "[   =]",
            ],
            SpinnerType::Grow => &["◯", "◉"],
        }
    }
}

/// Spinner widget configuration
#[derive(Debug, Clone)]
pub struct Spinner {
    /// Spinner type
    spinner_type: SpinnerType,
    /// Current frame index
    frame: usize,
    /// Text to display next to spinner
    text: Option<String>,
    /// Spinner style
    style: Style,
    /// Text style
    text_style: Style,
}

impl Spinner {
    /// Create a new spinner
    pub fn new() -> Self {
        Spinner {
            spinner_type: SpinnerType::Dots,
            frame: 0,
            text: None,
            style: Style::default().fg(Color::Cyan),
            text_style: Style::default(),
        }
    }

    /// Create with specific type
    pub fn with_type(spinner_type: SpinnerType) -> Self {
        Spinner {
            spinner_type,
            frame: 0,
            text: None,
            style: Style::default().fg(Color::Cyan),
            text_style: Style::default(),
        }
    }

    /// Set spinner type
    pub fn spinner_type(mut self, t: SpinnerType) -> Self {
        self.spinner_type = t;
        self
    }

    /// Set text
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set style
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Set text style
    pub fn text_style(mut self, style: Style) -> Self {
        self.text_style = style;
        self
    }

    /// Advance to next frame
    pub fn next(&mut self) {
        let frames = self.spinner_type.frames().len();
        self.frame = (self.frame + 1) % frames;
    }

    /// Get current frame content
    pub fn current_frame(&self) -> &'static str {
        self.spinner_type.frames()[self.frame]
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Spinner {
    fn render(&self, _area: Rect) -> Element {
        let frame = self.current_frame();

        let spinner_el = Element::Text {
            content: frame.to_string(),
            style: self.style,
        };

        if let Some(ref text) = self.text {
            Element::Container {
                children: vec![
                    spinner_el,
                    Element::Text {
                        content: format!(" {}", text),
                        style: self.text_style,
                    },
                ],
                layout: crate::layout::Layout::horizontal(vec![
                    crate::layout::Constraint::Length(2),
                    crate::layout::Constraint::Fill,
                ]),
                style: Style::default(),
            }
        } else {
            spinner_el
        }
    }

    fn min_size(&self) -> Size {
        let text_width = self.text.as_ref().map(|t| t.len() + 1).unwrap_or(0);
        Size {
            width: 2 + text_width as u16,
            height: 1,
        }
    }
}

/// Stateful spinner that tracks animation state
#[derive(Debug, Clone)]
pub struct StatefulSpinner {
    config: Spinner,
}

impl StatefulSpinner {
    /// Create new stateful spinner
    pub fn new(config: Spinner) -> Self {
        StatefulSpinner { config }
    }

    /// Advance animation
    pub fn tick(&mut self) {
        self.config.next();
    }

    /// Get current frame
    pub fn frame(&self) -> usize {
        self.config.frame
    }

    /// Set text
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.config.text = Some(text.into());
    }
}

impl Widget for StatefulSpinner {
    fn render(&self, area: Rect) -> Element {
        self.config.render(area)
    }
}

/// Multi-frame spinner for different states
#[derive(Debug, Clone)]
pub enum SpinnerState {
    /// Still loading
    Loading(Spinner),
    /// Success
    Success(String),
    /// Failed
    Failed(String),
    /// Warning
    Warning(String),
}

impl SpinnerState {
    /// Create loading state
    pub fn loading() -> Self {
        SpinnerState::Loading(Spinner::new())
    }

    /// Create success state
    pub fn success(text: impl Into<String>) -> Self {
        SpinnerState::Success(text.into())
    }

    /// Create failed state
    pub fn failed(text: impl Into<String>) -> Self {
        SpinnerState::Failed(text.into())
    }

    /// Advance if loading
    pub fn tick(&mut self) {
        if let SpinnerState::Loading(ref mut s) = self {
            s.next();
        }
    }
}

impl Widget for SpinnerState {
    fn render(&self, area: Rect) -> Element {
        match self {
            SpinnerState::Loading(s) => s.render(area),
            SpinnerState::Success(t) => Element::Text {
                content: format!("✓ {}", t),
                style: Style::default().fg(Color::Green),
            },
            SpinnerState::Failed(t) => Element::Text {
                content: format!("✗ {}", t),
                style: Style::default().fg(Color::Red),
            },
            SpinnerState::Warning(t) => Element::Text {
                content: format!("⚠ {}", t),
                style: Style::default().fg(Color::Yellow),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_frames() {
        let mut spinner = Spinner::new();
        let frame0 = spinner.current_frame();
        spinner.next();
        let frame1 = spinner.current_frame();
        assert_ne!(frame0, frame1);
    }

    #[test]
    fn spinner_types() {
        assert_eq!(SpinnerType::Dots.frames().len(), 10);
        assert_eq!(SpinnerType::Line.frames().len(), 4);
        assert_eq!(SpinnerType::Circle.frames().len(), 4);
    }

    #[test]
    fn spinner_state_transitions() {
        let mut state = SpinnerState::loading();
        state.tick();
        state.tick();
        // Should still be loading
        assert!(matches!(state, SpinnerState::Loading(_)));
    }
}
