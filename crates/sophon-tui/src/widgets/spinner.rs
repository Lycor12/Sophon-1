//! Spinner widget for loading states

use crate::element::{Element, ElementKind};
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
                "[ ]", "[= ]", "[== ]", "[=== ]", "[====]", "[ ===]", "[ ==]", "[ =]",
            ],
            SpinnerType::Grow => &["◯", "◉"],
        }
    }

    /// Get the frame count
    pub fn frame_count(&self) -> usize {
        self.frames().len()
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
    pub fn tick(&mut self) {
        self.frame = (self.frame + 1) % self.spinner_type.frame_count();
    }

    /// Set frame directly
    pub fn set_frame(&mut self, frame: usize) {
        self.frame = frame % self.spinner_type.frame_count();
    }

    /// Get current frame as string
    pub fn current_frame(&self) -> &'static str {
        self.spinner_type.frames()[self.frame]
    }

    /// Get current frame index
    pub fn frame_index(&self) -> usize {
        self.frame
    }

    /// Create dots spinner
    pub fn dots() -> Self {
        Self::new()
    }

    /// Create line spinner
    pub fn line() -> Self {
        Self::with_type(SpinnerType::Line)
    }

    /// Create circle spinner
    pub fn circle() -> Self {
        Self::with_type(SpinnerType::Circle)
    }

    /// Create square spinner
    pub fn square() -> Self {
        Self::with_type(SpinnerType::Square)
    }

    /// Create grow spinner
    pub fn grow() -> Self {
        Self::with_type(SpinnerType::Grow)
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Spinner {
    fn render(&self, _area: Rect) -> Element {
        let spinner_text = self.current_frame().to_string();
        let spinner_el = Element {
            id: None,
            kind: ElementKind::Text(spinner_text),
            style: self.style,
            children: vec![],
            layout: None,
        };

        if let Some(text) = &self.text {
            Element {
                id: None,
                kind: ElementKind::Row,
                style: Style::default(),
                children: vec![
                    spinner_el,
                    Element {
                        id: None,
                        kind: ElementKind::Text(format!(" {}", text)),
                        style: self.text_style,
                        children: vec![],
                        layout: None,
                    },
                ],
                layout: None,
            }
        } else {
            spinner_el
        }
    }

    fn min_size(&self) -> Size {
        let spinner_width = self.current_frame().chars().count();
        let text_width = self.text.as_ref().map(|t| t.len() + 1).unwrap_or(0);
        Size {
            width: spinner_width + text_width,
            height: 1,
        }
    }
}

/// Spinner with state for async operations
#[derive(Debug, Clone)]
pub enum SpinnerState {
    /// Still loading
    Loading(String),
    /// Completed successfully
    Success(String),
    /// Failed with error
    Failed(String),
    /// Completed with warning
    Warning(String),
}

impl SpinnerState {
    /// Check if loading
    pub fn is_loading(&self) -> bool {
        matches!(self, SpinnerState::Loading(_))
    }

    /// Check if success
    pub fn is_success(&self) -> bool {
        matches!(self, SpinnerState::Success(_))
    }

    /// Check if failed
    pub fn is_failed(&self) -> bool {
        matches!(self, SpinnerState::Failed(_))
    }

    /// Check if warning
    pub fn is_warning(&self) -> bool {
        matches!(self, SpinnerState::Warning(_))
    }

    /// Get message
    pub fn message(&self) -> &str {
        match self {
            SpinnerState::Loading(m) => m,
            SpinnerState::Success(m) => m,
            SpinnerState::Failed(m) => m,
            SpinnerState::Warning(m) => m,
        }
    }

    /// Set to success
    pub fn succeed(&mut self, message: impl Into<String>) {
        *self = SpinnerState::Success(message.into());
    }

    /// Set to failed
    pub fn fail(&mut self, message: impl Into<String>) {
        *self = SpinnerState::Failed(message.into());
    }

    /// Set to warning
    pub fn warn(&mut self, message: impl Into<String>) {
        *self = SpinnerState::Warning(message.into());
    }
}

impl Default for SpinnerState {
    fn default() -> Self {
        SpinnerState::Loading(String::new())
    }
}

/// Stateful spinner that shows different states
#[derive(Debug, Clone)]
pub struct StatusSpinner {
    spinner: Spinner,
    state: SpinnerState,
}

impl StatusSpinner {
    /// Create new status spinner
    pub fn new(message: impl Into<String>) -> Self {
        StatusSpinner {
            spinner: Spinner::new(),
            state: SpinnerState::Loading(message.into()),
        }
    }

    /// Set state
    pub fn set_state(&mut self, state: SpinnerState) {
        self.state = state;
    }

    /// Get state
    pub fn state(&self) -> &SpinnerState {
        &self.state
    }

    /// Tick the spinner (if loading)
    pub fn tick(&mut self) {
        if self.state.is_loading() {
            self.spinner.tick();
        }
    }
}

impl Widget for StatusSpinner {
    fn render(&self, _area: Rect) -> Element {
        let (icon, message, style) = match &self.state {
            SpinnerState::Loading(msg) => {
                let spinner_text = self.spinner.current_frame().to_string();
                return Element::row(vec![
                    Element {
                        id: None,
                        kind: ElementKind::Text(spinner_text),
                        style: self.spinner.style,
                        children: vec![],
                        layout: None,
                    },
                    Element {
                        id: None,
                        kind: ElementKind::Text(format!(" {}", msg)),
                        style: self.spinner.text_style,
                        children: vec![],
                        layout: None,
                    },
                ]);
            }
            SpinnerState::Success(msg) => ("✓", msg, Style::default().fg(Color::Green)),
            SpinnerState::Failed(msg) => ("✗", msg, Style::default().fg(Color::Red)),
            SpinnerState::Warning(msg) => ("⚠", msg, Style::default().fg(Color::Yellow)),
        };

        Element {
            id: None,
            kind: ElementKind::Row,
            style: Style::default(),
            children: vec![
                Element {
                    id: None,
                    kind: ElementKind::Text(icon.to_string()),
                    style,
                    children: vec![],
                    layout: None,
                },
                Element {
                    id: None,
                    kind: ElementKind::Text(format!(" {}", message)),
                    style,
                    children: vec![],
                    layout: None,
                },
            ],
            layout: None,
        }
    }

    fn min_size(&self) -> Size {
        let msg_len = self.state.message().len();
        Size {
            width: msg_len + 2,
            height: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_creation() {
        let spinner = Spinner::new();
        assert_eq!(spinner.frame, 0);
        assert!(matches!(spinner.spinner_type, SpinnerType::Dots));
    }

    #[test]
    fn spinner_with_type() {
        let spinner = Spinner::with_type(SpinnerType::Line);
        assert!(matches!(spinner.spinner_type, SpinnerType::Line));
    }

    #[test]
    fn spinner_tick() {
        let mut spinner = Spinner::new();
        assert_eq!(spinner.frame, 0);

        spinner.tick();
        assert_eq!(spinner.frame, 1);

        // Wrap around
        for _ in 0..spinner.spinner_type.frame_count() {
            spinner.tick();
        }
        assert_eq!(spinner.frame, 0);
    }

    #[test]
    fn spinner_set_frame() {
        let mut spinner = Spinner::new();
        spinner.set_frame(3);
        assert_eq!(spinner.frame, 3);

        // Should wrap
        spinner.set_frame(100);
        assert_eq!(spinner.frame, 100 % spinner.spinner_type.frame_count());
    }

    #[test]
    fn spinner_current_frame() {
        let spinner = Spinner::new();
        let frame = spinner.current_frame();
        assert!(!frame.is_empty());
    }

    #[test]
    fn spinner_types() {
        assert_eq!(SpinnerType::Dots.frame_count(), 10);
        assert_eq!(SpinnerType::Line.frame_count(), 4);
        assert_eq!(SpinnerType::Circle.frame_count(), 4);
        assert_eq!(SpinnerType::Square.frame_count(), 8);
        assert_eq!(SpinnerType::Grow.frame_count(), 2);
    }

    #[test]
    fn spinner_convenience_constructors() {
        assert!(matches!(Spinner::dots().spinner_type, SpinnerType::Dots));
        assert!(matches!(Spinner::line().spinner_type, SpinnerType::Line));
        assert!(matches!(
            Spinner::circle().spinner_type,
            SpinnerType::Circle
        ));
        assert!(matches!(
            Spinner::square().spinner_type,
            SpinnerType::Square
        ));
        assert!(matches!(Spinner::grow().spinner_type, SpinnerType::Grow));
    }

    #[test]
    fn spinner_text() {
        let spinner = Spinner::new().text("Loading...");
        assert_eq!(spinner.text, Some("Loading...".to_string()));
    }

    #[test]
    fn spinner_styles() {
        let style = Style::default().fg(Color::Red);
        let text_style = Style::default().fg(Color::Green);

        let spinner = Spinner::new().style(style).text_style(text_style);
        assert_eq!(spinner.style.fg, Some(Color::Red));
        assert_eq!(spinner.text_style.fg, Some(Color::Green));
    }

    #[test]
    fn spinner_state_is_loading() {
        assert!(SpinnerState::Loading("test".to_string()).is_loading());
        assert!(!SpinnerState::Success("test".to_string()).is_loading());
        assert!(!SpinnerState::Failed("test".to_string()).is_loading());
        assert!(!SpinnerState::Warning("test".to_string()).is_loading());
    }

    #[test]
    fn spinner_state_is_success() {
        assert!(SpinnerState::Success("test".to_string()).is_success());
        assert!(!SpinnerState::Loading("test".to_string()).is_success());
    }

    #[test]
    fn spinner_state_is_failed() {
        assert!(SpinnerState::Failed("test".to_string()).is_failed());
        assert!(!SpinnerState::Success("test".to_string()).is_failed());
    }

    #[test]
    fn spinner_state_is_warning() {
        assert!(SpinnerState::Warning("test".to_string()).is_warning());
        assert!(!SpinnerState::Success("test".to_string()).is_warning());
    }

    #[test]
    fn spinner_state_message() {
        let state = SpinnerState::Loading("Loading...".to_string());
        assert_eq!(state.message(), "Loading...");
    }

    #[test]
    fn spinner_state_transitions() {
        let mut state = SpinnerState::Loading("test".to_string());

        state.succeed("Done!");
        assert!(state.is_success());
        assert_eq!(state.message(), "Done!");

        state.fail("Failed!");
        assert!(state.is_failed());
        assert_eq!(state.message(), "Failed!");

        state.warn("Warning!");
        assert!(state.is_warning());
        assert_eq!(state.message(), "Warning!");
    }

    #[test]
    fn status_spinner_creation() {
        let status = StatusSpinner::new("Loading");
        assert!(status.state.is_loading());
        assert_eq!(status.state.message(), "Loading");
    }

    #[test]
    fn status_spinner_tick() {
        let mut status = StatusSpinner::new("Loading");
        let frame_before = status.spinner.frame;
        status.tick();
        assert_eq!(status.spinner.frame, frame_before + 1);

        // Should not tick if not loading
        status.set_state(SpinnerState::Success("Done".to_string()));
        let frame_before = status.spinner.frame;
        status.tick();
        assert_eq!(status.spinner.frame, frame_before);
    }

    #[test]
    fn status_spinner_set_state() {
        let mut status = StatusSpinner::new("Loading");
        status.set_state(SpinnerState::Success("Done".to_string()));
        assert!(status.state.is_success());
    }

    #[test]
    fn spinner_render() {
        let spinner = Spinner::new().text("Loading");
        let area = Rect::new(0, 0, 20, 1);
        let el = spinner.render(area);

        assert!(matches!(el.kind, ElementKind::Row));
    }

    #[test]
    fn status_spinner_render_loading() {
        let status = StatusSpinner::new("Loading");
        let area = Rect::new(0, 0, 20, 1);
        let el = status.render(area);

        assert!(matches!(el.kind, ElementKind::Row));
    }

    #[test]
    fn status_spinner_render_success() {
        let status = StatusSpinner {
            spinner: Spinner::new(),
            state: SpinnerState::Success("Done!".to_string()),
        };
        let area = Rect::new(0, 0, 20, 1);
        let el = status.render(area);

        assert!(matches!(el.kind, ElementKind::Row));
    }
}
