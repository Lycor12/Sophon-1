//! Progress bar widget

use crate::element::{Element, ElementKind};
use crate::layout::{Rect, Size};
use crate::style::{Color, Style};
use crate::widgets::Widget;

/// Progress bar configuration
#[derive(Debug, Clone)]
pub struct ProgressBar {
    /// Current value
    value: f64,
    /// Maximum value
    max: f64,
    /// Character to use for filled portion
    fill_char: char,
    /// Character to use for empty portion
    empty_char: char,
    /// Whether to show percentage
    show_percent: bool,
    /// Label (optional)
    label: Option<String>,
    /// Bar style
    bar_style: Style,
    /// Label style
    label_style: Style,
}

impl ProgressBar {
    /// Create a new progress bar
    pub fn new() -> Self {
        ProgressBar {
            value: 0.0,
            max: 100.0,
            fill_char: '█',
            empty_char: '░',
            show_percent: true,
            label: None,
            bar_style: Style::default().fg(Color::Cyan),
            label_style: Style::default(),
        }
    }

    /// Set value (clamped to max)
    pub fn value(mut self, value: f64) -> Self {
        self.value = value.min(self.max).max(0.0);
        self
    }

    /// Set maximum
    pub fn max(mut self, max: f64) -> Self {
        self.max = max;
        self.value = self.value.min(max);
        self
    }

    /// Set both value and max
    pub fn with_value(self, value: f64, max: f64) -> Self {
        self.max(max).value(value)
    }

    /// Set fill character
    pub fn fill_char(mut self, ch: char) -> Self {
        self.fill_char = ch;
        self
    }

    /// Set empty character
    pub fn empty_char(mut self, ch: char) -> Self {
        self.empty_char = ch;
        self
    }

    /// Set whether to show percentage
    pub fn show_percent(mut self, show: bool) -> Self {
        self.show_percent = show;
        self
    }

    /// Set label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set bar style
    pub fn bar_style(mut self, style: Style) -> Self {
        self.bar_style = style;
        self
    }

    /// Set label style
    pub fn label_style(mut self, style: Style) -> Self {
        self.label_style = style;
        self
    }

    /// Get percentage (0-100)
    pub fn percent(&self) -> f64 {
        if self.max <= 0.0 {
            return 0.0;
        }
        (self.value / self.max * 100.0).clamp(0.0, 100.0)
    }

    /// Get ratio (0.0-1.0)
    pub fn ratio(&self) -> f64 {
        if self.max <= 0.0 {
            return 0.0;
        }
        (self.value / self.max).clamp(0.0, 1.0)
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.value >= self.max
    }

    /// Create a progress bar from ratio (0.0-1.0)
    pub fn from_ratio(ratio: f64) -> Self {
        Self::new().value(ratio * 100.0).max(100.0)
    }

    /// Create with custom characters (e.g., ascii: [=] style)
    pub fn ascii() -> Self {
        Self::new().fill_char('=').empty_char('-')
    }

    /// Create with dots style
    pub fn dots() -> Self {
        Self::new().fill_char('●').empty_char('○')
    }

    /// Create with blocks style
    pub fn blocks() -> Self {
        Self::new().fill_char('█').empty_char('░')
    }
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for ProgressBar {
    fn render(&self, area: Rect) -> Element {
        let mut children = Vec::new();

        // Add label if present
        if let Some(label) = &self.label {
            children.push(Element {
                id: None,
                kind: ElementKind::Text(label.clone()),
                style: self.label_style,
                children: vec![],
                layout: None,
            });
        }

        // Calculate bar width
        let percent_str = if self.show_percent {
            format!(" {:.1}%", self.percent())
        } else {
            String::new()
        };

        let bar_width = area.width.saturating_sub(percent_str.len() as u16);
        let filled = (self.ratio() * bar_width as f64) as usize;
        let empty = bar_width as usize - filled;

        // Build bar string
        let mut bar = String::new();
        for _ in 0..filled {
            bar.push(self.fill_char);
        }
        for _ in 0..empty {
            bar.push(self.empty_char);
        }

        children.push(Element {
            id: None,
            kind: ElementKind::Text(bar),
            style: self.bar_style,
            children: vec![],
            layout: None,
        });

        // Add percentage if shown
        if self.show_percent {
            children.push(Element {
                id: None,
                kind: ElementKind::Text(percent_str.trim().to_string()),
                style: self.bar_style,
                children: vec![],
                layout: None,
            });
        }

        Element {
            id: None,
            kind: ElementKind::Row,
            style: Style::default(),
            children,
            layout: None,
        }
    }

    fn min_size(&self) -> Size {
        let label_width = self.label.as_ref().map(|l| l.len()).unwrap_or(0);
        let percent_width = if self.show_percent { 6 } else { 0 };
        Size {
            width: label_width + 10 + percent_width,
            height: 1,
        }
    }
}

/// Multi-step progress for tracking multiple stages
#[derive(Debug, Clone)]
pub struct MultiProgress {
    /// Current step (0-indexed)
    current: usize,
    /// Total steps
    total: usize,
    /// Step descriptions
    steps: Vec<String>,
    /// Style for completed steps
    completed_style: Style,
    /// Style for current step
    current_style: Style,
    /// Style for pending steps
    pending_style: Style,
}

impl MultiProgress {
    /// Create a new multi-step progress tracker
    pub fn new(steps: Vec<String>) -> Self {
        MultiProgress {
            current: 0,
            total: steps.len(),
            steps,
            completed_style: Style::default().fg(Color::Green),
            current_style: Style::default().fg(Color::Cyan).bold(),
            pending_style: Style::default().fg(Color::DarkGrey),
        }
    }

    /// Set current step
    pub fn current(mut self, step: usize) -> Self {
        self.current = step.min(self.total);
        self
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        if self.current < self.total {
            self.current += 1;
        }
    }

    /// Check if all steps complete
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }

    /// Get progress percentage
    pub fn percent(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.current as f64 / self.total as f64 * 100.0).min(100.0)
    }

    /// Set styles
    pub fn styles(mut self, completed: Style, current: Style, pending: Style) -> Self {
        self.completed_style = completed;
        self.current_style = current;
        self.pending_style = pending;
        self
    }
}

impl Widget for MultiProgress {
    fn render(&self, _area: Rect) -> Element {
        let mut children = Vec::new();

        for (idx, step) in self.steps.iter().enumerate() {
            let prefix = if idx < self.current {
                "✓ "
            } else if idx == self.current {
                "→ "
            } else {
                "  "
            };

            let style = if idx < self.current {
                self.completed_style
            } else if idx == self.current {
                self.current_style
            } else {
                self.pending_style
            };

            children.push(Element {
                id: None,
                kind: ElementKind::Text(format!("{}{}", prefix, step)),
                style,
                children: vec![],
                layout: None,
            });
        }

        Element {
            id: None,
            kind: ElementKind::Column,
            style: Style::default(),
            children,
            layout: None,
        }
    }

    fn min_size(&self) -> Size {
        Size {
            width: self.steps.iter().map(|s| s.len()).max().unwrap_or(0) + 2,
            height: self.steps.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_bar_creation() {
        let pb = ProgressBar::new();
        assert_eq!(pb.value, 0.0);
        assert_eq!(pb.max, 100.0);
    }

    #[test]
    fn progress_bar_value() {
        let pb = ProgressBar::new().value(50.0);
        assert_eq!(pb.value, 50.0);
        assert_eq!(pb.percent(), 50.0);
    }

    #[test]
    fn progress_bar_clamping() {
        let pb = ProgressBar::new().value(150.0);
        assert_eq!(pb.value, 100.0);
        assert_eq!(pb.percent(), 100.0);

        let pb = ProgressBar::new().value(-10.0);
        assert_eq!(pb.value, 0.0);
        assert_eq!(pb.percent(), 0.0);
    }

    #[test]
    fn progress_bar_ratio() {
        let pb = ProgressBar::new().value(25.0).max(50.0);
        assert_eq!(pb.ratio(), 0.5);
    }

    #[test]
    fn progress_bar_custom_chars() {
        let pb = ProgressBar::new().fill_char('=').empty_char('-');
        assert_eq!(pb.fill_char, '=');
        assert_eq!(pb.empty_char, '-');
    }

    #[test]
    fn progress_bar_ascii() {
        let pb = ProgressBar::ascii();
        assert_eq!(pb.fill_char, '=');
        assert_eq!(pb.empty_char, '-');
    }

    #[test]
    fn progress_bar_dots() {
        let pb = ProgressBar::dots();
        assert_eq!(pb.fill_char, '●');
        assert_eq!(pb.empty_char, '○');
    }

    #[test]
    fn progress_bar_label() {
        let pb = ProgressBar::new().label("Loading");
        assert_eq!(pb.label, Some("Loading".to_string()));
    }

    #[test]
    fn progress_bar_complete() {
        let pb = ProgressBar::new().value(100.0);
        assert!(pb.is_complete());

        let pb = ProgressBar::new().value(50.0);
        assert!(!pb.is_complete());
    }

    #[test]
    fn progress_bar_from_ratio() {
        let pb = ProgressBar::from_ratio(0.75);
        assert_eq!(pb.percent(), 75.0);
    }

    #[test]
    fn progress_bar_show_percent() {
        let pb = ProgressBar::new().show_percent(false);
        assert!(!pb.show_percent);
    }

    #[test]
    fn multi_progress_creation() {
        let steps = vec![
            "Step 1".to_string(),
            "Step 2".to_string(),
            "Step 3".to_string(),
        ];
        let mp = MultiProgress::new(steps);
        assert_eq!(mp.total, 3);
        assert_eq!(mp.current, 0);
    }

    #[test]
    fn multi_progress_advance() {
        let steps = vec!["A".to_string(), "B".to_string()];
        let mut mp = MultiProgress::new(steps);
        assert_eq!(mp.current, 0);

        mp.advance();
        assert_eq!(mp.current, 1);

        mp.advance();
        assert_eq!(mp.current, 2);
        assert!(mp.is_complete());

        mp.advance(); // Should not exceed total
        assert_eq!(mp.current, 2);
    }

    #[test]
    fn multi_progress_percent() {
        let steps = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let mut mp = MultiProgress::new(steps).current(2);
        assert_eq!(mp.percent(), 50.0);
    }

    #[test]
    fn progress_bar_render() {
        let pb = ProgressBar::new().value(50.0);
        let area = Rect::new(0, 0, 20, 1);
        let el = pb.render(area);

        assert!(matches!(el.kind, ElementKind::Row));
    }

    #[test]
    fn multi_progress_render() {
        let steps = vec!["Step 1".to_string(), "Step 2".to_string()];
        let mp = MultiProgress::new(steps).current(1);
        let area = Rect::new(0, 0, 20, 2);
        let el = mp.render(area);

        assert!(matches!(el.kind, ElementKind::Column));
        assert_eq!(el.children.len(), 2);
    }
}
