//! Progress bar widget

use crate::element::Element;
use crate::layout::{Constraint, Rect, Size};
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
            0.0
        } else {
            (self.value / self.max * 100.0).min(100.0).max(0.0)
        }
    }

    /// Update the value
    pub fn set_value(&mut self, value: f64) {
        self.value = value.min(self.max).max(0.0);
    }
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for ProgressBar {
    fn render(&self, area: Rect) -> Element {
        let percent = self.percent();
        let bar_width = if self.show_percent {
            area.width.saturating_sub(8) // Leave room for " 100.0%"
        } else {
            area.width
        };

        if bar_width == 0 {
            return Element::Empty;
        }

        let filled = (percent / 100.0 * bar_width as f64) as u16;
        let empty = bar_width.saturating_sub(filled);

        let bar: String = std::iter::repeat(self.fill_char)
            .take(filled as usize)
            .chain(std::iter::repeat(self.empty_char).take(empty as usize))
            .collect();

        let mut children = Vec::new();

        // Label if present
        if let Some(ref label) = self.label {
            children.push(Element::Text {
                content: label.clone(),
                style: self.label_style,
            });
        }

        // Progress bar
        children.push(Element::Text {
            content: bar.clone(),
            style: self.bar_style,
        });

        // Percentage
        if self.show_percent {
            let percent_str = format!("{:>6.1}%", percent);
            children.push(Element::Text {
                content: percent_str,
                style: self.label_style,
            });
        }

        if children.len() == 1 {
            children.pop().unwrap()
        } else {
            Element::Container {
                children,
                layout: crate::layout::Layout::horizontal(vec![
                    Constraint::Fill,
                    Constraint::Length(bar_width),
                    Constraint::Length(8),
                ]),
                style: Style::default(),
            }
        }
    }

    fn min_size(&self) -> Size {
        Size {
            width: if self.show_percent { 10 } else { 5 },
            height: 1,
        }
    }
}

/// Stateful progress bar for tracking ongoing progress
#[derive(Debug, Clone)]
pub struct StatefulProgressBar {
    config: ProgressBar,
}

impl StatefulProgressBar {
    /// Create new stateful progress bar
    pub fn new(config: ProgressBar) -> Self {
        StatefulProgressBar { config }
    }

    /// Update progress
    pub fn set_progress(&mut self, value: f64) {
        self.config.set_value(value);
    }

    /// Finish the progress bar
    pub fn finish(&mut self) {
        self.config.value = self.config.max;
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.config.value >= self.config.max
    }
}

impl Widget for StatefulProgressBar {
    fn render(&self, area: Rect) -> Element {
        self.config.render(area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_calculation() {
        let pb = ProgressBar::new().with_value(50.0, 100.0);
        assert!((pb.percent() - 50.0).abs() < 0.1);

        let pb = ProgressBar::new().with_value(25.0, 100.0);
        assert!((pb.percent() - 25.0).abs() < 0.1);
    }

    #[test]
    fn progress_bounds() {
        let pb = ProgressBar::new().max(100.0).value(150.0);
        assert_eq!(pb.percent(), 100.0);

        let pb = ProgressBar::new().max(100.0).value(-10.0);
        assert_eq!(pb.percent(), 0.0);
    }
}
