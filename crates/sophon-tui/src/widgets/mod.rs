//! Pre-built widgets for common UI patterns
//!
//! This module provides ready-to-use widgets that can be composed
//! into larger applications.

pub mod list;
pub mod table;
pub mod progress;
pub mod spinner;
pub mod chart;

pub use list::List;
pub use table::Table;
pub use progress::ProgressBar;
pub use spinner::Spinner;
pub use chart::{Chart, Dataset};

/// Widget trait - all widgets implement this
pub trait Widget {
    /// Render the widget to an element
    fn render(&self, area: crate::layout::Rect) -> crate::element::Element;

    /// Get the minimum size required by this widget
    fn min_size(&self) -> crate::layout::Size {
        crate::layout::Size::default()
    }

    /// Get the ideal size for this widget
    fn ideal_size(&self) -> crate::layout::Size {
        self.min_size()
    }
}

/// Stateful widget trait for widgets that maintain internal state
pub trait StatefulWidget {
    /// State type
    type State;

    /// Render with state
    fn render(&self, area: crate::layout::Rect, state: &mut Self::State) -> crate::element::Element;
}

/// Helper function to create a styled box
pub fn styled_box(
    title: impl Into<String>,
    content: crate::element::Element,
    style: crate::style::Style,
) -> crate::element::Element {
    use crate::element::Element;
    use crate::layout::Constraint;
    use crate::style::BorderStyle;

    Element::Container {
        children: vec![
            Element::Text {
                content: title.into(),
                style,
            },
            content,
        ],
        layout: crate::layout::Layout::vertical(vec![
            Constraint::Length(1),
            Constraint::Fill,
        ]),
        style,
    }
}

/// Helper function to create a centered element
pub fn centered(element: crate::element::Element, container_size: crate::layout::Size) -> crate::element::Element {
    use crate::element::Element;
    use crate::layout::{Constraint, Layout};

    Element::Container {
        children: vec![element],
        layout: Layout::centered(container_size),
        style: crate::style::Style::default(),
    }
}
