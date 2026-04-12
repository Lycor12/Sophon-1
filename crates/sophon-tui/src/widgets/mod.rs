//! Pre-built widgets for common UI patterns
//!
//! This module provides ready-to-use widgets that can be composed
//! into larger applications.

pub mod chart;
pub mod list;
pub mod progress;
pub mod spinner;
pub mod table;

pub use chart::{Chart, ChartType, Dataset};
pub use list::List;
pub use progress::{MultiProgress, ProgressBar};
pub use spinner::{Spinner, SpinnerState, SpinnerType, StatusSpinner};
pub use table::{Alignment, Column, Row, Table};

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
    fn render(&self, area: crate::layout::Rect, state: &mut Self::State)
        -> crate::element::Element;
}

/// Helper function to create a styled box with border
pub fn styled_box(
    title: impl Into<String>,
    content: crate::element::Element,
    style: crate::style::Style,
) -> crate::element::Element {
    use crate::element::{Element, ElementKind};
    use crate::style::BorderStyle;

    Element {
        id: None,
        kind: ElementKind::Border(BorderStyle::Single),
        style,
        children: vec![content],
        layout: None,
    }
}

/// Helper function to create a centered element
pub fn centered(
    element: crate::element::Element,
    container_size: crate::layout::Size,
) -> crate::element::Element {
    use crate::element::{Element, ElementKind};

    let element_size = element.min_size();
    let x = (container_size.width - element_size.width) / 2;
    let y = (container_size.height - element_size.height) / 2;

    Element::row(vec![
        Element {
            id: None,
            kind: ElementKind::Spacer,
            style: crate::style::Style::default(),
            children: vec![],
            layout: None,
        }, // Left padding
        Element::column(vec![
            Element {
                id: None,
                kind: ElementKind::Spacer,
                style: crate::style::Style::default(),
                children: vec![],
                layout: None,
            }, // Top padding
            element,
            Element {
                id: None,
                kind: ElementKind::Spacer,
                style: crate::style::Style::default(),
                children: vec![],
                layout: None,
            }, // Bottom padding
        ]),
        Element {
            id: None,
            kind: ElementKind::Spacer,
            style: crate::style::Style::default(),
            children: vec![],
            layout: None,
        }, // Right padding
    ])
}
