//! Element tree - the virtual DOM of Sophon TUI
//!
//! Elements are lightweight descriptions of what should be rendered.
//! They form a tree structure that is diffed and rendered to the terminal.

use crate::layout::{Rect, Size};
use crate::style::{BorderStyle, Color, Style};

/// Unique identifier for elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementId(pub usize);

/// The kind of element - determines rendering behavior
pub enum ElementKind {
    /// Text content
    Text(String),
    /// Single child container
    Box,
    /// Vertical stack
    Column,
    /// Horizontal stack
    Row,
    /// Container with border
    Border(BorderStyle),
    /// Interactive button
    Button {
        label: String,
        on_press: Option<Box<dyn Fn() + 'static>>,
    },
    /// Text input field
    Input { value: String, placeholder: String },
    /// Scrollable viewport
    Scroll { offset: usize },
    /// Empty spacer
    Spacer,
    /// Custom component
    Component { name: String },
}

impl std::fmt::Debug for ElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementKind::Text(s) => f.debug_tuple("Text").field(s).finish(),
            ElementKind::Box => f.debug_struct("Box").finish(),
            ElementKind::Column => f.debug_struct("Column").finish(),
            ElementKind::Row => f.debug_struct("Row").finish(),
            ElementKind::Border(style) => f.debug_tuple("Border").field(style).finish(),
            ElementKind::Button { label, .. } => {
                f.debug_struct("Button").field("label", label).finish()
            }
            ElementKind::Input { value, placeholder } => f
                .debug_struct("Input")
                .field("value", value)
                .field("placeholder", placeholder)
                .finish(),
            ElementKind::Scroll { offset } => {
                f.debug_struct("Scroll").field("offset", offset).finish()
            }
            ElementKind::Spacer => f.debug_struct("Spacer").finish(),
            ElementKind::Component { name } => {
                f.debug_struct("Component").field("name", name).finish()
            }
        }
    }
}

impl Clone for ElementKind {
    fn clone(&self) -> Self {
        match self {
            ElementKind::Text(s) => ElementKind::Text(s.clone()),
            ElementKind::Box => ElementKind::Box,
            ElementKind::Column => ElementKind::Column,
            ElementKind::Row => ElementKind::Row,
            ElementKind::Border(style) => ElementKind::Border(*style),
            ElementKind::Button { label, .. } => ElementKind::Button {
                label: label.clone(),
                on_press: None, // Can't clone closures
            },
            ElementKind::Input { value, placeholder } => ElementKind::Input {
                value: value.clone(),
                placeholder: placeholder.clone(),
            },
            ElementKind::Scroll { offset } => ElementKind::Scroll { offset: *offset },
            ElementKind::Spacer => ElementKind::Spacer,
            ElementKind::Component { name } => ElementKind::Component { name: name.clone() },
        }
    }
}

impl PartialEq for ElementKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ElementKind::Text(a), ElementKind::Text(b)) => a == b,
            (ElementKind::Box, ElementKind::Box) => true,
            (ElementKind::Column, ElementKind::Column) => true,
            (ElementKind::Row, ElementKind::Row) => true,
            (ElementKind::Border(a), ElementKind::Border(b)) => a == b,
            (ElementKind::Button { label: a, .. }, ElementKind::Button { label: b, .. }) => a == b,
            (
                ElementKind::Input {
                    value: a,
                    placeholder: ap,
                },
                ElementKind::Input {
                    value: b,
                    placeholder: bp,
                },
            ) => a == b && ap == bp,
            (ElementKind::Scroll { offset: a }, ElementKind::Scroll { offset: b }) => a == b,
            (ElementKind::Spacer, ElementKind::Spacer) => true,
            (ElementKind::Component { name: a }, ElementKind::Component { name: b }) => a == b,
            _ => false,
        }
    }
}

/// An element in the virtual tree
#[derive(Debug, Clone)]
pub struct Element {
    pub id: Option<ElementId>,
    pub kind: ElementKind,
    pub style: Style,
    pub children: Vec<Element>,
    pub layout: Option<Rect>,
}

impl Element {
    /// Create a text element
    pub fn text<S: Into<String>>(content: S) -> Self {
        Element {
            id: None,
            kind: ElementKind::Text(content.into()),
            style: Style::default(),
            children: vec![],
            layout: None,
        }
    }

    /// Create a boxed container
    pub fn boxed(child: Element) -> Self {
        Element {
            id: None,
            kind: ElementKind::Box,
            style: Style::default(),
            children: vec![child],
            layout: None,
        }
    }

    /// Create a vertical column
    pub fn column(children: Vec<Element>) -> Self {
        Element {
            id: None,
            kind: ElementKind::Column,
            style: Style::default(),
            children,
            layout: None,
        }
    }

    /// Create a horizontal row
    pub fn row(children: Vec<Element>) -> Self {
        Element {
            id: None,
            kind: ElementKind::Row,
            style: Style::default(),
            children,
            layout: None,
        }
    }

    /// Create a bordered container
    pub fn bordered(child: Element, style: BorderStyle) -> Self {
        Element {
            id: None,
            kind: ElementKind::Border(style),
            style: Style::default(),
            children: vec![child],
            layout: None,
        }
    }

    /// Create a button
    pub fn button<F: Fn() + 'static>(label: &str, on_press: F) -> Self {
        Element {
            id: None,
            kind: ElementKind::Button {
                label: label.to_string(),
                on_press: Some(Box::new(on_press)),
            },
            style: Style::default(),
            children: vec![],
            layout: None,
        }
    }

    /// Create a text input
    pub fn input(value: &str, placeholder: &str) -> Self {
        Element {
            id: None,
            kind: ElementKind::Input {
                value: value.to_string(),
                placeholder: placeholder.to_string(),
            },
            style: Style::default(),
            children: vec![],
            layout: None,
        }
    }

    /// Create a spacer
    pub fn spacer() -> Self {
        Element {
            id: None,
            kind: ElementKind::Spacer,
            style: Style::default(),
            children: vec![],
            layout: None,
        }
    }

    /// Set foreground color
    pub fn color(mut self, color: Color) -> Self {
        self.style.fg = Some(color);
        self
    }

    /// Set background color
    pub fn bg(mut self, color: Color) -> Self {
        self.style.bg = Some(color);
        self
    }

    /// Make text bold
    pub fn bold(mut self) -> Self {
        self.style.bold = true;
        self
    }

    /// Make text dim
    pub fn dim(mut self) -> Self {
        self.style.dim = true;
        self
    }

    /// Make text italic
    pub fn italic(mut self) -> Self {
        self.style.italic = true;
        self
    }

    /// Make text underlined
    pub fn underline(mut self) -> Self {
        self.style.underline = true;
        self
    }

    /// Make text blink
    pub fn blink(mut self) -> Self {
        self.style.blink = true;
        self
    }

    /// Set element ID
    pub fn with_id(mut self, id: ElementId) -> Self {
        self.id = Some(id);
        self
    }

    /// Apply a style transformation function
    pub fn with_style<F>(mut self, f: F) -> Self
    where
        F: FnOnce(Style) -> Style,
    {
        self.style = f(self.style);
        self
    }

    /// Get minimum size required by this element
    pub fn min_size(&self) -> Size {
        match &self.kind {
            ElementKind::Text(text) => {
                let lines: Vec<&str> = text.lines().collect();
                let height = lines.len();
                let width = lines.iter().map(|l| l.len()).max().unwrap_or(0);
                Size { width, height }
            }
            ElementKind::Spacer => Size {
                width: 0,
                height: 0,
            },
            _ => {
                // For containers, compute from children
                let mut width = 0;
                let mut height = 0;
                for child in &self.children {
                    let child_size = child.min_size();
                    match self.kind {
                        ElementKind::Column => {
                            width = width.max(child_size.width);
                            height += child_size.height;
                        }
                        ElementKind::Row => {
                            width += child_size.width;
                            height = height.max(child_size.height);
                        }
                        _ => {
                            width = width.max(child_size.width);
                            height = height.max(child_size.height);
                        }
                    }
                }
                // Account for border
                match &self.kind {
                    ElementKind::Border(_) => Size {
                        width: width + 2,
                        height: height + 2,
                    },
                    _ => Size { width, height },
                }
            }
        }
    }

    /// Count total elements in tree
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }

    /// Find element by ID
    pub fn find(&self, id: ElementId) -> Option<&Element> {
        if self.id == Some(id) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find element by ID (mutable)
    pub fn find_mut(&mut self, id: ElementId) -> Option<&mut Element> {
        if self.id == Some(id) {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(id) {
                return Some(found);
            }
        }
        None
    }
}

impl Default for Element {
    fn default() -> Self {
        Element::spacer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_element_creation() {
        let el = Element::text("Hello, World!");
        assert!(matches!(el.kind, ElementKind::Text(s) if s == "Hello, World!"));
    }

    #[test]
    fn column_layout_computation() {
        let col = Element::column(vec![
            Element::text("Line 1"),
            Element::text("Line 2 is longer"),
            Element::text("L3"),
        ]);

        let size = col.min_size();
        assert_eq!(size.height, 3); // 3 lines
        assert_eq!(size.width, 16); // "Line 2 is longer"
    }

    #[test]
    fn row_layout_computation() {
        let row = Element::row(vec![
            Element::text("A"),
            Element::text("BC"),
            Element::text("D"),
        ]);

        let size = row.min_size();
        assert_eq!(size.width, 4); // "A" + "BC" + "D"
        assert_eq!(size.height, 1); // max height
    }

    #[test]
    fn element_tree_count() {
        let tree = Element::column(vec![
            Element::text("A"),
            Element::row(vec![Element::text("B"), Element::text("C")]),
        ]);

        assert_eq!(tree.count(), 5); // col + 2 children + row + 2 children
    }

    #[test]
    fn style_chaining() {
        let el = Element::text("test")
            .color(Color::Red)
            .bold()
            .bg(Color::Blue);

        assert_eq!(el.style.fg, Some(Color::Red));
        assert_eq!(el.style.bg, Some(Color::Blue));
        assert!(el.style.bold);
    }

    #[test]
    fn bordered_element_size() {
        let inner = Element::text("content");
        let bordered = Element::bordered(inner, BorderStyle::Single);

        let size = bordered.min_size();
        assert_eq!(size.width, 9); // 7 + 2 border
        assert_eq!(size.height, 3); // 1 + 2 border
    }
}
