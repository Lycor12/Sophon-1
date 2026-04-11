//! List widget for displaying scrollable items

use crate::element::Element;
use crate::layout::{Constraint, Rect, Size};
use crate::style::{Color, Style, TextStyle};
use crate::widgets::Widget;

/// List widget configuration
#[derive(Debug, Clone)]
pub struct List {
    /// List items
    items: Vec<String>,
    /// Currently selected index (optional)
    selected: Option<usize>,
    /// Offset for scrolling
    offset: usize,
    /// Style for unselected items
    style: Style,
    /// Style for selected item
    selected_style: Style,
    /// Whether to show scrollbar
    show_scrollbar: bool,
    /// Start symbol (e.g., "> ")
    start_symbol: String,
    /// End symbol (e.g., " <")
    end_symbol: String,
}

impl List {
    /// Create a new list
    pub fn new(items: Vec<String>) -> Self {
        List {
            items,
            selected: None,
            offset: 0,
            style: Style::default(),
            selected_style: Style::default()
                .fg(Color::Cyan)
                .bg(Color::DarkGrey)
                .text_style(TextStyle::Bold),
            show_scrollbar: true,
            start_symbol: "> ".to_string(),
            end_symbol: String::new(),
        }
    }

    /// Set selected index
    pub fn selected(mut self, index: usize) -> Self {
        self.selected = Some(index);
        self
    }

    /// Set offset for scrolling
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set style
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Set selected item style
    pub fn selected_style(mut self, style: Style) -> Self {
        self.selected_style = style;
        self
    }

    /// Set scrollbar visibility
    pub fn show_scrollbar(mut self, show: bool) -> Self {
        self.show_scrollbar = show;
        self
    }

    /// Set start symbol
    pub fn start_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.start_symbol = symbol.into();
        self
    }

    /// Set end symbol
    pub fn end_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.end_symbol = symbol.into();
        self
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get selected index
    pub fn selected_index(&self) -> Option<usize> {
        self.selected
    }

    /// Move selection up
    pub fn up(&mut self) {
        if let Some(sel) = self.selected {
            if sel > 0 {
                self.selected = Some(sel - 1);
                if sel - 1 < self.offset {
                    self.offset = sel - 1;
                }
            }
        } else if !self.items.is_empty() {
            self.selected = Some(self.items.len() - 1);
        }
    }

    /// Move selection down
    pub fn down(&mut self) {
        if let Some(sel) = self.selected {
            if sel + 1 < self.items.len() {
                self.selected = Some(sel + 1);
                let visible_height = 10; // Default, should be passed in
                if sel + 1 >= self.offset + visible_height {
                    self.offset = sel + 1 - visible_height + 1;
                }
            }
        } else if !self.items.is_empty() {
            self.selected = Some(0);
        }
    }

    fn render_item(&self, item: &str, is_selected: bool, width: u16) -> Element {
        let content = if is_selected {
            format!("{}{}{}", self.start_symbol, item, self.end_symbol)
        } else {
            format!("  {}", item)
        };

        // Truncate or pad to fit width
        let display = if content.len() > width as usize {
            content.chars().take(width as usize).collect()
        } else {
            content
        };

        Element::Text {
            content: display,
            style: if is_selected {
                self.selected_style
            } else {
                self.style
            },
        }
    }
}

impl Widget for List {
    fn render(&self, area: Rect) -> Element {
        let visible_items = area.height as usize;
        let mut children = Vec::new();

        for (i, item) in self
            .items
            .iter()
            .enumerate()
            .skip(self.offset)
            .take(visible_items)
        {
            let is_selected = self.selected == Some(i);
            children.push(self.render_item(item, is_selected, area.width));
        }

        // Fill remaining space with empty lines
        let rendered_count = children.len();
        for _ in rendered_count..visible_items {
            children.push(Element::Text {
                content: String::new(),
                style: self.style,
            });
        }

        Element::Container {
            children,
            layout: crate::layout::Layout::vertical(vec![Constraint::Length(1); visible_items]),
            style: Style::default(),
        }
    }

    fn min_size(&self) -> Size {
        let max_width = self.items.iter().map(|s| s.len()).max().unwrap_or(0) + 4;
        Size {
            width: max_width as u16,
            height: self.items.len().min(10) as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_creation() {
        let items = vec![
            "Item 1".to_string(),
            "Item 2".to_string(),
            "Item 3".to_string(),
        ];
        let list = List::new(items);
        assert_eq!(list.len(), 3);
        assert!(list.selected.is_none());
    }

    #[test]
    fn list_selection() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut list = List::new(items).selected(0);
        assert_eq!(list.selected, Some(0));

        list.down();
        assert_eq!(list.selected, Some(1));

        list.up();
        assert_eq!(list.selected, Some(0));
    }
}
