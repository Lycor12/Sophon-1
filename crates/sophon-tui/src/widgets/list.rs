//! List widget for displaying scrollable items

use crate::element::{Element, ElementKind};
use crate::layout::{Rect, Size};
use crate::style::{Color, Style};
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
            selected_style: Style::default().fg(Color::Cyan).bg(Color::DarkGrey).bold(),
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

    /// Scroll to next item
    pub fn next(&mut self) {
        if let Some(selected) = self.selected {
            if selected < self.items.len().saturating_sub(1) {
                self.selected = Some(selected + 1);
                if self.selected.unwrap() >= self.offset + 10 {
                    self.offset += 1;
                }
            }
        } else if !self.items.is_empty() {
            self.selected = Some(0);
        }
    }

    /// Scroll to previous item
    pub fn previous(&mut self) {
        if let Some(selected) = self.selected {
            if selected > 0 {
                self.selected = Some(selected - 1);
                if self.selected.unwrap() < self.offset {
                    self.offset = self.offset.saturating_sub(1);
                }
            }
        }
    }

    /// Get visible items count
    pub fn visible_count(&self, height: usize) -> usize {
        height.min(self.items.len())
    }
}

impl Widget for List {
    fn render(&self, area: Rect) -> Element {
        if self.items.is_empty() {
            return Element {
                id: None,
                kind: ElementKind::Text("(empty list)".to_string()),
                style: self.style,
                children: vec![],
                layout: None,
            };
        }

        let visible_count = self.visible_count(area.height as usize);
        let end = (self.offset + visible_count).min(self.items.len());

        let mut children = Vec::new();
        let max_width = self.items.iter().map(|s| s.len()).max().unwrap_or(0);

        for (idx, item) in self.items[self.offset..end].iter().enumerate() {
            let actual_idx = self.offset + idx;
            let is_selected = self.selected == Some(actual_idx);

            let mut content = String::new();
            if is_selected {
                content.push_str(&self.start_symbol);
            } else {
                content.push_str(&" ".repeat(self.start_symbol.len()));
            }
            content.push_str(item);

            // Pad to max width
            let padding = max_width.saturating_sub(item.len());
            content.push_str(&" ".repeat(padding));
            content.push_str(&self.end_symbol);

            let el = Element {
                id: None,
                kind: ElementKind::Text(content),
                style: if is_selected {
                    self.selected_style
                } else {
                    self.style
                },
                children: vec![],
                layout: None,
            };
            children.push(el);
        }

        let height = children.len() as u16;
        Element {
            id: None,
            kind: ElementKind::Column,
            style: self.style,
            children,
            layout: Some(Rect {
                x: area.x,
                y: area.y,
                width: (max_width + self.start_symbol.len() + self.end_symbol.len()) as u16,
                height,
            }),
        }
    }

    fn min_size(&self) -> Size {
        let max_width = self.items.iter().map(|s| s.len()).max().unwrap_or(0);
        Size {
            width: max_width + self.start_symbol.len() + self.end_symbol.len(),
            height: self.items.len().min(10),
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
        let list = List::new(items.clone());
        assert_eq!(list.len(), 3);
        assert_eq!(list.items, items);
    }

    #[test]
    fn list_empty() {
        let list: List = List::new(vec![]);
        assert!(list.is_empty());
    }

    #[test]
    fn list_selected() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let list = List::new(items).selected(1);
        assert_eq!(list.selected_index(), Some(1));
    }

    #[test]
    fn list_navigation() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut list = List::new(items).selected(0);

        list.next();
        assert_eq!(list.selected_index(), Some(1));

        list.next();
        assert_eq!(list.selected_index(), Some(2));

        list.next(); // Should stay at 2
        assert_eq!(list.selected_index(), Some(2));

        list.previous();
        assert_eq!(list.selected_index(), Some(1));

        list.previous();
        list.previous();
        assert_eq!(list.selected_index(), Some(0));

        list.previous(); // Should stay at 0
        assert_eq!(list.selected_index(), Some(0));
    }

    #[test]
    fn list_custom_symbols() {
        let list = List::new(vec!["Item".to_string()])
            .start_symbol(">>>")
            .end_symbol("<<<");
        assert_eq!(list.start_symbol, ">>>");
        assert_eq!(list.end_symbol, "<<<");
    }

    #[test]
    fn list_styles() {
        let style = Style::default().fg(Color::Red);
        let selected_style = Style::default().fg(Color::Green).bold();

        let list = List::new(vec!["A".to_string()])
            .style(style)
            .selected_style(selected_style);

        assert_eq!(list.style.fg, Some(Color::Red));
        assert_eq!(list.selected_style.fg, Some(Color::Green));
        assert!(list.selected_style.bold);
    }

    #[test]
    fn list_visible_count() {
        let items: Vec<String> = (0..100).map(|i| i.to_string()).collect();
        let list = List::new(items);
        assert_eq!(list.visible_count(10), 10);
        assert_eq!(list.visible_count(200), 100);
    }

    #[test]
    fn list_render() {
        let items = vec!["First".to_string(), "Second".to_string()];
        let list = List::new(items);
        let area = Rect::new(0, 0, 20, 5);
        let el = list.render(area);

        assert!(matches!(el.kind, ElementKind::Column));
        assert_eq!(el.children.len(), 2);
    }

    #[test]
    fn list_render_selected() {
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let list = List::new(items).selected(1);
        let area = Rect::new(0, 0, 20, 5);
        let el = list.render(area);

        assert_eq!(el.children.len(), 3);
        // Second item should have selected style
        assert_eq!(el.children[1].style.fg, Some(Color::Cyan));
        assert!(el.children[1].style.bold);
    }

    #[test]
    fn list_scroll_offset() {
        let items: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let mut list = List::new(items).selected(0);
        list.offset = 5;

        let area = Rect::new(0, 0, 20, 5);
        let el = list.render(area);

        // Should render items 5-9 (offset 5, visible count 5)
        assert_eq!(el.children.len(), 5);
    }
}
