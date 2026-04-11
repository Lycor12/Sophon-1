//! Table widget for displaying tabular data

use crate::element::Element;
use crate::layout::{Constraint, Rect, Size};
use crate::style::{BorderStyle, Color, Style};
use crate::widgets::Widget;

/// Table column definition
#[derive(Debug, Clone)]
pub struct Column {
    /// Column header
    pub header: String,
    /// Width constraint
    pub width: Constraint,
    /// Alignment
    pub alignment: Alignment,
}

impl Column {
    /// Create a new column
    pub fn new(header: impl Into<String>, width: Constraint) -> Self {
        Column {
            header: header.into(),
            width,
            alignment: Alignment::Left,
        }
    }

    /// Set alignment
    pub fn alignment(mut self, align: Alignment) -> Self {
        self.alignment = align;
        self
    }
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Left-aligned
    Left,
    /// Center-aligned
    Center,
    /// Right-aligned
    Right,
}

/// Table row
#[derive(Debug, Clone)]
pub struct Row {
    /// Cell data
    cells: Vec<String>,
    /// Row style
    style: Option<Style>,
}

impl Row {
    /// Create a new row
    pub fn new(cells: Vec<String>) -> Self {
        Row { cells, style: None }
    }

    /// Create from slice
    pub fn from_slice(cells: &[&str]) -> Self {
        Row {
            cells: cells.iter().map(|s| s.to_string()).collect(),
            style: None,
        }
    }

    /// Set row style
    pub fn style(mut self, style: Style) -> Self {
        self.style = Some(style);
        self
    }
}

/// Table widget
#[derive(Debug, Clone)]
pub struct Table {
    /// Column definitions
    columns: Vec<Column>,
    /// Data rows
    rows: Vec<Row>,
    /// Header style
    header_style: Style,
    /// Row style
    row_style: Style,
    /// Alternate row style (for zebra striping)
    alt_row_style: Option<Style>,
    /// Show border
    border: BorderStyle,
    /// Column separator
    separator: String,
}

impl Table {
    /// Create a new table
    pub fn new(columns: Vec<Column>) -> Self {
        Table {
            columns,
            rows: Vec::new(),
            header_style: Style::default().fg(Color::White).bg(Color::DarkGrey).bold(),
            row_style: Style::default(),
            alt_row_style: Some(Style::default().bg(Color::Rgb(40, 40, 40))),
            border: BorderStyle::Single,
            separator: " │ ".to_string(),
        }
    }

    /// Add a row
    pub fn row(mut self, row: Row) -> Self {
        self.rows.push(row);
        self
    }

    /// Add multiple rows
    pub fn rows(mut self, rows: Vec<Row>) -> Self {
        self.rows.extend(rows);
        self
    }

    /// Set header style
    pub fn header_style(mut self, style: Style) -> Self {
        self.header_style = style;
        self
    }

    /// Set row style
    pub fn row_style(mut self, style: Style) -> Self {
        self.row_style = style;
        self
    }

    /// Set alternate row style
    pub fn alt_row_style(mut self, style: Option<Style>) -> Self {
        self.alt_row_style = style;
        self
    }

    /// Set border style
    pub fn border(mut self, border: BorderStyle) -> Self {
        self.border = border;
        self
    }

    /// Set separator
    pub fn separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    fn align_text(text: &str, width: usize, alignment: Alignment) -> String {
        let text_len = text.chars().count();
        if text_len >= width {
            text.chars().take(width).collect()
        } else {
            let padding = width - text_len;
            match alignment {
                Alignment::Left => format!("{}{}", text, " ".repeat(padding)),
                Alignment::Center => {
                    let left = padding / 2;
                    let right = padding - left;
                    format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
                }
                Alignment::Right => format!("{}{}", " ".repeat(padding), text),
            }
        }
    }

    fn render_header(&self, widths: &[usize]) -> Element {
        let mut cells = Vec::new();
        for (col, width) in self.columns.iter().zip(widths.iter()) {
            let aligned = Self::align_text(&col.header, *width, col.alignment);
            cells.push(Element::Text {
                content: aligned,
                style: self.header_style,
            });
        }

        Element::Container {
            children: cells,
            layout: crate::layout::Layout::horizontal(
                widths
                    .iter()
                    .map(|w| Constraint::Length(*w as u16))
                    .collect(),
            ),
            style: self.header_style,
        }
    }

    fn render_row(&self, row: &Row, widths: &[usize], row_idx: usize) -> Element {
        let mut cells = Vec::new();
        let style = if let Some(ref s) = row.style {
            *s
        } else if let Some(alt_style) = self.alt_row_style {
            if row_idx % 2 == 1 {
                alt_style
            } else {
                self.row_style
            }
        } else {
            self.row_style
        };

        for (cell, width) in row.cells.iter().zip(widths.iter()) {
            let col = self.columns.get(cells.len()).unwrap();
            let aligned = Self::align_text(cell, *width, col.alignment);
            cells.push(Element::Text {
                content: aligned,
                style,
            });
        }

        Element::Container {
            children: cells,
            layout: crate::layout::Layout::horizontal(
                widths
                    .iter()
                    .map(|w| Constraint::Length(*w as u16))
                    .collect(),
            ),
            style,
        }
    }
}

impl Widget for Table {
    fn render(&self, area: Rect) -> Element {
        // Calculate column widths
        let mut widths: Vec<usize> = self
            .columns
            .iter()
            .map(|c| match c.width {
                Constraint::Length(n) => n as usize,
                Constraint::Percentage(p) => {
                    ((area.width as f32 * p as f32 / 100.0) as u16).max(3) as usize
                }
                _ => 10,
            })
            .collect();

        // Ensure widths fit in area
        let total_width: usize = widths.iter().sum();
        let available = area.width as usize - (self.columns.len() - 1) * self.separator.len();
        if total_width > available {
            // Scale down proportionally
            let scale = available as f32 / total_width as f32;
            for w in &mut widths {
                *w = ((*w as f32 * scale) as usize).max(3);
            }
        }

        let mut children = Vec::new();

        // Header
        children.push(self.render_header(&widths));

        // Rows
        let max_rows = (area.height as usize).saturating_sub(1);
        for (idx, row) in self.rows.iter().take(max_rows).enumerate() {
            children.push(self.render_row(row, &widths, idx));
        }

        Element::Container {
            children,
            layout: crate::layout::Layout::vertical(
                std::iter::repeat(Constraint::Length(1))
                    .take(children.len())
                    .collect(),
            ),
            style: Style::default(),
        }
    }

    fn min_size(&self) -> Size {
        let width = self.columns.len() * 10 + (self.columns.len() - 1) * self.separator.len();
        let height = self.rows.len() + 1;
        Size {
            width: width as u16,
            height: height as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_creation() {
        let cols = vec![
            Column::new("Name", Constraint::Length(20)),
            Column::new("Age", Constraint::Length(5)),
        ];
        let table = Table::new(cols)
            .row(Row::from_slice(&["Alice", "30"]))
            .row(Row::from_slice(&["Bob", "25"]));

        assert_eq!(table.columns.len(), 2);
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn text_alignment() {
        assert_eq!(Table::align_text("hi", 5, Alignment::Left), "hi   ");
        assert_eq!(Table::align_text("hi", 5, Alignment::Center), " hi  ");
        assert_eq!(Table::align_text("hi", 5, Alignment::Right), "   hi");
    }
}
