//! Table widget for displaying tabular data

use crate::element::{Element, ElementKind};
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

impl Alignment {
    /// Align text within width
    fn align(&self, text: &str, width: usize) -> String {
        match self {
            Alignment::Left => format!("{}{}", text, " ".repeat(width.saturating_sub(text.len()))),
            Alignment::Right => format!("{}{}", " ".repeat(width.saturating_sub(text.len())), text),
            Alignment::Center => {
                let padding = width.saturating_sub(text.len());
                let left = padding / 2;
                let right = padding - left;
                format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
            }
        }
    }
}

/// Table row
#[derive(Debug, Clone)]
pub struct Row {
    /// Cell data
    pub cells: Vec<String>,
    /// Row style
    pub style: Option<Style>,
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
    /// Show header
    show_header: bool,
    /// Currently selected row
    selected: Option<usize>,
    /// Selected row style
    selected_style: Style,
}

impl Table {
    /// Create a new table
    pub fn new(columns: Vec<Column>) -> Self {
        Table {
            columns,
            rows: Vec::new(),
            header_style: Style::default().bold(),
            row_style: Style::default(),
            alt_row_style: None,
            border: BorderStyle::None,
            separator: " ".to_string(),
            show_header: true,
            selected: None,
            selected_style: Style::default().fg(Color::White).bg(Color::Blue),
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
    pub fn alt_row_style(mut self, style: Style) -> Self {
        self.alt_row_style = Some(style);
        self
    }

    /// Set border style
    pub fn border(mut self, border: BorderStyle) -> Self {
        self.border = border;
        self
    }

    /// Set column separator
    pub fn separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Show/hide header
    pub fn header(mut self, show: bool) -> Self {
        self.show_header = show;
        self
    }

    /// Set selected row
    pub fn selected(mut self, index: usize) -> Self {
        self.selected = Some(index);
        self
    }

    /// Set selected row style
    pub fn selected_style(mut self, style: Style) -> Self {
        self.selected_style = style;
        self
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Scroll to next row
    pub fn next(&mut self) {
        if let Some(selected) = self.selected {
            if selected < self.rows.len().saturating_sub(1) {
                self.selected = Some(selected + 1);
            }
        } else if !self.rows.is_empty() {
            self.selected = Some(0);
        }
    }

    /// Scroll to previous row
    pub fn previous(&mut self) {
        if let Some(selected) = self.selected {
            if selected > 0 {
                self.selected = Some(selected - 1);
            }
        }
    }

    /// Get selected row index
    pub fn selected_index(&self) -> Option<usize> {
        self.selected
    }

    /// Calculate column widths based on constraints
    fn calculate_widths(&self) -> Vec<usize> {
        let num_cols = self.columns.len();
        let mut widths = vec![0usize; num_cols];

        // Start with header widths
        for (i, col) in self.columns.iter().enumerate() {
            match col.width {
                Constraint::Length(n) => widths[i] = n as usize,
                Constraint::Min(n) => widths[i] = n as usize,
                Constraint::Max(n) => widths[i] = col.header.len().min(n as usize),
                _ => widths[i] = col.header.len(),
            }
        }

        // Adjust for cell content
        for row in &self.rows {
            for (i, cell) in row.cells.iter().enumerate().take(num_cols) {
                if i < num_cols {
                    match self.columns[i].width {
                        Constraint::Length(_n) => {} // Fixed, don't change
                        Constraint::Min(n) => widths[i] = widths[i].max(cell.len().max(n as usize)),
                        Constraint::Max(n) => widths[i] = widths[i].max(cell.len()).min(n as usize),
                        _ => widths[i] = widths[i].max(cell.len()),
                    }
                }
            }
        }

        widths
    }
}

impl Widget for Table {
    fn render(&self, _area: Rect) -> Element {
        let widths = self.calculate_widths();
        let num_cols = self.columns.len();

        // Build header row
        let mut children = Vec::new();

        if self.show_header && !self.columns.is_empty() {
            let mut header_cells = Vec::new();
            for (i, col) in self.columns.iter().enumerate() {
                let text = col.alignment.align(&col.header, widths[i]);
                header_cells.push(Element {
                    id: None,
                    kind: ElementKind::Text(text),
                    style: self.header_style,
                    children: vec![],
                    layout: None,
                });
            }
            children.push(Element {
                id: None,
                kind: ElementKind::Row,
                style: self.header_style,
                children: header_cells,
                layout: None,
            });
        }

        // Build data rows
        for (row_idx, row) in self.rows.iter().enumerate() {
            let is_selected = self.selected == Some(row_idx);
            let row_style = if is_selected {
                self.selected_style
            } else if let Some(alt_style) = &self.alt_row_style {
                if row_idx % 2 == 1 {
                    *alt_style
                } else {
                    row.style.unwrap_or(self.row_style)
                }
            } else {
                row.style.unwrap_or(self.row_style)
            };

            let mut cells = Vec::new();
            for (i, cell) in row.cells.iter().enumerate().take(num_cols) {
                let text = if i < num_cols {
                    self.columns[i].alignment.align(cell, widths[i])
                } else {
                    cell.clone()
                };
                cells.push(Element {
                    id: None,
                    kind: ElementKind::Text(text),
                    style: row_style,
                    children: vec![],
                    layout: None,
                });
            }
            children.push(Element {
                id: None,
                kind: ElementKind::Row,
                style: row_style,
                children: cells,
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
        let num_cols = self.columns.len();
        let num_rows = self.rows.len();

        // Calculate column widths based on constraints
        let mut widths = vec![0usize; num_cols];
        for (i, col) in self.columns.iter().enumerate() {
            widths[i] = widths[i].max(col.header.len());
        }

        for row in &self.rows {
            for (i, cell) in row.cells.iter().enumerate().take(num_cols) {
                if i < num_cols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let total_width: usize =
            widths.iter().sum::<usize>() + (num_cols.saturating_sub(1) * self.separator.len());
        let total_height = num_rows.saturating_add(if self.show_header { 1 } else { 0 });

        Size {
            width: total_width,
            height: total_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_column_creation() {
        let col = Column::new("Name", Constraint::Length(20));
        assert_eq!(col.header, "Name");
        assert!(matches!(col.width, Constraint::Length(20)));
    }

    #[test]
    fn table_column_alignment() {
        let col = Column::new("Amount", Constraint::Length(10)).alignment(Alignment::Right);
        assert!(matches!(col.alignment, Alignment::Right));
    }

    #[test]
    fn table_row_creation() {
        let row = Row::new(vec!["Cell 1".to_string(), "Cell 2".to_string()]);
        assert_eq!(row.cells.len(), 2);
        assert_eq!(row.cells[0], "Cell 1");
    }

    #[test]
    fn table_row_from_slice() {
        let row = Row::from_slice(&["A", "B", "C"]);
        assert_eq!(row.cells, vec!["A", "B", "C"]);
    }

    #[test]
    fn table_row_style() {
        let row = Row::new(vec!["Test".to_string()]).style(Style::default().fg(Color::Red));
        assert_eq!(row.style.unwrap().fg, Some(Color::Red));
    }

    #[test]
    fn table_creation() {
        let columns = vec![
            Column::new("ID", Constraint::Length(5)),
            Column::new("Name", Constraint::Length(20)),
        ];
        let table = Table::new(columns);
        assert_eq!(table.columns.len(), 2);
        assert!(table.rows.is_empty());
    }

    #[test]
    fn table_row() {
        let columns = vec![
            Column::new("A", Constraint::Length(5)),
            Column::new("B", Constraint::Length(5)),
        ];
        let table = Table::new(columns).row(Row::new(vec!["1".to_string(), "2".to_string()]));
        assert_eq!(table.rows.len(), 1);
    }

    #[test]
    fn table_rows() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let table = Table::new(columns).rows(vec![
            Row::new(vec!["Row 1".to_string()]),
            Row::new(vec!["Row 2".to_string()]),
        ]);
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn table_styles() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let header_style = Style::default().bold();
        let row_style = Style::default().fg(Color::Green);
        let alt_style = Style::default().fg(Color::DarkGrey);

        let table = Table::new(columns)
            .header_style(header_style)
            .row_style(row_style)
            .alt_row_style(alt_style);

        assert!(table.header_style.bold);
        assert_eq!(table.row_style.fg, Some(Color::Green));
        assert_eq!(table.alt_row_style.unwrap().fg, Some(Color::DarkGrey));
    }

    #[test]
    fn table_border() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let table = Table::new(columns).border(BorderStyle::Double);
        assert!(matches!(table.border, BorderStyle::Double));
    }

    #[test]
    fn table_separator() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let table = Table::new(columns).separator(" | ");
        assert_eq!(table.separator, " | ");
    }

    #[test]
    fn table_header() {
        let columns = vec![
            Column::new("ID", Constraint::Length(5)),
            Column::new("Name", Constraint::Length(10)),
        ];
        let table = Table::new(columns).header(true);
        assert!(table.show_header);
    }

    #[test]
    fn table_selected() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let table = Table::new(columns).selected(2);
        assert_eq!(table.selected, Some(2));
    }

    #[test]
    fn table_selected_style() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let selected_style = Style::default().bg(Color::Blue);
        let table = Table::new(columns).selected_style(selected_style);
        assert_eq!(table.selected_style.bg, Some(Color::Blue));
    }

    #[test]
    fn table_selection_navigation() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let rows = vec![
            Row::new(vec!["1".to_string()]),
            Row::new(vec!["2".to_string()]),
            Row::new(vec!["3".to_string()]),
        ];
        let mut table = Table::new(columns).rows(rows).selected(0);

        table.next();
        assert_eq!(table.selected, Some(1));

        table.next();
        assert_eq!(table.selected, Some(2));

        table.next(); // Should stop at last
        assert_eq!(table.selected, Some(2));

        table.previous();
        assert_eq!(table.selected, Some(1));

        table.previous();
        table.previous();
        assert_eq!(table.selected, Some(0));

        table.previous(); // Should stop at first
        assert_eq!(table.selected, Some(0));
    }

    #[test]
    fn table_empty_render() {
        let columns = vec![Column::new("Col", Constraint::Length(10))];
        let table = Table::new(columns);
        let area = Rect::new(0, 0, 20, 5);
        let el = table.render(area);

        assert!(matches!(el.kind, ElementKind::Column));
    }

    #[test]
    fn table_with_data_render() {
        let columns = vec![
            Column::new("ID", Constraint::Length(3)),
            Column::new("Name", Constraint::Length(10)),
        ];
        let table = Table::new(columns)
            .row(Row::new(vec!["1".to_string(), "Alice".to_string()]))
            .row(Row::new(vec!["2".to_string(), "Bob".to_string()]));
        let area = Rect::new(0, 0, 20, 5);
        let el = table.render(area);

        assert!(matches!(el.kind, ElementKind::Column));
    }

    #[test]
    fn table_min_size() {
        let columns = vec![
            Column::new("ID", Constraint::Length(3)),
            Column::new("Name", Constraint::Length(10)),
        ];
        let table = Table::new(columns).row(Row::new(vec!["1".to_string(), "Alice".to_string()]));
        let size = table.min_size();

        assert!(size.width >= 13); // ID(3) + Name(10)
        assert!(size.height >= 2); // Header + 1 row
    }

    #[test]
    fn alignment_left() {
        let aligned = Alignment::Left.align("test", 10);
        assert_eq!(aligned, "test      ");
    }

    #[test]
    fn alignment_right() {
        let aligned = Alignment::Right.align("test", 10);
        assert_eq!(aligned, "      test");
    }

    #[test]
    fn alignment_center() {
        let aligned = Alignment::Center.align("test", 10);
        assert_eq!(aligned, "   test   ");
    }

    #[test]
    fn alignment_long_text() {
        let aligned = Alignment::Left.align("verylongtext", 5);
        assert_eq!(aligned, "verylongtext");
    }
}
