//! Chart widget for rendering simple data visualizations

use crate::element::{Element, ElementKind};
use crate::layout::{Rect, Size};
use crate::style::{Color, Style};
use crate::widgets::Widget;

/// Chart dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Data points
    pub data: Vec<f64>,
    /// Dataset name/label
    pub name: String,
    /// Color
    pub color: Color,
    /// Character to use for drawing
    pub symbol: char,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(name: impl Into<String>, data: Vec<f64>) -> Self {
        Dataset {
            name: name.into(),
            data,
            color: Color::Cyan,
            symbol: '█',
        }
    }

    /// Set color
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set symbol
    pub fn symbol(mut self, symbol: char) -> Self {
        self.symbol = symbol;
        self
    }

    /// Get min/max values
    pub fn range(&self) -> (f64, f64) {
        if self.data.is_empty() {
            return (0.0, 1.0);
        }
        let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if min == max {
            (min, min + 1.0)
        } else {
            (min, max)
        }
    }
}

/// Chart type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    /// Bar chart
    Bar,
    /// Line chart
    Line,
    /// Scatter plot
    Scatter,
}

/// Chart widget
#[derive(Debug, Clone)]
pub struct Chart {
    /// Chart type
    chart_type: ChartType,
    /// Datasets
    datasets: Vec<Dataset>,
    /// Show legend
    show_legend: bool,
    /// Show labels
    show_labels: bool,
    /// Chart style
    style: Style,
    /// Axis style
    axis_style: Style,
}

impl Chart {
    /// Create a new bar chart
    pub fn bar() -> Self {
        Chart {
            chart_type: ChartType::Bar,
            datasets: Vec::new(),
            show_legend: true,
            show_labels: true,
            style: Style::default(),
            axis_style: Style::default().fg(Color::DarkGrey),
        }
    }

    /// Create a new line chart
    pub fn line() -> Self {
        Chart {
            chart_type: ChartType::Line,
            datasets: Vec::new(),
            show_legend: true,
            show_labels: true,
            style: Style::default(),
            axis_style: Style::default().fg(Color::DarkGrey),
        }
    }

    /// Create a new scatter chart
    pub fn scatter() -> Self {
        Chart {
            chart_type: ChartType::Scatter,
            datasets: Vec::new(),
            show_legend: true,
            show_labels: true,
            style: Style::default(),
            axis_style: Style::default().fg(Color::DarkGrey),
        }
    }

    /// Add a dataset
    pub fn dataset(mut self, dataset: Dataset) -> Self {
        self.datasets.push(dataset);
        self
    }

    /// Show/hide legend
    pub fn show_legend(mut self, show: bool) -> Self {
        self.show_legend = show;
        self
    }

    /// Show/hide labels
    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Set style
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Set axis style
    pub fn axis_style(mut self, style: Style) -> Self {
        self.axis_style = style;
        self
    }

    /// Render bar chart
    fn render_bar(&self, area: Rect) -> Element {
        if self.datasets.is_empty() || area.width < 4 || area.height < 4 {
            return Element {
                id: None,
                kind: ElementKind::Text("No data".to_string()),
                style: self.style,
                children: vec![],
                layout: None,
            };
        }

        let mut lines = Vec::new();
        let chart_height = area.height.saturating_sub(2);
        let _chart_width = area.width.saturating_sub(4);

        // Find global range
        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;
        for ds in &self.datasets {
            let (min, max) = ds.range();
            global_min = global_min.min(min);
            global_max = global_max.max(max);
        }
        let range = (global_max - global_min).max(1.0);

        // Render bars
        for row in 0..chart_height {
            let mut line = String::new();
            for ds in &self.datasets {
                if ds.data.is_empty() {
                    continue;
                }
                let bar_height = ((ds.data[0] - global_min) / range * chart_height as f64) as usize;
                if (chart_height.saturating_sub(row) as usize) <= bar_height {
                    line.push(ds.symbol);
                } else {
                    line.push(' ');
                }
            }
            lines.push(Element {
                id: None,
                kind: ElementKind::Text(line),
                style: self.style,
                children: vec![],
                layout: None,
            });
        }

        Element::column(lines)
    }

    /// Render line chart
    fn render_line(&self, area: Rect) -> Element {
        if self.datasets.is_empty() || area.width < 4 || area.height < 4 {
            return Element {
                id: None,
                kind: ElementKind::Text("No data".to_string()),
                style: self.style,
                children: vec![],
                layout: None,
            };
        }

        let chart_height = area.height.saturating_sub(2);
        let chart_width = area.width.saturating_sub(4);

        // Find global range
        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;
        for ds in &self.datasets {
            let (min, max) = ds.range();
            global_min = global_min.min(min);
            global_max = global_max.max(max);
        }
        let range = (global_max - global_min).max(1.0);

        // Build grid
        let mut grid = vec![vec![' '; chart_width as usize]; chart_height as usize];

        for ds in &self.datasets {
            let points: Vec<(usize, usize)> = ds
                .data
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| {
                    let x = (i * chart_width as usize) / ds.data.len().max(1);
                    let y = (chart_height.saturating_sub(1) as usize)
                        - ((v - global_min) / range * (chart_height.saturating_sub(1)) as f64)
                            as usize;
                    if x < chart_width as usize && y < chart_height as usize {
                        Some((x, y))
                    } else {
                        None
                    }
                })
                .collect();

            // Draw points and lines
            for (i, &(x, y)) in points.iter().enumerate() {
                grid[y][x] = ds.symbol;
                if i > 0 {
                    let prev = points[i - 1];
                    // Simple line drawing
                    let dx = x as isize - prev.0 as isize;
                    let dy = y as isize - prev.1 as isize;
                    if dx.abs() > dy.abs() {
                        let step = if dx > 0 { 1 } else { -1 };
                        let mut cx = prev.0 as isize;
                        while cx != x as isize {
                            cx += step;
                            let cy = prev.1 as isize + dy * (cx - prev.0 as isize) / dx.max(1);
                            if cx >= 0
                                && cx < chart_width as isize
                                && cy >= 0
                                && cy < chart_height as isize
                            {
                                grid[cy as usize][cx as usize] = '─';
                            }
                        }
                    }
                }
            }
        }

        // Convert grid to elements
        let lines: Vec<Element> = grid
            .into_iter()
            .map(|row| Element {
                id: None,
                kind: ElementKind::Text(row.into_iter().collect::<String>()),
                style: self.style,
                children: vec![],
                layout: None,
            })
            .collect();

        Element::column(lines)
    }

    /// Render scatter chart
    fn render_scatter(&self, area: Rect) -> Element {
        if self.datasets.is_empty() || area.width < 4 || area.height < 4 {
            return Element {
                id: None,
                kind: ElementKind::Text("No data".to_string()),
                style: self.style,
                children: vec![],
                layout: None,
            };
        }

        let chart_height = area.height.saturating_sub(2);
        let chart_width = area.width.saturating_sub(4);

        let mut grid = vec![vec![' '; chart_width as usize]; chart_height as usize];

        for ds in &self.datasets {
            let x_range = ds.data.len() as f64;
            let (y_min, y_max) = ds.range();
            let y_range = (y_max - y_min).max(1.0);

            for (i, &v) in ds.data.iter().enumerate() {
                let x = ((i as f64 / x_range) * (chart_width.saturating_sub(1)) as f64) as usize;
                let y = (chart_height.saturating_sub(1) as usize)
                    - (((v - y_min) / y_range) * (chart_height.saturating_sub(1)) as f64) as usize;
                if x < chart_width as usize && y < chart_height as usize {
                    grid[y][x] = '●';
                }
            }
        }

        let lines: Vec<Element> = grid
            .into_iter()
            .map(|row| Element {
                id: None,
                kind: ElementKind::Text(row.into_iter().collect::<String>()),
                style: self.style,
                children: vec![],
                layout: None,
            })
            .collect();

        Element::column(lines)
    }

    /// Build legend element
    fn build_legend(&self) -> Element {
        let items: Vec<Element> = self
            .datasets
            .iter()
            .map(|ds| {
                Element::row(vec![
                    Element::text(ds.symbol.to_string()).color(ds.color),
                    Element::text(format!(" {}", ds.name)),
                ])
            })
            .collect();
        Element::column(items)
    }
}

impl Widget for Chart {
    fn render(&self, area: Rect) -> Element {
        let chart_element = match self.chart_type {
            ChartType::Bar => self.render_bar(area),
            ChartType::Line => self.render_line(area),
            ChartType::Scatter => self.render_scatter(area),
        };

        if self.show_legend && !self.datasets.is_empty() {
            Element::row(vec![chart_element, self.build_legend()])
        } else {
            chart_element
        }
    }

    fn min_size(&self) -> Size {
        Size {
            width: 20,
            height: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_creation() {
        let ds = Dataset::new("Test", vec![1.0, 2.0, 3.0]);
        assert_eq!(ds.name, "Test");
        assert_eq!(ds.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dataset_range() {
        let ds = Dataset::new("Test", vec![5.0, 1.0, 3.0, 10.0, 2.0]);
        let (min, max) = ds.range();
        assert_eq!(min, 1.0);
        assert_eq!(max, 10.0);
    }

    #[test]
    fn chart_building() {
        let chart = Chart::bar()
            .dataset(Dataset::new("A", vec![1.0, 2.0, 3.0]))
            .dataset(Dataset::new("B", vec![2.0, 3.0, 4.0]));

        assert_eq!(chart.datasets.len(), 2);
        assert!(matches!(chart.chart_type, ChartType::Bar));
    }

    #[test]
    fn chart_line() {
        let chart = Chart::line().dataset(Dataset::new("Series", vec![1.0, 2.0, 3.0]));
        assert!(matches!(chart.chart_type, ChartType::Line));
    }

    #[test]
    fn chart_scatter() {
        let chart = Chart::scatter().dataset(Dataset::new("Points", vec![1.0, 2.0, 3.0]));
        assert!(matches!(chart.chart_type, ChartType::Scatter));
    }

    #[test]
    fn chart_render_empty() {
        let chart = Chart::bar();
        let area = Rect::new(0, 0, 20, 10);
        let el = chart.render(area);
        assert!(matches!(el.kind, crate::element::ElementKind::Column));
    }

    #[test]
    fn chart_with_legend() {
        let chart = Chart::bar()
            .dataset(Dataset::new("A", vec![1.0, 2.0, 3.0]))
            .show_legend(true);
        let area = Rect::new(0, 0, 40, 10);
        let el = chart.render(area);
        // Should be row with chart and legend
        assert!(matches!(el.kind, crate::element::ElementKind::Row));
    }
}
