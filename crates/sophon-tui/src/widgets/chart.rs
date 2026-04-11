//! Chart widget for rendering simple data visualizations

use crate::element::Element;
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
            axis_style: Style::default().fg(Color::Grey),
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
            axis_style: Style::default().fg(Color::Grey),
        }
    }

    /// Add a dataset
    pub fn dataset(mut self, dataset: Dataset) -> Self {
        self.datasets.push(dataset);
        self
    }

    /// Set legend visibility
    pub fn show_legend(mut self, show: bool) -> Self {
        self.show_legend = show;
        self
    }

    /// Set labels visibility
    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Get global data range
    fn global_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for dataset in &self.datasets {
            let (dmin, dmax) = dataset.range();
            min = min.min(dmin);
            max = max.max(dmax);
        }

        if min == f64::INFINITY {
            (0.0, 1.0)
        } else if min == max {
            (min, min + 1.0)
        } else {
            (min, max)
        }
    }

    fn render_bar(&self, area: Rect) -> Element {
        if self.datasets.is_empty() {
            return Element::Text {
                content: "No data".to_string(),
                style: self.style,
            };
        }

        let (min, max) = self.global_range();
        let range = max - min;

        let chart_height =
            area.height
                .saturating_sub(if self.show_legend { 2 } else { 0 }) as usize;
        let bar_width = 1u16;
        let max_bars = (area.width / (bar_width + 1)) as usize;

        let dataset = &self.datasets[0]; // Simple: render first dataset
        let data = &dataset.data;

        if data.is_empty() {
            return Element::Text {
                content: "No data".to_string(),
                style: self.style,
            };
        }

        // Sample data to fit width
        let sample_every = (data.len() / max_bars).max(1);
        let sampled: Vec<f64> = data
            .chunks(sample_every)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .take(max_bars)
            .collect();

        // Build chart rows from top down
        let mut rows: Vec<Vec<char>> = vec![vec![' '; sampled.len()]; chart_height];

        for (i, &value) in sampled.iter().enumerate() {
            let normalized = ((value - min) / range).max(0.0).min(1.0);
            let bar_height = (normalized * chart_height as f64) as usize;

            for h in 0..bar_height {
                let row = chart_height - 1 - h;
                if row < chart_height {
                    rows[row][i] = dataset.symbol;
                }
            }
        }

        // Convert to elements
        let mut children: Vec<Element> = rows
            .into_iter()
            .map(|row| Element::Text {
                content: row.into_iter().collect(),
                style: Style::default().fg(dataset.color),
            })
            .collect();

        // Add legend
        if self.show_legend {
            children.push(Element::Text {
                content: format!("{}: {} to {}", dataset.name, min, max),
                style: Style::default().fg(dataset.color),
            });
        }

        Element::Container {
            children,
            layout: crate::layout::Layout::vertical(
                std::iter::repeat(crate::layout::Constraint::Length(1))
                    .take(children.len())
                    .collect(),
            ),
            style: self.style,
        }
    }

    fn render_line(&self, area: Rect) -> Element {
        // Simplified line chart - just show data points
        if self.datasets.is_empty() {
            return Element::Text {
                content: "No data".to_string(),
                style: self.style,
            };
        }

        let mut content = String::new();
        for dataset in &self.datasets {
            content.push_str(&format!("{}: {:?}\n", dataset.name, dataset.data));
        }

        Element::Text {
            content: content.trim_end().to_string(),
            style: self.style,
        }
    }
}

impl Widget for Chart {
    fn render(&self, area: Rect) -> Element {
        match self.chart_type {
            ChartType::Bar => self.render_bar(area),
            ChartType::Line => self.render_line(area),
            ChartType::Scatter => self.render_line(area), // Simplified
        }
    }

    fn min_size(&self) -> Size {
        Size {
            width: 10,
            height: 5,
        }
    }
}

/// Sparkline - minimal chart for inline use
#[derive(Debug, Clone)]
pub struct Sparkline {
    /// Data points
    data: Vec<f64>,
    /// Max value (auto-detected if None)
    max: Option<f64>,
    /// Sparkline characters
    symbols: &'static str,
}

impl Sparkline {
    /// Create a new sparkline
    pub fn new(data: Vec<f64>) -> Self {
        Sparkline {
            data,
            max: None,
            symbols: "▁▂▃▄▅▆▇█",
        }
    }

    /// Set max value
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Render to string
    pub fn render_to_string(&self) -> String {
        if self.data.is_empty() {
            return String::new();
        }

        let max = self
            .max
            .unwrap_or_else(|| self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max));

        let symbols: Vec<char> = self.symbols.chars().collect();
        let symbol_count = symbols.len() as f64;

        self.data
            .iter()
            .map(|&v| {
                if max <= 0.0 {
                    symbols[0]
                } else {
                    let idx = ((v / max) * (symbol_count - 1.0)) as usize;
                    symbols[idx.min(symbols.len() - 1)]
                }
            })
            .collect()
    }
}

impl Widget for Sparkline {
    fn render(&self, _area: Rect) -> Element {
        Element::Text {
            content: self.render_to_string(),
            style: Style::default(),
        }
    }

    fn min_size(&self) -> Size {
        Size {
            width: self.data.len() as u16,
            height: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_range() {
        let ds = Dataset::new("test", vec![1.0, 5.0, 3.0, 10.0]);
        let (min, max) = ds.range();
        assert_eq!(min, 1.0);
        assert_eq!(max, 10.0);
    }

    #[test]
    fn sparkline_render() {
        let spark = Sparkline::new(vec![1.0, 5.0, 10.0]);
        let s = spark.render_to_string();
        assert!(!s.is_empty());
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn chart_dataset() {
        let ds = Dataset::new("sales", vec![10.0, 20.0, 30.0]).color(Color::Green);
        assert_eq!(ds.color, Color::Green);
    }
}
