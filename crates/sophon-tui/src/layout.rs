//! Layout engine - calculates positions and sizes for elements

/// A rectangle in terminal coordinates (x, y, width, height)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    /// Create a new rectangle
    pub fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if point is inside rectangle
    pub fn contains(&self, x: u16, y: u16) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get area
    pub fn area(&self) -> u16 {
        self.width * self.height
    }

    /// Get right edge (exclusive)
    pub fn right(&self) -> u16 {
        self.x + self.width
    }

    /// Get bottom edge (exclusive)
    pub fn bottom(&self) -> u16 {
        self.y + self.height
    }

    /// Shrink by amount on all sides
    pub fn shrink(&self, amount: u16) -> Self {
        Rect {
            x: self.x + amount,
            y: self.y + amount,
            width: self.width.saturating_sub(amount * 2),
            height: self.height.saturating_sub(amount * 2),
        }
    }
}

/// Size in terminal cells
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Size {
    pub width: usize,
    pub height: usize,
}

/// Layout constraints
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constraint {
    /// Fixed size
    Length(u16),
    /// Percentage of available space
    Percentage(u16),
    /// Minimum size
    Min(u16),
    /// Maximum size
    Max(u16),
    /// Take remaining space
    Fill,
}

/// Layout engine
pub struct LayoutEngine;

impl LayoutEngine {
    /// Solve layout constraints to produce final sizes
    pub fn solve(constraints: &[Constraint], available: u16) -> Vec<u16> {
        let mut sizes = vec![0u16; constraints.len()];
        let mut remaining = available;

        // First pass: handle fixed constraints
        for (i, c) in constraints.iter().enumerate() {
            match c {
                Constraint::Length(n) => {
                    sizes[i] = *n;
                    remaining = remaining.saturating_sub(*n);
                }
                Constraint::Min(n) => {
                    sizes[i] = *n;
                    remaining = remaining.saturating_sub(*n);
                }
                _ => {}
            }
        }

        // Second pass: handle percentages
        for (i, c) in constraints.iter().enumerate() {
            if let Constraint::Percentage(p) = c {
                let size = (available as u32 * *p as u32 / 100) as u16;
                sizes[i] = size;
                remaining = remaining.saturating_sub(size);
            }
        }

        // Third pass: handle max constraints
        for (i, c) in constraints.iter().enumerate() {
            if let Constraint::Max(n) = c {
                if sizes[i] > *n {
                    sizes[i] = *n;
                }
            }
        }

        // Final pass: distribute remaining to Fill constraints
        let fill_count = constraints
            .iter()
            .filter(|c| matches!(c, Constraint::Fill))
            .count() as u16;
        if fill_count > 0 {
            let per_fill = remaining / fill_count;
            let mut extra = remaining % fill_count;
            for (i, c) in constraints.iter().enumerate() {
                if matches!(c, Constraint::Fill) {
                    sizes[i] = per_fill
                        + if extra > 0 {
                            extra -= 1;
                            1
                        } else {
                            0
                        };
                }
            }
        }

        sizes
    }
}

/// Layout direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Horizontal,
    Vertical,
}

/// Layout configuration
#[derive(Debug, Clone)]
pub struct Layout {
    pub direction: Direction,
    pub constraints: Vec<Constraint>,
}

impl Layout {
    /// Create a horizontal layout
    pub fn horizontal(constraints: Vec<Constraint>) -> Self {
        Layout {
            direction: Direction::Horizontal,
            constraints,
        }
    }

    /// Create a vertical layout
    pub fn vertical(constraints: Vec<Constraint>) -> Self {
        Layout {
            direction: Direction::Vertical,
            constraints,
        }
    }

    /// Split a rectangle into children based on constraints
    pub fn split(&self, area: Rect) -> Vec<Rect> {
        let sizes = LayoutEngine::solve(
            &self.constraints,
            match self.direction {
                Direction::Horizontal => area.width,
                Direction::Vertical => area.height,
            },
        );

        let mut result = Vec::new();
        let mut current = 0u16;

        for size in sizes {
            let rect = match self.direction {
                Direction::Horizontal => Rect {
                    x: area.x + current,
                    y: area.y,
                    width: size,
                    height: area.height,
                },
                Direction::Vertical => Rect {
                    x: area.x,
                    y: area.y + current,
                    width: area.width,
                    height: size,
                },
            };
            result.push(rect);
            current += size;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_contains() {
        let rect = Rect::new(10, 10, 20, 10);
        assert!(rect.contains(15, 15));
        assert!(!rect.contains(5, 15));
        assert!(!rect.contains(15, 25));
    }

    #[test]
    fn layout_horizontal() {
        let layout = Layout::horizontal(vec![
            Constraint::Length(10),
            Constraint::Fill,
            Constraint::Length(5),
        ]);

        let areas = layout.split(Rect::new(0, 0, 100, 1));
        assert_eq!(areas.len(), 3);
        assert_eq!(areas[0].width, 10);
        assert_eq!(areas[1].width, 85); // 100 - 10 - 5
        assert_eq!(areas[2].width, 5);
    }

    #[test]
    fn layout_vertical() {
        let layout = Layout::vertical(vec![Constraint::Percentage(30), Constraint::Percentage(70)]);

        let areas = layout.split(Rect::new(0, 0, 80, 100));
        assert_eq!(areas.len(), 2);
        assert_eq!(areas[0].height, 30);
        assert_eq!(areas[1].height, 70);
    }
}
