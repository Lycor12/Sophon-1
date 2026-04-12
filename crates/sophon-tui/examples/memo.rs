//! Memo Example - Demonstrates use_memo hook
//!
//! Run with: cargo run --example memo --package sophon-tui

use sophon_tui::{clear_hooks, init_hooks, use_memo, Color, Element, ElementKind, Style};

/// Expensive computation example
fn expensive_calculation(n: i32) -> i32 {
    // Simulate expensive work
    let mut result = 0;
    for i in 0..n {
        result += i * i;
    }
    result
}

/// Component with memoized computation
fn memoized_component() -> Element {
    let base_value = 1000;

    // Memoize the expensive calculation
    let memoized_result = use_memo(|| expensive_calculation(base_value));

    Element::column(vec![
        Element::text("Memo Example").with_style(|s| s.fg(Color::Cyan).bold()),
        Element::text(format!("Base value: {}", base_value)),
        Element::text(format!(
            "Sum of squares (0 to {}): {}",
            base_value, memoized_result
        )),
        Element::text("This calculation is memoized and only recomputed when dependencies change.")
            .with_style(|s| s.fg(Color::DarkGrey)),
    ])
}

/// Fibonacci calculation (naturally recursive)
fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// Memoized fibonacci component
fn fibonacci_component(n: u32) -> Element {
    // Without memoization, this would be O(2^n) every render
    // With memoization, it's O(n) on first call, O(1) on subsequent renders
    let result = use_memo(|| fibonacci(n));

    Element::column(vec![
        Element::text(format!("Fibonacci({}) = {}", n, result)).with_style(|s| s.fg(Color::Green)),
        Element::text(format!("Calculated fib({})", n)).with_style(|s| s.fg(Color::DarkGrey)),
    ])
}

/// String formatting example
fn formatted_data_component() -> Element {
    let data = vec![1.5, 2.7, 3.14, 4.2, 5.0];

    // Memoize formatted string to avoid reallocation
    let formatted = use_memo(|| {
        data.iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join(", ")
    });

    Element::column(vec![
        Element::text("Formatted Data").color(Color::Magenta),
        Element::text(format!("[{}]", formatted)),
    ])
}

fn main() {
    println!("Memo Example - TUI Hooks Demo");
    println!("==============================");

    init_hooks();

    let memo = memoized_component();
    println!("\nMemoized component:");
    println!("  Kind: {:?}", memo.kind);
    println!("  Children: {}", memo.children.len());

    clear_hooks();
    init_hooks();

    let fib = fibonacci_component(10);
    println!("\nFibonacci component:");
    println!("  Kind: {:?}", fib.kind);
    println!("  Children: {}", fib.children.len());

    clear_hooks();
    init_hooks();

    let formatted = formatted_data_component();
    println!("\nFormatted data component:");
    println!("  Kind: {:?}", formatted.kind);
    println!("  Children: {}", formatted.children.len());

    clear_hooks();
    println!("\nDone!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memoized_calculation() {
        let result = expensive_calculation(100);
        assert_eq!(result, 328350); // sum of squares 0..100
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn test_fibonacci_component() {
        init_hooks();

        let element = fibonacci_component(10);
        assert!(matches!(element.kind, ElementKind::Column));

        clear_hooks();
    }
}
