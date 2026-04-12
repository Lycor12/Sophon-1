//! Counter Example - Demonstrates use_state hook
//!
//! Run with: cargo run --example counter --package sophon-tui

use sophon_tui::{clear_hooks, init_hooks, use_state, Color, Element, ElementKind, Style};

/// Counter component using use_state hook
fn counter_component() -> Element {
    // Initialize state with count = 0
    let (count, set_count) = use_state(0);

    // Build the UI
    Element::column(vec![
        Element::text("Counter Example").with_style(|s| s.fg(Color::Cyan).bold()),
        Element::text(format!("Current count: {}", count)),
        Element::row(vec![
            Element::text("[+] Increment").with_style(|s| s.fg(Color::Green)),
            Element::text(" [-] Decrement").with_style(|s| s.fg(Color::Red)),
            Element::text(" [q] Quit").with_style(|s| s.fg(Color::Yellow)),
        ]),
    ])
}

/// Alternative: Counter with step size
fn counter_with_step(initial: i32, step: i32) -> Element {
    let (count, _set_count) = use_state(initial);

    Element::column(vec![
        Element::text(format!("Counter (step: {})", step)).with_style(|s| s.fg(Color::Magenta)),
        Element::text(format!("Value: {}", count)),
        Element::text(format!(
            "Next values: {}, {}",
            count + step,
            count + 2 * step
        )),
    ])
}

/// Counter with history tracking
fn counter_with_history() -> Element {
    let (count, _set_count) = use_state(0);
    let (history, _set_history) = use_state::<Vec<i32>>(vec![]);

    // Add current count to history (simulated)
    let mut history_display = format!("History: {:?}", history);
    if history_display.len() > 50 {
        history_display = format!("{}...", &history_display[..50]);
    }

    Element::column(vec![
        Element::text("Counter with History").with_style(|s| s.fg(Color::Cyan).bold()),
        Element::text(format!("Count: {}", count)),
        Element::text(history_display).with_style(|s| s.fg(Color::DarkGrey)),
    ])
}

fn main() {
    println!("Counter Example - TUI Hooks Demo");
    println!("=================================");

    // Initialize hooks
    init_hooks();

    // Render components (demonstrate structure)
    let counter = counter_component();
    println!("\nCounter component structure:");
    println!("  Kind: {:?}", counter.kind);
    println!("  Children: {}", counter.children.len());

    let counter_step = counter_with_step(10, 5);
    println!("\nCounter with step structure:");
    println!("  Kind: {:?}", counter_step.kind);
    println!("  Children: {}", counter_step.children.len());

    // Cleanup
    clear_hooks();

    println!("\nDone!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_renders() {
        // Initialize hooks system
        init_hooks();

        // Render the component
        let element = counter_component();

        // Verify structure
        assert!(matches!(element.kind, ElementKind::Column));
        assert_eq!(element.children.len(), 3);

        // Cleanup
        clear_hooks();
    }

    #[test]
    fn test_counter_with_step() {
        init_hooks();

        let element = counter_with_step(10, 5);

        assert!(matches!(element.kind, ElementKind::Column));
        assert_eq!(element.children.len(), 3);

        clear_hooks();
    }
}
