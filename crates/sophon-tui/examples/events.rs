//! Events Example - Demonstrates event handling
//!
//! Run with: cargo run --example events

use sophon_tui::{
    use_state, Color, Element, Event, EventListener, KeyCode, KeyEvent, KeyModifiers, MouseButton,
    MouseEvent, MouseEventKind, PollEventSource,
};
use std::time::Duration;

/// Key event handler component
fn key_event_component() -> Element {
    let (last_key, _set_last_key) = use_state::<Option<String>>(None);
    let (modifiers, _set_modifiers) = use_state::<String>(String::new());

    Element::column(vec![
        Element::text("Key Event Handler").color(Color::Cyan).bold(),
        Element::text(format!(
            "Last key: {}",
            last_key.as_deref().unwrap_or("None")
        )),
        Element::text(format!("Modifiers: {}", modifiers)),
        Element::text("Press keys to see events").color(Color::DarkGrey),
    ])
}

/// Mouse event handler component
fn mouse_event_component() -> Element {
    let (mouse_pos, _set_mouse_pos) = use_state::<Option<(u16, u16)>>(None);
    let (last_button, _set_last_button) = use_state::<Option<String>>(None);

    Element::column(vec![
        Element::text("Mouse Event Handler")
            .color(Color::Green)
            .bold(),
        Element::text(format!(
            "Position: {:?}",
            mouse_pos.map(|(column, row)| format!("({}, {})", column, row))
        )),
        Element::text(format!(
            "Last button: {}",
            last_button.as_deref().unwrap_or("None")
        )),
        Element::text("Click and move mouse to see events").color(Color::DarkGrey),
    ])
}

/// Combined input handler
fn input_handler_component() -> Element {
    let (event_count, _set_event_count) = use_state(0i32);
    let (event_log, _set_event_log) = use_state::<Vec<String>>(vec![]);

    // Display last 5 events
    let log_display = event_log.join("\n");

    Element::column(vec![
        Element::text("Input Handler").color(Color::Magenta).bold(),
        Element::text(format!("Total events: {}", event_count)),
        Element::text("Recent events:").underline(),
        Element::text(if log_display.is_empty() {
            "No events yet".to_string()
        } else {
            log_display
        })
        .color(Color::DarkGrey),
    ])
}

/// Poll event source demo
fn poll_event_source_demo() -> Element {
    // Create a poll event source
    let source = PollEventSource::new(Duration::from_millis(16));

    let (tick_count, _set_tick_count) = use_state(0i32);

    // Check if it's time for a tick
    let tick = false; // In real usage, this would be checked in the event loop

    let final_count = if tick { tick_count + 1 } else { tick_count };

    Element::column(vec![
        Element::text("Poll Event Source")
            .color(Color::Yellow)
            .bold(),
        Element::text(format!("Tick rate: 16ms (~60 FPS)")),
        Element::text(format!("Tick count: {}", final_count)),
        Element::text("Ticks fire at regular intervals").color(Color::DarkGrey),
    ])
}

/// Event listener demo
fn event_listener_demo() -> Element {
    let mut listener = EventListener::new();

    // Register event handlers
    listener.on_event(|event| {
        match event {
            Event::Key(key) => println!("Key event: {:?}", key.code),
            Event::Mouse(mouse) => println!("Mouse event: {:?}", mouse),
            Event::Resize(w, h) => println!("Resize: {}x{}", w, h),
            Event::Tick => println!("Tick"),
            _ => {}
        }
        false // Event not consumed, pass to next handler
    });

    Element::column(vec![
        Element::text("Event Listener").color(Color::Cyan).bold(),
        Element::text("Multiple handlers registered"),
        Element::text("Events dispatched in order").color(Color::DarkGrey),
    ])
}

/// Example: Create a key event
fn create_key_event_example() -> KeyEvent {
    KeyEvent {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::empty(),
        kind: sophon_tui::KeyEventKind::Press,
    }
}

/// Example: Create a mouse event
fn create_mouse_event_example() -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 10,
        row: 20,
        modifiers: KeyModifiers::empty(),
    }
}

/// Example: Check mouse event kind
fn is_mouse_click(event: &MouseEvent) -> bool {
    matches!(event.kind, MouseEventKind::Down(_))
}

/// Example: Check key
fn is_quit_key(event: &KeyEvent) -> bool {
    matches!(event.code, KeyCode::Char('q')) && event.modifiers == KeyModifiers::empty()
}

fn main() {
    println!("Events Example - TUI Event Handling Demo");
    println!("========================================");

    sophon_tui::init_hooks();

    let key_handler = key_event_component();
    println!("\nKey event component:");
    println!(" Kind: {:?}", key_handler.kind);
    println!(" Children: {}", key_handler.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let mouse_handler = mouse_event_component();
    println!("\nMouse event component:");
    println!(" Kind: {:?}", mouse_handler.kind);
    println!(" Children: {}", mouse_handler.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let input_handler = input_handler_component();
    println!("\nInput handler component:");
    println!(" Kind: {:?}", input_handler.kind);
    println!(" Children: {}", input_handler.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let poll_demo = poll_event_source_demo();
    println!("\nPoll event source demo:");
    println!(" Kind: {:?}", poll_demo.kind);
    println!(" Children: {}", poll_demo.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let listener_demo = event_listener_demo();
    println!("\nEvent listener demo:");
    println!(" Kind: {:?}", listener_demo.kind);
    println!(" Children: {}", listener_demo.children.len());

    sophon_tui::clear_hooks();
    println!("\nDone!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_event_creation() {
        let event = create_key_event_example();
        assert_eq!(event.code, KeyCode::Enter);
    }

    #[test]
    fn test_mouse_event_creation() {
        let event = create_mouse_event_example();
        assert_eq!(event.column, 10);
        assert_eq!(event.row, 20);
        assert!(matches!(
            event.kind,
            MouseEventKind::Down(MouseButton::Left)
        ));
    }

    #[test]
    fn test_quit_key_detection() {
        let quit = KeyEvent {
            code: KeyCode::Char('q'),
            modifiers: KeyModifiers::empty(),
            kind: sophon_tui::KeyEventKind::Press,
        };
        assert!(is_quit_key(&quit));

        let not_quit = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::empty(),
            kind: sophon_tui::KeyEventKind::Press,
        };
        assert!(!is_quit_key(&not_quit));
    }

    #[test]
    fn test_poll_event_source() {
        let source = PollEventSource::new(Duration::from_millis(16));
        // Note: should_tick will depend on timing, so we just verify it compiles
        // In practice, you would mock time or wait
    }

    #[test]
    fn test_event_listener() {
        let mut listener = EventListener::new();

        let mut received = false;
        listener.on_event(|_event| {
            received = true;
            true // Consume the event
        });

        // Dispatch a tick event
        let event = Event::Tick;
        let consumed = listener.dispatch(&event);

        assert!(consumed);
        assert!(received);
    }

    #[test]
    fn test_key_event_component() {
        sophon_tui::init_hooks();

        let element = key_event_component();
        assert!(matches!(element.kind, sophon_tui::ElementKind::Column));

        sophon_tui::clear_hooks();
    }
}
