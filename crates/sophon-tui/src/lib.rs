//! # Sophon TUI - React+Ink Inspired Terminal UI Framework
//!
//! A custom-built, zero-dependency terminal UI framework for the Sophon AGI system.
//! Inspired by React's component model and Ink's terminal rendering.
//!
//! ## Core Concepts
//!
//! - **Components**: Functions that return `Element` trees (like React components)
//! - **Hooks**: `use_state`, `use_effect`, `use_memo`, `use_ref` for state management
//! - **Reconciliation**: Efficient diffing and patching of the virtual element tree
//! - **Effects**: Side-effect management with cleanup
//! - **Terminal**: Buffered output with ANSI escape sequences
//!
//! ## Example
//!
//! ```rust
//! use sophon_tui::{render, component, use_state, Element, Color};
//!
//! #[component]
//! fn counter() -> Element {
//!     let (count, set_count) = use_state(0);
//!     
//!     Element::column(vec![
//!         Element::text("Counter App").bold().color(Color::Cyan),
//!         Element::text(format!("Count: {}", count)),
//!         Element::row(vec![
//!             Element::button("+", move || set_count(count + 1)),
//!             Element::button("-", move || set_count(count - 1)),
//!         ]),
//!     ])
//! }
//!
//! fn main() {
//!     render(counter);
//! }
//! ```

#![forbid(unsafe_code)]

mod ansi;
mod component;
mod effect;
mod element;
mod hook;
mod input;
mod layout;
mod render;
mod renderer;
mod style;
mod terminal;

pub use ansi::{style_to_ansi, AnsiBuffer, AnsiCode, RESET};
pub use component::{Component, ComponentId, ComponentRegistry, RenderContext};
pub use effect::{Dependency, Effect, EffectQueue, EffectType};
pub use element::{Element, ElementId, ElementKind};
pub use hook::{HookId, HookState, Hooks};
pub use input::{Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent};
pub use layout::{Constraint, Layout, Rect, Size};
pub use render::{render_element, render_to_string, RenderBuffer};
pub use renderer::{RenderMode, Renderer};
pub use style::{BorderStyle, Color, Style, TextWrap};
pub use terminal::{Capabilities, Terminal, TerminalBuffer};

/// Re-export commonly used functions
pub mod prelude {
    pub use crate::element::{Element, ElementKind};
    pub use crate::layout::{Constraint, Rect};
    pub use crate::render::render_to_string;
    pub use crate::style::{BorderStyle, Color, Style};
}

pub mod widgets;

/// Current TUI version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the TUI system
pub fn init() {
    terminal::init_terminal();
}

/// Cleanup TUI resources
pub fn cleanup() {
    terminal::cleanup_terminal();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_tree_building() {
        let tree = Element::column(vec![Element::text("Hello"), Element::text("World")]);

        assert!(matches!(tree.kind, ElementKind::Column));
    }

    #[test]
    fn style_chaining() {
        let styled = Element::text("test").color(Color::Red).bold().underline();

        assert_eq!(styled.style.fg, Some(Color::Red));
        assert!(styled.style.bold);
        assert!(styled.style.underline);
    }
}
