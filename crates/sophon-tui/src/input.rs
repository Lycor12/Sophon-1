//! Input event handling for keyboard and mouse
//!
//! Provides cross-platform input event processing with support for:
//! - Keyboard input (including special keys)
//! - Mouse events (clicks, movement, scroll)
//! - Resize events

use std::io;
use std::time::{Duration, Instant};

/// Input event types
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Keyboard key press
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Terminal resize
    Resize(u16, u16),
    /// Paste event
    Paste(String),
    /// Focus gained
    FocusGained,
    /// Focus lost
    FocusLost,
    /// Tick event (for animations)
    Tick,
}

/// Keyboard event
#[derive(Debug, Clone, PartialEq)]
pub struct KeyEvent {
    /// The key code
    pub code: KeyCode,
    /// Modifier keys
    pub modifiers: KeyModifiers,
    /// Key event kind
    pub kind: KeyEventKind,
}

impl KeyEvent {
    /// Create a new key event
    pub fn new(code: KeyCode) -> Self {
        KeyEvent {
            code,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
        }
    }

    /// Create a key event with modifiers
    pub fn with_modifiers(code: KeyCode, modifiers: KeyModifiers) -> Self {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
        }
    }

    /// Check if Ctrl key is pressed
    pub fn ctrl(&self) -> bool {
        self.modifiers.contains(KeyModifiers::CONTROL)
    }

    /// Check if Alt key is pressed
    pub fn alt(&self) -> bool {
        self.modifiers.contains(KeyModifiers::ALT)
    }

    /// Check if Shift key is pressed
    pub fn shift(&self) -> bool {
        self.modifiers.contains(KeyModifiers::SHIFT)
    }
}

/// Key codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// Backspace key
    Backspace,
    /// Enter key
    Enter,
    /// Left arrow
    Left,
    /// Right arrow
    Right,
    /// Up arrow
    Up,
    /// Down arrow
    Down,
    /// Home key
    Home,
    /// End key
    End,
    /// Page up
    PageUp,
    /// Page down
    PageDown,
    /// Tab key
    Tab,
    /// Delete key
    Delete,
    /// Insert key
    Insert,
    /// Escape key
    Esc,
    /// Function key F1-F24
    F(u8),
    /// Character key
    Char(char),
    /// Null
    Null,
}

/// Key modifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeyModifiers(u8);

impl KeyModifiers {
    /// Shift key
    pub const SHIFT: Self = KeyModifiers(0b0001);
    /// Control key
    pub const CONTROL: Self = KeyModifiers(0b0010);
    /// Alt key
    pub const ALT: Self = KeyModifiers(0b0100);
    /// Super/Meta/Windows key
    pub const SUPER: Self = KeyModifiers(0b1000);

    /// Empty modifiers
    pub const fn empty() -> Self {
        KeyModifiers(0)
    }

    /// Check if modifiers contains flag
    pub fn contains(&self, other: KeyModifiers) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Add modifiers
    pub fn add(&mut self, other: KeyModifiers) {
        self.0 |= other.0
    }

    /// Check if no modifiers are set
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for KeyModifiers {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        KeyModifiers(self.0 | rhs.0)
    }
}

/// Key event kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyEventKind {
    /// Key press
    Press,
    /// Key release
    Release,
    /// Key repeat
    Repeat,
}

/// Mouse event
#[derive(Debug, Clone, PartialEq)]
pub struct MouseEvent {
    /// Mouse event kind
    pub kind: MouseEventKind,
    /// Column position (0-indexed)
    pub column: u16,
    /// Row position (0-indexed)
    pub row: u16,
    /// Modifier keys
    pub modifiers: KeyModifiers,
}

/// Mouse event kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseEventKind {
    /// Mouse button down
    Down(MouseButton),
    /// Mouse button up
    Up(MouseButton),
    /// Mouse drag
    Drag(MouseButton),
    /// Mouse moved
    Moved,
    /// Scroll up
    ScrollUp,
    /// Scroll down
    ScrollDown,
    /// Scroll left
    ScrollLeft,
    /// Scroll right
    ScrollRight,
}

/// Mouse buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    /// Left button
    Left,
    /// Right button
    Right,
    /// Middle button
    Middle,
}

/// Event handler trait
pub trait EventHandler {
    /// Handle an event, returns true if event was consumed
    fn handle_event(&mut self, event: Event) -> bool;
}

/// Simple event listener
pub struct EventListener {
    handlers: Vec<Box<dyn FnMut(&Event) -> bool>>,
}

impl EventListener {
    /// Create a new event listener
    pub fn new() -> Self {
        EventListener {
            handlers: Vec::new(),
        }
    }

    /// Add an event handler
    pub fn on_event<F>(&mut self, handler: F)
    where
        F: FnMut(&Event) -> bool + 'static,
    {
        self.handlers.push(Box::new(handler));
    }

    /// Dispatch an event to all handlers
    pub fn dispatch(&mut self, event: &Event) -> bool {
        for handler in &mut self.handlers {
            if handler(event) {
                return true;
            }
        }
        false
    }
}

impl Default for EventListener {
    fn default() -> Self {
        Self::new()
    }
}

/// Event source trait for reading events from different sources
pub trait EventSource {
    /// Try to read an event (non-blocking)
    fn try_read(&mut self) -> io::Result<Option<Event>>;

    /// Read an event (blocking)
    fn read(&mut self) -> io::Result<Event>;

    /// Set timeout for blocking reads
    fn set_timeout(&mut self, timeout: Duration);
}

/// Polling event source
pub struct PollEventSource {
    timeout: Duration,
    tick_rate: Duration,
    last_tick: Instant,
}

impl PollEventSource {
    /// Create a new poll event source
    pub fn new(tick_rate: Duration) -> Self {
        PollEventSource {
            timeout: Duration::from_millis(100),
            tick_rate,
            last_tick: Instant::now(),
        }
    }

    /// Check if it's time for a tick
    pub fn should_tick(&mut self) -> bool {
        if self.last_tick.elapsed() >= self.tick_rate {
            self.last_tick = Instant::now();
            true
        } else {
            false
        }
    }
}

impl Default for PollEventSource {
    fn default() -> Self {
        Self::new(Duration::from_millis(16)) // ~60 FPS
    }
}

/// Helper function to parse escape sequences (basic implementation)
pub fn parse_escape_sequence(buf: &[u8]) -> Option<(Event, usize)> {
    if buf.is_empty() {
        return None;
    }

    // CSI sequences
    if buf[0] == b'\x1b' && buf.len() >= 3 && buf[1] == b'[' {
        match buf[2] {
            b'A' => return Some((Event::Key(KeyEvent::new(KeyCode::Up)), 3)),
            b'B' => return Some((Event::Key(KeyEvent::new(KeyCode::Down)), 3)),
            b'C' => return Some((Event::Key(KeyEvent::new(KeyCode::Right)), 3)),
            b'D' => return Some((Event::Key(KeyEvent::new(KeyCode::Left)), 3)),
            b'H' => return Some((Event::Key(KeyEvent::new(KeyCode::Home)), 3)),
            b'F' => return Some((Event::Key(KeyEvent::new(KeyCode::End)), 3)),
            b'3' if buf.len() >= 4 && buf[3] == b'~' => {
                return Some((Event::Key(KeyEvent::new(KeyCode::Delete)), 4))
            }
            b'5' if buf.len() >= 4 && buf[3] == b'~' => {
                return Some((Event::Key(KeyEvent::new(KeyCode::PageUp)), 4))
            }
            b'6' if buf.len() >= 4 && buf[3] == b'~' => {
                return Some((Event::Key(KeyEvent::new(KeyCode::PageDown)), 4))
            }
            _ => {}
        }
    }

    // Simple escape
    if buf[0] == b'\x1b' && buf.len() == 1 {
        return Some((Event::Key(KeyEvent::new(KeyCode::Esc)), 1));
    }

    None
}
