//! React-style hooks for state management
//!
//! Implements a hook system similar to React's hooks but for synchronous
//! terminal rendering. Hooks are stored in a thread-local registry.

use std::cell::RefCell;

/// Hook identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookId(pub usize);

/// Hook state storage (manual Debug because of Fn trait objects)
pub enum HookState {
    /// use_state: (value, setter)
    State(Box<dyn std::any::Any>),
    /// use_ref: mutable reference
    Ref(Box<dyn std::any::Any>),
    /// use_memo: cached value
    Memo {
        value: Box<dyn std::any::Any>,
        deps: Vec<MemoDep>,
    },
    /// use_effect: effect queue
    Effect {
        callback: Box<dyn Fn()>,
        cleanup: Option<Box<dyn Fn()>>,
    },
}

impl std::fmt::Debug for HookState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookState::State(_) => f.debug_struct("State").field("type", &"Any").finish(),
            HookState::Ref(_) => f.debug_struct("Ref").field("type", &"Any").finish(),
            HookState::Memo { .. } => f.debug_struct("Memo").finish(),
            HookState::Effect {
                callback: _,
                cleanup,
            } => f
                .debug_struct("Effect")
                .field("has_callback", &true)
                .field("has_cleanup", &cleanup.is_some())
                .finish(),
        }
    }
}

/// Collection of hooks for a component
#[derive(Debug, Default)]
pub struct Hooks {
    states: Vec<HookState>,
    current_index: RefCell<usize>,
}

impl Hooks {
    /// Create new hooks collection
    pub fn new() -> Self {
        Hooks {
            states: Vec::new(),
            current_index: RefCell::new(0),
        }
    }

    /// Reset the hook index (called before each render)
    pub fn reset(&self) {
        *self.current_index.borrow_mut() = 0;
    }

    /// Get next hook index
    fn next_index(&self) -> usize {
        let idx = *self.current_index.borrow();
        *self.current_index.borrow_mut() += 1;
        idx
    }

    /// Get or create a hook at current index
    pub fn get_or_create<T: 'static>(&mut self, default: impl FnOnce() -> T) -> &mut T {
        let idx = self.next_index();

        if idx >= self.states.len() {
            self.states.push(HookState::State(Box::new(default())));
        }

        match &mut self.states[idx] {
            HookState::State(val) => val.downcast_mut::<T>().unwrap(),
            _ => panic!("Hook type mismatch at index {}", idx),
        }
    }

    /// Get hook count
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Clear all hooks
    pub fn clear(&mut self) {
        self.states.clear();
        *self.current_index.borrow_mut() = 0;
    }
}

/// Hook trait for custom hooks
pub trait Hook: 'static {
    type Value;
    fn init(&self) -> Self::Value;
    fn update(&mut self, value: Self::Value);
}

/// use_state hook - returns a value and setter function
pub fn use_state<T: Clone + 'static>(initial: T) -> (T, impl Fn(T)) {
    let idx = HOOKS.with(|h| {
        let hooks = h.borrow_mut();
        hooks.next_index()
    });

    let needs_init = HOOKS.with(|h| {
        let hooks = h.borrow();
        idx >= hooks.states.len()
    });

    if needs_init {
        HOOKS.with(|h| {
            let mut hooks = h.borrow_mut();
            hooks
                .states
                .push(HookState::State(Box::new(initial.clone())));
        });
    }

    let value = HOOKS.with(|h| {
        let hooks = h.borrow();
        match &hooks.states[idx] {
            HookState::State(val) => val.downcast_ref::<T>().unwrap().clone(),
            _ => panic!("use_state hook type mismatch"),
        }
    });

    let setter = move |new_val: T| {
        HOOKS.with(|h| {
            let mut hooks = h.borrow_mut();
            if idx < hooks.states.len() {
                hooks.states[idx] = HookState::State(Box::new(new_val));
            }
        });
    };

    (value, setter)
}

/// use_ref hook - returns a mutable reference
pub fn use_ref<T: Clone + 'static>(initial: T) -> RefCell<T> {
    let idx = HOOKS.with(|h| {
        let hooks = h.borrow_mut();
        hooks.next_index()
    });

    let needs_init = HOOKS.with(|h| {
        let hooks = h.borrow();
        idx >= hooks.states.len()
    });

    if needs_init {
        HOOKS.with(|h| {
            let mut hooks = h.borrow_mut();
            hooks
                .states
                .push(HookState::Ref(Box::new(RefCell::new(initial))));
        });
    }

    HOOKS.with(|h| {
        let hooks = h.borrow();
        match &hooks.states[idx] {
            HookState::Ref(val) => {
                // Clone the value from the RefCell
                let inner_val = val.downcast_ref::<RefCell<T>>().unwrap().borrow().clone();
                RefCell::new(inner_val)
            }
            _ => panic!("use_ref hook type mismatch"),
        }
    })
}

/// Dependency for use_memo hook
#[derive(Debug, Clone)]
pub struct MemoDep {
    type_id: std::any::TypeId,
    hash: u64,
}

impl PartialEq for MemoDep {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id && self.hash == other.hash
    }
}

/// use_memo hook - memoized computation (simplified version)
pub fn use_memo<T: Clone + 'static, F: Fn() -> T>(compute: F) -> T {
    let idx = HOOKS.with(|h| {
        let hooks = h.borrow_mut();
        hooks.next_index()
    });

    let has_existing = HOOKS.with(|h| {
        let hooks = h.borrow();
        idx < hooks.states.len()
    });

    if has_existing {
        let existing = HOOKS.with(|h| {
            let hooks = h.borrow();
            if let HookState::Memo { value, .. } = &hooks.states[idx] {
                Some(value.downcast_ref::<T>().unwrap().clone())
            } else {
                None
            }
        });

        if let Some(val) = existing {
            return val;
        }
    }

    let result = compute();
    HOOKS.with(|h| {
        let mut hooks = h.borrow_mut();
        // Use empty deps since we can't compare Box<dyn Any>
        hooks.states[idx] = HookState::Memo {
            value: Box::new(result.clone()),
            deps: Vec::new(),
        };
    });
    result
}

thread_local! {
    /// Thread-local hooks storage
    static HOOKS: RefCell<Hooks> = RefCell::new(Hooks::new());
}

/// Initialize hooks for a component
pub fn init_hooks() {
    HOOKS.with(|h| h.borrow_mut().reset());
}

/// Clear all hooks
pub fn clear_hooks() {
    HOOKS.with(|h| h.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hooks_collection() {
        let mut hooks = Hooks::new();
        let val: &mut i32 = hooks.get_or_create(|| 0);
        assert_eq!(*val, 0);
        *val = 5;

        hooks.reset();
        let val2: &mut i32 = hooks.get_or_create(|| 0);
        assert_eq!(*val2, 5);
    }

    #[test]
    fn use_state_basic() {
        clear_hooks();
        let (count, set_count) = use_state(0);
        assert_eq!(count, 0);

        // Update the state
        set_count(5);

        // Access the state again - note: in real usage, you'd call
        // use_state at the same position in the component each render
        // Here we verify the hook storage was updated
        let stored_value = HOOKS.with(|h| {
            let hooks = h.borrow();
            if let HookState::State(val) = &hooks.states[0] {
                let stored: &i32 = val.downcast_ref().unwrap();
                Some(*stored)
            } else {
                None
            }
        });
        assert_eq!(stored_value, Some(5));
    }
}
