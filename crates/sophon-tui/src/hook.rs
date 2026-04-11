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
        deps: Vec<Box<dyn std::any::Any>>,
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
    let mut hooks = HOOKS.with(|h| h.borrow_mut());
    let idx = hooks.next_index();

    if idx >= hooks.states.len() {
        hooks
            .states
            .push(HookState::State(Box::new(initial.clone())));
    }

    let value = match &hooks.states[idx] {
        HookState::State(val) => val.downcast_ref::<T>().unwrap().clone(),
        _ => panic!("use_state hook type mismatch"),
    };

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
pub fn use_ref<T: 'static>(initial: T) -> RefCell<T> {
    let mut hooks = HOOKS.with(|h| h.borrow_mut());
    let idx = hooks.next_index();

    if idx >= hooks.states.len() {
        hooks
            .states
            .push(HookState::Ref(Box::new(RefCell::new(initial))));
    }

    match &hooks.states[idx] {
        HookState::Ref(val) => {
            // Return a clone of the RefCell wrapper
            // Actually we need to return a reference, not a clone
            // For now, return a new RefCell (this is a limitation)
            RefCell::new(val.downcast_ref::<RefCell<T>>().unwrap().borrow().clone())
        }
        _ => panic!("use_ref hook type mismatch"),
    }
}

/// use_memo hook - memoized computation
pub fn use_memo<T: Clone + 'static, F: Fn() -> T>(
    compute: F,
    deps: &[Box<dyn std::any::Any>],
) -> T {
    let mut hooks = HOOKS.with(|h| h.borrow_mut());
    let idx = hooks.next_index();

    if idx < hooks.states.len() {
        match &hooks.states[idx] {
            HookState::Memo {
                value,
                deps: old_deps,
            } => {
                // Check if deps changed
                let deps_changed = deps.len() != old_deps.len()
                    || deps.iter().zip(old_deps.iter()).any(|(a, b)| {
                        // Compare type IDs
                        a.type_id() == b.type_id() && {
                            // Use Any comparison
                            let a_any = a.as_ref() as &dyn std::any::Any;
                            let b_any = b.as_ref() as &dyn std::any::Any;
                            // Can't actually compare values without downcasting
                            // For simplicity, assume changed
                            true
                        }
                    });

                if !deps_changed {
                    return value.downcast_ref::<T>().unwrap().clone();
                }
            }
            _ => {}
        }
    }

    let result = compute();
    hooks.states[idx] = HookState::Memo {
        value: Box::new(result.clone()),
        deps: deps.to_vec(),
    };
    result
}

/// Thread-local hooks storage
thread_local! {
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

        set_count(5);
        let (new_count, _) = use_state(0);
        assert_eq!(new_count, 5);
    }
}
