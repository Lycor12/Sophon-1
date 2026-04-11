//! Effect system for side effects in components
//!
//! Effects are executed after rendering completes, similar to React's useEffect.
//! They can return cleanup functions that run before the next effect execution
//! or when the component unmounts.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Effect type
#[derive(Debug)]
pub enum EffectType {
    /// Run once after initial render (componentDidMount)
    Mount,
    /// Run after every render
    Update,
    /// Run when specific dependencies change
    Conditional(Vec<Dependency>),
    /// Run before unmount (componentWillUnmount)
    Unmount,
}

/// Dependency for conditional effects
#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    /// Dependency key
    pub key: String,
    /// Dependency value (stringified for comparison)
    pub value: String,
}

impl Dependency {
    /// Create a new dependency
    pub fn new(key: impl Into<String>, value: impl ToString) -> Self {
        Dependency {
            key: key.into(),
            value: value.to_string(),
        }
    }
}

/// Effect callback type
pub type EffectCallback = Box<dyn FnMut() + 'static>;

/// Effect cleanup type
pub type EffectCleanup = Box<dyn FnOnce() + 'static>;

/// Effect entry (manual Debug impl because EffectCallback doesn't implement Debug)
pub struct Effect {
    /// Effect type
    pub effect_type: EffectType,
    /// Last run dependencies (for conditional effects)
    pub last_deps: Option<Vec<Dependency>>,
    /// Whether effect has run
    pub has_run: bool,
    /// Cleanup function (set when effect runs)
    pub cleanup: Option<EffectCleanup>,
}

impl std::fmt::Debug for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Effect")
            .field("effect_type", &self.effect_type)
            .field("last_deps", &self.last_deps)
            .field("has_run", &self.has_run)
            .field("cleanup", &self.cleanup.is_some())
            .finish()
    }
}

impl Effect {
    /// Create a new mount effect
    pub fn mount() -> Self {
        Effect {
            effect_type: EffectType::Mount,
            last_deps: None,
            has_run: false,
            cleanup: None,
        }
    }

    /// Create a new update effect
    pub fn update() -> Self {
        Effect {
            effect_type: EffectType::Update,
            last_deps: None,
            has_run: false,
            cleanup: None,
        }
    }

    /// Create a new conditional effect
    pub fn conditional(deps: Vec<Dependency>) -> Self {
        Effect {
            effect_type: EffectType::Conditional(deps.clone()),
            last_deps: None,
            has_run: false,
            cleanup: None,
        }
    }

    /// Create a new unmount effect
    pub fn unmount() -> Self {
        Effect {
            effect_type: EffectType::Unmount,
            last_deps: None,
            has_run: false,
            cleanup: None,
        }
    }

    /// Check if effect should run
    pub fn should_run(&self, current_deps: &[Dependency]) -> bool {
        match &self.effect_type {
            EffectType::Mount => !self.has_run,
            EffectType::Update => true,
            EffectType::Conditional(_) => {
                if let Some(last) = &self.last_deps {
                    last != current_deps
                } else {
                    true
                }
            }
            EffectType::Unmount => false, // Unmount effects handled separately
        }
    }
}

/// Effect queue for batching effects
#[derive(Debug, Default)]
pub struct EffectQueue {
    queue: RefCell<VecDeque<Effect>>,
}

impl EffectQueue {
    /// Create a new effect queue
    pub fn new() -> Self {
        EffectQueue {
            queue: RefCell::new(VecDeque::new()),
        }
    }

    /// Push an effect to the queue
    pub fn push(&self, effect: Effect) {
        self.queue.borrow_mut().push_back(effect);
    }

    /// Pop the next effect
    pub fn pop(&self) -> Option<Effect> {
        self.queue.borrow_mut().pop_front()
    }

    /// Clear all effects
    pub fn clear(&self) {
        self.queue.borrow_mut().clear();
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.borrow().is_empty()
    }

    /// Run all effects in the queue
    pub fn run_all(&self) {
        while let Some(mut effect) = self.pop() {
            // Run cleanup if exists
            if let Some(cleanup) = effect.cleanup.take() {
                cleanup();
            }
            effect.has_run = true;
        }
    }
}

/// Hook for side effects
#[derive(Debug)]
pub struct UseEffect {
    effects: RefCell<Vec<Effect>>,
}

impl UseEffect {
    /// Create a new use_effect hook
    pub fn new() -> Self {
        UseEffect {
            effects: RefCell::new(Vec::new()),
        }
    }

    /// Register an effect
    pub fn register(&self, effect: Effect) -> usize {
        let mut effects = self.effects.borrow_mut();
        let idx = effects.len();
        effects.push(effect);
        idx
    }

    /// Run all effects
    pub fn run_effects(&self) {
        let mut effects = self.effects.borrow_mut();
        for effect in effects.iter_mut() {
            if let Some(cleanup) = effect.cleanup.take() {
                cleanup();
            }
            effect.has_run = true;
        }
    }

    /// Cleanup all effects (call on unmount)
    pub fn cleanup(&self) {
        let effects = self.effects.borrow();
        for effect in effects.iter() {
            if let Some(cleanup) = &effect.cleanup {
                cleanup();
            }
        }
    }
}

impl Default for UseEffect {
    fn default() -> Self {
        Self::new()
    }
}
