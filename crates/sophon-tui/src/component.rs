//! Component trait and registry for reusable UI components
//!
//! Components are the building blocks of TUI applications. Each component
//! has its own state (via hooks) and renders to a virtual DOM element tree.

use crate::element::Element;
use crate::hook::Hooks;
use crate::layout::Rect;
use std::collections::HashMap;

/// Component ID for registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(pub usize);

/// Component trait - implement for custom components
pub trait Component: std::fmt::Debug {
    /// Unique identifier for this component type
    fn type_id(&self) -> &'static str;

    /// Called when component is mounted
    fn mount(&mut self) {}

    /// Called before each render
    fn update(&mut self, props: &dyn std::any::Any) {}

    /// Render the component to an element tree
    fn render(&self, hooks: &Hooks, area: Rect) -> Element;

    /// Called when component is unmounted
    fn unmount(&mut self) {}
}

/// Component registry for managing component instances
#[derive(Debug, Default)]
pub struct ComponentRegistry {
    components: HashMap<ComponentId, Box<dyn Component>>,
    next_id: usize,
}

impl ComponentRegistry {
    /// Create a new component registry
    pub fn new() -> Self {
        ComponentRegistry {
            components: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a component and return its ID
    pub fn register(&mut self, component: Box<dyn Component>) -> ComponentId {
        let id = ComponentId(self.next_id);
        self.next_id += 1;
        self.components.insert(id, component);
        id
    }

    /// Get a component by ID
    pub fn get(&self, id: ComponentId) -> Option<&dyn Component> {
        self.components.get(&id).map(|b| b.as_ref())
    }

    /// Get a mutable component by ID
    pub fn get_mut(&mut self, id: ComponentId) -> Option<&mut dyn Component> {
        self.components.get_mut(&id).map(|b| b.as_mut())
    }

    /// Remove a component
    pub fn unregister(&mut self, id: ComponentId) -> Option<Box<dyn Component>> {
        self.components.remove(&id)
    }

    /// Count of registered components
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

/// Function component wrapper - allows using functions as components
pub struct FunctionComponent<F> {
    render_fn: F,
    _props: std::marker::PhantomData<dyn std::any::Any>,
}

impl<F> FunctionComponent<F> {
    /// Create a new function component
    pub fn new(render_fn: F) -> Self {
        FunctionComponent {
            render_fn,
            _props: std::marker::PhantomData,
        }
    }
}

impl<F> std::fmt::Debug for FunctionComponent<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionComponent")
            .field("type", &std::any::type_name::<F>())
            .finish()
    }
}

/// Macro for creating function components
#[macro_export]
macro_rules! component {
    ($name:ident, |$hooks:ident, $area:ident| $body:expr) => {
        pub fn $name(
            $hooks: &$crate::hook::Hooks,
            $area: $crate::layout::Rect,
        ) -> $crate::element::Element {
            $body
        }
    };
}

/// Component props trait - derive for type-safe props
pub trait Props: Clone + std::fmt::Debug + 'static {
    /// Compare props for equality
    fn eq(&self, other: &dyn Props) -> bool;
}

impl Props for () {
    fn eq(&self, _other: &dyn Props) -> bool {
        true
    }
}

/// Component context - passed during render
#[derive(Debug)]
pub struct RenderContext {
    /// Available area for rendering
    pub area: Rect,
    /// Whether component has focus
    pub has_focus: bool,
    /// Parent component ID
    pub parent_id: Option<ComponentId>,
}

impl RenderContext {
    /// Create a new render context
    pub fn new(area: Rect) -> Self {
        RenderContext {
            area,
            has_focus: false,
            parent_id: None,
        }
    }

    /// Set focus state
    pub fn with_focus(mut self, focus: bool) -> Self {
        self.has_focus = focus;
        self
    }

    /// Set parent ID
    pub fn with_parent(mut self, id: ComponentId) -> Self {
        self.parent_id = Some(id);
        self
    }
}
