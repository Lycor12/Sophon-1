//! Effects Example - Demonstrates use_effect hook
//!
//! Run with: cargo run --example effects

use sophon_tui::{use_effect, use_state, Color, Dependency, EffectType, Element};

/// Component with mount effect
fn mount_effect_component() -> Element {
    let (mounted, _set_mounted) = use_state(false);

    // Effect that runs on mount
    use_effect(
        move || {
            println!("Component mounted!");
        },
        EffectType::Mount,
    );

    Element::column(vec![
        Element::text("Mount Effect Example")
            .color(Color::Cyan)
            .bold(),
        Element::text(if mounted {
            "✓ Component has mounted"
        } else {
            "Component mounting..."
        }),
    ])
}

/// Component with update effect
fn update_effect_component() -> Element {
    let (count, _set_count) = use_state::<i32>(0);
    let (update_count, _set_update_count) = use_state::<i32>(0);

    // Effect that runs on every update (simulated - just logs)
    use_effect(
        move || {
            println!("Component updated! Count: {}", count);
            println!("Update count: {}", update_count);
        },
        EffectType::Update,
    );

    Element::column(vec![
        Element::text("Update Effect Example").color(Color::Green),
        Element::text(format!("State count: {}", count)),
        Element::text(format!("Update effect ran {} times", update_count)),
        Element::text("Updates on every render").color(Color::DarkGrey),
    ])
}

/// Component with conditional effect
fn conditional_effect_component() -> Element {
    let (value, _set_value) = use_state(0i32);
    let (effect_count, _set_effect_count) = use_state(0i32);

    // Effect that only runs when 'value' changes
    use_effect(
        move || {
            println!("Value changed to: {}", value);
            println!("Effect count: {}", effect_count);
        },
        EffectType::Conditional(vec![Dependency::new("value", value)]),
    );

    Element::column(vec![
        Element::text("Conditional Effect Example")
            .color(Color::Magenta)
            .bold(),
        Element::text(format!("Value: {}", value)),
        Element::text(format!("Effect triggered {} times", effect_count)),
        Element::text("Effect only runs when value changes").color(Color::DarkGrey),
    ])
}

/// Component with cleanup effect
fn cleanup_effect_component() -> Element {
    let (resource_count, _set_resource_count) = use_state(0i32);

    // Effect with cleanup
    use_effect(
        move || {
            // Simulate acquiring a resource
            println!("Acquiring resource #{}!", resource_count + 1);
            println!("(Cleanup would run here before next execution)");
        },
        EffectType::Mount,
    );

    Element::column(vec![
        Element::text("Cleanup Effect Example")
            .color(Color::Yellow)
            .bold(),
        Element::text(format!("Resources acquired: {}", resource_count)),
        Element::text("Cleanup runs before unmount").color(Color::DarkGrey),
    ])
}

fn main() {
    println!("Effects Example - TUI Hooks Demo");
    println!("=================================");

    sophon_tui::init_hooks();

    let mount = mount_effect_component();
    println!("\nMount effect component:");
    println!(" Kind: {:?}", mount.kind);
    println!(" Children: {}", mount.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let update = update_effect_component();
    println!("\nUpdate effect component:");
    println!(" Kind: {:?}", update.kind);
    println!(" Children: {}", update.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let conditional = conditional_effect_component();
    println!("\nConditional effect component:");
    println!(" Kind: {:?}", conditional.kind);
    println!(" Children: {}", conditional.children.len());

    sophon_tui::clear_hooks();
    sophon_tui::init_hooks();

    let cleanup = cleanup_effect_component();
    println!("\nCleanup effect component:");
    println!(" Kind: {:?}", cleanup.kind);
    println!(" Children: {}", cleanup.children.len());

    sophon_tui::clear_hooks();
    println!("\nDone!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_effect_component() {
        sophon_tui::init_hooks();

        // Note: Effects run after render, so we just verify the component renders
        let element = mount_effect_component();
        assert!(matches!(element.kind, sophon_tui::ElementKind::Column));

        sophon_tui::clear_hooks();
    }

    #[test]
    fn test_conditional_effect_component() {
        sophon_tui::init_hooks();

        let element = conditional_effect_component();
        assert!(matches!(element.kind, sophon_tui::ElementKind::Column));
        assert_eq!(element.children.len(), 4);

        sophon_tui::clear_hooks();
    }
}
