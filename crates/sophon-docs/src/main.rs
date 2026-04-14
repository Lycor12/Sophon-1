//! Documentation generator CLI
//!
//! Commands:
//! - sophon-docs generate  - Generate all documentation
//! - sophon-docs serve     - Start local server
//! - sophon-docs search    - Search documentation
//! - sophon-docs diagram   - Generate architecture diagrams

use std::env;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("generate");

    match command {
        "generate" => cmd_generate(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "diagram" => cmd_diagram(&args[2..]),
        _ => print_help(),
    }
}

fn print_help() {
    println!("sophon-docs - Documentation generator for Sophon AGI");
    println!();
    println!("Commands:");
    println!("  generate [output_dir]  Generate all documentation");
    println!("  serve [port]           Start documentation server");
    println!("  search <query>         Search documentation");
    println!("  diagram                Generate architecture diagrams");
    println!();
    println!("Examples:");
    println!("  sophon-docs generate docs/");
    println!("  sophon-docs serve 8080");
}

fn cmd_generate(args: &[String]) {
    let output = args.first().map(|s| s.as_str()).unwrap_or("docs");
    let mut generator = sophon_docs::DocGenerator::new(output);

    // Add each crate directory under crates/
    let crates_dir = std::path::Path::new("crates");
    if let Ok(entries) = std::fs::read_dir(crates_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("src").exists() {
                generator.add_root(&path);
            }
        }
    }

    match generator.generate() {
        Ok(()) => println!("Documentation generated in {}", output),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_serve(args: &[String]) {
    let port = args
        .first()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3000);

    println!("Documentation server would start on port {}", port);
    println!("(Server functionality requires async runtime - see sophon-runtime)");
}

fn cmd_search(args: &[String]) {
    let query = args.join(" ");
    if query.is_empty() {
        eprintln!("Usage: sophon-docs search <query>");
        return;
    }

    println!("Searching for: {}", query);
    println!("(Search functionality requires index - run 'sophon-docs generate' first)");
}

fn cmd_diagram(_args: &[String]) {
    let diagram = sophon_docs::generate_architecture_diagram(&[]);
    println!("{}", diagram);
}
