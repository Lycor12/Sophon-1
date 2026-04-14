//! HTML Documentation Generation
//!
//! Generates beautiful HTML documentation from Rust source code.
//! Uses Handlebars-style templating for structure, CSS for styling.

use crate::{CrateDocs, DocError, ItemDocs, ItemKind, ModuleDocs};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// HTML documentation generator
pub struct HtmlGenerator {
    /// Output directory
    output: PathBuf,
    /// Templates directory
    templates_dir: PathBuf,
    /// Assets directory
    assets_dir: PathBuf,
    /// Search index
    search_index: Vec<SearchEntry>,
}

/// Search index entry
#[derive(Debug, Clone)]
struct SearchEntry {
    title: String,
    path: String,
    content: String,
    kind: String,
}

impl HtmlGenerator {
    /// Create new HTML generator
    pub fn new(output: impl AsRef<Path>) -> Self {
        let output = output.as_ref().to_path_buf();
        HtmlGenerator {
            templates_dir: PathBuf::from("templates"),
            assets_dir: PathBuf::from("assets"),
            output,
            search_index: Vec::new(),
        }
    }

    /// Generate all HTML documentation
    pub fn generate(&mut self, crates: &HashMap<String, CrateDocs>) -> Result<(), DocError> {
        // Create output directory
        fs::create_dir_all(&self.output).map_err(|e| DocError::Io(e.to_string()))?;

        // Create assets directory
        let assets_dir = self.output.join("assets");
        fs::create_dir_all(&assets_dir).map_err(|e| DocError::Io(e.to_string()))?;

        // Copy CSS
        self.copy_stylesheet()?;

        // Generate JavaScript files
        self.generate_js()?;

        // Generate main index
        self.generate_main_index(crates)?;

        // Generate crate documentation
        for (name, crate_docs) in crates {
            self.generate_crate_docs(name, crate_docs, crates)?;
        }

        // Generate search index
        self.generate_search_json()?;

        Ok(())
    }

    /// Copy CSS stylesheet
    fn copy_stylesheet(&self) -> Result<(), DocError> {
        let css_path = self.output.join("assets").join("styles.css");

        // Create styles.css inline
        let css = include_str!("../templates/styles.css");

        fs::write(&css_path, css).map_err(|e| DocError::Io(e.to_string()))?;

        Ok(())
    }

    /// Generate JavaScript files
    fn generate_js(&self) -> Result<(), DocError> {
        // Theme toggle script
        let theme_js = r#"(function() {
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', theme);
    
    document.getElementById('theme-toggle').addEventListener('click', function() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });
})();"#;

        fs::write(self.output.join("assets/theme.js"), theme_js)
            .map_err(|e| DocError::Io(e.to_string()))?;

        // Mobile menu toggle
        let menu_js = r#"(function() {
    const toggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    
    if (toggle && sidebar) {
        toggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });
    }
})();"#;

        fs::write(self.output.join("assets/menu.js"), menu_js)
            .map_err(|e| DocError::Io(e.to_string()))?;

        // Search functionality
        let search_js = r#"(function() {
    const searchInput = document.getElementById('search-input');
    const modal = document.getElementById('search-overlay');
    const modalInput = document.getElementById('modal-search-input');
    const results = document.getElementById('search-results');
    const closeBtn = document.getElementById('search-close');
    
    let searchData = [];
    
    // Load search index
    fetch('assets/search.json')
        .then(r => r.json())
        .then(data => { searchData = data; })
        .catch(e => console.error('Failed to load search index:', e));
    
    function openModal() {
        modal.classList.add('active');
        modalInput.focus();
        modalInput.value = searchInput.value;
        performSearch(modalInput.value);
    }
    
    function closeModal() {
        modal.classList.remove('active');
    }
    
    function performSearch(query) {
        if (!query || query.length < 2) {
            results.innerHTML = '<div class="search-result"><div class="search-result-title">Type to search...</div></div>';
            return;
        }
        
        const q = query.toLowerCase();
        const matches = searchData
            .filter(item => 
                item.title.toLowerCase().includes(q) || 
                item.content.toLowerCase().includes(q)
            )
            .slice(0, 10);
        
        if (matches.length === 0) {
            results.innerHTML = '<div class="search-result"><div class="search-result-title">No results found</div></div>';
            return;
        }
        
        results.innerHTML = matches.map(item => 
            '<div class="search-result" onclick="location.href=\'' + item.path + '\'">' +
            '<div class="search-result-title">' + escapeHtml(item.title) + '</div>' +
            '<div class="search-result-path">' + escapeHtml(item.path) + '</div>' +
            '</div>'
        ).join('');
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    if (searchInput) {
        searchInput.addEventListener('focus', openModal);
    }
    
    if (modalInput) {
        modalInput.addEventListener('input', function() {
            performSearch(this.value);
        });
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeModal);
    }
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            openModal();
        }
    });
})();"#;

        fs::write(self.output.join("assets/search.js"), search_js)
            .map_err(|e| DocError::Io(e.to_string()))?;

        Ok(())
    }

    /// Generate main index page
    fn generate_main_index(&mut self, crates: &HashMap<String, CrateDocs>) -> Result<(), DocError> {
        let mut crate_list = String::new();

        for (name, docs) in crates {
            let mut modules_html = String::new();
            for module in &docs.modules {
                modules_html.push_str(&format!(
                    r#"<li><a href="crates/{}/{}.html">{}</a></li>"#,
                    name, module.path, module.path
                ));
            }

            crate_list.push_str(&format!(
                r#"<div class="crate-card">
                    <h3><a href="crates/{0}/index.html">{0}</a></h3>
                    <p>{1}</p>
                    <ul>{2}</ul>
                </div>"#,
                name,
                markdown_to_html(&docs.description),
                modules_html
            ));

            // Add to search index
            self.search_index.push(SearchEntry {
                title: name.clone(),
                path: format!("crates/{}/index.html", name),
                content: docs.description.clone(),
                kind: "crate".to_string(),
            });
        }

        let content = format!(
            r#"<h1>Sophon AGI Documentation</h1>
            <p class="lead">A complete production-ready AGI system with state-space models, hyperdimensional computing, and adversarial learning.</p>
            
            <div class="crates-grid">
                {crate_list}
            </div>
            
            <h2>Getting Started</h2>
            <div class="getting-started">
                <div class="step">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <h3>Install</h3>
                        <pre><code>cargo install sophon</code></pre>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <h3>Run Demo</h3>
                        <pre><code>sophon demo</code></pre>
                    </div>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <h3>Explore</h3>
                        <pre><code>sophon --help</code></pre>
                    </div>
                </div>
            </div>"#
        );

        let html = self.render_page(
            "Sophon AGI Documentation",
            "",
            &content,
            vec![],
            true,
            crates,
        )?;

        fs::write(self.output.join("index.html"), html).map_err(|e| DocError::Io(e.to_string()))?;

        Ok(())
    }

    /// Generate documentation for a single crate
    fn generate_crate_docs(
        &mut self,
        name: &str,
        crate_docs: &CrateDocs,
        all_crates: &HashMap<String, CrateDocs>,
    ) -> Result<(), DocError> {
        let crate_dir = self.output.join("crates").join(name);
        fs::create_dir_all(&crate_dir).map_err(|e| DocError::Io(e.to_string()))?;

        // Generate crate index
        let mut content = format!(
            r#"<h1>Crate: {}</h1>
            <p class="crate-description">{}</p>
            "#,
            name,
            markdown_to_html(&crate_docs.description)
        );

        // Modules section
        if !crate_docs.modules.is_empty() {
            content.push_str(r#"<h2>Modules</h2><div class="modules-grid">"#);
            for module in &crate_docs.modules {
                content.push_str(&format!(
                    r#"<div class="module-card">
                        <h3><a href="{}.html">{}</a></h3>
                        <p>{}</p>
                    </div>"#,
                    module.path,
                    module.path,
                    if module.docs.is_empty() {
                        "No documentation available."
                    } else {
                        &module.docs.lines().next().unwrap_or("")[..60.min(module.docs.len())]
                    }
                ));
            }
            content.push_str("</div>");
        }

        // Items section
        if !crate_docs.items.is_empty() {
            content.push_str(r#"<h2>Public Items</h2><div class="items-list">"#);
            for item in &crate_docs.items {
                content.push_str(&format!(
                    r#"<div class="api-item" id="{}">
                        <div class="api-header">
                            <span class="api-kind {}">{}</span>
                            <code class="api-signature">{}</code>
                        </div>
                        <div class="api-description">{}</div>
                        <a class="api-source" href="{}#L{}">View source →</a>
                    </div>"#,
                    item.name,
                    item.kind.to_string(),
                    item.kind,
                    html_escape(&item.signature),
                    markdown_to_html(&item.docs),
                    item.source.display(),
                    item.line
                ));

                // Add to search index
                self.search_index.push(SearchEntry {
                    title: format!("{}::{}", name, item.name),
                    path: format!("crates/{}/index.html#{}", name, item.name),
                    content: item.docs.clone(),
                    kind: item.kind.to_string(),
                });
            }
            content.push_str("</div>");
        }

        let html = self.render_page(
            &format!("Crate: {} - Sophon AGI", name),
            &format!("crates/{}/index.html", name),
            &content,
            vec![("crates".to_string(), name.to_string())],
            false,
            all_crates,
        )?;

        fs::write(crate_dir.join("index.html"), html).map_err(|e| DocError::Io(e.to_string()))?;

        // Generate module pages
        for module in &crate_docs.modules {
            let mut module_content = format!(
                r#"<h1>Module: {}</h1>
                <p class="module-docs">{}</p>
                "#,
                module.path,
                markdown_to_html(&module.docs)
            );

            if !module.items.is_empty() {
                module_content.push_str(r#"<h2>Items</h2><div class="items-list">"#);
                for item in &module.items {
                    module_content.push_str(&format!(
                        r#"<div class="api-item" id="{}">
                            <div class="api-header">
                                <span class="api-kind {}">{}</span>
                                <code class="api-signature">{}</code>
                            </div>
                            <div class="api-description">{}</div>
                            <a class="api-source" href="{}#L{}">View source →</a>
                        </div>"#,
                        item.name,
                        item.kind.to_string(),
                        item.kind,
                        html_escape(&item.signature),
                        markdown_to_html(&item.docs),
                        item.source.display(),
                        item.line
                    ));
                }
                module_content.push_str("</div>");
            }

            let module_html = self.render_page(
                &format!("Module: {}::{} - Sophon AGI", name, module.path),
                &format!("crates/{}/{}.html", name, module.path),
                &module_content,
                vec![
                    ("crates".to_string(), name.to_string()),
                    ("modules".to_string(), module.path.clone()),
                ],
                false,
                all_crates,
            )?;

            fs::write(crate_dir.join(format!("{}.html", module.path)), module_html)
                .map_err(|e| DocError::Io(e.to_string()))?;
        }

        Ok(())
    }

    /// Generate search index JSON
    fn generate_search_json(&self) -> Result<(), DocError> {
        let search_data: Vec<_> = self
            .search_index
            .iter()
            .map(|e| {
                format!(
                    r#"{{"title":"{}","path":"{}","content":"{}","kind":"{}"}}"#,
                    json_escape(&e.title),
                    json_escape(&e.path),
                    json_escape(&e.content[..200.min(e.content.len())]),
                    e.kind
                )
            })
            .collect();

        let json = format!("[{}]", search_data.join(","));

        fs::write(self.output.join("assets/search.json"), json)
            .map_err(|e| DocError::Io(e.to_string()))?;

        Ok(())
    }

    /// Render a complete HTML page
    fn render_page(
        &self,
        title: &str,
        path: &str,
        content: &str,
        breadcrumbs: Vec<(String, String)>,
        is_index: bool,
        crates: &HashMap<String, CrateDocs>,
    ) -> Result<String, DocError> {
        // Calculate base path for relative URLs
        let depth = path.matches('/').count();
        let base_path = if depth == 0 { "" } else { &"../".repeat(depth) };

        // Build breadcrumbs HTML
        let mut breadcrumbs_html = String::from(r#"<a href="index.html">Docs</a>"#);
        for (i, (name, link)) in breadcrumbs.iter().enumerate() {
            breadcrumbs_html.push_str(r#"<span class="separator">/</span>"#);
            if i == breadcrumbs.len() - 1 {
                breadcrumbs_html.push_str(&html_escape(name));
            } else {
                breadcrumbs_html.push_str(&format!(
                    r#"<a href="{}{}">{}</a>"#,
                    base_path,
                    link,
                    html_escape(name)
                ));
            }
        }

        // Build crate navigation
        let mut crates_nav = String::new();
        for (crate_name, crate_docs) in crates {
            let active = if path.contains(&format!("crates/{}/", crate_name)) {
                " active"
            } else {
                ""
            };

            let mut modules_nav = String::new();
            for module in &crate_docs.modules {
                modules_nav.push_str(&format!(
                    r#"<li><a href="{}crates/{}/{}.html">{}</a></li>"#,
                    base_path, crate_name, module.path, module.path
                ));
            }

            crates_nav.push_str(&format!(
                r#"<li class="nav-item">
                    <a href="{}crates/{}/index.html" class="nav-link{}">{}</a>
                    <ul class="nav-sublist">{}</ul>
                </li>"#,
                base_path, crate_name, active, crate_name, modules_nav
            ));
        }

        // Render full HTML page
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <link rel="stylesheet" href="{}assets/styles.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
    <div class="layout">
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <a href="{}index.html" class="logo">
                    <svg class="logo-icon" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="16" cy="16" r="14" stroke="currentColor" stroke-width="2"/>
                        <circle cx="16" cy="16" r="6" fill="currentColor"/>
                        <path d="M16 2V8M16 24V30M2 16H8M24 16H30" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    <span class="logo-text">Sophon</span>
                </a>
                <div class="version">v0.1.0</div>
            </div>
            
            <nav class="sidebar-nav">
                <div class="search-box">
                    <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="M21 21l-4.35-4.35"/>
                    </svg>
                    <input type="text" id="search-input" placeholder="Search documentation..." autocomplete="off">
                </div>
                
                <div class="nav-section">
                    <h3 class="nav-section-title">Getting Started</h3>
                    <ul class="nav-list">
                        <li><a href="{}index.html" class="nav-link{}">Introduction</a></li>
                    </ul>
                </div>
                
                <div class="nav-section">
                    <h3 class="nav-section-title">Crates</h3>
                    <ul class="nav-list">
                        {}
                    </ul>
                </div>
            </nav>
            
            <div class="sidebar-footer">
                <a href="https://github.com/sophon-agi/sophon" class="external-link" target="_blank">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                    GitHub
                </a>
            </div>
        </aside>
        
        <main class="main">
            <header class="header">
                <button class="menu-toggle" id="menu-toggle" aria-label="Toggle menu">
                    <span></span><span></span><span></span>
                </button>
                <div class="breadcrumbs">{}</div>
                <div class="header-actions">
                    <button class="theme-toggle" id="theme-toggle" aria-label="Toggle theme">
                        <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="5"/>
                            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                        </svg>
                        <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                        </svg>
                    </button>
                </div>
            </header>
            
            <article class="content">
                {}
            </article>
            
            <footer class="footer">
                <div class="footer-content">
                    <p>Generated by <a href="https://github.com/sophon-agi/sophon">sophon-docs</a></p>
                    <p>Licensed under MIT or Apache-2.0</p>
                </div>
            </footer>
        </main>
        
        <div class="search-overlay" id="search-overlay">
            <div class="search-modal">
                <div class="search-header">
                    <input type="text" id="modal-search-input" placeholder="Search..." autocomplete="off">
                    <button class="search-close" id="search-close">ESC</button>
                </div>
                <div class="search-results" id="search-results"></div>
            </div>
        </div>
    </div>
    
    <script src="{}assets/search.js"></script>
    <script src="{}assets/theme.js"></script>
    <script src="{}assets/menu.js"></script>
</body>
</html>"#,
            title,
            base_path,
            base_path,
            base_path,
            if is_index { " active" } else { "" },
            crates_nav,
            breadcrumbs_html,
            content,
            base_path,
            base_path,
            base_path
        );

        Ok(html)
    }
}

/// Simple markdown to HTML conversion
fn markdown_to_html(md: &str) -> String {
    let mut html = String::new();
    let mut in_code = false;
    let mut code_lang = String::new();

    for line in md.lines() {
        if line.starts_with("```") {
            if in_code {
                html.push_str("</code></pre>");
                in_code = false;
            } else {
                code_lang = line[3..].trim().to_string();
                html.push_str(&format!(r#"<pre><code class="language-{}">"#, code_lang));
                in_code = true;
            }
        } else if in_code {
            html.push_str(&html_escape(line));
            html.push('\n');
        } else if line.starts_with("# ") {
            html.push_str(&format!("<h1>{}</h1>", html_escape(&line[2..])));
        } else if line.starts_with("## ") {
            html.push_str(&format!("<h2>{}</h2>", html_escape(&line[3..])));
        } else if line.starts_with("### ") {
            html.push_str(&format!("<h3>{}</h3>", html_escape(&line[4..])));
        } else if line.starts_with("- ") {
            html.push_str(&format!("<li>{}</li>", html_escape(&line[2..])));
        } else if line.trim().is_empty() {
            html.push_str("<p>");
        } else {
            // Inline code
            let mut result = String::new();
            let mut chars = line.chars().peekable();
            while let Some(ch) = chars.next() {
                if ch == '`' {
                    if let Some(end) = chars
                        .by_ref()
                        .take_while(|c| *c != '`')
                        .collect::<String>()
                        .into()
                    {
                        result.push_str(&format!("<code>{}</code>", html_escape(&end)));
                    }
                } else {
                    result.push(ch);
                }
            }
            html.push_str(&format!("<p>{}</p>", html_escape(line)));
        }
    }

    html
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Escape JSON special characters
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("&"), "&amp;");
        assert_eq!(html_escape("\"test\""), "&quot;test&quot;");
    }

    #[test]
    fn test_markdown_to_html() {
        let md = "# Title\n\nSome text\n";
        let html = markdown_to_html(md);
        assert!(html.contains("<h1>"));
        assert!(html.contains("Title"));
    }
}
