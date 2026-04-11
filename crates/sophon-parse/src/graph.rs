//! Semantic Graph Extraction — Call graphs, data flow, and type hierarchies.
//!
//! Implements MSCA Level 3: Semantic analysis and cross-reference tracking.
//! Builds dependency graphs that enable whole-program analysis.

use crate::ast::*;
use crate::ParseError;
use std::collections::{HashMap, HashSet};

/// Collection of all semantic graphs for a source file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphBundle {
    pub call_graph: CallGraph,
    pub data_flow: DataFlowGraph,
    pub type_graph: TypeGraph,
}

/// Call graph: tracks which functions call which other functions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CallGraph {
    /// Function name -> set of functions it calls
    pub edges: HashMap<String, HashSet<String>>,
    /// Function name -> source location
    pub locations: HashMap<String, (String, usize)>, // (file, line)
    /// Reverse edges: callee -> callers
    pub reverse_edges: HashMap<String, HashSet<String>>,
}

/// Data flow graph: tracks variable definitions and uses.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DataFlowGraph {
    /// Variable name -> definition locations
    pub definitions: HashMap<String, Vec<SourceLocation>>,
    /// Variable name -> use locations
    pub uses: HashMap<String, Vec<SourceLocation>>,
    /// Function -> local variable names
    pub locals_per_function: HashMap<String, HashSet<String>>,
    /// Function -> parameter names
    pub params_per_function: HashMap<String, Vec<String>>,
}

/// Type graph: tracks struct/enum/typedef hierarchies and dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeGraph {
    /// Type name -> definition location
    pub type_locations: HashMap<String, (String, usize)>,
    /// Type name -> fields (for structs/unions)
    pub struct_fields: HashMap<String, Vec<(String, String)>>, // (field_name, field_type_string)
    /// Type name -> base types it depends on
    pub dependencies: HashMap<String, HashSet<String>>,
    /// Reverse: type -> types that depend on it
    pub reverse_deps: HashMap<String, HashSet<String>>,
}

/// Source location for error reporting and cross-referencing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub entity: String, // function or variable name
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a call edge from `caller` to `callee`.
    pub fn add_call(&mut self, caller: &str, callee: &str) {
        self.edges
            .entry(caller.to_string())
            .or_default()
            .insert(callee.to_string());
        self.reverse_edges
            .entry(callee.to_string())
            .or_default()
            .insert(caller.to_string());
    }

    /// Set the source location for a function.
    pub fn set_location(&mut self, func: &str, file: &str, line: usize) {
        self.locations
            .insert(func.to_string(), (file.to_string(), line));
    }

    /// Get all functions called by `func` (direct callees).
    pub fn callees(&self, func: &str) -> Option<&HashSet<String>> {
        self.edges.get(func)
    }

    /// Get all functions that call `func` (direct callers).
    pub fn callers(&self, func: &str) -> Option<&HashSet<String>> {
        self.reverse_edges.get(func)
    }

    /// Get all callees recursively (transitive closure).
    pub fn transitive_callees(&self, func: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![func.to_string()];

        while let Some(current) = stack.pop() {
            if let Some(callees) = self.edges.get(&current) {
                for callee in callees {
                    if visited.insert(callee.clone()) {
                        stack.push(callee.clone());
                    }
                }
            }
        }

        visited.remove(func);
        visited
    }

    /// Get all callers recursively (reverse transitive closure).
    pub fn transitive_callers(&self, func: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![func.to_string()];

        while let Some(current) = stack.pop() {
            if let Some(callers) = self.reverse_edges.get(&current) {
                for caller in callers {
                    if visited.insert(caller.clone()) {
                        stack.push(caller.clone());
                    }
                }
            }
        }

        visited.remove(func);
        visited
    }

    /// Check if `caller` can reach `callee` (is there a call path?).
    pub fn can_reach(&self, caller: &str, callee: &str) -> bool {
        self.transitive_callees(caller).contains(callee)
    }

    /// Detect potential recursion (direct or mutual).
    pub fn is_recursive(&self, func: &str) -> bool {
        self.can_reach(func, func)
    }

    /// Compute call depth from entry points (functions with no callers).
    pub fn compute_depths(&self) -> HashMap<String, usize> {
        let mut depths: HashMap<String, usize> = HashMap::new();
        let mut changed = true;

        // Initialize entry points to depth 0
        for func in self.edges.keys() {
            if self
                .reverse_edges
                .get(func)
                .map(|s| s.is_empty())
                .unwrap_or(true)
            {
                depths.insert(func.clone(), 0);
            }
        }

        // Propagate depths
        while changed {
            changed = false;
            for (caller, callees) in &self.edges {
                let caller_depth = *depths.get(caller).unwrap_or(&0);
                for callee in callees {
                    let new_depth = caller_depth + 1;
                    let current = depths.get(callee).copied().unwrap_or(0);
                    if new_depth > current {
                        depths.insert(callee.clone(), new_depth);
                        changed = true;
                    }
                }
            }
        }

        depths
    }

    /// Find disconnected functions (dead code candidates).
    pub fn find_orphans(&self) -> HashSet<String> {
        let mut orphans = HashSet::new();
        for func in self.edges.keys() {
            if self
                .reverse_edges
                .get(func)
                .map(|s| s.is_empty())
                .unwrap_or(true)
            {
                // No one calls this function (might be main or dead code)
                if func != "main" {
                    orphans.insert(func.clone());
                }
            }
        }
        orphans
    }
}

impl DataFlowGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a variable definition.
    pub fn add_definition(&mut self, var: &str, loc: SourceLocation) {
        self.definitions
            .entry(var.to_string())
            .or_default()
            .push(loc);
    }

    /// Record a variable use.
    pub fn add_use(&mut self, var: &str, loc: SourceLocation) {
        self.uses.entry(var.to_string()).or_default().push(loc);
    }

    /// Add local variable for a function.
    pub fn add_local(&mut self, func: &str, var: &str) {
        self.locals_per_function
            .entry(func.to_string())
            .or_default()
            .insert(var.to_string());
    }

    /// Add parameter for a function.
    pub fn add_param(&mut self, func: &str, param: &str) {
        self.params_per_function
            .entry(func.to_string())
            .or_default()
            .push(param.to_string());
    }

    /// Check if a variable is used before defined (potential bug).
    pub fn use_before_def(&self, var: &str) -> Option<Vec<SourceLocation>> {
        let defs = self.definitions.get(var)?;
        let uses = self.uses.get(var)?;

        let earliest_def = defs.iter().map(|d| d.line).min()?;
        let uses_before: Vec<_> = uses
            .iter()
            .filter(|u| u.line < earliest_def)
            .cloned()
            .collect();

        if uses_before.is_empty() {
            None
        } else {
            Some(uses_before)
        }
    }

    /// Find unused variables (defined but never used).
    pub fn find_unused(&self) -> Vec<String> {
        self.definitions
            .keys()
            .filter(|var| !self.uses.contains_key(*var) || self.uses[*var].is_empty())
            .cloned()
            .collect()
    }

    /// Find variables that escape their defining function (address taken).
    pub fn find_escaping(&self) -> Vec<String> {
        // Simplified: variables with "&" operator in uses
        self.uses
            .keys()
            .filter(|var| {
                // Check if any use involves address-of
                self.uses
                    .get(*var)
                    .map(|uses| uses.iter().any(|u| u.entity.contains("&")))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }
}

impl TypeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a struct/union type.
    pub fn add_struct(&mut self, name: &str, fields: &[(String, String)]) {
        self.struct_fields.insert(name.to_string(), fields.to_vec());
    }

    /// Set location for a type.
    pub fn set_location(&mut self, name: &str, file: &str, line: usize) {
        self.type_locations
            .insert(name.to_string(), (file.to_string(), line));
    }

    /// Add a type dependency.
    pub fn add_dependency(&mut self, dependent: &str, dependency: &str) {
        self.dependencies
            .entry(dependent.to_string())
            .or_default()
            .insert(dependency.to_string());
        self.reverse_deps
            .entry(dependency.to_string())
            .or_default()
            .insert(dependent.to_string());
    }

    /// Get all types a given type depends on (transitive).
    pub fn transitive_deps(&self, ty: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![ty.to_string()];

        while let Some(current) = stack.pop() {
            if let Some(deps) = self.dependencies.get(&current) {
                for dep in deps {
                    if visited.insert(dep.clone()) {
                        stack.push(dep.clone());
                    }
                }
            }
        }

        visited.remove(ty);
        visited
    }

    /// Find circular type dependencies.
    pub fn find_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();
        let mut path = Vec::new();

        fn dfs(
            node: &str,
            graph: &TypeGraph,
            visited: &mut HashSet<String>,
            recursion_stack: &mut HashSet<String>,
            path: &mut Vec<String>,
            cycles: &mut Vec<Vec<String>>,
        ) {
            visited.insert(node.to_string());
            recursion_stack.insert(node.to_string());
            path.push(node.to_string());

            if let Some(deps) = graph.dependencies.get(node) {
                for dep in deps {
                    if !visited.contains(dep) {
                        dfs(dep, graph, visited, recursion_stack, path, cycles);
                    } else if recursion_stack.contains(dep) {
                        // Found cycle
                        let cycle_start = path.iter().position(|n| n == dep).unwrap();
                        cycles.push(path[cycle_start..].to_vec());
                    }
                }
            }

            path.pop();
            recursion_stack.remove(node);
        }

        for ty in self.dependencies.keys() {
            if !visited.contains(ty) {
                dfs(
                    ty,
                    self,
                    &mut visited,
                    &mut recursion_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// Find all types that would need to be recompiled if `ty` changes.
    pub fn affected_by_change(&self, ty: &str) -> HashSet<String> {
        self.transitive_reverse_deps(ty)
    }

    fn transitive_reverse_deps(&self, ty: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![ty.to_string()];

        while let Some(current) = stack.pop() {
            if let Some(deps) = self.reverse_deps.get(&current) {
                for dep in deps {
                    if visited.insert(dep.clone()) {
                        stack.push(dep.clone());
                    }
                }
            }
        }

        visited.remove(ty);
        visited
    }
}

/// Extract all semantic graphs from an AST.
pub fn extract_graphs(root: &AstRoot) -> Result<GraphBundle, ParseError> {
    let mut call_graph = CallGraph::new();
    let mut data_flow = DataFlowGraph::new();
    let mut type_graph = TypeGraph::new();

    // Process each function
    for func in &root.functions {
        process_function(func, &mut call_graph, &mut data_flow, &mut type_graph)?;
    }

    // Process struct declarations
    for struct_decl in &root.structs {
        process_struct(struct_decl, &mut type_graph)?;
    }

    // Process typedefs
    for typedef in &root.typedefs {
        process_typedef(typedef, &mut type_graph)?;
    }

    Ok(GraphBundle {
        call_graph,
        data_flow,
        type_graph,
    })
}

fn process_function(
    func: &FunctionDecl,
    call_graph: &mut CallGraph,
    data_flow: &mut DataFlowGraph,
    type_graph: &mut TypeGraph,
) -> Result<(), ParseError> {
    let func_name = &func.name;

    // Set location (line 0 for now, would be populated from tokens)
    call_graph.set_location(func_name, "<unknown>", 0);

    // Add parameters to data flow
    for param in &func.params {
        if let Some(name) = &param.name {
            data_flow.add_param(func_name, name);
        }
    }

    // Process function body
    if let Some(ref body) = func.body {
        for stmt in body {
            process_stmt(func_name, stmt, call_graph, data_flow)?;
        }
    }

    // Track return type dependencies
    track_type_deps(func_name, &func.return_type, type_graph);

    Ok(())
}

fn process_stmt(
    func_name: &str,
    stmt: &Stmt,
    call_graph: &mut CallGraph,
    data_flow: &mut DataFlowGraph,
) -> Result<(), ParseError> {
    match stmt {
        Stmt::Node(node) => process_node(func_name, node, call_graph, data_flow),
        Stmt::Label(_, inner) => process_stmt(func_name, inner, call_graph, data_flow),
    }
}

fn process_node(
    func_name: &str,
    node: &AstNode,
    call_graph: &mut CallGraph,
    data_flow: &mut DataFlowGraph,
) -> Result<(), ParseError> {
    match node {
        AstNode::CompoundStmt(stmts) => {
            for stmt in stmts {
                process_stmt(func_name, stmt, call_graph, data_flow)?;
            }
        }
        AstNode::IfStmt(if_stmt) => {
            process_expr(func_name, &if_stmt.cond, call_graph, data_flow);
            process_stmt(func_name, &if_stmt.then_branch, call_graph, data_flow)?;
            if let Some(else_branch) = &if_stmt.else_branch {
                process_stmt(func_name, else_branch, call_graph, data_flow)?;
            }
        }
        AstNode::WhileStmt(while_stmt) => {
            process_expr(func_name, &while_stmt.cond, call_graph, data_flow);
            process_stmt(func_name, &while_stmt.body, call_graph, data_flow)?;
        }
        AstNode::ForStmt(for_stmt) => {
            if let Some(init) = &for_stmt.init {
                process_node(func_name, init, call_graph, data_flow)?;
            }
            if let Some(cond) = &for_stmt.cond {
                process_expr(func_name, cond, call_graph, data_flow);
            }
            if let Some(step) = &for_stmt.step {
                process_expr(func_name, step, call_graph, data_flow);
            }
            process_stmt(func_name, &for_stmt.body, call_graph, data_flow)?;
        }
        AstNode::ReturnStmt(Some(expr)) => {
            process_expr(func_name, expr, call_graph, data_flow);
        }
        AstNode::ExprStmt(expr) => {
            process_expr(func_name, expr, call_graph, data_flow);
        }
        AstNode::VarDecl(var) => {
            data_flow.add_local(func_name, &var.name);
            if let Some(init) = &var.init {
                process_expr(func_name, init, call_graph, data_flow);
            }
        }
        _ => {}
    }
    Ok(())
}

fn process_expr(
    func_name: &str,
    expr: &Expr,
    call_graph: &mut CallGraph,
    data_flow: &mut DataFlowGraph,
) {
    process_expr_node(func_name, &expr.node, call_graph, data_flow);
}

fn process_expr_node(
    func_name: &str,
    node: &AstNode,
    call_graph: &mut CallGraph,
    data_flow: &mut DataFlowGraph,
) {
    match node {
        AstNode::CallExpr(call) => {
            // Track function call
            if let AstNode::Identifier(func_id) = call.func.node.as_ref() {
                call_graph.add_call(func_name, func_id);
            }
            // Process arguments
            for arg in &call.args {
                process_expr(func_name, arg, call_graph, data_flow);
            }
        }
        AstNode::BinaryExpr(be) => {
            process_expr(func_name, &be.left, call_graph, data_flow);
            process_expr(func_name, &be.right, call_graph, data_flow);
        }
        AstNode::UnaryExpr(ue) => {
            process_expr(func_name, &ue.operand, call_graph, data_flow);
        }
        AstNode::MemberExpr(me) => {
            process_expr(func_name, &me.base, call_graph, data_flow);
        }
        AstNode::IndexExpr(ie) => {
            process_expr(func_name, &ie.base, call_graph, data_flow);
            process_expr(func_name, &ie.index, call_graph, data_flow);
        }
        AstNode::ConditionalExpr(ce) => {
            process_expr(func_name, &ce.cond, call_graph, data_flow);
            process_expr(func_name, &ce.then_branch, call_graph, data_flow);
            process_expr(func_name, &ce.else_branch, call_graph, data_flow);
        }
        AstNode::Identifier(name) => {
            data_flow.add_use(
                name,
                SourceLocation {
                    file: func_name.to_string(),
                    line: 0,
                    col: 0,
                    entity: name.clone(),
                },
            );
        }
        AstNode::BinaryExpr(be)
            if matches!(
                be.op,
                crate::ast::BinaryOp::Assign
                    | crate::ast::BinaryOp::AssignAdd
                    | crate::ast::BinaryOp::AssignSub
                    | crate::ast::BinaryOp::AssignMul
                    | crate::ast::BinaryOp::AssignDiv
                    | crate::ast::BinaryOp::AssignMod
                    | crate::ast::BinaryOp::AssignAnd
                    | crate::ast::BinaryOp::AssignOr
                    | crate::ast::BinaryOp::AssignXor
                    | crate::ast::BinaryOp::AssignShl
                    | crate::ast::BinaryOp::AssignShr
            ) =>
        {
            // LHS is a definition, RHS uses
            process_expr(func_name, &be.right, call_graph, data_flow);
        }
        AstNode::Assignment(assign) => {
            // LHS is a definition, RHS is uses
            process_expr(func_name, &assign.rhs, call_graph, data_flow);
        }
        _ => {}
    }
}

fn process_struct(struct_decl: &StructDecl, type_graph: &mut TypeGraph) -> Result<(), ParseError> {
    let name = struct_decl.name.as_deref().unwrap_or("<anonymous>");
    let fields: Vec<(String, String)> = struct_decl
        .fields
        .iter()
        .map(|f| (f.name.clone(), format!("{:?}", f.ty)))
        .collect();

    type_graph.add_struct(name, &fields);

    // Track field type dependencies
    for field in &struct_decl.fields {
        track_type_deps(name, &field.ty, type_graph);
    }

    Ok(())
}

fn process_typedef(typedef: &TypedefDecl, type_graph: &mut TypeGraph) -> Result<(), ParseError> {
    track_type_deps(&typedef.new_name, &typedef.underlying, type_graph);
    Ok(())
}

fn track_type_deps(dependent: &str, ty: &Type, type_graph: &mut TypeGraph) {
    let deps = extract_type_deps(ty);
    for dep in deps {
        type_graph.add_dependency(dependent, &dep);
    }
}

fn extract_type_deps(ty: &Type) -> Vec<String> {
    let mut deps = Vec::new();
    match ty {
        Type::Struct(name) | Type::Union(name) | Type::Enum(name) | Type::Typedef(name)
            if !name.is_empty() =>
        {
            deps.push(name.clone());
        }
        Type::Pointer(inner)
        | Type::Array(inner, _)
        | Type::Signed(inner)
        | Type::Unsigned(inner)
        | Type::Const(inner)
        | Type::Volatile(inner)
        | Type::Restrict(inner) => {
            deps.extend(extract_type_deps(inner));
        }
        Type::Function(ret, params) => {
            deps.extend(extract_type_deps(ret));
            for param in params {
                deps.extend(extract_type_deps(param));
            }
        }
        _ => {}
    }
    deps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::parse_c_file;

    #[test]
    fn extract_simple_call_graph() {
        let source = r#"
            void bar(void);
            void baz(void);

            void foo(void) {
                bar();
                baz();
            }

            int main(void) {
                foo();
                return 0;
            }
        "#;
        let root = parse_c_file(source).unwrap();
        let graphs = extract_graphs(&root).unwrap();

        assert!(graphs.call_graph.can_reach("main", "foo"));
        assert!(graphs.call_graph.can_reach("foo", "bar"));
        assert!(graphs.call_graph.can_reach("foo", "baz"));
        assert!(graphs.call_graph.can_reach("main", "bar"));
    }

    #[test]
    fn detect_recursion() {
        let source = r#"
            int factorial(int n) {
                if (n <= 1) return 1;
                return n * factorial(n - 1);
            }
        "#;
        let root = parse_c_file(source).unwrap();
        let graphs = extract_graphs(&root).unwrap();

        assert!(graphs.call_graph.is_recursive("factorial"));
    }

    #[test]
    fn find_unused_variables() {
        let source = r#"
            void foo(void) {
                int used = 1;
                int unused = 2;
                int x = used;
            }
        "#;
        let root = parse_c_file(source).unwrap();
        let graphs = extract_graphs(&root).unwrap();

        let unused = graphs.data_flow.find_unused();
        assert!(unused.contains(&"unused".to_string()));
        assert!(!unused.contains(&"used".to_string()));
    }

    #[test]
    fn type_dependency_tracking() {
        let source = r#"
            struct Point { int x; int y; };
            struct Line { struct Point start; struct Point end; };
        "#;
        let root = parse_c_file(source).unwrap();
        let graphs = extract_graphs(&root).unwrap();

        let line_deps = graphs.type_graph.transitive_deps("Line");
        assert!(line_deps.contains("Point"));
    }
}
