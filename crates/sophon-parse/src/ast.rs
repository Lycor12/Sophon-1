//! C AST Parser — Recursive descent building owned AST nodes.
//!
//! Implements MSCA Level 2: Abstract Syntax Tree construction.
//! Builds a full AST with functions, structs, typedefs, statements, expressions.
//! Supports forward declarations, type annotations, and cross-reference tracking.

use crate::lexer::{Token, TokenKind};
use crate::ParseError;

/// AST node types for C programs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstNode {
    // Top-level declarations
    Function(FunctionDecl),
    Struct(StructDecl),
    Typedef(TypedefDecl),
    Enum(EnumDecl),
    GlobalVar(VarDecl),
    PreprocessorDirective(String), // #include, #define, etc.

    // Statements
    CompoundStmt(Vec<Stmt>),
    IfStmt(IfStmt),
    WhileStmt(WhileStmt),
    ForStmt(ForStmt),
    ReturnStmt(Option<Expr>),
    BreakStmt,
    ContinueStmt,
    SwitchStmt(SwitchStmt),
    CaseStmt(i64), // case value
    DefaultStmt,
    ExprStmt(Expr),
    VarDecl(VarDecl),

    // Expressions
    BinaryExpr(BinaryExpr),
    UnaryExpr(UnaryExpr),
    CallExpr(CallExpr),
    MemberExpr(MemberExpr),           // a.b or a->b
    IndexExpr(IndexExpr),             // a[i]
    CastExpr(CastExpr),               // (type) expr
    ConditionalExpr(ConditionalExpr), // a ? b : c
    SizeofExpr(SizeofExpr),
    Literal(Literal),
    Identifier(String),
}

/// Function declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub name: String,
    pub return_type: Type,
    pub params: Vec<Param>,
    pub body: Option<Vec<Stmt>>, // None for forward declaration
    pub is_static: bool,
    pub is_inline: bool,
}

/// Struct declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructDecl {
    pub name: Option<String>, // None for anonymous struct
    pub fields: Vec<Field>,
    pub is_union: bool,
}

/// Typedef declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedefDecl {
    pub new_name: String,
    pub underlying: Type,
}

/// Enum declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDecl {
    pub name: Option<String>,
    pub variants: Vec<(String, Option<i64>)>, // name, optional value
}

/// Variable/field declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDecl {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
}

/// Function parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: Option<String>, // can be omitted in prototypes
    pub ty: Type,
}

/// Struct field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

/// C type representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Bool,
    Signed(Box<Type>),
    Unsigned(Box<Type>),
    Pointer(Box<Type>),
    Array(Box<Type>, Option<usize>), // element type, optional size
    Function(Box<Type>, Vec<Type>),  // return type, param types
    Struct(String),                  // struct name
    Union(String),                   // union name
    Enum(String),                    // enum name
    Typedef(String),                 // typedef name
    Const(Box<Type>),
    Volatile(Box<Type>),
    Restrict(Box<Type>),
}

/// Statement wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Node(AstNode),
    Label(String, Box<Stmt>),
}

/// Expression wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub node: Box<AstNode>,
    pub ty: Option<Type>, // Type annotation (filled by semantic analysis)
}

/// Binary expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub left: Expr,
    pub right: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LogicalAnd,
    LogicalOr,
    Assign,
    AssignAdd,
    AssignSub,
    AssignMul,
    AssignDiv,
    AssignMod,
    AssignAnd,
    AssignOr,
    AssignXor,
    AssignShl,
    AssignShr,
    Comma,
}

/// Unary expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Deref,
    Addr,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
    Sizeof,
}

/// Function call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpr {
    pub func: Expr,
    pub args: Vec<Expr>,
}

/// Member access (a.b or a->b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberExpr {
    pub base: Expr,
    pub member: String,
    pub is_arrow: bool, // true for ->, false for .
}

/// Array index (a[i]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexExpr {
    pub base: Expr,
    pub index: Expr,
}

/// Type cast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastExpr {
    pub target: Type,
    pub expr: Expr,
}

/// Ternary conditional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalExpr {
    pub cond: Expr,
    pub then_branch: Expr,
    pub else_branch: Expr,
}

/// Sizeof expression or type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeofExpr {
    Type(Type),
    Expr(Expr),
}

/// Literal values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Int(i64, IntBase), // value, base
    Float(f64),
    Char(u8),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBase {
    Decimal,
    Hex,
    Octal,
}

/// If statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfStmt {
    pub cond: Expr,
    pub then_branch: Box<Stmt>,
    pub else_branch: Option<Box<Stmt>>,
}

/// While loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhileStmt {
    pub cond: Expr,
    pub body: Box<Stmt>,
}

/// For loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForStmt {
    pub init: Option<Box<AstNode>>, // VarDecl or ExprStmt
    pub cond: Option<Expr>,
    pub step: Option<Expr>,
    pub body: Box<Stmt>,
}

/// Switch statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchStmt {
    pub expr: Expr,
    pub cases: Vec<(i64, Vec<Stmt>)>, // value, statements
    pub default: Option<Vec<Stmt>>,
}

/// Complete AST root.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AstRoot {
    pub declarations: Vec<AstNode>,
    pub functions: Vec<FunctionDecl>,
    pub structs: Vec<StructDecl>,
    pub typedefs: Vec<TypedefDecl>,
    pub enums: Vec<EnumDecl>,
    pub globals: Vec<VarDecl>,
}

/// Parser state.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Create a new parser from a vector of tokens.
    ///
    /// The parser takes ownership of the tokens and starts at position 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use sophon_parse::lexer::{lex_c, Token};
    /// use sophon_parse::ast::Parser;
    ///
    /// let source = "int x = 42;";
    /// let tokens = lex_c(source).unwrap();
    /// let parser = Parser::new(tokens);
    /// ```
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Get a reference to the token at the current position.
    ///
    /// If the current position is past the end of the token stream,
    /// returns the last token (typically EOF).
    ///
    /// # Returns
    ///
    /// A reference to the current token.
    fn current(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or(&self.tokens[self.tokens.len() - 1])
    }

    /// Look ahead in the token stream without advancing.
    ///
    /// # Arguments
    ///
    /// * `offset` - The number of positions to look ahead (0 = current)
    ///
    /// # Returns
    ///
    /// A reference to the token at `pos + offset`, or the last token
    /// if the offset goes past the end of the stream.
    fn peek(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.pos + offset)
            .unwrap_or(&self.tokens[self.tokens.len() - 1])
    }

    /// Advance the parser position by one token.
    ///
    /// Does nothing if already at the end of the token stream.
    fn advance(&mut self) {
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
    }

    /// Expect a specific token kind at the current position.
    ///
    /// If the current token matches the expected kind, consumes it and
    /// returns a clone of the token. Otherwise, returns a parse error.
    ///
    /// # Arguments
    ///
    /// * `kind` - The expected token kind
    ///
    /// # Returns
    ///
    /// `Ok(Token)` if the token matches, `Err(ParseError)` otherwise.
    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        if self.current().kind == kind {
            let token = self.current().clone();
            self.advance();
            Ok(token)
        } else {
            Err(ParseError::ParseError {
                line: self.current().pos.line,
                col: self.current().pos.col,
                msg: format!("Expected {:?}, found {:?}", kind, self.current().kind),
            })
        }
    }

    /// Check if the current token matches a specific kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The token kind to check against
    ///
    /// # Returns
    ///
    /// `true` if the current token's kind equals `kind`, `false` otherwise.
    fn match_kind(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

    /// Check if the current token matches any of the given kinds.
    ///
    /// # Arguments
    ///
    /// * `kinds` - A slice of token kinds to check against
    ///
    /// # Returns
    ///
    /// `true` if the current token's kind is in `kinds`, `false` otherwise.
    fn match_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|k| self.match_kind(*k))
    }

    /// Skip whitespace, newlines, and comments.
    ///
    /// Advances the parser past any tokens that are not significant
    /// for parsing (whitespace, newlines, line comments, block comments).
    fn skip_whitespace_and_comments(&mut self) {
        while matches!(
            self.current().kind,
            TokenKind::Whitespace
                | TokenKind::Newline
                | TokenKind::LineComment
                | TokenKind::BlockComment
        ) {
            self.advance();
        }
    }

    /// Parse complete source into AST root.
    pub fn parse(&mut self) -> Result<AstRoot, ParseError> {
        let mut root = AstRoot::default();

        while !self.match_kind(TokenKind::Eof) {
            self.skip_whitespace_and_comments();
            if self.match_kind(TokenKind::Eof) {
                break;
            }

            match self.parse_top_level()? {
                AstNode::Function(f) => {
                    root.declarations.push(AstNode::Function(f.clone()));
                    root.functions.push(f);
                }
                AstNode::Struct(s) => {
                    root.declarations.push(AstNode::Struct(s.clone()));
                    root.structs.push(s);
                }
                AstNode::Typedef(t) => {
                    root.declarations.push(AstNode::Typedef(t.clone()));
                    root.typedefs.push(t);
                }
                AstNode::Enum(e) => {
                    root.declarations.push(AstNode::Enum(e.clone()));
                    root.enums.push(e);
                }
                AstNode::GlobalVar(v) => {
                    root.declarations.push(AstNode::GlobalVar(v.clone()));
                    root.globals.push(v);
                }
                AstNode::PreprocessorDirective(d) => {
                    root.declarations.push(AstNode::PreprocessorDirective(d));
                }
                _ => {}
            }
        }

        Ok(root)
    }

    fn parse_top_level(&mut self) -> Result<AstNode, ParseError> {
        self.skip_whitespace_and_comments();

        // Preprocessor directive
        if self.match_kind(TokenKind::Preprocessor) {
            let text = self.current().text.clone();
            self.advance();
            return Ok(AstNode::PreprocessorDirective(text));
        }

        // Typedef
        if self.match_kind(TokenKind::Typedef) {
            return self.parse_typedef();
        }

        // Struct/union
        if self.match_kind(TokenKind::Struct) || self.match_kind(TokenKind::Union) {
            return self.parse_struct_or_union();
        }

        // Enum
        if self.match_kind(TokenKind::Enum) {
            return self.parse_enum();
        }

        // Function or global variable
        self.parse_function_or_global()
    }

    fn parse_typedef(&mut self) -> Result<AstNode, ParseError> {
        self.expect(TokenKind::Typedef)?;

        let base_type = self.parse_type()?;
        let name = self.expect(TokenKind::Identifier)?.text.clone();

        self.expect(TokenKind::Semi)?;

        Ok(AstNode::Typedef(TypedefDecl {
            new_name: name,
            underlying: base_type,
        }))
    }

    fn parse_struct_or_union(&mut self) -> Result<AstNode, ParseError> {
        let is_union = self.match_kind(TokenKind::Union);
        self.advance();

        let name = if self.match_kind(TokenKind::Identifier) {
            Some(self.current().text.clone())
        } else {
            None
        };

        if name.is_some() {
            self.advance();
        }

        let mut fields = Vec::new();

        if self.match_kind(TokenKind::LBrace) {
            self.advance();

            while !self.match_kind(TokenKind::RBrace) && !self.match_kind(TokenKind::Eof) {
                self.skip_whitespace_and_comments();
                if self.match_kind(TokenKind::RBrace) {
                    break;
                }

                let field_ty = self.parse_type()?;
                let field_name = self.expect(TokenKind::Identifier)?.text.clone();

                // Array size?
                if self.match_kind(TokenKind::LBracket) {
                    self.advance();
                    let size = self.parse_const_expr()?.to_const_int();
                    self.expect(TokenKind::RBracket)?;
                    // Type is already parsed, need to wrap in Array - simplified for now
                }

                self.expect(TokenKind::Semi)?;
                fields.push(Field {
                    name: field_name,
                    ty: field_ty,
                });
            }

            self.expect(TokenKind::RBrace)?;
        }

        // Optional variable declaration or just forward declaration
        if self.match_kind(TokenKind::Semi) {
            self.advance();
        } else if self.match_kind(TokenKind::Identifier) {
            // Variable name after struct definition
            self.advance();
            self.expect(TokenKind::Semi)?;
        }

        Ok(AstNode::Struct(StructDecl {
            name,
            fields,
            is_union,
        }))
    }

    fn parse_enum(&mut self) -> Result<AstNode, ParseError> {
        self.expect(TokenKind::Enum)?;

        let name = if self.match_kind(TokenKind::Identifier) {
            let n = self.current().text.clone();
            self.advance();
            Some(n)
        } else {
            None
        };

        let mut variants = Vec::new();

        if self.match_kind(TokenKind::LBrace) {
            self.advance();

            while !self.match_kind(TokenKind::RBrace) && !self.match_kind(TokenKind::Eof) {
                self.skip_whitespace_and_comments();
                if self.match_kind(TokenKind::RBrace) {
                    break;
                }

                let variant_name = self.expect(TokenKind::Identifier)?.text.clone();

                let value = if self.match_kind(TokenKind::Eq) {
                    self.advance();
                    Some(self.parse_const_expr()?.to_const_int())
                } else {
                    None
                };

                variants.push((variant_name, value));

                if self.match_kind(TokenKind::Comma) {
                    self.advance();
                }
            }

            self.expect(TokenKind::RBrace)?;
        }

        if self.match_kind(TokenKind::Semi) {
            self.advance();
        }

        Ok(AstNode::Enum(EnumDecl { name, variants }))
    }

    fn parse_function_or_global(&mut self) -> Result<AstNode, ParseError> {
        // Check for storage class specifiers
        let is_static = self.match_kind(TokenKind::Static);
        let is_inline = self.match_kind(TokenKind::Inline);

        if is_static {
            self.advance();
            self.skip_whitespace_and_comments();
        }
        if is_inline {
            self.advance();
            self.skip_whitespace_and_comments();
        }

        let return_type = self.parse_type()?;
        let name = self.expect(TokenKind::Identifier)?.text.clone();

        self.expect(TokenKind::LParen)?;

        // Parameter list
        let mut params = Vec::new();

        if !self.match_kind(TokenKind::RParen) {
            loop {
                self.skip_whitespace_and_comments();

                if self.match_kind(TokenKind::Void) && params.is_empty() {
                    self.advance();
                    break;
                }

                let param_ty = self.parse_type()?;
                let param_name = if self.match_kind(TokenKind::Identifier) {
                    let n = self.current().text.clone();
                    self.advance();
                    Some(n)
                } else {
                    None
                };

                params.push(Param {
                    name: param_name,
                    ty: param_ty,
                });

                if self.match_kind(TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        // Function body or forward declaration
        let body = if self.match_kind(TokenKind::LBrace) {
            Some(self.parse_compound_stmt()?.into_vec())
        } else {
            self.expect(TokenKind::Semi)?;
            None
        };

        Ok(AstNode::Function(FunctionDecl {
            name,
            return_type,
            params,
            body,
            is_static,
            is_inline,
        }))
    }

    fn parse_compound_stmt(&mut self) -> Result<AstNode, ParseError> {
        self.expect(TokenKind::LBrace)?;
        let mut stmts = Vec::new();

        while !self.match_kind(TokenKind::RBrace) && !self.match_kind(TokenKind::Eof) {
            self.skip_whitespace_and_comments();
            if self.match_kind(TokenKind::RBrace) {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }

        self.expect(TokenKind::RBrace)?;
        Ok(AstNode::CompoundStmt(stmts))
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.skip_whitespace_and_comments();

        // Labeled statement
        if self.peek(0).kind == TokenKind::Identifier && self.peek(1).kind == TokenKind::Colon {
            let label = self.current().text.clone();
            self.advance();
            self.advance(); // consume :
            let stmt = self.parse_stmt()?;
            return Ok(Stmt::Label(label, Box::new(stmt)));
        }

        // Variable declaration (needs lookahead for type)
        if self.is_type_start() {
            let var = self.parse_var_decl()?;
            self.expect(TokenKind::Semi)?;
            return Ok(Stmt::Node(AstNode::VarDecl(var)));
        }

        // If statement
        if self.match_kind(TokenKind::If) {
            return self.parse_if_stmt();
        }

        // While loop
        if self.match_kind(TokenKind::While) {
            return self.parse_while_stmt();
        }

        // For loop
        if self.match_kind(TokenKind::For) {
            return self.parse_for_stmt();
        }

        // Return statement
        if self.match_kind(TokenKind::Return) {
            return self.parse_return_stmt();
        }

        // Break statement
        if self.match_kind(TokenKind::Break) {
            self.advance();
            self.expect(TokenKind::Semi)?;
            return Ok(Stmt::Node(AstNode::BreakStmt));
        }

        // Continue statement
        if self.match_kind(TokenKind::Continue) {
            self.advance();
            self.expect(TokenKind::Semi)?;
            return Ok(Stmt::Node(AstNode::ContinueStmt));
        }

        // Compound statement
        if self.match_kind(TokenKind::LBrace) {
            return Ok(Stmt::Node(self.parse_compound_stmt()?));
        }

        // Switch statement
        if self.match_kind(TokenKind::Switch) {
            return self.parse_switch_stmt();
        }

        // Case/Default (inside switch)
        if self.match_kind(TokenKind::Case) {
            self.advance();
            let val = self.parse_const_expr()?.to_const_int();
            self.expect(TokenKind::Colon)?;
            return Ok(Stmt::Node(AstNode::CaseStmt(val)));
        }

        if self.match_kind(TokenKind::Default) {
            self.advance();
            self.expect(TokenKind::Colon)?;
            return Ok(Stmt::Node(AstNode::DefaultStmt));
        }

        // Expression statement
        let expr = self.parse_expr()?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::Node(AstNode::ExprStmt(expr)))
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::If)?;
        self.expect(TokenKind::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        let then_branch = Box::new(self.parse_stmt()?);

        let else_branch = if self.match_kind(TokenKind::Else) {
            self.advance();
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };

        Ok(Stmt::Node(AstNode::IfStmt(IfStmt {
            cond,
            then_branch,
            else_branch,
        })))
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::While)?;
        self.expect(TokenKind::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::Node(AstNode::WhileStmt(WhileStmt { cond, body })))
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::For)?;
        self.expect(TokenKind::LParen)?;

        let init = if self.match_kind(TokenKind::Semi) {
            None
        } else if self.is_type_start() {
            Some(Box::new(AstNode::VarDecl(self.parse_var_decl()?)))
        } else {
            let expr = self.parse_expr()?;
            self.expect(TokenKind::Semi)?;
            Some(Box::new(AstNode::ExprStmt(expr)))
        };

        if init.is_none() {
            self.expect(TokenKind::Semi)?;
        }

        let cond = if self.match_kind(TokenKind::Semi) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(TokenKind::Semi)?;

        let step = if self.match_kind(TokenKind::RParen) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(TokenKind::RParen)?;

        let body = Box::new(self.parse_stmt()?);

        Ok(Stmt::Node(AstNode::ForStmt(ForStmt {
            init,
            cond,
            step,
            body,
        })))
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::Return)?;
        let expr = if self.match_kind(TokenKind::Semi) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::Node(AstNode::ReturnStmt(expr)))
    }

    fn parse_switch_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::Switch)?;
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        // Parse body - should be compound with case/default labels
        self.expect(TokenKind::LBrace)?;
        let mut cases = Vec::new();
        let mut default = None;

        while !self.match_kind(TokenKind::RBrace) && !self.match_kind(TokenKind::Eof) {
            self.skip_whitespace_and_comments();

            if self.match_kind(TokenKind::Case) {
                self.advance();
                let val = self.parse_const_expr()?.to_const_int();
                self.expect(TokenKind::Colon)?;

                let mut stmts = Vec::new();
                while !self.match_kind(TokenKind::Case)
                    && !self.match_kind(TokenKind::Default)
                    && !self.match_kind(TokenKind::RBrace)
                {
                    stmts.push(self.parse_stmt()?);
                    self.skip_whitespace_and_comments();
                }
                cases.push((val, stmts));
            } else if self.match_kind(TokenKind::Default) {
                self.advance();
                self.expect(TokenKind::Colon)?;

                let mut stmts = Vec::new();
                while !self.match_kind(TokenKind::Case)
                    && !self.match_kind(TokenKind::Default)
                    && !self.match_kind(TokenKind::RBrace)
                {
                    stmts.push(self.parse_stmt()?);
                    self.skip_whitespace_and_comments();
                }
                default = Some(stmts);
            } else {
                // Regular statement (fall through from previous case)
                self.parse_stmt()?;
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(Stmt::Node(AstNode::SwitchStmt(SwitchStmt {
            expr,
            cases,
            default,
        })))
    }

    fn parse_var_decl(&mut self) -> Result<VarDecl, ParseError> {
        let ty = self.parse_type()?;
        let name = self.expect(TokenKind::Identifier)?.text.clone();

        // Array size?
        let ty = if self.match_kind(TokenKind::LBracket) {
            self.advance();
            let size = self.parse_const_expr()?.to_const_int() as usize;
            self.expect(TokenKind::RBracket)?;
            Type::Array(Box::new(ty), Some(size))
        } else {
            ty
        };

        let init = if self.match_kind(TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(VarDecl { name, ty, init })
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_assign_expr()
    }

    fn parse_assign_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_conditional_expr()?;

        if self.match_any(&[
            TokenKind::Eq,
            TokenKind::PlusEq,
            TokenKind::MinusEq,
            TokenKind::StarEq,
            TokenKind::SlashEq,
            TokenKind::PercentEq,
            TokenKind::AmpEq,
            TokenKind::PipeEq,
            TokenKind::CaretEq,
            TokenKind::LtLtEq,
            TokenKind::GtGtEq,
        ]) {
            let op = match self.current().kind {
                TokenKind::Eq => BinaryOp::Assign,
                TokenKind::PlusEq => BinaryOp::AssignAdd,
                TokenKind::MinusEq => BinaryOp::AssignSub,
                TokenKind::StarEq => BinaryOp::AssignMul,
                TokenKind::SlashEq => BinaryOp::AssignDiv,
                TokenKind::PercentEq => BinaryOp::AssignMod,
                TokenKind::AmpEq => BinaryOp::AssignAnd,
                TokenKind::PipeEq => BinaryOp::AssignOr,
                TokenKind::CaretEq => BinaryOp::AssignXor,
                TokenKind::LtLtEq => BinaryOp::AssignShl,
                TokenKind::GtGtEq => BinaryOp::AssignShr,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_assign_expr()?;
            expr = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op,
                    left: expr,
                    right,
                })),
                ty: None,
            };
        }

        Ok(expr)
    }

    fn parse_conditional_expr(&mut self) -> Result<Expr, ParseError> {
        let cond = self.parse_or_expr()?;

        if self.match_kind(TokenKind::Question) {
            self.advance();
            let then_branch = self.parse_expr()?;
            self.expect(TokenKind::Colon)?;
            let else_branch = self.parse_conditional_expr()?;
            Ok(Expr {
                node: Box::new(AstNode::ConditionalExpr(ConditionalExpr {
                    cond,
                    then_branch,
                    else_branch,
                })),
                ty: None,
            })
        } else {
            Ok(cond)
        }
    }

    fn parse_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and_expr()?;

        while self.match_kind(TokenKind::OrOr) {
            self.advance();
            let right = self.parse_and_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op: BinaryOp::LogicalOr,
                    left,
                    right,
                })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_or_expr()?;

        while self.match_kind(TokenKind::AndAnd) {
            self.advance();
            let right = self.parse_bit_or_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op: BinaryOp::LogicalAnd,
                    left,
                    right,
                })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_bit_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_xor_expr()?;

        while self.match_kind(TokenKind::Pipe) {
            self.advance();
            let right = self.parse_bit_xor_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op: BinaryOp::Or,
                    left,
                    right,
                })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_bit_xor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bit_and_expr()?;

        while self.match_kind(TokenKind::Caret) {
            self.advance();
            let right = self.parse_bit_and_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op: BinaryOp::Xor,
                    left,
                    right,
                })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_bit_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_equality_expr()?;

        while self.match_kind(TokenKind::Ampersand) {
            self.advance();
            let right = self.parse_equality_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr {
                    op: BinaryOp::And,
                    left,
                    right,
                })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_equality_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_relational_expr()?;

        while self.match_any(&[TokenKind::EqEq, TokenKind::NotEq]) {
            let op = if self.match_kind(TokenKind::EqEq) {
                self.advance();
                BinaryOp::Eq
            } else {
                self.advance();
                BinaryOp::Ne
            };
            let right = self.parse_relational_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr { op, left, right })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_relational_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_shift_expr()?;

        while self.match_any(&[
            TokenKind::Lt,
            TokenKind::Gt,
            TokenKind::LtEq,
            TokenKind::GtEq,
        ]) {
            let op = match self.current().kind {
                TokenKind::Lt => {
                    self.advance();
                    BinaryOp::Lt
                }
                TokenKind::Gt => {
                    self.advance();
                    BinaryOp::Gt
                }
                TokenKind::LtEq => {
                    self.advance();
                    BinaryOp::Le
                }
                TokenKind::GtEq => {
                    self.advance();
                    BinaryOp::Ge
                }
                _ => unreachable!(),
            };
            let right = self.parse_shift_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr { op, left, right })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_shift_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive_expr()?;

        while self.match_any(&[TokenKind::LtLt, TokenKind::GtGt]) {
            let op = if self.match_kind(TokenKind::LtLt) {
                self.advance();
                BinaryOp::Shl
            } else {
                self.advance();
                BinaryOp::Shr
            };
            let right = self.parse_additive_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr { op, left, right })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative_expr()?;

        while self.match_any(&[TokenKind::Plus, TokenKind::Minus]) {
            let op = if self.match_kind(TokenKind::Plus) {
                self.advance();
                BinaryOp::Add
            } else {
                self.advance();
                BinaryOp::Sub
            };
            let right = self.parse_multiplicative_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr { op, left, right })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_cast_expr()?;

        while self.match_any(&[TokenKind::Star, TokenKind::Slash, TokenKind::Percent]) {
            let op = match self.current().kind {
                TokenKind::Star => {
                    self.advance();
                    BinaryOp::Mul
                }
                TokenKind::Slash => {
                    self.advance();
                    BinaryOp::Div
                }
                TokenKind::Percent => {
                    self.advance();
                    BinaryOp::Mod
                }
                _ => unreachable!(),
            };
            let right = self.parse_cast_expr()?;
            left = Expr {
                node: Box::new(AstNode::BinaryExpr(BinaryExpr { op, left, right })),
                ty: None,
            };
        }

        Ok(left)
    }

    fn parse_cast_expr(&mut self) -> Result<Expr, ParseError> {
        if self.match_kind(TokenKind::LParen) && self.is_type_lookahead() {
            self.advance();
            let target = self.parse_type()?;
            self.expect(TokenKind::RParen)?;
            let expr = self.parse_cast_expr()?;
            return Ok(Expr {
                node: Box::new(AstNode::CastExpr(CastExpr { target, expr })),
                ty: None,
            });
        }
        self.parse_unary_expr()
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        match self.current().kind {
            TokenKind::Plus => {
                self.advance();
                self.parse_unary_expr() // Unary plus is no-op
            }
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::Neg,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Bang => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::Not,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::BitNot,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Star => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::Deref,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Ampersand => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::Addr,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Sizeof => {
                self.advance();
                if self.match_kind(TokenKind::LParen) && self.is_type_lookahead() {
                    self.advance();
                    let ty = self.parse_type()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr {
                        node: Box::new(AstNode::SizeofExpr(SizeofExpr::Type(ty))),
                        ty: None,
                    })
                } else {
                    let expr = self.parse_unary_expr()?;
                    Ok(Expr {
                        node: Box::new(AstNode::SizeofExpr(SizeofExpr::Expr(expr))),
                        ty: None,
                    })
                }
            }
            TokenKind::Inc => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::PreInc,
                        operand,
                    })),
                    ty: None,
                })
            }
            TokenKind::Dec => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::PreDec,
                        operand,
                    })),
                    ty: None,
                })
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            if self.match_kind(TokenKind::LBracket) {
                self.advance();
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                expr = Expr {
                    node: Box::new(AstNode::IndexExpr(IndexExpr { base: expr, index })),
                    ty: None,
                };
            } else if self.match_kind(TokenKind::LParen) {
                self.advance();
                let mut args = Vec::new();
                if !self.match_kind(TokenKind::RParen) {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.match_kind(TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                expr = Expr {
                    node: Box::new(AstNode::CallExpr(CallExpr { func: expr, args })),
                    ty: None,
                };
            } else if self.match_kind(TokenKind::Dot) {
                self.advance();
                let member = self.expect(TokenKind::Identifier)?.text.clone();
                expr = Expr {
                    node: Box::new(AstNode::MemberExpr(MemberExpr {
                        base: expr,
                        member,
                        is_arrow: false,
                    })),
                    ty: None,
                };
            } else if self.match_kind(TokenKind::Arrow) {
                self.advance();
                let member = self.expect(TokenKind::Identifier)?.text.clone();
                expr = Expr {
                    node: Box::new(AstNode::MemberExpr(MemberExpr {
                        base: expr,
                        member,
                        is_arrow: true,
                    })),
                    ty: None,
                };
            } else if self.match_kind(TokenKind::Inc) {
                self.advance();
                expr = Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::PostInc,
                        operand: expr,
                    })),
                    ty: None,
                };
            } else if self.match_kind(TokenKind::Dec) {
                self.advance();
                expr = Expr {
                    node: Box::new(AstNode::UnaryExpr(UnaryExpr {
                        op: UnaryOp::PostDec,
                        operand: expr,
                    })),
                    ty: None,
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        self.skip_whitespace_and_comments();

        match self.current().kind {
            TokenKind::Identifier => {
                let name = self.current().text.clone();
                self.advance();
                Ok(Expr {
                    node: Box::new(AstNode::Identifier(name)),
                    ty: None,
                })
            }
            TokenKind::IntLiteral => {
                let text = self.current().text.clone();
                let (value, base) = Self::parse_int_literal(&text);
                self.advance();
                Ok(Expr {
                    node: Box::new(AstNode::Literal(Literal::Int(value, base))),
                    ty: Some(Type::Int),
                })
            }
            TokenKind::FloatLiteral => {
                let text = self.current().text.clone();
                let value = text.parse::<f64>().unwrap_or(0.0);
                self.advance();
                Ok(Expr {
                    node: Box::new(AstNode::Literal(Literal::Float(value))),
                    ty: Some(Type::Double),
                })
            }
            TokenKind::CharLiteral => {
                let text = self.current().text.clone();
                let value = Self::parse_char_literal(&text);
                self.advance();
                Ok(Expr {
                    node: Box::new(AstNode::Literal(Literal::Char(value))),
                    ty: Some(Type::Char),
                })
            }
            TokenKind::StringLiteral => {
                let text = self.current().text.clone();
                self.advance();
                Ok(Expr {
                    node: Box::new(AstNode::Literal(Literal::String(text))),
                    ty: Some(Type::Pointer(Box::new(Type::Char))),
                })
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            _ => Err(ParseError::ParseError {
                line: self.current().pos.line,
                col: self.current().pos.col,
                msg: format!("Unexpected token in expression: {:?}", self.current().kind),
            }),
        }
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let mut ty = self.parse_type_specifier()?;

        // Parse pointer/restrict qualifiers
        loop {
            self.skip_whitespace_and_comments();
            if self.match_kind(TokenKind::Star) {
                self.advance();
                // Handle const/restrict/volatile
                let mut pointer_ty = Type::Pointer(Box::new(ty));
                loop {
                    self.skip_whitespace_and_comments();
                    if self.match_kind(TokenKind::Const) {
                        self.advance();
                        pointer_ty = Type::Const(Box::new(pointer_ty));
                    } else if self.match_kind(TokenKind::Restrict) {
                        self.advance();
                        pointer_ty = Type::Restrict(Box::new(pointer_ty));
                    } else if self.match_kind(TokenKind::Volatile) {
                        self.advance();
                        pointer_ty = Type::Volatile(Box::new(pointer_ty));
                    } else {
                        break;
                    }
                }
                ty = pointer_ty;
            } else {
                break;
            }
        }

        // Array dimensions
        while self.match_kind(TokenKind::LBracket) {
            self.advance();
            let size = if self.match_kind(TokenKind::RBracket) {
                None
            } else {
                Some(self.parse_const_expr()?.to_const_int() as usize)
            };
            self.expect(TokenKind::RBracket)?;
            ty = Type::Array(Box::new(ty), size);
        }

        Ok(ty)
    }

    fn parse_type_specifier(&mut self) -> Result<Type, ParseError> {
        // Check for struct/union/enum
        if self.match_kind(TokenKind::Struct) {
            self.advance();
            if self.match_kind(TokenKind::Identifier) {
                let name = self.current().text.clone();
                self.advance();
                Ok(Type::Struct(name))
            } else {
                Ok(Type::Struct(String::new())) // Anonymous
            }
        } else if self.match_kind(TokenKind::Union) {
            self.advance();
            if self.match_kind(TokenKind::Identifier) {
                let name = self.current().text.clone();
                self.advance();
                Ok(Type::Union(name))
            } else {
                Ok(Type::Union(String::new()))
            }
        } else if self.match_kind(TokenKind::Enum) {
            self.advance();
            if self.match_kind(TokenKind::Identifier) {
                let name = self.current().text.clone();
                self.advance();
                Ok(Type::Enum(name))
            } else {
                Ok(Type::Enum(String::new()))
            }
        } else {
            // Simple type keywords
            let mut signed = false;
            let mut unsigned = false;
            let mut base = None;

            loop {
                self.skip_whitespace_and_comments();
                match self.current().kind {
                    TokenKind::Signed => {
                        signed = true;
                        self.advance();
                    }
                    TokenKind::Unsigned => {
                        unsigned = true;
                        self.advance();
                    }
                    TokenKind::Void => {
                        base = Some(Type::Void);
                        self.advance();
                        break;
                    }
                    TokenKind::Char => {
                        base = Some(Type::Char);
                        self.advance();
                        break;
                    }
                    TokenKind::Short => {
                        base = Some(Type::Short);
                        self.advance();
                        break;
                    }
                    TokenKind::Int => {
                        base = Some(Type::Int);
                        self.advance();
                        break;
                    }
                    TokenKind::Long => {
                        self.advance();
                        if self.match_kind(TokenKind::Long) || self.match_kind(TokenKind::Int) {
                            base = Some(Type::Long);
                            self.advance();
                        } else {
                            base = Some(Type::Long);
                        }
                        break;
                    }
                    TokenKind::Float => {
                        base = Some(Type::Float);
                        self.advance();
                        break;
                    }
                    TokenKind::Double => {
                        base = Some(Type::Double);
                        self.advance();
                        break;
                    }
                    TokenKind::Bool => {
                        base = Some(Type::Bool);
                        self.advance();
                        break;
                    }
                    TokenKind::Identifier => {
                        // Typedef'd type
                        let name = self.current().text.clone();
                        base = Some(Type::Typedef(name));
                        self.advance();
                        break;
                    }
                    TokenKind::Const => {
                        base = Some(Type::Const(Box::new(self.parse_type()?)));
                        self.advance();
                        break;
                    }
                    TokenKind::Volatile => {
                        base = Some(Type::Volatile(Box::new(self.parse_type()?)));
                        self.advance();
                        break;
                    }
                    _ => break,
                }
            }

            let mut ty = base.ok_or_else(|| ParseError::ParseError {
                line: self.current().pos.line,
                col: self.current().pos.col,
                msg: "Expected type specifier".to_string(),
            })?;

            if unsigned {
                ty = Type::Unsigned(Box::new(ty));
            } else if signed {
                ty = Type::Signed(Box::new(ty));
            }

            Ok(ty)
        }
    }

    fn parse_const_expr(&mut self) -> Result<ConstExpr, ParseError> {
        // Simplified: just parse as regular expression and evaluate if constant
        let expr = self.parse_expr()?;
        Ok(ConstExpr(expr))
    }

    fn is_type_start(&self) -> bool {
        self.skip_whitespace_and_comments();
        matches!(
            self.current().kind,
            TokenKind::Void
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Bool
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                | TokenKind::Const
                | TokenKind::Volatile
                | TokenKind::Static
                | TokenKind::Extern
                | TokenKind::Inline
                | TokenKind::Typedef
                | TokenKind::Identifier
        ) && !self.is_postfix_operator()
    }

    fn is_type_lookahead(&self) -> bool {
        matches!(
            self.peek(1).kind,
            TokenKind::Void
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Bool
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
        )
    }

    fn is_postfix_operator(&self) -> bool {
        matches!(
            self.peek(1).kind,
            TokenKind::LParen | TokenKind::LBracket | TokenKind::Dot | TokenKind::Arrow
        )
    }

    fn parse_int_literal(text: &str) -> (i64, IntBase) {
        let text = text.trim();
        if text.starts_with("0x") || text.starts_with("0X") {
            let val = i64::from_str_radix(&text[2..], 16).unwrap_or(0);
            (val, IntBase::Hex)
        } else if text.starts_with("0") && text.len() > 1 {
            let val = i64::from_str_radix(&text[1..], 8).unwrap_or(0);
            (val, IntBase::Octal)
        } else {
            let val = text.parse::<i64>().unwrap_or(0);
            (val, IntBase::Decimal)
        }
    }

    fn parse_char_literal(text: &str) -> u8 {
        // Extract character from 'x' or '\x' format
        let chars: Vec<char> = text.chars().collect();
        if chars.len() >= 3 && chars[0] == '\'' && chars[chars.len() - 1] == '\'' {
            if chars[1] == '\\' && chars.len() > 3 {
                // Escape sequence
                match chars[2] {
                    'n' => b'\n',
                    't' => b'\t',
                    'r' => b'\r',
                    '0' => b'\0',
                    '\\' => b'\\',
                    '\'' => b'\'',
                    '"' => b'"',
                    _ => chars[2] as u8,
                }
            } else {
                chars[1] as u8
            }
        } else {
            0
        }
    }
}

/// Wrapper for constant expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstExpr(Expr);

impl ConstExpr {
    /// Evaluate constant expression to integer (simplified).
    pub fn to_const_int(&self) -> i64 {
        Self::eval_const(&self.0.node)
    }

    fn eval_const(node: &AstNode) -> i64 {
        match node {
            AstNode::Literal(Literal::Int(n, _)) => *n,
            AstNode::BinaryExpr(be) => {
                let left = Self::eval_const(&be.left.node);
                let right = Self::eval_const(&be.right.node);
                match be.op {
                    BinaryOp::Add => left + right,
                    BinaryOp::Sub => left - right,
                    BinaryOp::Mul => left * right,
                    BinaryOp::Div => {
                        if right != 0 {
                            left / right
                        } else {
                            0
                        }
                    }
                    BinaryOp::Mod => {
                        if right != 0 {
                            left % right
                        } else {
                            0
                        }
                    }
                    BinaryOp::And => left & right,
                    BinaryOp::Or => left | right,
                    BinaryOp::Xor => left ^ right,
                    BinaryOp::Shl => left << right,
                    BinaryOp::Shr => left >> right,
                    _ => 0,
                }
            }
            AstNode::UnaryExpr(ue) => {
                let val = Self::eval_const(&ue.operand.node);
                match ue.op {
                    UnaryOp::Neg => -val,
                    UnaryOp::Not => {
                        if val == 0 {
                            1
                        } else {
                            0
                        }
                    }
                    UnaryOp::BitNot => !val,
                    _ => 0,
                }
            }
            _ => 0,
        }
    }
}

trait IntoVec {
    fn into_vec(self) -> Vec<Stmt>;
}

impl IntoVec for AstNode {
    fn into_vec(self) -> Vec<Stmt> {
        match self {
            AstNode::CompoundStmt(stmts) => stmts,
            other => vec![Stmt::Node(other)],
        }
    }
}

/// Parse a C source file into an AST.
pub fn parse_c_file(source: &str) -> Result<AstRoot, ParseError> {
    let tokens = crate::lexer::lex_c(source)?;
    let mut parser = Parser::new(tokens);
    parser.parse()
}

/// Parse a C expression fragment.
pub fn parse_c_fragment(source: &str) -> Result<Expr, ParseError> {
    let tokens = crate::lexer::lex_c(source)?;
    let mut parser = Parser::new(tokens);
    parser.parse_expr()
}

/// Parse tokens into an AST.
pub fn parse_tokens(tokens: &[Token]) -> Result<AstRoot, ParseError> {
    let mut parser = Parser::new(tokens.to_vec());
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_function() {
        let source = r#"
            int main(void) {
                return 0;
            }
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.functions.len(), 1);
        assert_eq!(root.functions[0].name, "main");
        assert!(root.functions[0].body.is_some());
    }

    #[test]
    fn parse_function_with_params() {
        let source = r#"
            int add(int a, int b) {
                return a + b;
            }
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.functions[0].params.len(), 2);
        assert_eq!(root.functions[0].params[0].name.as_deref(), Some("a"));
    }

    #[test]
    fn parse_struct_declaration() {
        let source = r#"
            struct Point {
                int x;
                int y;
            };
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.structs.len(), 1);
        assert_eq!(root.structs[0].name.as_deref(), Some("Point"));
        assert_eq!(root.structs[0].fields.len(), 2);
    }

    #[test]
    fn parse_typedef() {
        let source = r#"typedef unsigned int uint32_t;"#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.typedefs.len(), 1);
        assert_eq!(root.typedefs[0].new_name, "uint32_t");
    }

    #[test]
    fn parse_if_statement() {
        let source = r#"
            int test(int x) {
                if (x > 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.functions.len(), 1);
    }

    #[test]
    fn parse_while_loop() {
        let source = r#"
            void foo() {
                int i = 0;
                while (i < 10) {
                    i = i + 1;
                }
            }
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.functions.len(), 1);
    }

    #[test]
    fn parse_binary_expressions() {
        let source = "a + b * c - d / e";
        let expr = parse_c_fragment(source).unwrap();
        // Should parse as ((a + (b * c)) - (d / e))
        assert!(matches!(expr.node.as_ref(), AstNode::BinaryExpr(_)));
    }

    #[test]
    fn parse_function_call() {
        let source = "foo(a, b + c)";
        let expr = parse_c_fragment(source).unwrap();
        assert!(matches!(expr.node.as_ref(), AstNode::CallExpr(_)));
    }

    #[test]
    fn parse_array_access() {
        let source = "arr[i + 1]";
        let expr = parse_c_fragment(source).unwrap();
        assert!(matches!(expr.node.as_ref(), AstNode::IndexExpr(_)));
    }

    #[test]
    fn parse_member_access() {
        let source = "ptr->field";
        let expr = parse_c_fragment(source).unwrap();
        assert!(matches!(expr.node.as_ref(), AstNode::MemberExpr(me) if me.is_arrow));
    }

    #[test]
    fn parse_typedef_enum() {
        let source = r#"
            typedef enum {
                RED = 0,
                GREEN = 1,
                BLUE = 2
            } Color;
        "#;
        let root = parse_c_file(source).unwrap();
        assert_eq!(root.enums.len(), 1);
        assert_eq!(root.enums[0].variants.len(), 3);
    }
}
