//! C/C++ Lexer — Zero-dependency recursive descent tokenizer.
//!
//! Implements MSCA Level 1: lexical tokenization.
//! Handles C99 subset including: keywords, identifiers, literals,
//! operators, punctuation, comments, and preprocessor directives.

use crate::ParseError;

/// Position in source for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourcePos {
    pub line: usize,
    pub col: usize,
    pub byte_offset: usize,
}

/// C token classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    Auto,
    Break,
    Case,
    Char,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    Float,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Register,
    Restrict,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
    // C99 bool/complex
    Bool,
    Complex,
    Imaginary,

    // Preprocessor
    Preprocessor, // starts with #

    // Literals
    Identifier,
    IntLiteral,
    FloatLiteral,
    CharLiteral,
    StringLiteral,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent, // + - * / %
    Ampersand,
    Pipe,
    Caret,
    Tilde,
    Bang, // & | ^ ~ !
    Lt,
    Gt,
    Eq, // < > =
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq, // += -= *= /= %=
    AmpEq,
    PipeEq,
    CaretEq, // &= |= ^=
    LtLt,
    GtGt, // << >>
    LtLtEq,
    GtGtEq, // <<= >>=
    EqEq,
    NotEq, // == !=
    LtEq,
    GtEq, // <= >=
    Arrow,
    Dot, // -> .
    Inc,
    Dec, // ++ --
    AndAnd,
    OrOr, // && ||
    Question,
    Colon, // ? :
    Comma,
    Semi, // , ;
    LBrace,
    RBrace, // { }
    LBracket,
    RBracket, // [ ]
    LParen,
    RParen, // ( )

    // Comments (preserved as tokens for documentation extraction)
    LineComment,
    BlockComment,

    // Special
    Whitespace,
    Newline,
    Eof,
}

/// C source token with position information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub pos: SourcePos,
}

impl Token {
    pub fn new(kind: TokenKind, text: String, pos: SourcePos) -> Self {
        Self { kind, text, pos }
    }

    /// Is this a keyword?
    pub fn is_keyword(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Auto
                | TokenKind::Break
                | TokenKind::Case
                | TokenKind::Char
                | TokenKind::Const
                | TokenKind::Continue
                | TokenKind::Default
                | TokenKind::Do
                | TokenKind::Double
                | TokenKind::Else
                | TokenKind::Enum
                | TokenKind::Extern
                | TokenKind::Float
                | TokenKind::For
                | TokenKind::Goto
                | TokenKind::If
                | TokenKind::Inline
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Register
                | TokenKind::Restrict
                | TokenKind::Return
                | TokenKind::Short
                | TokenKind::Signed
                | TokenKind::Sizeof
                | TokenKind::Static
                | TokenKind::Struct
                | TokenKind::Switch
                | TokenKind::Typedef
                | TokenKind::Union
                | TokenKind::Unsigned
                | TokenKind::Void
                | TokenKind::Volatile
                | TokenKind::While
                | TokenKind::Bool
                | TokenKind::Complex
                | TokenKind::Imaginary
        )
    }

    /// Is this a type keyword?
    pub fn is_type_keyword(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Char
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::Float
                | TokenKind::Double
                | TokenKind::Void
                | TokenKind::Bool
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Struct
                | TokenKind::Union
                | TokenKind::Enum
                | TokenKind::Typedef
        )
    }
}

/// C lexer with source tracking.
pub struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn current(&self) -> u8 {
        self.source.get(self.pos).copied().unwrap_or(0)
    }

    fn peek(&self, offset: usize) -> u8 {
        self.source.get(self.pos + offset).copied().unwrap_or(0)
    }

    fn advance(&mut self) {
        if self.pos < self.source.len() {
            if self.source[self.pos] == b'\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn skip_while(&mut self, pred: impl Fn(u8) -> bool) {
        while self.pos < self.source.len() && pred(self.current()) {
            self.advance();
        }
    }

    fn source_pos(&self) -> SourcePos {
        SourcePos {
            line: self.line,
            col: self.col,
            byte_offset: self.pos,
        }
    }

    fn text_from(&self, start: usize) -> String {
        String::from_utf8_lossy(&self.source[start..self.pos]).to_string()
    }

    /// Lex a complete C source file into tokens.
    pub fn lex(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();

        while self.pos < self.source.len() {
            let token = self.next_token()?;
            if token.kind == TokenKind::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Get next token.
    fn next_token(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        let pos = self.source_pos();

        // EOF check
        if self.pos >= self.source.len() {
            return Ok(Token::new(TokenKind::Eof, String::new(), pos));
        }

        let c = self.current();

        // Newline
        if c == b'\n' {
            self.advance();
            return Ok(Token::new(TokenKind::Newline, "\n".to_string(), pos));
        }

        // Whitespace (excluding newline)
        if c.is_ascii_whitespace() {
            self.skip_while(|c| c.is_ascii_whitespace() && c != b'\n');
            return Ok(Token::new(
                TokenKind::Whitespace,
                self.text_from(start),
                pos,
            ));
        }

        // Preprocessor directive
        if c == b'#' {
            self.advance();
            // Skip whitespace after #
            self.skip_while(|c| c.is_ascii_whitespace() && c != b'\n');
            // Read rest of line
            while self.pos < self.source.len() && self.current() != b'\n' {
                // Handle line continuation
                if self.current() == b'\\' && self.peek(1) == b'\n' {
                    self.advance(); // \
                    self.advance(); // \n
                } else {
                    self.advance();
                }
            }
            return Ok(Token::new(
                TokenKind::Preprocessor,
                self.text_from(start),
                pos,
            ));
        }

        // Line comment
        if c == b'/' && self.peek(1) == b'/' {
            self.advance();
            self.advance();
            while self.pos < self.source.len() && self.current() != b'\n' {
                self.advance();
            }
            return Ok(Token::new(
                TokenKind::LineComment,
                self.text_from(start),
                pos,
            ));
        }

        // Block comment
        if c == b'/' && self.peek(1) == b'*' {
            self.advance();
            self.advance();
            while self.pos < self.source.len() {
                if self.current() == b'*' && self.peek(1) == b'/' {
                    self.advance();
                    self.advance();
                    break;
                }
                self.advance();
            }
            return Ok(Token::new(
                TokenKind::BlockComment,
                self.text_from(start),
                pos,
            ));
        }

        // Character literal
        if c == b'\'' {
            return self.lex_char_literal(pos);
        }

        // String literal
        if c == b'"' {
            return self.lex_string_literal(pos);
        }

        // Number literal
        if c.is_ascii_digit() {
            return self.lex_number(pos);
        }

        // Identifier or keyword
        if c.is_ascii_alphabetic() || c == b'_' {
            return self.lex_identifier(pos);
        }

        // Multi-character operators
        match (c, self.peek(1)) {
            (b'+', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::PlusEq, "+=".to_string(), pos));
            }
            (b'-', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::MinusEq, "-=".to_string(), pos));
            }
            (b'*', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::StarEq, "*=".to_string(), pos));
            }
            (b'/', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::SlashEq, "/=".to_string(), pos));
            }
            (b'%', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::PercentEq, "%=".to_string(), pos));
            }
            (b'&', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::AmpEq, "&=".to_string(), pos));
            }
            (b'|', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::PipeEq, "|=".to_string(), pos));
            }
            (b'^', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::CaretEq, "^=".to_string(), pos));
            }
            (b'+', b'+') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::Inc, "++".to_string(), pos));
            }
            (b'-', b'-') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::Dec, "--".to_string(), pos));
            }
            (b'-', b'>') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::Arrow, "->".to_string(), pos));
            }
            (b'=', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::EqEq, "==".to_string(), pos));
            }
            (b'!', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::NotEq, "!=".to_string(), pos));
            }
            (b'<', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::LtEq, "<=".to_string(), pos));
            }
            (b'>', b'=') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::GtEq, ">=".to_string(), pos));
            }
            (b'<', b'<') if self.peek(2) != b'=' => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::LtLt, "<<".to_string(), pos));
            }
            (b'>', b'>') if self.peek(2) != b'=' => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::GtGt, ">>".to_string(), pos));
            }
            (b'<', b'<') if self.peek(2) == b'=' => {
                self.advance();
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::LtLtEq, "<<=".to_string(), pos));
            }
            (b'>', b'>') if self.peek(2) == b'=' => {
                self.advance();
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::GtGtEq, ">>=".to_string(), pos));
            }
            (b'&', b'&') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::AndAnd, "&&".to_string(), pos));
            }
            (b'|', b'|') => {
                self.advance();
                self.advance();
                return Ok(Token::new(TokenKind::OrOr, "||".to_string(), pos));
            }
            _ => {}
        }

        // Single-character tokens
        let kind = match c {
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'*' => TokenKind::Star,
            b'/' => TokenKind::Slash,
            b'%' => TokenKind::Percent,
            b'&' => TokenKind::Ampersand,
            b'|' => TokenKind::Pipe,
            b'^' => TokenKind::Caret,
            b'~' => TokenKind::Tilde,
            b'!' => TokenKind::Bang,
            b'<' => TokenKind::Lt,
            b'>' => TokenKind::Gt,
            b'=' => TokenKind::Eq,
            b'.' => TokenKind::Dot,
            b'?' => TokenKind::Question,
            b':' => TokenKind::Colon,
            b',' => TokenKind::Comma,
            b';' => TokenKind::Semi,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            _ => {
                return Err(ParseError::LexError {
                    line: self.line,
                    col: self.col,
                    msg: format!("Unexpected character: '{}'", c as char),
                });
            }
        };

        self.advance();
        Ok(Token::new(
            kind,
            String::from_utf8_lossy(&[c]).to_string(),
            pos,
        ))
    }

    fn lex_identifier(&mut self, pos: SourcePos) -> Result<Token, ParseError> {
        let start = self.pos;
        self.skip_while(|c| c.is_ascii_alphanumeric() || c == b'_');
        let text = self.text_from(start);
        let kind = match text.as_str() {
            "auto" => TokenKind::Auto,
            "break" => TokenKind::Break,
            "case" => TokenKind::Case,
            "char" => TokenKind::Char,
            "const" => TokenKind::Const,
            "continue" => TokenKind::Continue,
            "default" => TokenKind::Default,
            "do" => TokenKind::Do,
            "double" => TokenKind::Double,
            "else" => TokenKind::Else,
            "enum" => TokenKind::Enum,
            "extern" => TokenKind::Extern,
            "float" => TokenKind::Float,
            "for" => TokenKind::For,
            "goto" => TokenKind::Goto,
            "if" => TokenKind::If,
            "inline" => TokenKind::Inline,
            "int" => TokenKind::Int,
            "long" => TokenKind::Long,
            "register" => TokenKind::Register,
            "restrict" => TokenKind::Restrict,
            "return" => TokenKind::Return,
            "short" => TokenKind::Short,
            "signed" => TokenKind::Signed,
            "sizeof" => TokenKind::Sizeof,
            "static" => TokenKind::Static,
            "struct" => TokenKind::Struct,
            "switch" => TokenKind::Switch,
            "typedef" => TokenKind::Typedef,
            "union" => TokenKind::Union,
            "unsigned" => TokenKind::Unsigned,
            "void" => TokenKind::Void,
            "volatile" => TokenKind::Volatile,
            "while" => TokenKind::While,
            "_Bool" => TokenKind::Bool,
            "_Complex" => TokenKind::Complex,
            "_Imaginary" => TokenKind::Imaginary,
            _ => TokenKind::Identifier,
        };
        Ok(Token::new(kind, text, pos))
    }

    fn lex_number(&mut self, pos: SourcePos) -> Result<Token, ParseError> {
        let start = self.pos;
        let mut is_float = false;

        // Handle hex/octal prefix
        if self.current() == b'0' {
            self.advance();
            if self.current() == b'x' || self.current() == b'X' {
                // Hex literal
                self.advance();
                self.skip_while(|c| c.is_ascii_hexdigit());
            } else if self.current().is_ascii_digit() {
                // Octal literal
                self.skip_while(|c| c.is_ascii_digit() && c <= b'7');
            }
        }

        // Decimal part
        self.skip_while(|c| c.is_ascii_digit());

        // Fractional part
        if self.current() == b'.' {
            is_float = true;
            self.advance();
            self.skip_while(|c| c.is_ascii_digit());
        }

        // Exponent
        if self.current() == b'e' || self.current() == b'E' {
            is_float = true;
            self.advance();
            if self.current() == b'+' || self.current() == b'-' {
                self.advance();
            }
            self.skip_while(|c| c.is_ascii_digit());
        }

        // Suffixes (u, U, l, L, f, F, ul, etc.)
        while matches!(self.current(), b'u' | b'U' | b'l' | b'L' | b'f' | b'F') {
            if self.current() == b'f' || self.current() == b'F' {
                is_float = true;
            }
            self.advance();
        }

        let text = self.text_from(start);
        let kind = if is_float {
            TokenKind::FloatLiteral
        } else {
            TokenKind::IntLiteral
        };
        Ok(Token::new(kind, text, pos))
    }

    fn lex_char_literal(&mut self, pos: SourcePos) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // opening quote

        while self.pos < self.source.len() && self.current() != b'\'' {
            if self.current() == b'\\' {
                self.advance();
                if self.pos < self.source.len() {
                    self.advance(); // consume escape char
                }
            } else {
                self.advance();
            }
        }

        if self.current() != b'\'' {
            return Err(ParseError::LexError {
                line: self.line,
                col: self.col,
                msg: "Unterminated character literal".to_string(),
            });
        }
        self.advance(); // closing quote

        Ok(Token::new(
            TokenKind::CharLiteral,
            self.text_from(start),
            pos,
        ))
    }

    fn lex_string_literal(&mut self, pos: SourcePos) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // opening quote

        while self.pos < self.source.len() && self.current() != b'"' {
            if self.current() == b'\\' {
                self.advance();
                if self.pos < self.source.len() {
                    self.advance(); // consume escape char
                }
            } else {
                self.advance();
            }
        }

        if self.current() != b'"' {
            return Err(ParseError::LexError {
                line: self.line,
                col: self.col,
                msg: "Unterminated string literal".to_string(),
            });
        }
        self.advance(); // closing quote

        Ok(Token::new(
            TokenKind::StringLiteral,
            self.text_from(start),
            pos,
        ))
    }
}

/// Convenience: lex a complete C source file.
pub fn lex_c(source: &str) -> Result<Vec<Token>, ParseError> {
    let mut lexer = Lexer::new(source);
    lexer.lex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_keywords() {
        let source = "int main(void) { return 0; }";
        let tokens = lex_c(source).unwrap();

        let kinds: Vec<TokenKind> = tokens
            .iter()
            .filter(|t| t.kind != TokenKind::Whitespace && t.kind != TokenKind::Eof)
            .map(|t| t.kind)
            .collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::Int,
                TokenKind::Identifier,
                TokenKind::LParen,
                TokenKind::Void,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::Return,
                TokenKind::IntLiteral,
                TokenKind::Semi,
                TokenKind::RBrace,
            ]
        );
    }

    #[test]
    fn lex_operators() {
        let source = "a += b; c->d; e == f; g && h;";
        let tokens = lex_c(source).unwrap();

        let ops: Vec<&str> = tokens
            .iter()
            .filter(|t| {
                t.kind != TokenKind::Identifier
                    && t.kind != TokenKind::Whitespace
                    && t.kind != TokenKind::Semi
                    && t.kind != TokenKind::Eof
            })
            .map(|t| t.text.as_str())
            .collect();

        assert_eq!(ops, vec!["+=", "->", "==", "&&"]);
    }

    #[test]
    fn lex_preprocessor() {
        let source = "#include <stdio.h>\n#define MAX 100";
        let tokens = lex_c(source).unwrap();

        assert_eq!(tokens[0].kind, TokenKind::Preprocessor);
        assert!(tokens[0].text.starts_with("#include"));
    }

    #[test]
    fn lex_comments() {
        let source = "// line comment\n/* block\ncomment */";
        let tokens = lex_c(source).unwrap();

        let comments: Vec<_> = tokens
            .iter()
            .filter(|t| matches!(t.kind, TokenKind::LineComment | TokenKind::BlockComment))
            .collect();

        assert_eq!(comments.len(), 2);
    }

    #[test]
    fn lex_literals() {
        let source = r#"123 3.14 'a' "hello" 0x1a"#;
        let tokens = lex_c(source).unwrap();

        let literals: Vec<_> = tokens
            .iter()
            .filter(|t| {
                matches!(
                    t.kind,
                    TokenKind::IntLiteral
                        | TokenKind::FloatLiteral
                        | TokenKind::CharLiteral
                        | TokenKind::StringLiteral
                )
            })
            .collect();

        assert_eq!(literals.len(), 4);
        assert_eq!(literals[0].text, "123");
        assert_eq!(literals[1].text, "3.14");
    }
}
