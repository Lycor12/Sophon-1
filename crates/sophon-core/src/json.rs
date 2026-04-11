//! Minimal JSON parser for Lean JSON-RPC communication.
//!
//! Implements a lightweight JSON parser supporting the subset needed for
//! JSON-RPC 2.0: objects, arrays, strings, numbers, booleans, and null.
//!
//! # Examples
//!
//! ```
//! use sophon_core::parse_json;
//!
//! // Parse a JSON object
//! let json = r#"{"name": "sophon", "version": 1.0}"#;
//! let value = parse_json(json).unwrap();
//!
//! // Access fields
//! assert_eq!(value.get("name").unwrap().as_str(), Some("sophon"));
//! assert_eq!(value.get("version").unwrap().as_f64(), Some(1.0));
//!
//! // Parse arrays
//! let arr = parse_json("[1, 2, 3]").unwrap();
//! assert_eq!(arr.as_f64(), None); // Not a number
//!
//! // Parse null
//! let null = parse_json("null").unwrap();
//! assert_eq!(null.as_f64(), None);
//!```

use std::collections::HashMap;
use std::fmt;

/// JSON value types.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

/// JSON parsing error.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonError {
    UnexpectedEnd,
    UnexpectedChar {
        expected: String,
        found: char,
        pos: usize,
    },
    InvalidEscape {
        pos: usize,
    },
    InvalidNumber {
        pos: usize,
    },
    InvalidUnicode {
        pos: usize,
    },
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnexpectedChar {
                expected,
                found,
                pos,
            } => {
                write!(
                    f,
                    "expected '{}' but found '{}' at position {}",
                    expected, found, pos
                )
            }
            Self::InvalidEscape { pos } => write!(f, "invalid escape sequence at position {}", pos),
            Self::InvalidNumber { pos } => write!(f, "invalid number at position {}", pos),
            Self::InvalidUnicode { pos } => write!(f, "invalid unicode escape at position {}", pos),
        }
    }
}

impl std::error::Error for JsonError {}

impl JsonValue {
    /// Get a field from an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        if let JsonValue::Object(obj) = self {
            obj.get(key)
        } else {
            None
        }
    }

    /// Get string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get number value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get bool value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Parse a JSON string into a JsonValue.
pub fn parse(input: &str) -> Result<JsonValue, JsonError> {
    let mut parser = Parser::new(input);
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.peek().is_some() {
        return Err(JsonError::UnexpectedChar {
            expected: "end of input".to_string(),
            found: parser.peek().unwrap(),
            pos: parser.pos,
        });
    }
    Ok(value)
}

/// Serialize a JsonValue to a JSON string.
pub fn stringify(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(true) => "true".to_string(),
        JsonValue::Bool(false) => "false".to_string(),
        JsonValue::Number(n) => {
            if n.is_finite() {
                n.to_string()
            } else if n.is_nan() {
                "null".to_string()
            } else if n.is_sign_negative() {
                "-Infinity".to_string()
            } else {
                "Infinity".to_string()
            }
        }
        JsonValue::String(s) => encode_string(s),
        JsonValue::Array(arr) => {
            let mut result = String::from("[");
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                result.push_str(&stringify(v));
            }
            result.push(']');
            result
        }
        JsonValue::Object(obj) => {
            let mut result = String::from("{");
            for (i, (k, v)) in obj.iter().enumerate() {
                if i > 0 {
                    result.push(',');
                }
                result.push_str(&encode_string(k));
                result.push(':');
                result.push_str(&stringify(v));
            }
            result.push('}');
            result
        }
    }
}

fn encode_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\u{0008}' => result.push_str("\\b"),
            '\u{000C}' => result.push_str("\\f"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_ascii_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

struct Parser<'a> {
    input: &'a str,
    chars: std::str::Chars<'a>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.clone().next()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.next();
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), JsonError> {
        self.skip_whitespace();
        match self.next() {
            Some(c) if c == expected => Ok(()),
            Some(c) => Err(JsonError::UnexpectedChar {
                expected: expected.to_string(),
                found: c,
                pos: self.pos - 1,
            }),
            None => Err(JsonError::UnexpectedEnd),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonError> {
        self.skip_whitespace();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_string(),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(JsonError::UnexpectedChar {
                expected: "value".to_string(),
                found: c,
                pos: self.pos,
            }),
            None => Err(JsonError::UnexpectedEnd),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonError> {
        self.expect('{')?;
        self.skip_whitespace();

        let mut obj = HashMap::new();

        if self.peek() == Some('}') {
            self.next();
            return Ok(JsonValue::Object(obj));
        }

        loop {
            self.skip_whitespace();
            let key = self.parse_string_raw()?;
            self.expect(':')?;
            let value = self.parse_value()?;
            obj.insert(key, value);

            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.next();
                    continue;
                }
                Some('}') => {
                    self.next();
                    break;
                }
                Some(c) => {
                    return Err(JsonError::UnexpectedChar {
                        expected: "} or ,".to_string(),
                        found: c,
                        pos: self.pos,
                    })
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }

        Ok(JsonValue::Object(obj))
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonError> {
        self.expect('[')?;
        self.skip_whitespace();

        let mut arr = Vec::new();

        if self.peek() == Some(']') {
            self.next();
            return Ok(JsonValue::Array(arr));
        }

        loop {
            self.skip_whitespace();
            arr.push(self.parse_value()?);

            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.next();
                    continue;
                }
                Some(']') => {
                    self.next();
                    break;
                }
                Some(c) => {
                    return Err(JsonError::UnexpectedChar {
                        expected: "] or ,".to_string(),
                        found: c,
                        pos: self.pos,
                    })
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }

        Ok(JsonValue::Array(arr))
    }

    fn parse_string(&mut self) -> Result<JsonValue, JsonError> {
        Ok(JsonValue::String(self.parse_string_raw()?))
    }

    fn parse_string_raw(&mut self) -> Result<String, JsonError> {
        self.expect('"')?;
        let mut result = String::new();

        while let Some(c) = self.peek() {
            match c {
                '"' => {
                    self.next();
                    return Ok(result);
                }
                '\\' => {
                    self.next();
                    match self.next() {
                        Some('"') => result.push('"'),
                        Some('\\') => result.push('\\'),
                        Some('/') => result.push('/'),
                        Some('b') => result.push('\u{0008}'),
                        Some('f') => result.push('\u{000C}'),
                        Some('n') => result.push('\n'),
                        Some('r') => result.push('\r'),
                        Some('t') => result.push('\t'),
                        Some('u') => {
                            let mut code = 0u32;
                            for _ in 0..4 {
                                match self.next() {
                                    Some(c) if c.is_ascii_hexdigit() => {
                                        code = code * 16 + c.to_digit(16).unwrap();
                                    }
                                    _ => return Err(JsonError::InvalidUnicode { pos: self.pos }),
                                }
                            }
                            match char::from_u32(code) {
                                Some(c) => result.push(c),
                                None => return Err(JsonError::InvalidUnicode { pos: self.pos }),
                            }
                        }
                        _ => return Err(JsonError::InvalidEscape { pos: self.pos }),
                    }
                }
                c => {
                    self.next();
                    result.push(c);
                }
            }
        }

        Err(JsonError::UnexpectedEnd)
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;

        if self.peek() == Some('-') {
            self.next();
        }

        let mut has_digits = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                has_digits = true;
                self.next();
            } else {
                break;
            }
        }

        if !has_digits {
            return Err(JsonError::InvalidNumber { pos: self.pos });
        }

        if self.peek() == Some('.') {
            self.next();
            let mut frac_digits = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    frac_digits = true;
                    self.next();
                } else {
                    break;
                }
            }
            if !frac_digits {
                return Err(JsonError::InvalidNumber { pos: self.pos });
            }
        }

        if let Some(c) = self.peek() {
            if c == 'e' || c == 'E' {
                self.next();
                if let Some(c) = self.peek() {
                    if c == '+' || c == '-' {
                        self.next();
                    }
                }
                let mut exp_digits = false;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        exp_digits = true;
                        self.next();
                    } else {
                        break;
                    }
                }
                if !exp_digits {
                    return Err(JsonError::InvalidNumber { pos: self.pos });
                }
            }
        }

        let num_str = &self.input[start..self.pos];
        match num_str.parse::<f64>() {
            Ok(n) => Ok(JsonValue::Number(n)),
            Err(_) => Err(JsonError::InvalidNumber { pos: start }),
        }
    }

    fn parse_bool(&mut self) -> Result<JsonValue, JsonError> {
        if self.input[self.pos..].starts_with("true") {
            for _ in 0..4 {
                self.next();
            }
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with("false") {
            for _ in 0..5 {
                self.next();
            }
            Ok(JsonValue::Bool(false))
        } else {
            Err(JsonError::UnexpectedChar {
                expected: "true or false".to_string(),
                found: self.peek().unwrap_or('\0'),
                pos: self.pos,
            })
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, JsonError> {
        if self.input[self.pos..].starts_with("null") {
            for _ in 0..4 {
                self.next();
            }
            Ok(JsonValue::Null)
        } else {
            Err(JsonError::UnexpectedChar {
                expected: "null".to_string(),
                found: self.peek().unwrap_or('\0'),
                pos: self.pos,
            })
        }
    }
}

/// JSON-RPC 2.0 request structure.
#[derive(Debug, Clone)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<JsonValue>,
    pub id: Option<JsonValue>,
}

/// JSON-RPC 2.0 response structure.
#[derive(Debug, Clone)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub result: Option<JsonValue>,
    pub error: Option<JsonRpcError>,
    pub id: JsonValue,
}

/// JSON-RPC 2.0 error structure.
#[derive(Debug, Clone)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<JsonValue>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request.
    pub fn new(method: &str, params: Option<JsonValue>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: Some(JsonValue::Number(1.0)),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        let mut obj = HashMap::new();
        obj.insert("jsonrpc".to_string(), JsonValue::String("2.0".to_string()));
        obj.insert("method".to_string(), JsonValue::String(self.method.clone()));

        if let Some(ref params) = self.params {
            obj.insert("params".to_string(), params.clone());
        }

        if let Some(ref id) = self.id {
            obj.insert("id".to_string(), id.clone());
        }

        stringify(&JsonValue::Object(obj))
    }
}

impl JsonRpcResponse {
    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self, JsonError> {
        let value = parse(json)?;

        let JsonValue::Object(obj) = value else {
            return Err(JsonError::UnexpectedChar {
                expected: "{".to_string(),
                found: '{',
                pos: 0,
            });
        };

        let jsonrpc = obj
            .get("jsonrpc")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let result = obj.get("result").cloned();
        let id = obj.get("id").cloned().unwrap_or(JsonValue::Null);

        let error = obj.get("error").and_then(|e| {
            if let JsonValue::Object(err_obj) = e {
                let code = err_obj
                    .get("code")
                    .and_then(|v| v.as_f64())
                    .map(|n| n as i64)?;
                let message = err_obj.get("message").and_then(|v| v.as_str())?.to_string();
                let data = err_obj.get("data").cloned();
                Some(JsonRpcError {
                    code,
                    message,
                    data,
                })
            } else {
                None
            }
        });

        Ok(Self {
            jsonrpc: jsonrpc.unwrap_or_else(|| "2.0".to_string()),
            result,
            error,
            id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_null() {
        assert_eq!(parse("null").unwrap(), JsonValue::Null);
    }

    #[test]
    fn parse_bool() {
        assert_eq!(parse("true").unwrap(), JsonValue::Bool(true));
        assert_eq!(parse("false").unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn parse_number() {
        assert_eq!(parse("42").unwrap(), JsonValue::Number(42.0));
        assert_eq!(parse("-3.14").unwrap(), JsonValue::Number(-3.14));
        assert_eq!(parse("1e10").unwrap(), JsonValue::Number(1e10));
    }

    #[test]
    fn parse_string() {
        assert_eq!(
            parse("\"hello\"").unwrap(),
            JsonValue::String("hello".to_string())
        );
    }

    #[test]
    fn parse_array() {
        let arr = parse("[1, 2, 3]").unwrap();
        if let JsonValue::Array(v) = arr {
            assert_eq!(v.len(), 3);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn parse_object() {
        let obj = parse(r#"{"a": 1, "b": "test"}"#).unwrap();
        if let JsonValue::Object(map) = obj {
            assert_eq!(map.len(), 2);
            assert_eq!(map.get("a"), Some(&JsonValue::Number(1.0)));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn stringify_value() {
        let mut obj = HashMap::new();
        obj.insert("name".to_string(), JsonValue::String("test".to_string()));
        obj.insert("value".to_string(), JsonValue::Number(42.0));
        let json = stringify(&JsonValue::Object(obj));
        assert!(json.contains("name"));
        assert!(json.contains("test"));
        assert!(json.contains("42"));
    }

    #[test]
    fn rpc_request_roundtrip() {
        let req = JsonRpcRequest::new("test", Some(JsonValue::Array(vec![JsonValue::Number(1.0)])));
        let json = req.to_json();
        assert!(json.contains("test"));
        assert!(json.contains("2.0"));
    }
}
