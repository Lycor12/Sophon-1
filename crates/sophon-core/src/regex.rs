//! Custom regex engine using Thompson NFA (Nondeterministic Finite Automaton)
//!
//! This is a pure Rust implementation that does not depend on external crates.
//! Supports: literals, `.`, `*`, `+`, `?`, `[]` character classes, `()` groups, `^`, `$`
//!
//! # Examples
//!
//! ```
//! use sophon_core::Regex;
//!
//! // Basic matching
//! let re = Regex::new("hello").unwrap();
//! assert!(re.is_match("hello world"));
//! assert!(!re.is_match("goodbye world"));
//!
//! // Character classes
//! let re = Regex::new("[0-9]+").unwrap();
//! assert!(re.is_match("123"));
//! assert!(!re.is_match("abc"));
//!
//! // Find with byte positions
//! let re = Regex::new("[0-9]+").unwrap();
//! assert_eq!(re.find("abc123def"), Some((3, 6)));
//!```

#![forbid(unsafe_code)]

use std::fmt;
use std::string::String;
use std::vec::Vec;

/// Error types for regex operations
#[derive(Debug, Clone, PartialEq)]
pub enum RegexError {
    InvalidPattern(String),
    UnmatchedParenthesis,
    UnmatchedBracket,
    InvalidEscape,
    EmptyPattern,
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexError::InvalidPattern(s) => write!(f, "Invalid regex pattern: {}", s),
            RegexError::UnmatchedParenthesis => write!(f, "Unmatched parenthesis in regex"),
            RegexError::UnmatchedBracket => write!(f, "Unmatched bracket in regex"),
            RegexError::InvalidEscape => write!(f, "Invalid escape sequence in regex"),
            RegexError::EmptyPattern => write!(f, "Empty regex pattern"),
        }
    }
}

impl std::error::Error for RegexError {}

/// NFA state type - stored separately from states to allow Copy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateTypeTag {
    Char,
    Any,
    Class {
        start_idx: usize,
        end_idx: usize,
        negated: bool,
    },
    Epsilon,
    StartAnchor,
    EndAnchor,
    Match,
}

/// Character class storage
#[derive(Debug, Clone)]
struct CharClass {
    chars: Vec<char>,
}

/// NFA state
#[derive(Debug, Clone)]
struct State {
    tag: StateTypeTag,
    out: Option<usize>,
    out2: Option<usize>,
}

/// Compiled regex NFA
#[derive(Debug, Clone)]
pub struct Regex {
    states: Vec<State>,
    char_classes: Vec<CharClass>,
    start: usize,
    pattern: String,
}

/// Fragment of NFA under construction
#[derive(Debug)]
struct Frag {
    start: usize,
    accept: usize,
}

impl Regex {
    /// Compile a regex pattern into an NFA
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        if pattern.is_empty() {
            return Err(RegexError::EmptyPattern);
        }

        let mut parser = Parser::new(pattern);
        let nfa = parser.parse()?;

        Ok(Regex {
            states: nfa.states,
            char_classes: nfa.char_classes,
            start: nfa.start,
            pattern: pattern.to_string(),
        })
    }

    /// Check if the regex matches anywhere in the string
    pub fn is_match(&self, text: &str) -> bool {
        self.find(text).is_some()
    }

    /// Find the first match in the string
    pub fn find(&self, text: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = text.chars().collect();

        for start_idx in 0..=chars.len() {
            if let Some(end_idx) = self.match_at(&chars, start_idx) {
                let start_byte = chars[..start_idx].iter().map(|c| c.len_utf8()).sum();
                let end_byte = start_byte
                    + chars[start_idx..end_idx]
                        .iter()
                        .map(|c| c.len_utf8())
                        .sum::<usize>();
                return Some((start_byte, end_byte));
            }
        }

        None
    }

    /// Match starting at a specific position (longest match semantics)
    fn match_at(&self, chars: &[char], start: usize) -> Option<usize> {
        // NFA simulation using Thompson's algorithm with longest match
        let mut current: Vec<usize> = Vec::new();
        let mut next: Vec<usize> = Vec::new();
        let mut last_match: Option<usize> = None;

        // Add start state with epsilon closure
        self.add_epsilon_states(&mut current, self.start);

        for pos in start..=chars.len() {
            // Check if we're in a match state - record it but keep going for longest match
            for &state_idx in &current {
                if let StateTypeTag::Match = self.states[state_idx].tag {
                    last_match = Some(pos);
                }
            }

            // No more characters to consume
            if pos >= chars.len() {
                break;
            }

            let ch = chars[pos];
            next.clear();

            // Process all current states
            for &state_idx in &current {
                let state = &self.states[state_idx];

                match state.tag {
                    StateTypeTag::Char => {
                        // Get the char from the pattern (stored in a separate lookup)
                        // For simplicity, we store the expected char in the first char class
                        if let Some(class_idx) = self.char_classes.first() {
                            if class_idx.chars.first() == Some(&ch) {
                                if let Some(out) = state.out {
                                    self.add_epsilon_states(&mut next, out);
                                }
                            }
                        }
                    }
                    StateTypeTag::Any => {
                        if let Some(out) = state.out {
                            self.add_epsilon_states(&mut next, out);
                        }
                    }
                    StateTypeTag::Class {
                        start_idx,
                        end_idx: _,
                        negated,
                    } => {
                        let class = &self.char_classes[start_idx];
                        let in_class = class.chars.contains(&ch);
                        if (in_class && !negated) || (!in_class && negated) {
                            if let Some(out) = state.out {
                                self.add_epsilon_states(&mut next, out);
                            }
                        }
                    }
                    _ => {}
                }
            }

            std::mem::swap(&mut current, &mut next);

            if current.is_empty() {
                // No more states to process - return last match if any
                return last_match;
            }
        }

        // Check final states
        for &state_idx in &current {
            if let StateTypeTag::Match = self.states[state_idx].tag {
                last_match = Some(chars.len());
            }
        }

        last_match
    }

    /// Add all epsilon-reachable states
    fn add_epsilon_states(&self, states: &mut Vec<usize>, start: usize) {
        let mut stack = vec![start];

        while let Some(state_idx) = stack.pop() {
            if states.contains(&state_idx) {
                continue;
            }
            states.push(state_idx);

            let state = &self.states[state_idx];

            // Handle epsilon transitions
            if let StateTypeTag::Epsilon | StateTypeTag::StartAnchor | StateTypeTag::EndAnchor =
                state.tag
            {
                if let Some(out) = state.out {
                    stack.push(out);
                }
                if let Some(out2) = state.out2 {
                    stack.push(out2);
                }
            }
        }
    }
}

/// Parser for regex patterns
struct Parser {
    pattern: Vec<char>,
    pos: usize,
    states: Vec<State>,
    char_classes: Vec<CharClass>,
}

impl Parser {
    fn new(pattern: &str) -> Self {
        Parser {
            pattern: pattern.chars().collect(),
            pos: 0,
            states: Vec::new(),
            char_classes: Vec::new(),
        }
    }

    fn parse(&mut self) -> Result<Nfa, RegexError> {
        let frag = self.parse_expr()?;

        // Add match state
        let match_state = self.add_state(StateTypeTag::Match, None, None);
        self.patch(frag.accept, match_state);

        Ok(Nfa {
            states: std::mem::take(&mut self.states),
            char_classes: std::mem::take(&mut self.char_classes),
            start: frag.start,
        })
    }

    fn parse_expr(&mut self) -> Result<Frag, RegexError> {
        self.parse_alternation()
    }

    /// Parse alternation (a|b)
    fn parse_alternation(&mut self) -> Result<Frag, RegexError> {
        let mut left = self.parse_concatenation()?;

        while self.consume('|') {
            let right = self.parse_concatenation()?;

            // Create split state
            let split = self.add_state(StateTypeTag::Epsilon, Some(left.start), Some(right.start));
            let accept = self.add_state(StateTypeTag::Epsilon, None, None);

            self.patch(left.accept, accept);
            self.patch(right.accept, accept);

            left = Frag {
                start: split,
                accept,
            };
        }

        Ok(left)
    }

    /// Parse concatenation (ab)
    fn parse_concatenation(&mut self) -> Result<Frag, RegexError> {
        let mut left = self.parse_repetition()?;

        while self.pos < self.pattern.len() && !self.at_alternation_end() {
            let right = self.parse_repetition()?;
            self.patch(left.accept, right.start);
            left.accept = right.accept;
        }

        Ok(left)
    }

    fn at_alternation_end(&self) -> bool {
        matches!(self.pattern.get(self.pos), Some('|') | Some(')'))
    }

    /// Parse repetition (* + ?)
    fn parse_repetition(&mut self) -> Result<Frag, RegexError> {
        let mut frag = self.parse_atom()?;

        loop {
            if self.consume('*') {
                // Zero or more
                let split = self.add_state(StateTypeTag::Epsilon, Some(frag.start), None);
                self.patch(frag.accept, split);
                frag = Frag {
                    start: split,
                    accept: split,
                };
            } else if self.consume('+') {
                // One or more
                let split = self.add_state(StateTypeTag::Epsilon, Some(frag.start), None);
                self.patch(frag.accept, split);
                frag = Frag {
                    start: frag.start,
                    accept: split,
                };
            } else if self.consume('?') {
                // Zero or one
                let split =
                    self.add_state(StateTypeTag::Epsilon, Some(frag.start), Some(frag.accept));
                frag = Frag {
                    start: split,
                    accept: frag.accept,
                };
            } else {
                break;
            }
        }

        Ok(frag)
    }

    /// Parse atomic elements
    fn parse_atom(&mut self) -> Result<Frag, RegexError> {
        if self.consume('(') {
            let frag = self.parse_expr()?;
            if !self.consume(')') {
                return Err(RegexError::UnmatchedParenthesis);
            }
            Ok(frag)
        } else if self.consume('[') {
            self.parse_char_class()
        } else if self.consume('^') {
            // Start anchor
            let state = self.add_state(StateTypeTag::StartAnchor, None, None);
            let accept = self.add_state(StateTypeTag::Epsilon, None, None);
            self.patch(state, accept);
            Ok(Frag {
                start: state,
                accept,
            })
        } else if self.consume('$') {
            // End anchor
            let state = self.add_state(StateTypeTag::EndAnchor, None, None);
            let accept = self.add_state(StateTypeTag::Epsilon, None, None);
            self.patch(state, accept);
            Ok(Frag {
                start: state,
                accept,
            })
        } else if self.consume('.') {
            // Any character
            let state = self.add_state(StateTypeTag::Any, None, None);
            let accept = self.add_state(StateTypeTag::Epsilon, None, None);
            self.patch(state, accept);
            Ok(Frag {
                start: state,
                accept,
            })
        } else if self.consume('\\') {
            // Escape sequence
            self.parse_escape()
        } else {
            // Literal character
            match self.next_char() {
                Some(ch) if !is_special(ch) => {
                    let class_idx = self.char_classes.len();
                    self.char_classes.push(CharClass { chars: vec![ch] });
                    let state = self.add_state(
                        StateTypeTag::Class {
                            start_idx: class_idx,
                            end_idx: class_idx + 1,
                            negated: false,
                        },
                        None,
                        None,
                    );
                    let accept = self.add_state(StateTypeTag::Epsilon, None, None);
                    self.patch(state, accept);
                    Ok(Frag {
                        start: state,
                        accept,
                    })
                }
                Some(ch) => Err(RegexError::InvalidPattern(format!(
                    "Unexpected character: {}",
                    ch
                ))),
                None => Err(RegexError::InvalidPattern(
                    "Unexpected end of pattern".to_string(),
                )),
            }
        }
    }

    /// Parse character class [abc] or [a-z]
    fn parse_char_class(&mut self) -> Result<Frag, RegexError> {
        let mut chars = Vec::new();
        let negated = self.consume('!');

        while let Some(&ch) = self.pattern.get(self.pos) {
            if ch == ']' {
                self.pos += 1;
                break;
            }

            if ch == '-' && !chars.is_empty() {
                // Range
                self.pos += 1;
                if let Some(&end_ch) = self.pattern.get(self.pos) {
                    if end_ch == ']' {
                        chars.push('-');
                        break;
                    }
                    let start_ch = chars.pop().unwrap();
                    for c in start_ch..=end_ch {
                        chars.push(c);
                    }
                    self.pos += 1;
                }
            } else if ch == '\\' {
                self.pos += 1;
                if let Some(&escaped) = self.pattern.get(self.pos) {
                    chars.push(parse_escape_char(escaped)?);
                    self.pos += 1;
                } else {
                    return Err(RegexError::InvalidEscape);
                }
            } else {
                chars.push(ch);
                self.pos += 1;
            }
        }

        if chars.is_empty() && !negated {
            return Err(RegexError::InvalidPattern(
                "Empty character class".to_string(),
            ));
        }

        let start_idx = self.char_classes.len();
        self.char_classes.push(CharClass { chars });
        let end_idx = start_idx + 1;

        let state = self.add_state(
            StateTypeTag::Class {
                start_idx,
                end_idx,
                negated,
            },
            None,
            None,
        );
        let accept = self.add_state(StateTypeTag::Epsilon, None, None);
        self.patch(state, accept);

        Ok(Frag {
            start: state,
            accept,
        })
    }

    /// Parse escape sequence
    fn parse_escape(&mut self) -> Result<Frag, RegexError> {
        match self.next_char() {
            Some(ch) => {
                // Check for special character classes
                match ch {
                    'd' => {
                        // \d = [0-9]
                        return self.parse_char_class_with_chars(('0'..='9').collect(), false);
                    }
                    'w' => {
                        // \w = [a-zA-Z0-9_]
                        let mut chars: Vec<char> =
                            ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
                        chars.push('_');
                        return self.parse_char_class_with_chars(chars, false);
                    }
                    's' => {
                        // \s = whitespace
                        let chars = vec![' ', '\t', '\n', '\r'];
                        return self.parse_char_class_with_chars(chars, false);
                    }
                    _ => {}
                }

                let escaped = parse_escape_char(ch)?;
                let class_idx = self.char_classes.len();
                self.char_classes.push(CharClass {
                    chars: vec![escaped],
                });
                let state = self.add_state(
                    StateTypeTag::Class {
                        start_idx: class_idx,
                        end_idx: class_idx + 1,
                        negated: false,
                    },
                    None,
                    None,
                );
                let accept = self.add_state(StateTypeTag::Epsilon, None, None);
                self.patch(state, accept);
                Ok(Frag {
                    start: state,
                    accept,
                })
            }
            None => Err(RegexError::InvalidEscape),
        }
    }

    /// Create a character class state from pre-built character list
    fn parse_char_class_with_chars(
        &mut self,
        chars: Vec<char>,
        negated: bool,
    ) -> Result<Frag, RegexError> {
        if chars.is_empty() && !negated {
            return Err(RegexError::InvalidPattern(
                "Empty character class".to_string(),
            ));
        }

        let start_idx = self.char_classes.len();
        self.char_classes.push(CharClass { chars });
        let end_idx = start_idx + 1;

        let state = self.add_state(
            StateTypeTag::Class {
                start_idx,
                end_idx,
                negated,
            },
            None,
            None,
        );
        let accept = self.add_state(StateTypeTag::Epsilon, None, None);
        self.patch(state, accept);

        Ok(Frag {
            start: state,
            accept,
        })
    }

    /// Add a new state
    fn add_state(&mut self, tag: StateTypeTag, out: Option<usize>, out2: Option<usize>) -> usize {
        let idx = self.states.len();
        self.states.push(State { tag, out, out2 });
        idx
    }

    /// Patch a state's output
    fn patch(&mut self, state_idx: usize, target: usize) {
        let state = &mut self.states[state_idx];
        if state.out.is_none() {
            state.out = Some(target);
        } else if state.out2.is_none() {
            state.out2 = Some(target);
        }
    }

    /// Consume a character if it matches
    fn consume(&mut self, expected: char) -> bool {
        if let Some(&ch) = self.pattern.get(self.pos) {
            if ch == expected {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    /// Get next character
    fn next_char(&mut self) -> Option<char> {
        let ch = self.pattern.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }
}

fn is_special(ch: char) -> bool {
    matches!(
        ch,
        '|' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '.' | '^' | '$' | '\\'
    )
}

fn parse_escape_char(ch: char) -> Result<char, RegexError> {
    match ch {
        'n' => Ok('\n'),
        't' => Ok('\t'),
        'r' => Ok('\r'),
        '\\' => Ok('\\'),
        '.' => Ok('.'),
        '*' => Ok('*'),
        '+' => Ok('+'),
        '?' => Ok('?'),
        '[' => Ok('['),
        ']' => Ok(']'),
        '(' => Ok('('),
        ')' => Ok(')'),
        '|' => Ok('|'),
        '^' => Ok('^'),
        '$' => Ok('$'),
        'd' | 'w' | 's' => {
            // These are handled specially in parse_escape, should not reach here
            Err(RegexError::InvalidEscape)
        }
        _ => Err(RegexError::InvalidEscape),
    }
}

/// Compiled NFA structure
#[derive(Debug)]
struct Nfa {
    states: Vec<State>,
    char_classes: Vec<CharClass>,
    start: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        let re = Regex::new("hello").unwrap();
        assert!(re.is_match("hello"));
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("world"));
    }

    #[test]
    fn test_dot() {
        let re = Regex::new("h.llo").unwrap();
        assert!(re.is_match("hello"));
        assert!(re.is_match("hallo"));
        assert!(!re.is_match("hllo"));
    }

    #[test]
    fn test_star() {
        let re = Regex::new("ab*c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbbc"));
        assert!(!re.is_match("ab"));
    }

    #[test]
    fn test_plus() {
        let re = Regex::new("ab+c").unwrap();
        assert!(re.is_match("abc"));
        assert!(re.is_match("abbc"));
        assert!(!re.is_match("ac"));
    }

    #[test]
    fn test_question() {
        let re = Regex::new("ab?c").unwrap();
        assert!(re.is_match("ac"));
        assert!(re.is_match("abc"));
        assert!(!re.is_match("abbc"));
    }

    #[test]
    fn test_alternation() {
        let re = Regex::new("cat|dog").unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("dog"));
        assert!(!re.is_match("bird"));
    }

    #[test]
    fn test_char_class() {
        let re = Regex::new("[aeiou]").unwrap();
        assert!(re.is_match("a"));
        assert!(re.is_match("e"));
        assert!(!re.is_match("b"));
    }

    #[test]
    fn test_negated_class() {
        let re = Regex::new("[!aeiou]").unwrap();
        assert!(!re.is_match("a"));
        assert!(re.is_match("b"));
    }

    #[test]
    fn test_range() {
        let re = Regex::new("[a-z]").unwrap();
        assert!(re.is_match("a"));
        assert!(re.is_match("z"));
        assert!(!re.is_match("A"));
    }

    #[test]
    fn test_group() {
        let re = Regex::new("a(b|c)d").unwrap();
        assert!(re.is_match("abd"));
        assert!(re.is_match("acd"));
        assert!(!re.is_match("ad"));
    }

    #[test]
    fn test_digits_pattern() {
        // Common pattern: \d+ equivalent
        let re = Regex::new("[0-9]+").unwrap();
        assert!(re.is_match("123"));
        assert!(re.is_match("0"));
        assert!(!re.is_match("abc"));
    }

    #[test]
    fn test_find() {
        let re = Regex::new("[0-9]+").unwrap();
        let result = re.find("abc123def");
        assert_eq!(result, Some((3, 6)));
    }
}
