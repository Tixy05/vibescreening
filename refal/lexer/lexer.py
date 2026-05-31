import re
from dataclasses import dataclass
from typing import Any, List


def _unescape_quoted_body(raw: str) -> str:
    """Decode escapes allowed by _Q_ESC (must stay in sync with the regex)."""
    i = 0
    out: List[str] = []
    n = len(raw)
    while i < n:
        if raw[i] == "\\" and i + 1 < n:
            e = raw[i + 1]
            if e in "'\"\\":
                out.append(e)
                i += 2
                continue
            if e == "x":
                j = i + 2
                digits: List[str] = []
                while j < n and raw[j] in "0123456789abcdefABCDEF":
                    digits.append(raw[j])
                    j += 1
                if digits:
                    v = int("".join(digits), 16)
                    out.append(chr(v) if v <= 0x10FFFF else chr(v & 0xFF))
                    i = j
                    continue
            if e in "tnrabfv":
                out.append(
                    {"t": "\t", "n": "\n", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}[e]
                )
                i += 2
                continue
        out.append(raw[i])
        i += 1
    return "".join(out)


def _semantic_quoted_content(source: str) -> str:
    """Strip outer quotes and decode escapes (so a backslash-n pair becomes one newline character)."""
    return _unescape_quoted_body(source[1:-1])


@dataclass(frozen=True)
class Position:
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"

    def to_location(self, filename: str) -> "Location":
        return Location(filename=filename, position=self)

    def pos_after(self, s: str) -> "Position":
        text_lines = s.splitlines()
        d_lines = len(text_lines) - 1
        if "\n"*len(text_lines) == s or "\r\n"*len(text_lines) == s:
            d_lines += 1
        d_cols = len(text_lines[-1])
        return Position(line=self.line + d_lines, col=d_cols + (self.col if d_lines == 0 else 0))


@dataclass(frozen=True)
class Location:
    filename: str
    position: Position

    def __str__(self) -> str:
        return f"{self.filename}:{self.position}"


@dataclass(frozen=True)
class Span:
    start: Position
    end: Position

    def __str__(self) -> str:
        return f"{self.start} - {self.end}"


@dataclass
class Token:
    span: Span

    @property
    def value(self) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.span}: {type(self).__name__} valued `{self.value}`"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "Token":
        raise NotImplementedError


@dataclass
class EVarToken(Token):
    var_name: str

    @property
    def value(self) -> str:
        return self.var_name


    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "EVarToken":
        return EVarToken(span=Span(start=start, end=start.pos_after(source)), var_name=source[2:])


@dataclass
class TVarToken(Token):
    var_name: str

    @property
    def value(self) -> str:
        return self.var_name

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "TVarToken":
        return TVarToken(span=Span(start=start, end=start.pos_after(source)), var_name=source[2:])


@dataclass
class SVarToken(Token):
    var_name: str

    @property
    def value(self) -> str:
        return self.var_name

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "SVarToken":
        return SVarToken(span=Span(start=start, end=start.pos_after(source)), var_name=source[2:])


@dataclass
class QuotedSequenceToken(Token):
    sequence: str

    @property
    def value(self) -> str:
        return self.sequence

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "QuotedSequenceToken":
        return QuotedSequenceToken(span=Span(start=start, end=start.pos_after(source)), sequence=_semantic_quoted_content(source))


@dataclass
class DoubleQuotedNameToken(Token):
    name: str

    @property
    def value(self) -> str:
        return self.name

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "DoubleQuotedNameToken":
        return DoubleQuotedNameToken(span=Span(start=start, end=start.pos_after(source)), name=_semantic_quoted_content(source))


@dataclass
class LParenthesisToken(Token):
    @property
    def value(self) -> str:
        return "("

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "LParenthesisToken":
        return LParenthesisToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class RParenthesisToken(Token):
    @property
    def value(self) -> str:
        return ")"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "RParenthesisToken":
        return RParenthesisToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class LBraceToken(Token):
    @property
    def value(self) -> str:
        return "{"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "LBraceToken":
        return LBraceToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class RBraceToken(Token):
    @property
    def value(self) -> str:
        return "}"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "RBraceToken":
        return RBraceToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class LAngleToken(Token):
    @property
    def value(self) -> str:
        return "<"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "LAngleToken":
        return LAngleToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class RAngleToken(Token):
    @property
    def value(self) -> str:
        return ">"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "RAngleToken":
        return RAngleToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class EqSignToken(Token):
    @property
    def value(self) -> str:
        return "="

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "EqSignToken":
        return EqSignToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class SemicolonToken(Token):
    @property
    def value(self) -> str:
        return ";"

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "SemicolonToken":
        return SemicolonToken(span=Span(start=start, end=start.pos_after(source)))


@dataclass
class NumberToken(Token):
    number: int

    @property
    def value(self) -> int:
        return self.number

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "NumberToken":
        return NumberToken(span=Span(start=start, end=start.pos_after(source)), number=int(source))


@dataclass
class NameToken(Token):
    name: str

    @property
    def value(self) -> str:
        return self.name

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "NameToken":
        return NameToken(span=Span(start=start, end=start.pos_after(source)), name=source)


@dataclass
class CommentToken(Token):
    comment: str

    @property
    def value(self) -> str:
        return self.comment

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "CommentToken":
        return CommentToken(span=Span(start=start, end=start.pos_after(source)), comment=source[2:-2])


@dataclass
class WhitespaceToken(Token):
    spaces: str

    @property
    def value(self) -> str:
        return self.spaces

    @staticmethod
    def from_string_and_start_pos(start: Position, source: str) -> "WhitespaceToken":
        return WhitespaceToken(span=Span(start=start, end=start.pos_after(source)), spaces=source)

###############

_VAR_SUFFIX = r"([A-Za-z0-9_][A-Za-z0-9_\-]*)"

# Inside '...' / "..." (single line only): non-quote non-backslash run, or escapes:
# \' \" \\  \x89A (hex)  \t \n \r \a \b \f \v
_Q_ESC = r"\\(?:['\"]|x[0-9A-Fa-f]+|[tnrabfv]|\\)"
_SQ_BODY = rf"(?:[^'\\\n]|{_Q_ESC})*"
_DQ_BODY = rf'(?:[^"\\\n]|{_Q_ESC})*'

token_map: dict[type[Token], re.Pattern[str]] = {
    EVarToken: re.compile(rf"e\.{_VAR_SUFFIX}"),
    TVarToken: re.compile(rf"t\.{_VAR_SUFFIX}"),
    SVarToken: re.compile(rf"s\.{_VAR_SUFFIX}"),
    NumberToken: re.compile(r"[0-9]+"),
    NameToken: re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*"),
    CommentToken: re.compile(r"/\*[\s\S]*?\*/"),
    LParenthesisToken: re.compile(r"\("),
    RParenthesisToken: re.compile(r"\)"),
    LBraceToken: re.compile(r"\{"),
    RBraceToken: re.compile(r"\}"),
    LAngleToken: re.compile(r"<"),
    RAngleToken: re.compile(r">"),
    EqSignToken: re.compile(r"="),
    SemicolonToken: re.compile(r";"),
    QuotedSequenceToken: re.compile(rf"'{_SQ_BODY}'"),
    DoubleQuotedNameToken: re.compile(rf'"{_DQ_BODY}"'),
    WhitespaceToken: re.compile(r"\s+"),
}

def lex(
    source: str,
    ignore_comments: bool = False,
    ignore_whitespace: bool = False,
) -> List[Token]:
    tokens: List[Token] = []
    cur_pos = Position(line=1, col=1)
    cur_offset = 0
    while cur_offset < len(source):
        for t, regex in token_map.items():
            if m := regex.match(source[cur_offset:]):
                s = m.group(0)
                token = t.from_string_and_start_pos(cur_pos, s)
                if type(token) == WhitespaceToken and not ignore_whitespace:
                    tokens.append(token)
                elif type(token) == CommentToken and not ignore_comments:
                    tokens.append(token)
                else:
                    tokens.append(token)
                cur_offset += len(s)
                cur_pos = cur_pos.pos_after(s)
                break
        else:
            raise ValueError(f"Unexpected token at {cur_pos}")

    return tokens


def lex_file(filename: str) -> List[Token]:
    with open(filename, "r") as file:
        return lex(file.read())


#####

"""
Simple Refal Grammar

Program = Definition+
Definition = "Name" "{" Rule (";" Rule)* ";"? "}"
Rule = Pattern "=" Expression
Pattern = PatternElement+
PatternElement = Var | EMPTY | Symbol | "(" Pattern ")"
Expression = ExpressionElemnt+
ExpressionElemant = Var | EMPTY | Symbol | "(" Expression ")" | Call
Call = "<" "Name" Pattern ">"
Var = "EVar" | "TVar" | "SVar"
Symbol = "Name" | "Number" | "QuotedSequence" | "DoubleQuotedName"

"""


if __name__ == "__main__":
    source = """
e.x = 1; 
/* comment 

*/ j
k

l
"""
    tokens = lex(source)
    for token in tokens:
        print(token)