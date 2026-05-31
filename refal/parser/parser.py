from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import TypeAlias

from refal.lexer.lexer import *

# ── AST ──────────────────────────────────────────────────────

@dataclass
class Node:
    span: Span

    def pretty(self, indent: int = 0) -> str:
        raise NotImplementedError

    def _p(self, indent: int) -> str:
        return "  " * indent

class VarKind(enum.StrEnum):
    E = "e"
    T = "t"
    S = "s"

class SymbolTag(enum.StrEnum):
    NAME = "name"
    NUMBER = "number"
    QUOTED = "quoted"
    DOUBLE_QUOTED = "double_quoted"

@dataclass
class Var(Node):
    kind: VarKind
    name: str

    def pretty(self, indent: int = 0) -> str:
        return f"{self._p(indent)}{self.kind}.{self.name}"


@dataclass
class Symbol(Node):
    tag: SymbolTag
    value: str | int

    def pretty(self, indent: int = 0) -> str:
        v = self.value if self.tag == "name" else repr(self.value)
        return f"{self._p(indent)}{v}"


@dataclass
class ParenthesizedPattern(Node):
    children: list[PatternElement]

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        if not self.children:
            return f"{p}()"
        inner = "\n".join(c.pretty(indent + 1) for c in self.children)
        return f"{p}(\n{inner}\n{p})"


@dataclass
class Call(Node):
    name: str
    args: Expression

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        if not self.args.children:
            return f"{p}<{self.name}>"
        inner = "\n".join(a.pretty(indent + 1) for a in self.args.children)
        return f"{p}<{self.name}\n{inner}\n{p}>"


@dataclass
class ParenthesizedExpression(Node):
    children: list[ExpressionElement]

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        if not self.children:
            return f"{p}()"
        inner = "\n".join(c.pretty(indent + 1) for c in self.children)
        return f"{p}(\n{inner}\n{p})"


PatternElement: TypeAlias = Var | Symbol | ParenthesizedPattern
ExpressionElement: TypeAlias = Var | Symbol | ParenthesizedExpression | Call


@dataclass
class Pattern(Node):
    children: list[PatternElement]

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        if not self.children:
            return f"{p}(empty)"
        return "\n".join(c.pretty(indent) for c in self.children)


@dataclass
class Expression(Node):
    children: list[ExpressionElement]

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        if not self.children:
            return f"{p}(empty)"
        return "\n".join(c.pretty(indent) for c in self.children)


@dataclass
class Rule(Node):
    pattern: Pattern
    expression: Expression

    def pretty(self, indent: int = 0) -> str:
        p = self._p(indent)
        parts = [f"{p}Rule"]
        parts.append(f"{p}  pat:")
        parts.append(self.pattern.pretty(indent + 2))
        parts.append(f"{p}  = expr:")
        parts.append(self.expression.pretty(indent + 2))
        return "\n".join(parts)


@dataclass
class Definition(Node):
    name: str
    rules: list[Rule]

    def pretty(self, indent: int = 0) -> str:
        parts = [f"{self._p(indent)}Definition {self.name}"]
        parts.extend(r.pretty(indent + 1) for r in self.rules)
        return "\n".join(parts)


@dataclass
class Program(Node):
    definitions: list[Definition]

    def pretty(self, indent: int = 0) -> str:
        parts = [f"{self._p(indent)}Program"]
        parts.extend(d.pretty(indent + 1) for d in self.definitions)
        return "\n".join(parts)


# ── Parser ───────────────────────────────────────────────────

_VAR_TOKS = (EVarToken, TVarToken, SVarToken)
_SYM_TOKS = (NameToken, NumberToken, QuotedSequenceToken, DoubleQuotedNameToken)
_PAT_START = _VAR_TOKS + _SYM_TOKS + (LParenthesisToken,)
_EXPR_START = _PAT_START + (LAngleToken,)

_VAR_KIND: dict[type, VarKind] = {EVarToken: VarKind.E, TVarToken: VarKind.T, SVarToken: VarKind.S}
_SYM_TAG: dict[type, SymbolTag] = {
    NameToken: SymbolTag.NAME, NumberToken: SymbolTag.NUMBER,
    QuotedSequenceToken: SymbolTag.QUOTED, DoubleQuotedNameToken: SymbolTag.DOUBLE_QUOTED,
}


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = [t for t in tokens if not isinstance(t, (WhitespaceToken, CommentToken))]
        self.pos = 0

    def _peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]; self.pos += 1; return tok

    def _expect(self, typ: type[Token]) -> Token:
        tok = self._advance()
        assert isinstance(tok, typ), f"expected {typ.__name__}, got {type(tok).__name__} at {tok.span}"
        return tok

    def _at(self, *types: type[Token]) -> bool:
        return isinstance(self._peek(), types)

    # ── entry ──

    def parse(self) -> Program:
        defs: list[Definition] = []
        while self._peek() is not None:
            defs.append(self._definition())
        span = Span(defs[0].span.start, defs[-1].span.end) if defs else Span(Position(1, 1), Position(1, 1))
        return Program(span=span, definitions=defs)

    # ── grammar rules ──

    def _definition(self) -> Definition:
        name_tok = self._expect(NameToken)
        self._expect(LBraceToken)
        rules = [self._rule()]
        while self._at(SemicolonToken):
            self._advance()
            if self._at(RBraceToken):
                break
            rules.append(self._rule())
        end = self._expect(RBraceToken)
        return Definition(span=Span(name_tok.span.start, end.span.end), name=name_tok.value, rules=rules)

    def _rule(self) -> Rule:
        pat = self._pattern()
        self._expect(EqSignToken)
        expr = self._expression()
        start = pat.span.start
        end = expr.span.end
        return Rule(span=Span(start, end), pattern=pat, expression=expr)

    def _pattern(self) -> Pattern:
        elems: list[PatternElement] = []
        start_tok = self._peek()
        start = start_tok.span.start if start_tok else Position(1, 1)
        while self._at(*_PAT_START):
            elems.append(self._pattern_elem())
        end = elems[-1].span.end if elems else start
        return Pattern(span=Span(start, end), children=elems)

    def _pattern_elem(self) -> PatternElement:
        if self._at(*_VAR_TOKS):   return self._var()
        if self._at(*_SYM_TOKS):   return self._symbol()
        if self._at(LParenthesisToken): return self._pattern_group()
        assert False, f"unexpected {self._peek()} in pattern"

    def _expression(self) -> Expression:
        elems: list[ExpressionElement] = []
        start_tok = self._peek()
        start = start_tok.span.start if start_tok else Position(1, 1)
        while self._at(*_EXPR_START):
            elems.append(self._expr_elem())
        end = elems[-1].span.end if elems else start
        return Expression(span=Span(start, end), children=elems)

    def _expr_elem(self) -> ExpressionElement:
        if self._at(LAngleToken):       return self._call()
        if self._at(LParenthesisToken):  return self._expression_group()
        if self._at(*_VAR_TOKS):         return self._var()
        if self._at(*_SYM_TOKS):         return self._symbol()
        assert False, f"unexpected {self._peek()} in expression"

    # ── atoms / compounds ──

    def _var(self) -> Var:
        tok = self._advance()
        return Var(span=tok.span, kind=_VAR_KIND[type(tok)], name=tok.value)

    def _symbol(self) -> Symbol:
        tok = self._advance()
        return Symbol(span=tok.span, tag=_SYM_TAG[type(tok)], value=tok.value)

    def _pattern_group(self) -> ParenthesizedPattern:
        lp = self._expect(LParenthesisToken)
        children = self._pattern()
        rp = self._expect(RParenthesisToken)
        return ParenthesizedPattern(span=Span(lp.span.start, rp.span.end), children=children.children)

    def _expression_group(self) -> ParenthesizedExpression:
        lp = self._expect(LParenthesisToken)
        children = self._expression()
        rp = self._expect(RParenthesisToken)
        return ParenthesizedExpression(span=Span(lp.span.start, rp.span.end), children=children.children)

    def _call(self) -> Call:
        la = self._expect(LAngleToken)
        name_tok = self._expect(NameToken)
        # Grammar says Pattern here, but Expression lets you nest calls:
        #   <F <G e.x>>  — change to self._pattern() if you want strict grammar
        args = self._expression()
        ra = self._expect(RAngleToken)
        return Call(span=Span(la.span.start, ra.span.end), name=name_tok.value, args=args)


# ── Utility ──────────────────────────────────────────────────

def flat_parens_of_pattern(p: list[PatternElement]) -> list[ParenthesizedPattern]:
    result: list[ParenthesizedPattern] = []
    for elem in p:
        if isinstance(elem, ParenthesizedPattern):
            if all(not isinstance(c, ParenthesizedPattern) for c in elem.children):
                result.append(elem)
            else:
                result.extend(flat_parens_of_pattern(elem.children))
    return result

def flat_parens_of_function(d: Definition) -> list[ParenthesizedPattern]:
    result: list[ParenthesizedPattern] = []
    for rule in d.rules:
        flat_parens = flat_parens_of_pattern(rule.pattern.children)
        result.extend(flat_parens)
    return result

def build_ast_from_file(filename: str) -> Program:
    tokens = lex_file(filename)
    parser = Parser(tokens)
    return parser.parse()

def build_ast_from_string(source: str) -> Program:
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()