from __future__ import annotations

from dataclasses import dataclass

import refal.parser.parser as parser
from .common import *

@dataclass(frozen=True)
class AbstractPatternElement:
    pass


@dataclass(frozen=True)
class AbstractEVar(AbstractPatternElement):
    def __str__(self) -> str:
        return "e"


@dataclass(frozen=True)
class AbstractTvar(AbstractPatternElement):
    def __str__(self) -> str:
        return "t"


@dataclass(frozen=True)
class AbstractSvar(AbstractPatternElement):
    def __str__(self) -> str:
        return "s"


@dataclass(frozen=True)
class AbstractSymbol(AbstractPatternElement):
    value: str | int

    def __str__(self) -> str:
        return f"{self.value}"


@dataclass(frozen=True)
class AbstractedRegexedElement(AbstractPatternElement):
    alternation: Alternation

    def __str__(self) -> str:
        return f"{self.alternation}"


@dataclass(frozen=True)
class AbstractParenthesizedPattern(AbstractPatternElement):
    children: list[AbstractPatternElement]

    def __str__(self) -> str:
        return f"({' '.join(str(c) for c in self.children)})"

    def __hash__(self) -> int:
        return hash(tuple(self.children))


@dataclass
class AbstractPattern:
    children: list[AbstractPatternElement]

    def __str__(self) -> str:
        return " ".join(str(c) for c in self.children)

    
    @staticmethod
    def _from_concrete(
        node: list[parser.PatternElement],
        seen: dict[parser.VarKind, set[str]]
    ) -> list[AbstractPatternElement]:
        result: list[AbstractPatternElement] = []
        for e in node:
            match e:
                case parser.Symbol(_):
                    raise NotImplementedError("Abstractization of symbols is not supported yet!")
                case parser.Var(_, kind, name):
                    match kind:
                        case parser.VarKind.E:
                            if name in seen[kind]:
                                raise NotImplementedError(f"No support for repeated vars yet!")
                            seen[kind] |= {name}
                            result.append(AbstractEVar())
                        case parser.VarKind.T:
                            if name in seen[kind]:
                                raise NotImplementedError(f"No support for repeated vars yet!")
                            seen[kind] |= {name}
                            result.append(AbstractTvar())
                        case parser.VarKind.S:
                            if name in seen[kind]:
                                raise NotImplementedError(f"No support for repeated vars yet!")
                            seen[kind] |= {name}
                            result.append(AbstractSvar())
                case parser.ParenthesizedPattern(_, children):
                    result.append(
                        AbstractParenthesizedPattern(
                            children=AbstractPattern._from_concrete(children, seen)
                        )
                    )
        return result

    @staticmethod
    def from_concrete(node: parser.Pattern) -> AbstractPattern:
        return AbstractPattern(children=AbstractPattern._from_concrete(node.children, {
            parser.VarKind.E: set(),
            parser.VarKind.T: set(),
            parser.VarKind.S: set(),
        }))


def _flat_parens_of_abstract_pattern(
    elements: list[AbstractPatternElement],
) -> list[AbstractParenthesizedPattern]:
    result: list[AbstractParenthesizedPattern] = []
    for elem in elements:
        if isinstance(elem, AbstractParenthesizedPattern):
            if all(
                not isinstance(c, AbstractParenthesizedPattern) for c in elem.children
            ):
                result.append(elem)
            else:
                result.extend(_flat_parens_of_abstract_pattern(elem.children))
    return result

def flat_parens_of_abstract_pattern(
    ap: AbstractPattern,
) -> list[AbstractParenthesizedPattern]:
    return _flat_parens_of_abstract_pattern(ap.children)


def _rewrite(
    elements: list[AbstractPatternElement],
    mapping: dict[AbstractPatternElement, AbstractPatternElement],
) -> list[AbstractPatternElement]:
    result: list[AbstractPatternElement] = []
    for elem in elements:
        if m := mapping.get(elem):
            result.append(m)
        else:
            if isinstance(elem, AbstractParenthesizedPattern):
                result.append(AbstractParenthesizedPattern(
                    _rewrite(elem.children, mapping)
                    ))
            else:
                result.append(elem)
    return result

def rewrite(
    ap: AbstractPattern,
    mapping: dict[AbstractPatternElement, AbstractPatternElement],
) -> AbstractPattern:
    return AbstractPattern(_rewrite(ap.children, mapping))


def replace_at_path(
    pattern: AbstractPattern,
    path: tuple[int, ...],
    replacement: AbstractPatternElement,
) -> AbstractPattern:
    """Replace the node at *path* with *replacement* (or whole pattern if path empty)."""
    if not path:
        return AbstractPattern([replacement])
    children = list(pattern.children)
    _replace_in_children(children, path, replacement)
    return AbstractPattern(children)


def _replace_in_children(
    children: list[AbstractPatternElement],
    path: tuple[int, ...],
    replacement: AbstractPatternElement,
) -> None:
    i, *rest = path
    if not rest:
        children[i] = replacement
        return
    elem = children[i]
    if not isinstance(elem, AbstractParenthesizedPattern):
        raise ValueError(f"Invalid path {path}")
    sub = list(elem.children)
    _replace_in_children(sub, tuple(rest), replacement)
    children[i] = AbstractParenthesizedPattern(sub)


if __name__ == "__main__":
    # (e t) (s [a b] e)
    ap = AbstractPattern(children=[
        AbstractParenthesizedPattern(children=[
            AbstractEVar(),
            AbstractTvar(),
        ]),
        AbstractParenthesizedPattern(children=[
            AbstractSvar(),
            AbstractedRegexedElement(
                Alternation(alternatives=[
                    Letter(name="a"),
                    Letter(name="b"),
                ]),
            ),
            AbstractEVar(),
        ]),
    ])


    for p in flat_parens_of_abstract_pattern(ap):
        print(p)
