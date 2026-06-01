from __future__ import annotations

import enum
from dataclasses import dataclass

import refal.parser.parser as parser
from .common import *


class PatternKind(enum.StrEnum):
    GOOD = "good"
    BAD = "bad"


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
        children = AbstractPattern._from_concrete(node.children, {
            parser.VarKind.E: set(),
            parser.VarKind.T: set(),
            parser.VarKind.S: set(),
        })
        return AbstractPattern(children=_collapse_consecutive_evars(children))


def _walk_pattern_elements(
    elements: list[parser.PatternElement],
) -> list[parser.PatternElement]:
    out: list[parser.PatternElement] = []
    for e in elements:
        out.append(e)
        if isinstance(e, parser.ParenthesizedPattern):
            out.extend(_walk_pattern_elements(e.children))
    return out


def has_constants(pattern: parser.Pattern) -> bool:
    return any(isinstance(e, parser.Symbol) for e in _walk_pattern_elements(pattern.children))


def has_repeated_vars(pattern: parser.Pattern) -> bool:
    seen: set[str] = set()
    for e in _walk_pattern_elements(pattern.children):
        if isinstance(e, parser.Var):
            if e.name in seen:
                return True
            seen.add(e.name)
    return False


def classify_pattern(pattern: parser.Pattern) -> PatternKind:
    if has_constants(pattern) or has_repeated_vars(pattern):
        return PatternKind.BAD
    return PatternKind.GOOD


def _abstractize_relaxed_elements(
    node: list[parser.PatternElement],
) -> list[AbstractPatternElement]:
    result: list[AbstractPatternElement] = []
    for e in node:
        match e:
            case parser.Symbol(tag=parser.SymbolTag.NUMBER, value=_):
                result.append(AbstractSvar())
            case parser.Symbol(tag=_, value=sym_value):
                if isinstance(sym_value, str):
                    result.extend(AbstractSvar() for _ in sym_value)
                else:
                    result.append(AbstractSvar())
            case parser.Var(_, kind, _):
                match kind:
                    case parser.VarKind.E:
                        result.append(AbstractEVar())
                    case parser.VarKind.T:
                        result.append(AbstractTvar())
                    case parser.VarKind.S:
                        result.append(AbstractSvar())
            case parser.ParenthesizedPattern(_, children):
                result.append(
                    AbstractParenthesizedPattern(
                        children=_abstractize_relaxed_elements(children)
                    )
                )
    return result


def _collapse_consecutive_evars(
    elements: list[AbstractPatternElement],
) -> list[AbstractPatternElement]:
    """Merge runs of top-level e-vars into one (e e e ≡ e)."""
    result: list[AbstractPatternElement] = []
    for elem in elements:
        if isinstance(elem, AbstractParenthesizedPattern):
            result.append(
                AbstractParenthesizedPattern(
                    children=_collapse_consecutive_evars(elem.children),
                )
            )
        elif isinstance(elem, AbstractEVar):
            if result and isinstance(result[-1], AbstractEVar):
                continue
            result.append(elem)
        else:
            result.append(elem)
    return result


def abstractize_relaxed(pattern: parser.Pattern) -> AbstractPattern:
    """Abstractize pattern, allowing constants and repeated variable names."""
    children = _abstractize_relaxed_elements(pattern.children)
    return AbstractPattern(children=_collapse_consecutive_evars(children))


def abstractize_strict(pattern: parser.Pattern) -> AbstractPattern:
    """Abstractize a good pattern (no constants, no repeated names)."""
    if classify_pattern(pattern) != PatternKind.GOOD:
        raise ValueError("pattern is not strict-good")
    return abstractize_relaxed(pattern)


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
                Alternation(alternatives=(
                    Letter(name="a"),
                    Letter(name="b"),
                )),
            ),
            AbstractEVar(),
        ]),
    ])


    for p in flat_parens_of_abstract_pattern(ap):
        print(p)
