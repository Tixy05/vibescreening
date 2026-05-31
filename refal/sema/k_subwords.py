"""Extract k-subwords (nesting-level subpatterns) from abstract patterns."""

from __future__ import annotations

from dataclasses import dataclass

from .abstractizer import (
    AbstractParenthesizedPattern,
    AbstractPattern,
    AbstractPatternElement,
    AbstractedRegexedElement,
)


@dataclass(frozen=True)
class KSubword:
    """Canonical flat sequence representing a k-subword shape."""

    elements: tuple[AbstractPatternElement, ...]

    def __str__(self) -> str:
        return "".join(_elem_symbol(e) for e in self.elements)

    def __hash__(self) -> int:
        return hash(_canonical_key(self.elements))


def _canonical_key(elements: tuple[AbstractPatternElement, ...]) -> tuple:
    return tuple(_elem_key(e) for e in elements)


def _elem_key(elem: AbstractPatternElement) -> tuple:
    if isinstance(elem, AbstractParenthesizedPattern):
        return ("p", _canonical_key(tuple(elem.children)))
    if isinstance(elem, AbstractedRegexedElement):
        names = tuple(sorted(L.name for L in elem.alternation.alternatives))
        return ("r", names)
    return (type(elem).__name__,)


@dataclass(frozen=True)
class KSubwordOccurrence:
    """One occurrence of a k-subword in a pattern tree."""

    subword: KSubword
    index: int  # 1-based class index among distinct k-subwords at this level
    path: tuple[int, ...]  # indices from pattern root to the node to replace


def _elem_symbol(elem: AbstractPatternElement) -> str:
    if isinstance(elem, AbstractParenthesizedPattern):
        return f"({''.join(_elem_symbol(c) for c in elem.children)})"
    if isinstance(elem, AbstractedRegexedElement):
        return str(elem.alternation)
    return str(elem)


def max_paren_depth(elements: list[AbstractPatternElement]) -> int:
    """Maximum parenthesis nesting depth (0 if no parens)."""
    best = 0
    for elem in elements:
        if isinstance(elem, AbstractParenthesizedPattern):
            best = max(best, 1 + max_paren_depth(elem.children))
    return best


def max_paren_depth_pattern(pattern: AbstractPattern) -> int:
    return max_paren_depth(pattern.children)


def _has_nested_parens(elements: list[AbstractPatternElement]) -> bool:
    return any(isinstance(e, AbstractParenthesizedPattern) for e in elements)


def _collect_k_subwords_at_level(
    elements: list[AbstractPatternElement],
    k: int,
    max_depth: int,
    open_depth: int,
    path_prefix: tuple[int, ...],
    shapes: dict[KSubword, int],
    occurrences: list[KSubwordOccurrence],
) -> None:
    for i, elem in enumerate(elements):
        path = path_prefix + (i,)
        if isinstance(elem, AbstractParenthesizedPattern):
            group_depth = open_depth + 1
            children = elem.children
            if group_depth == k:
                if k == max_depth:
                    if not _has_nested_parens(children):
                        subword = KSubword(tuple(children))
                        _register(subword, path, shapes, occurrences)
                else:
                    subword = KSubword(tuple(children))
                    _register(subword, path, shapes, occurrences)
            _collect_k_subwords_at_level(
                children, k, max_depth, group_depth, path, shapes, occurrences
            )


def _register(
    subword: KSubword,
    path: tuple[int, ...],
    shapes: dict[KSubword, int],
    occurrences: list[KSubwordOccurrence],
) -> None:
    if subword not in shapes:
        shapes[subword] = len(shapes) + 1
    occurrences.append(
        KSubwordOccurrence(subword=subword, index=shapes[subword], path=path)
    )


def extract_k_subwords_at_level(
    pattern: AbstractPattern, k: int, max_depth: int | None = None
) -> tuple[list[KSubword], list[KSubwordOccurrence]]:
    """Return distinct k-subwords (1-based indices) and all occurrences in *pattern*."""
    if max_depth is None:
        max_depth = max_paren_depth_pattern(pattern)
    shapes: dict[KSubword, int] = {}
    occurrences: list[KSubwordOccurrence] = []

    if k == 0:
        subword = KSubword(tuple(pattern.children))
        _register(subword, (), shapes, occurrences)
    else:
        _collect_k_subwords_at_level(
            pattern.children, k, max_depth, 0, (), shapes, occurrences
        )

    ordered = sorted(shapes.keys(), key=lambda s: shapes[s])
    return ordered, occurrences


def extract_k_subwords_function(
    patterns: list[AbstractPattern], k: int
) -> tuple[list[KSubword], list[tuple[int, KSubwordOccurrence]]]:
    """Distinct k-subwords across all patterns; occurrences tagged with rule index."""
    max_d = max(max_paren_depth_pattern(p) for p in patterns) if patterns else 0
    shapes: dict[KSubword, int] = {}
    tagged: list[tuple[int, KSubwordOccurrence]] = []

    for rule_idx, pattern in enumerate(patterns):
        if k == 0:
            subword = KSubword(tuple(pattern.children))
            if subword not in shapes:
                shapes[subword] = len(shapes) + 1
            tagged.append(
                (
                    rule_idx,
                    KSubwordOccurrence(
                        subword=subword, index=shapes[subword], path=()
                    ),
                )
            )
        else:
            _, occs = extract_k_subwords_at_level(pattern, k, max_d)
            for occ in occs:
                if occ.subword not in shapes:
                    shapes[occ.subword] = len(shapes) + 1
                tagged.append(
                    (
                        rule_idx,
                        KSubwordOccurrence(
                            subword=occ.subword,
                            index=shapes[occ.subword],
                            path=occ.path,
                        ),
                    )
                )

    ordered = sorted(shapes.keys(), key=lambda s: shapes[s])
    return ordered, tagged
