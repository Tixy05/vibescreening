"""Cascade encoding of est-only patterns (main.pdf §4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import refal.parser.parser as parser
from refal.parser.parser import build_ast_from_string

if TYPE_CHECKING:
    from antidict import DFA

from .abstractizer import (
    AbstractParenthesizedPattern,
    AbstractPattern,
    AbstractPatternElement,
    AbstractedRegexedElement,
    Alternation,
    Letter,
    replace_at_path,
)
from .annotated_dfa import AnnotatedDFA
from .k_subwords import (
    KSubword,
    KSubwordOccurrence,
    extract_k_subwords_function,
    max_paren_depth_pattern,
)
from .regex_dfa import (
    BiClass,
    bi_alternation_for_index,
    build_step_dfa,
    terminal_alphabet_for_step,
    union_accepts_all,
)
from .trace_log import trace_step


@dataclass
class EncodingStep:
    level: int
    subwords: list[KSubword]
    bi_classes: list[BiClass]
    patterns: list[AbstractPattern]
    annotated_dfa_states: int = 0


@dataclass
class EncodingResult:
    steps: list[EncodingStep]
    final_patterns: list[AbstractPattern]


def function_t_alphabet(encoding: EncodingResult) -> frozenset[str]:
    """Γ for ``t`` at flat level (after cascade).

    * No cascade steps: top level ``{b, s}``.
    * After at least one encoding step: ``{s} ∪ {b_i}`` and ``b0`` if any step's
      annotated union DFA had a state with empty annotation (see ``quotient_to_bi_classes``).
    Bare ``b`` never appears once the cascade has run.
    """
    if not encoding.steps:
        return frozenset({"b", "s"})
    letters: set[str] = {"s"}
    for step in encoding.steps:
        for cls in step.bi_classes:
            letters.add(cls.label)
    return frozenset(letters)


def sorted_terminal_alphabet(letters: set[str] | frozenset[str]) -> list[str]:
    """``s``, then ``b1``…``bn``, then ``b0``, then bare ``b`` (top level only)."""
    s = [x for x in letters if x == "s"]
    bn = sorted(
        (x for x in letters if len(x) >= 2 and x[0] == "b" and x[1:].isdigit() and x != "b0"),
        key=lambda x: int(x[1:]),
    )
    b0 = [x for x in letters if x == "b0"]
    bare_b = [x for x in letters if x == "b"]
    return s + bn + b0 + bare_b


def flat_terminal_alphabet(
    encoding: EncodingResult,
    final_patterns: list[AbstractPattern],
) -> list[str]:
    """Level-0 Γ for regex / ``e*`` over flat patterns (``tmp.md``)."""
    if not encoding.steps:
        return sorted_terminal_alphabet({"b", "s"})
    from .pattern_sltl import alphabet_from_pattern

    letters: set[str] = set()
    for ap in final_patterns:
        letters |= alphabet_from_pattern(ap)
    return sorted_terminal_alphabet(letters)


def _alternation_for_index(index: int, classes: list[BiClass]) -> AbstractedRegexedElement:
    labels = bi_alternation_for_index(index, classes)
    return AbstractedRegexedElement(
        Alternation(tuple(Letter(name) for name in labels))
    )


def _apply_replacements(
    patterns: list[AbstractPattern],
    level: int,
    tagged: list[tuple[int, KSubwordOccurrence]],
    classes: list[BiClass],
) -> list[AbstractPattern]:
    result = [AbstractPattern(list(p.children)) for p in patterns]
    for rule_idx, occ in tagged:
        repl = _alternation_for_index(occ.index, classes)
        result[rule_idx] = replace_at_path(result[rule_idx], occ.path, repl)
    return result


def _flatten_remaining_parens(
    patterns: list[AbstractPattern],
    bi_labels: list[str],
) -> list[AbstractPattern]:
    """Replace leftover ``(...)`` with ``(b_i|...)`` — no level-0 union DFA."""
    labels = bi_labels if bi_labels else ["b0"]
    fallback = AbstractedRegexedElement(
        Alternation(tuple(Letter(name) for name in labels))
    )

    def rewrite_elem(elem: AbstractPatternElement) -> AbstractPatternElement:
        if isinstance(elem, AbstractParenthesizedPattern):
            return fallback
        return elem

    return [
        AbstractPattern([rewrite_elem(e) for e in p.children]) for p in patterns
    ]


def encode_patterns(
    patterns: list[AbstractPattern],
    *,
    name: str = "_",
    out_dir: Path | None = None,
) -> EncodingResult:
    """Encode patterns from max nesting depth down to 1 (no level-0 union DFA)."""
    if not patterns:
        return EncodingResult(steps=[], final_patterns=[])

    patterns = [AbstractPattern(list(p.children)) for p in patterns]
    max_d = max(max_paren_depth_pattern(p) for p in patterns)
    steps: list[EncodingStep] = []
    bi_labels: list[str] = []
    step_index = 0
    trace_step(f"encode {name}: max_paren_depth={max_d}, {len(patterns)} pattern(s)")

    for level in range(max_d, 0, -1):
        trace_step(f"encode {name} level {level}: extract k-subwords")
        subwords, tagged = extract_k_subwords_function(patterns, level)
        if not subwords:
            trace_step(f"encode {name} level {level}: skip (no subwords)")
            continue

        terminals = terminal_alphabet_for_step(step_index, bi_labels)
        trace_step(
            f"encode {name} level {level}: build_step_dfa "
            f"({len(subwords)} subword(s), alphabet size {len(terminals)})"
        )
        dfa, annotations, classes = build_step_dfa(subwords, terminals)
        trace_step(
            f"encode {name} level {level}: DFA done — "
            f"{dfa.num_states} states, {len(classes)} bi-class(es)"
        )
        bi_labels = [c.label for c in classes]

        if out_dir is not None:
            stem = out_dir / f"{name}_depth{level}"
            union_note = None
            if union_accepts_all(dfa):
                union_note = f"L(union) = ({'|'.join(dfa.alphabet)})*"
            AnnotatedDFA(
                dfa,
                annotations,
                depth=level,
                function_name=name,
                subwords=subwords,
                bi_classes=classes,
                union_note=union_note,
            ).write_outputs(stem)
            _write_bi_map(stem.with_suffix(".bi_map.txt"), classes, dfa)

        trace_step(f"encode {name} level {level}: apply replacements")
        patterns = _apply_replacements(patterns, level, tagged, classes)
        steps.append(
            EncodingStep(
                level=level,
                subwords=subwords,
                bi_classes=classes,
                patterns=[AbstractPattern(list(p.children)) for p in patterns],
                annotated_dfa_states=dfa.num_states,
            )
        )
        step_index += 1

    trace_step(f"encode {name}: flatten remaining parens (bi={len(bi_labels)})")
    patterns = _flatten_remaining_parens(patterns, bi_labels)

    trace_step(f"encode {name}: cascade finished ({len(steps)} step(s))")
    return EncodingResult(steps=steps, final_patterns=patterns)


def encode_abstract_pattern(
    ap: AbstractPattern,
    *,
    name: str = "_",
) -> AbstractPattern:
    """Encode a single abstract pattern through the cascade."""
    result = encode_patterns([ap], name=name)
    if not result.final_patterns:
        return AbstractPattern([])
    return result.final_patterns[0]


def encode_function(
    function: parser.Definition,
    *,
    out_dir: Path | None = None,
) -> EncodingResult:
    """Encode patterns from max nesting depth down to 1 (flat eAe… over bi)."""
    from .abstractizer import abstractize_relaxed

    patterns = [abstractize_relaxed(rule.pattern) for rule in function.rules]
    return encode_patterns(patterns, name=function.name, out_dir=out_dir)


def _write_bi_map(path: Path, classes: list[BiClass], dfa: "DFA") -> None:
    lines = [f"{c.label} -> {set(c.indices)}" for c in classes]
    if union_accepts_all(dfa):
        alpha = "|".join(dfa.alphabet)
        lines.append(f"# L(union) = ({alpha})*")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_encoding(name: str, result: EncodingResult) -> None:
    print(f"Function {name}: {len(result.steps)} encoding steps")
    for step in result.steps:
        print(f"  level {step.level}: subwords={[str(s) for s in step.subwords]}")
        print(f"    bi: {[(c.label, set(c.indices)) for c in step.bi_classes]}")
    print("Final patterns:")
    for i, p in enumerate(result.final_patterns):
        print(f"  rule {i}: {p}")


def check_function(function: parser.Definition) -> list[int]:
    """Return 0-based indices of screened rules in *function*."""
    from .screening import screen_function

    return screen_function(function).screened_indices


_LISTING3 = """\
f {
    t.1 (e.1 (s.1) s.2) = ;
    ((t.1) e.1 s.2) s.3 = ;
}
"""

_EMPTY_PLUS_ETA = """\
g {
    = ;
    e.1 t.2 e.3 = ;
}
"""


def main() -> None:
    out = Path("output/screening")
    for source in (_LISTING3, _EMPTY_PLUS_ETA):
        fn = build_ast_from_string(source).definitions[0]
        result = encode_function(fn, out_dir=out)
        _print_encoding(fn.name, result)
        print()


if __name__ == "__main__":
    main()
