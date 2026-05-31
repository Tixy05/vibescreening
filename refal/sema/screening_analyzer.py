"""Cascade encoding of est-only patterns (main.pdf §4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import refal.parser.parser as parser
from refal.parser.parser import build_ast_from_string

from .abstractizer import (
    AbstractPattern,
    AbstractedRegexedElement,
    Alternation,
    Letter,
    replace_at_path,
)
from .annotated_dfa import AnnotatedDFA
from .k_subwords import (
    KSubword,
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


@dataclass
class EncodingStep:
    level: int
    subwords: list[KSubword]
    bi_classes: list[BiClass]
    patterns: list[AbstractPattern]


@dataclass
class EncodingResult:
    steps: list[EncodingStep]
    final_patterns: list[AbstractPattern]


def _alternation_for_index(index: int, classes: list[BiClass]) -> AbstractedRegexedElement:
    labels = bi_alternation_for_index(index, classes)
    return AbstractedRegexedElement(
        Alternation(tuple(Letter(name) for name in labels))
    )


def _apply_replacements(
    patterns: list[AbstractPattern],
    level: int,
    tagged: list,
    classes: list[BiClass],
) -> list[AbstractPattern]:
    result = [AbstractPattern(list(p.children)) for p in patterns]
    for rule_idx, occ in tagged:
        repl = _alternation_for_index(occ.index, classes)
        result[rule_idx] = replace_at_path(result[rule_idx], occ.path, repl)
    return result


def encode_function(
    function: parser.Definition,
    *,
    out_dir: Path | None = None,
) -> EncodingResult:
    """Iteratively encode patterns from max nesting depth down to 0."""
    patterns = [AbstractPattern.from_concrete(rule.pattern) for rule in function.rules]
    if not patterns:
        return EncodingResult(steps=[], final_patterns=[])

    max_d = max(max_paren_depth_pattern(p) for p in patterns)
    steps: list[EncodingStep] = []
    bi_labels: list[str] = []
    step_index = 0

    for level in range(max_d, -1, -1):
        subwords, tagged = extract_k_subwords_function(patterns, level)
        if not subwords:
            continue

        terminals = terminal_alphabet_for_step(step_index, bi_labels)
        dfa, annotations, classes = build_step_dfa(subwords, terminals)
        bi_labels = [c.label for c in classes]

        if out_dir is not None:
            stem = out_dir / f"{function.name}_depth{level}"
            union_note = None
            if union_accepts_all(dfa):
                union_note = f"L(union) = ({'|'.join(dfa.alphabet)})*"
            AnnotatedDFA(
                dfa,
                annotations,
                depth=level,
                function_name=function.name,
                subwords=subwords,
                bi_classes=classes,
                union_note=union_note,
            ).write_outputs(stem)
            _write_bi_map(stem.with_suffix(".bi_map.txt"), classes, dfa)

        patterns = _apply_replacements(patterns, level, tagged, classes)
        steps.append(
            EncodingStep(
                level=level,
                subwords=subwords,
                bi_classes=classes,
                patterns=[AbstractPattern(list(p.children)) for p in patterns],
            )
        )
        step_index += 1

    return EncodingResult(steps=steps, final_patterns=patterns)


def _write_bi_map(path: Path, classes: list[BiClass], dfa) -> None:
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
    """Phase 1: run encoding; screening violations TBD."""
    encode_function(function)
    return []


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
