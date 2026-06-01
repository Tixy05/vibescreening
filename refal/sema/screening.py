"""Top-down screening detection (main.pdf §7)."""

from __future__ import annotations

from dataclasses import dataclass, field

import refal.parser.parser as parser

from .abstractizer import (
    PatternKind,
    abstractize_relaxed,
    classify_pattern,
)
from .pattern_sltl import (
    align_alphabet,
    anti_sltl_from_final_pattern,
    detect_form,
    function_encoding_alphabet,
    universe_sltl,
)
from .screening_analyzer import encode_function


@dataclass
class RuleScreening:
    index: int
    screened: bool
    good: bool
    discarded: bool
    approximated: bool = False
    defines_constraint: bool = False
    reason: str = ""


@dataclass
class FunctionScreening:
    name: str
    rules: list[RuleScreening] = field(default_factory=list)

    @property
    def screened_indices(self) -> list[int]:
        return [r.index for r in self.rules if r.screened]


def _defines_constraint(kind: PatternKind) -> bool:
    """Only good source patterns accumulate constraints in L (§7)."""
    return kind == PatternKind.GOOD


def screen_function(function: parser.Definition) -> FunctionScreening:
    """Walk rules top-to-bottom; detect screened rules via accumulated L ⊆ anti."""
    encoding = encode_function(function)
    dense_map = function_encoding_alphabet(encoding.final_patterns)
    dense_alpha = set(dense_map.values())
    abstract_patterns = [abstractize_relaxed(rule.pattern) for rule in function.rules]
    fallthrough = universe_sltl(dense_alpha)
    result = FunctionScreening(name=function.name)

    for i, rule in enumerate(function.rules):
        kind = classify_pattern(rule.pattern)
        defines = _defines_constraint(kind)
        abstract_ap = abstract_patterns[i]
        approximated = detect_form(abstract_ap) is None
        anti = anti_sltl_from_final_pattern(
            encoding.final_patterns[i],
            dense_map,
            abstract_ap=abstract_ap,
            peer_patterns=abstract_patterns,
            rule_index=i,
        )

        if anti is None:
            result.rules.append(
                RuleScreening(
                    index=i,
                    screened=False,
                    good=defines,
                    discarded=True,
                    approximated=False,
                    defines_constraint=False,
                    reason="could not build anti-SLTL",
                )
            )
            continue

        anti = align_alphabet(anti, fallthrough)
        if i == 0:
            screened = False
        else:
            # Prior fall-through language already covered by this rule's anti-lang.
            screened = fallthrough.is_subset_of(anti)
        if approximated:
            reason = "screened (approx)" if screened else "reachable (approx)"
        else:
            reason = "screened" if screened else "reachable"

        # Bad patterns may yield SLTL-shaped anti-languages after relaxed
        # abstractization, but they never narrow the accumulated language.
        if defines:
            fallthrough = fallthrough.intersect(anti)

        result.rules.append(
            RuleScreening(
                index=i,
                screened=screened,
                good=defines,
                discarded=False,
                approximated=approximated,
                defines_constraint=defines,
                reason=reason,
            )
        )

    return result
