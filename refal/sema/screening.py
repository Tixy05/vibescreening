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
from .file_log import DfaLevelStats, FunctionLog, RuleLogEntry
from .metrics import sltl_antidict_stats
from .screening_analyzer import EncodingResult, encode_function
from .trace_log import trace_step


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
    encoding: EncodingResult | None = None
    log: FunctionLog | None = None

    @property
    def screened_indices(self) -> list[int]:
        return [r.index for r in self.rules if r.screened]


def _defines_constraint(kind: PatternKind) -> bool:
    """Only good source patterns accumulate constraints in L (§7)."""
    return kind == PatternKind.GOOD


def screen_function(
    function: parser.Definition,
    *,
    collect_log: bool = False,
) -> FunctionScreening:
    """Walk rules top-to-bottom; detect screened rules via accumulated L ⊆ anti."""
    trace_step(
        f"screen {function.name}: start ({len(function.rules)} rule(s))"
    )
    trace_step(f"screen {function.name}: encoding cascade")
    encoding = encode_function(function)
    trace_step(
        f"screen {function.name}: encoding done "
        f"({', '.join(f'L{s.level}={s.annotated_dfa_states}' for s in encoding.steps) or 'flat'})"
    )
    dense_map = function_encoding_alphabet(encoding.final_patterns)
    dense_alpha = set(dense_map.values())
    abstract_patterns = [abstractize_relaxed(rule.pattern) for rule in function.rules]
    trace_step(f"screen {function.name}: universe_sltl (|Σ|={len(dense_alpha)})")
    fallthrough = universe_sltl(dense_alpha)
    result = FunctionScreening(name=function.name, encoding=encoding)
    log_entries: list[RuleLogEntry] = []

    for i, rule in enumerate(function.rules):
        kind = classify_pattern(rule.pattern)
        defines = _defines_constraint(kind)
        abstract_ap = abstract_patterns[i]
        approximated = detect_form(abstract_ap) is None
        trace_step(
            f"screen {function.name} rule {i}: "
            f"{kind}, approx={approximated}, build anti-SLTL"
        )
        anti = anti_sltl_from_final_pattern(
            encoding.final_patterns[i],
            dense_map,
            abstract_ap=abstract_ap,
            peer_patterns=abstract_patterns,
            rule_index=i,
        )

        if anti is None:
            trace_step(f"screen {function.name} rule {i}: anti-SLTL failed")
            if collect_log:
                log_entries.append(
                    RuleLogEntry(
                        index=i,
                        good=defines,
                        anti=None,
                        fallthrough=sltl_antidict_stats(fallthrough),
                    )
                )
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

        anti_stats = sltl_antidict_stats(anti)
        trace_step(
            f"screen {function.name} rule {i}: anti built "
            f"(P={anti_stats.prefixes} F={anti_stats.factors} "
            f"S={anti_stats.suffixes} W={anti_stats.sfw})"
        )
        trace_step(f"screen {function.name} rule {i}: align_alphabet")
        anti = align_alphabet(anti, fallthrough)
        if i == 0:
            screened = False
        else:
            trace_step(f"screen {function.name} rule {i}: is_subset_of (screening test)")
            # Prior fall-through language already covered by this rule's anti-lang.
            screened = fallthrough.is_subset_of(anti)
            trace_step(
                f"screen {function.name} rule {i}: screened={screened}"
            )
        if approximated:
            reason = "screened (approx)" if screened else "reachable (approx)"
        else:
            reason = "screened" if screened else "reachable"

        # Bad patterns may yield SLTL-shaped anti-languages after relaxed
        # abstractization, but they never narrow the accumulated language.
        if defines:
            ft_before = sltl_antidict_stats(fallthrough)
            trace_step(
                f"screen {function.name} rule {i}: intersect fall-through "
                f"(before W={ft_before.sfw})"
            )
            fallthrough = fallthrough.intersect(anti)
            ft_after = sltl_antidict_stats(fallthrough)
            trace_step(
                f"screen {function.name} rule {i}: intersect done "
                f"(after W={ft_after.sfw}, total={ft_after.total})"
            )

        if collect_log:
            log_entries.append(
                RuleLogEntry(
                    index=i,
                    good=defines,
                    anti=sltl_antidict_stats(anti),
                    fallthrough=sltl_antidict_stats(fallthrough),
                )
            )

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

    trace_step(f"screen {function.name}: finished")
    if collect_log:
        result.log = FunctionLog(
            name=function.name,
            dfa_by_level=tuple(
                DfaLevelStats(level=step.level, num_states=step.annotated_dfa_states)
                for step in encoding.steps
            ),
            rules=tuple(log_entries),
        )

    return result
