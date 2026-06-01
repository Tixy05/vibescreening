"""Write per-function analysis logs under log/<source-file>/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .metrics import SltlAntidictStats


@dataclass(frozen=True)
class DfaLevelStats:
    level: int
    num_states: int


@dataclass(frozen=True)
class RuleLogEntry:
    index: int
    good: bool
    anti: SltlAntidictStats | None
    fallthrough: SltlAntidictStats


@dataclass(frozen=True)
class FunctionLog:
    name: str
    dfa_by_level: tuple[DfaLevelStats, ...]
    rules: tuple[RuleLogEntry, ...]


def write_function_log(func_dir: Path, data: FunctionLog) -> None:
    func_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("=== Annotated DFA (per nesting level) ===")
    if data.dfa_by_level:
        for row in data.dfa_by_level:
            lines.append(f"level {row.level}: {row.num_states} states")
    else:
        lines.append("(no cascade steps)")

    lines.append("")
    lines.append("=== Anti-SLTL antidictionary (per rule) ===")
    for rule in data.rules:
        lines.append(f"rule {rule.index} ({'good' if rule.good else 'bad'}):")
        if rule.anti is None:
            lines.append("  (not built)")
        else:
            a = rule.anti
            lines.append(
                f"  prefixes={a.prefixes} factors={a.factors} "
                f"suffixes={a.suffixes} sfw={a.sfw} total={a.total}"
            )

    lines.append("")
    lines.append("=== Accumulated fall-through SLTL (after each rule) ===")
    for rule in data.rules:
        f = rule.fallthrough
        lines.append(
            f"after rule {rule.index}: "
            f"prefixes={f.prefixes} factors={f.factors} "
            f"suffixes={f.suffixes} sfw={f.sfw} total={f.total}"
        )

    (func_dir / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_root_for_source(path: Path, *, log_base: Path = Path("log")) -> Path:
    """``log/<file-name>/`` for a source path like ``refal/test/foo.ref``."""
    return log_base / path.name
