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
    anti_sltl_nf: str = ""
    fallthrough_sltl_nf: str = ""


@dataclass(frozen=True)
class FunctionLog:
    name: str
    dfa_by_level: tuple[DfaLevelStats, ...]
    level0: tuple["Level0PatternLogRow", ...]
    rules: tuple[RuleLogEntry, ...]


@dataclass(frozen=True)
class Level0PatternLogRow:
    index: int
    tag: str
    pattern: str
    regex: str
    sltl_nf: str


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
    lines.append("=== Level-0 pattern / regex / SLTL ===")
    if data.level0:
        for row in data.level0:
            lines.append(f"rule {row.index} [{row.tag}] pattern: {row.pattern}")
            lines.append(f"  regex: {row.regex}")
            lines.append(f"  SLTL: {row.sltl_nf}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("=== Anti-SLTL per rule ===")
    for rule in data.rules:
        lines.append(f"rule {rule.index} ({'good' if rule.good else 'bad'}):")
        if rule.anti is None:
            lines.append("  anti: (not built)")
        else:
            lines.append(f"  anti: {rule.anti_sltl_nf}")

    lines.append("")
    lines.append("=== Accumulated L (fall-through SLTL after each rule) ===")
    for rule in data.rules:
        lines.append(f"after rule {rule.index}:")
        lines.append(f"  L: {rule.fallthrough_sltl_nf}")

    (func_dir / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_root_for_source(path: Path, *, log_base: Path = Path("log")) -> Path:
    """``log/<file-name>/`` for a source path like ``refal/test/foo.ref``."""
    return log_base / path.name
