"""Annotated DFA visualization."""

from __future__ import annotations

import html
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from antidict import DFA

if TYPE_CHECKING:
    from .k_subwords import KSubword
    from .regex_dfa import BiClass


@dataclass
class AnnotatedDFA:
    dfa: DFA
    annotations: dict[int, frozenset[int]]
    depth: int | None = None
    function_name: str | None = None
    subwords: list[KSubword] = field(default_factory=list)
    bi_classes: list[BiClass] = field(default_factory=list)
    union_note: str | None = None

    def _format_ann(self, ann: frozenset[int]) -> str:
        if not ann:
            return "∅"
        return "{" + ",".join(str(i) for i in sorted(ann)) + "}"

    def _graph_title(self) -> str:
        parts: list[str] = []
        if self.function_name:
            parts.append(self.function_name)
        if self.depth is not None:
            parts.append(f"depth level {self.depth}")
        return " — ".join(parts) if parts else "Annotated DFA"

    def _pattern_for_index(self, index: int) -> str:
        if 1 <= index <= len(self.subwords):
            text = str(self.subwords[index - 1])
            return text if text else "ε"
        return "?"

    def _patterns_for_bi(self, bi: BiClass) -> str:
        if not bi.indices:
            return "∅"
        pats = [self._pattern_for_index(i) for i in sorted(bi.indices)]
        return " | ".join(pats)

    def _legend_html(self) -> str:
        rows: list[str] = []

        rows.append(
            '<tr><td colspan="2" bgcolor="#eeeeee">'
            "<b>index → pattern</b></td></tr>"
        )
        if self.subwords:
            for i, sw in enumerate(self.subwords, start=1):
                pat = str(sw) if str(sw) else "ε"
                rows.append(
                    f"<tr><td align=\"right\">{i}</td>"
                    f"<td align=\"left\">{html.escape(pat)}</td></tr>"
                )
        else:
            rows.append(
                '<tr><td colspan="2" align="center">—</td></tr>'
            )

        rows.append(
            '<tr><td colspan="2" bgcolor="#eeeeee">'
            "<b>b_i → pattern</b></td></tr>"
        )
        if self.bi_classes:
            for bi in self.bi_classes:
                pat = self._patterns_for_bi(bi)
                rows.append(
                    f"<tr><td align=\"right\">{html.escape(bi.label)}</td>"
                    f"<td align=\"left\">{html.escape(pat)}</td></tr>"
                )
        else:
            rows.append(
                '<tr><td colspan="2" align="center">—</td></tr>'
            )

        if self.union_note:
            rows.append(
                f'<tr><td colspan="2" align="center">'
                f"<i>{html.escape(self.union_note)}</i></td></tr>"
            )

        return (
            "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" "
            'cellpadding="4">'
            + "".join(rows)
            + "</table>"
        )

    def to_dot(self) -> str:
        title = html.escape(self._graph_title())
        legend = self._legend_html()

        lines: list[str] = [
            "digraph AnnotatedDFA {",
            "    rankdir=TB;",
            '    graph [fontname="Helvetica", fontsize=14, compound=true, '
            f'label=<{title}>, labelloc=t, labeljust=c];',
            '    node [shape=circle, fontname="Courier", fontsize=10];',
            '    edge [fontname="Courier", fontsize=10];',
            "",
            f"    legend [shape=plain, label=<{legend}>];",
            "",
            "    subgraph cluster_automaton {",
            '        label="Annotated DFA";',
            "        rankdir=LR;",
            '        node [shape=circle, fontname="Courier", fontsize=10];',
            '        edge [fontname="Courier", fontsize=10];',
            "",
            '        _start [shape=point, width=0.15];',
            f"        _start -> {self.dfa.initial};",
            "",
        ]

        dead_states = {
            s
            for s in range(self.dfa.num_states)
            if self.dfa._is_dead(s) and s not in self.dfa.accepting
        }

        for s in range(self.dfa.num_states):
            ann = self.annotations.get(s, frozenset())
            label = f"{s}\\n{self._format_ann(ann)}"
            shape = "doublecircle" if s in self.dfa.accepting else "circle"
            parts = [f"shape={shape}", f'label="{label}"']
            if s in dead_states:
                parts.append("style=filled")
                parts.append('fillcolor="#e0c4c4"')
            lines.append(f"        {s} [{', '.join(parts)}];")

        lines.append("")

        for s in range(self.dfa.num_states):
            row = self.dfa.transitions.get(s, {})
            target_syms: dict[int, list[str]] = {}
            for a in self.dfa.alphabet:
                t = row.get(a)
                if t is not None:
                    target_syms.setdefault(t, []).append(a)

            for idx, (t, syms) in enumerate(sorted(target_syms.items())):
                label = ",".join(syms)
                esc = label.replace("\\", "\\\\").replace('"', '\\"')
                col = DFA._EDGE_COLORS[idx % len(DFA._EDGE_COLORS)]
                lines.append(
                    f'        {s} -> {t} [label="{esc}", color={col}, '
                    f"fontcolor={col}];"
                )

        lines.extend([
            "    }",
            "",
            f"    legend -> {self.dfa.initial} "
            "[style=invis, weight=0, lhead=cluster_automaton];",
            "}",
        ])
        return "\n".join(lines)

    def to_svg(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dot_src = self.to_dot()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dot", delete=False,
        ) as tmp:
            tmp.write(dot_src)
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["dot", "-Tsvg", "-o", str(path), tmp_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Graphviz `dot` not found. Install it to render SVG."
            ) from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def write_outputs(self, stem: str | Path) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        dot_path = stem.with_suffix(".dot")
        dot_path.write_text(self.to_dot(), encoding="utf-8")
        self.to_svg(stem.with_suffix(".svg"))
