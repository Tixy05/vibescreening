"""SA-inspired completion structure.

Takes a built :class:`SuffixAutomaton` and a finite alphabet, then
"completes" every live (non-trap) state so it has outgoing transitions
for every letter in the alphabet.  Missing transitions are filled via a
first-character-stripping rule, processed in BFS order from init.

The result is generally an NFA (nondeterministic).
"""

from __future__ import annotations

import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from suffix_automaton import State, SuffixAutomaton


@dataclass
class NFAState:
    """A single state in the completed NFA."""

    id: int
    trans: dict[str, set[int]] = field(default_factory=dict)
    is_accepting: bool = False

    @property
    def is_trap(self) -> bool:
        return len(self.trans) == 0


class CompletedAutomaton:
    """Alphabet-completed automaton derived from a SuffixAutomaton.

    Every state that was "live" (had at least one outgoing transition) in
    the original SA gets transitions for every letter in *alphabet*.
    New edges are added by the first-character-stripping rule described
    in the plan.
    """

    def __init__(self, sa: SuffixAutomaton, alphabet: set[str]) -> None:
        self.alphabet: set[str] = alphabet
        self._states: list[NFAState] = []
        self._original_edges: set[tuple[int, str, int]] = set()
        self._added_edges: set[tuple[int, str, int]] = set()

        self._build(sa)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self, sa: SuffixAutomaton) -> None:
        sa_states = sa.states

        for st in sa_states:
            nfa = NFAState(
                id=st.id,
                trans={ch: {tgt} for ch, tgt in st.trans.items()},
                is_accepting=bool(st.words),
            )
            self._states.append(nfa)

        for st in sa_states:
            for ch, tgt in st.trans.items():
                self._original_edges.add((st.id, ch, tgt))

        self._complete_bfs(sa_states)

    def _read_nfa(self, word: str) -> set[int]:
        """Read *word* from init through the (possibly augmented) NFA.

        Returns the set of states reachable after consuming all characters.
        Because the automaton may be nondeterministic we track a frontier.
        """
        current: set[int] = {0}
        for ch in word:
            nxt: set[int] = set()
            for sid in current:
                nxt |= self._states[sid].trans.get(ch, set())
            if not nxt:
                return set()
            current = nxt
        return current

    def _complete_bfs(self, sa_states: list[State]) -> None:
        """BFS over the *original* SA graph; complete each live state."""

        # paths[state_id] = set of strings that reach this state from init
        # via original SA edges only.
        paths: dict[int, set[str]] = {0: {""}}
        visited: set[int] = {0}
        queue: deque[int] = deque([0])

        while queue:
            sid = queue.popleft()
            nfa_st = self._states[sid]

            if nfa_st.is_trap:
                continue

            missing = self.alphabet - set(nfa_st.trans.keys())

            if sid == 0:
                for c in missing:
                    nfa_st.trans.setdefault(c, set()).add(0)
                    self._added_edges.add((0, c, 0))
            else:
                for c in missing:
                    targets: set[int] = set()
                    for w in paths.get(sid, set()):
                        stripped = w[1:] + c  # drop first character
                        reached = self._read_nfa(stripped)
                        targets |= reached
                    for tgt in targets:
                        nfa_st.trans.setdefault(c, set()).add(tgt)
                        self._added_edges.add((sid, c, tgt))

            # Propagate paths along original SA edges and enqueue children.
            sa_st = sa_states[sid]
            for ch, tgt in sa_st.trans.items():
                child_paths = {w + ch for w in paths.get(sid, set())}
                if tgt not in paths:
                    paths[tgt] = set()
                paths[tgt] |= child_paths
                if tgt not in visited:
                    visited.add(tgt)
                    queue.append(tgt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_states(self) -> int:
        return len(self._states)

    @property
    def states(self) -> list[NFAState]:
        return list(self._states)

    @property
    def added_edges(self) -> set[tuple[int, str, int]]:
        return set(self._added_edges)

    # ------------------------------------------------------------------
    # Graphviz DOT export
    # ------------------------------------------------------------------
    _ORIG_COLORS = ("black", "darkred", "darkgreen", "darkblue")
    _ADDED_COLOR = "#b05000"
    _TRAP_FILL = "#e0c4c4"

    def to_dot(self) -> str:
        lines: list[str] = [
            "digraph CompletedAutomaton {",
            '    rankdir=LR;',
            '    node [shape=circle, fontname="Courier", fontsize=10];',
            '    edge [fontname="Courier", fontsize=10];',
            "",
        ]

        for st in self._states:
            shape = "doublecircle" if st.is_accepting else "circle"
            label = str(st.id)
            if st.id == 0:
                label += "\\ninit"
            parts = [f"shape={shape}", f'label="{label}"']
            if st.is_trap:
                parts.append("style=filled")
                parts.append(f'fillcolor="{self._TRAP_FILL}"')
            lines.append(f"    {st.id} [{', '.join(parts)}];")

        # Collect all edges: (src, ch, tgt, is_added)
        edges: list[tuple[int, str, int, bool]] = []
        for st in self._states:
            for ch, targets in sorted(st.trans.items()):
                for tgt in sorted(targets):
                    is_added = (st.id, ch, tgt) in self._added_edges
                    edges.append((st.id, ch, tgt, is_added))

        for src, grp in groupby(edges, key=lambda e: e[0]):
            orig_idx = 0
            for _, ch, tgt, is_added in grp:
                esc = ch.replace("\\", "\\\\").replace('"', '\\"')
                if is_added:
                    col = self._ADDED_COLOR
                    style = "dashed"
                else:
                    col = self._ORIG_COLORS[orig_idx % len(self._ORIG_COLORS)]
                    style = "solid"
                    orig_idx += 1
                lines.append(
                    f'    {src} -> {tgt} [label="{esc}", '
                    f"color=\"{col}\", fontcolor=\"{col}\", style={style}];"
                )

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SVG rendering
    # ------------------------------------------------------------------
    def to_svg(self, path: str | Path) -> None:
        dot_src = self.to_dot()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

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
        except FileNotFoundError:
            raise RuntimeError(
                "Graphviz `dot` not found. Install it with:\n"
                "  Ubuntu/Debian : sudo apt install graphviz\n"
                "  macOS (brew)  : brew install graphviz\n"
                "  pip           : pip install graphviz  (Python wrapper only)"
            ) from None
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    sa = SuffixAutomaton()
    words = ["abcbc", "cbc"]
    for w in words:
        sa.add_word(w)

    alphabet = {"a", "b", "c"}
    ca = CompletedAutomaton(sa, alphabet)

    print(f"Words      : {words}")
    print(f"Alphabet   : {sorted(alphabet)}")
    print(f"SA states  : {sa.num_states}")
    print(f"NFA states : {ca.num_states}")
    print(f"Added edges: {len(ca.added_edges)}")
    print()

    for st in ca.states:
        if st.is_trap:
            continue
        print(f"  State {st.id}: ", end="")
        for ch in sorted(st.trans):
            targets = sorted(st.trans[ch])
            print(f"  --{ch}--> {targets}", end="")
        print()

    print()
    print("--- DOT output ---")
    print(ca.to_dot())

    ca.to_svg("completed_automaton.svg")
    print("\nSVG written to completed_automaton.svg")
