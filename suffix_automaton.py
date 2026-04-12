import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class State:
    """A single state (node) in the suffix automaton."""

    id: int
    len: int = 0                          # length of the longest string in this equivalence class
    link: int = -1                        # suffix link  (-1 = none)
    trans: dict[str, int] = field(default_factory=dict)   # char → state id
    is_clone: bool = False                # True if this state was created by cloning
    words: set[int] = field(default_factory=set)          # which words end here (accepting info)


class SuffixAutomaton:
    """Generalized Suffix Automaton for a set of words.

    Reverse the transition graph via :meth:`predecessors` / :meth:`predecessor_edges`
    to walk from a state back toward the root (backtracking).
    """

    def __init__(self) -> None:
        # state 0 is the initial (root) state
        self._states: list[State] = [State(id=0, len=0, link=-1)]
        self._last: int = 0       # id of the state corresponding to the entire current prefix
        self._word_count: int = 0 # how many words have been added

    # ------------------------------------------------------------------
    # Core: extend the automaton by one character
    # ------------------------------------------------------------------
    def _sa_extend(self, c: str, word_id: int) -> None:
        """Append character *c* to the current word being built."""

        # If `last` already has a transition on c we can follow it
        # (happens when the current suffix is already present — common
        # in the generalised variant).
        if c in self._states[self._last].trans:
            q = self._states[self._last].trans[c]
            if self._states[q].len == self._states[self._last].len + 1:
                # q is already the correct state — just move there
                self._last = q
                return
            # Otherwise we need to clone q (same logic as the clone
            # branch below but triggered from a different entry point).
            clone = State(
                id=len(self._states),
                len=self._states[self._last].len + 1,
                link=self._states[q].link,
                trans=dict(self._states[q].trans),
                is_clone=True,
                words=set(self._states[q].words),
            )
            self._states.append(clone)
            # re-point q's suffix link
            self._states[q].link = clone.id
            # walk up suffix links and re-route transitions from q to clone
            p = self._last
            while p != -1 and self._states[p].trans.get(c) == q:
                self._states[p].trans[c] = clone.id
                p = self._states[p].link
            self._last = clone.id
            return

        # Standard case: create a brand-new state for the new character
        cur = State(id=len(self._states), len=self._states[self._last].len + 1)
        self._states.append(cur)

        p = self._last
        # Walk up suffix links, adding transitions on c → cur
        while p != -1 and c not in self._states[p].trans:
            self._states[p].trans[c] = cur.id
            p = self._states[p].link

        if p == -1:
            # Reached the root without finding an existing transition on c
            cur.link = 0
        else:
            q = self._states[p].trans[c]
            if self._states[q].len == self._states[p].len + 1:
                # q is the correct parent — just set the suffix link
                cur.link = q
            else:
                # Clone q to split its equivalence class
                clone = State(
                    id=len(self._states),
                    len=self._states[p].len + 1,
                    link=self._states[q].link,
                    trans=dict(self._states[q].trans),
                    is_clone=True,
                    words=set(self._states[q].words),
                )
                self._states.append(clone)
                # Re-route transitions: everything that pointed to q from
                # ancestors at the right depth now points to clone
                while p != -1 and self._states[p].trans.get(c) == q:
                    self._states[p].trans[c] = clone.id
                    p = self._states[p].link
                self._states[q].link = clone.id
                cur.link = clone.id

        self._last = cur.id

    # ------------------------------------------------------------------
    # Public: add a word
    # ------------------------------------------------------------------
    def add_word(self, w: str) -> None:
        """Insert all suffixes of *w* into the automaton."""
        word_id = self._word_count
        self._word_count += 1
        self._last = 0  # reset to root (re-rooting for generalised variant)
        for ch in w:
            self._sa_extend(ch, word_id)
        # Mark the terminal state AND all its suffix-link ancestors as accepting.
        # Walking suffix links from `last` reaches every state that corresponds
        # to a suffix of the word just inserted.
        s = self._last
        while s > 0:
            if word_id in self._states[s].words:
                break  # already propagated (from a previous shared suffix)
            self._states[s].words.add(word_id)
            s = self._states[s].link

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_states(self) -> int:
        return len(self._states)

    @property
    def states(self) -> list[State]:
        return list(self._states)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def accepts(self, s: str) -> bool:
        """Return True if *s* is a suffix of any added word."""
        cur = 0
        for ch in s:
            if ch not in self._states[cur].trans:
                return False
            cur = self._states[cur].trans[ch]
        # Every state reachable from root in a suffix automaton corresponds
        # to a substring — but only *suffixes* end in states that are on
        # the suffix-link chain from a terminal state.  We mark terminal
        # states in add_word, so walk suffix links.
        return self._is_accepting(cur)

    def _is_accepting(self, state_id: int) -> bool:
        """Check if a state is accepting (corresponds to a complete suffix)."""
        return bool(self._states[state_id].words)

    def _check_state_id(self, state_id: int) -> None:
        if state_id < 0 or state_id >= len(self._states):
            raise IndexError(
                f"state_id {state_id} is out of range [0, {len(self._states)})"
            )

    def predecessor_edges(self, state_id: int) -> list[tuple[int, str]]:
        """All forward transitions that end in ``state_id``: ``(source_state, label)``.

        Sorted by ``(source_state, label)`` for deterministic iteration. Use this to
        backtrack: from ``state_id``, each pair is one step backward along the trie
        part of the automaton.
        """
        self._check_state_id(state_id)
        out: list[tuple[int, str]] = []
        for st in self._states:
            for ch, tgt in st.trans.items():
                if tgt == state_id:
                    out.append((st.id, ch))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def predecessors(self, state_id: int) -> set[int]:
        """Set of state ids that have at least one transition into ``state_id``."""
        self._check_state_id(state_id)
        return {
            st.id
            for st in self._states
            for t in st.trans.values()
            if t == state_id
        }

    # ------------------------------------------------------------------
    # Graphviz DOT export
    # ------------------------------------------------------------------
    _EDGE_COLORS = ("black", "darkred", "darkgreen", "darkblue")
    _TRAP_FILL = "#e0c4c4"  # muted pastel rose for dead-end (no outgoing) states

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the automaton."""
        lines: list[str] = [
            "digraph SuffixAutomaton {",
            "    rankdir=LR;",
            "    node [shape=circle, fontname=\"Courier\", fontsize=10];",
            "    edge [fontname=\"Courier\", fontsize=10];",
            "",
        ]

        # Accepting = any state with .words (already propagated in add_word)
        accepting: set[int] = {st.id for st in self._states if st.words}

        for st in self._states:
            shape = "doublecircle" if st.id in accepting else "circle"
            label = str(st.id)
            if st.id == 0:
                label += "\\ninit"
            parts = [f"shape={shape}", f"label=\"{label}\""]
            if not st.trans:
                parts.append("style=filled")
                parts.append(f'fillcolor="{self._TRAP_FILL}"')
            lines.append(f"    {st.id} [{', '.join(parts)}];")

        # Transitions: per source state, cycle colors so dense graphs stay readable
        for st in self._states:
            for i, (ch, tgt) in enumerate(sorted(st.trans.items())):
                col = self._EDGE_COLORS[i % len(self._EDGE_COLORS)]
                esc = ch.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(
                    f"    {st.id} -> {tgt} [label=\"{esc}\", color={col}, fontcolor={col}];"
                )

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SVG rendering (requires `dot` from Graphviz)
    # ------------------------------------------------------------------
    def to_svg(self, path: str | Path) -> None:
        """Render the automaton to an SVG file using Graphviz `dot`."""
        dot_src = self.to_dot()
        path = Path(path)

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

    print(f"Words      : {words}")
    print(f"States     : {sa.num_states}")
    print()

    # Test acceptance of various strings
    tests = ["abcbc", "bcbc", "cbc", "bc", "c", "abc", "cb", "b", "abcb", "xyz", ""]
    for t in tests:
        print(f"  accepts({t!r:10s}) = {sa.accepts(t)}")

    print()
    print("--- DOT output ---")
    print(sa.to_dot())

    sa.to_svg("suffix_automaton.svg")
    print("\nSVG written to suffix_automaton.svg")