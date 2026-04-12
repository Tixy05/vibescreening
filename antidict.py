"""Build a DFA from an Extended Antidictionary (EAD).

An EAD is a triple (P, F, S) of finite string sets:
  P = forbidden prefixes
  F = forbidden factors (infixes)
  S = forbidden suffixes

The resulting DFA accepts exactly:
  L = { w in Sigma* | w has no prefix in P, no factor in F, no suffix in S }
"""

from __future__ import annotations

import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from itertools import product as iterproduct
from pathlib import Path


# ======================================================================
# Result type
# ======================================================================

@dataclass
class DFA:
    num_states: int
    alphabet: list[str]
    transitions: dict[int, dict[str, int]]
    initial: int
    accepting: set[int]

    _EDGE_COLORS = ("black", "darkred", "darkgreen", "darkblue", "darkorange")
    _DEAD_FILL = "#e0c4c4"

    def _is_dead(self, state: int) -> bool:
        """True when every transition from *state* is a self-loop (trap)."""
        row = self.transitions.get(state)
        if row is None:
            return False
        return all(t == state for t in row.values())

    def useless_states(self) -> set[int]:
        """States from which no accepting state is reachable (L(q) = {})."""
        rev: dict[int, set[int]] = {s: set() for s in range(self.num_states)}
        for s in range(self.num_states):
            for t in self.transitions.get(s, {}).values():
                rev[t].add(s)

        useful: set[int] = set(self.accepting)
        queue: deque[int] = deque(self.accepting)
        while queue:
            s = queue.popleft()
            for pred in rev.get(s, set()):
                if pred not in useful:
                    useful.add(pred)
                    queue.append(pred)
        return set(range(self.num_states)) - useful

    # ------------------------------------------------------------------
    # Hopcroft minimization
    # ------------------------------------------------------------------

    def minimize(self) -> DFA:
        """Return an equivalent minimal DFA (Hopcroft's algorithm).

        The returned DFA has contiguous state ids starting from 0, with
        the initial state always mapped to 0.
        """
        # 1. Remove unreachable states via BFS from initial.
        reachable: set[int] = set()
        bfs: deque[int] = deque([self.initial])
        reachable.add(self.initial)
        while bfs:
            s = bfs.popleft()
            for t in self.transitions.get(s, {}).values():
                if t not in reachable:
                    reachable.add(t)
                    bfs.append(t)

        # 2. Initial partition: accepting vs non-accepting (restricted to
        #    reachable states).
        acc = self.accepting & reachable
        rej = reachable - acc

        partitions: list[set[int]] = []
        if acc:
            partitions.append(acc)
        if rej:
            partitions.append(rej)

        # state -> index into `partitions`
        state_to_part: dict[int, int] = {}
        for idx, part in enumerate(partitions):
            for s in part:
                state_to_part[s] = idx

        # 3. Hopcroft refinement loop.
        worklist: deque[int] = deque(range(len(partitions)))

        while worklist:
            splitter_idx = worklist.popleft()
            splitter = partitions[splitter_idx]

            for a in self.alphabet:
                # States that transition into the splitter on symbol a
                pre_a: set[int] = set()
                for s in reachable:
                    t = self.transitions[s].get(a)
                    if t is not None and t in splitter:
                        pre_a.add(s)

                if not pre_a:
                    continue

                # Try to split each existing partition
                indices_to_check = list(set(state_to_part[s] for s in pre_a))
                for pi in indices_to_check:
                    part = partitions[pi]
                    intersection = part & pre_a
                    difference = part - pre_a

                    if not intersection or not difference:
                        continue

                    # Split: keep intersection at pi, create new for difference
                    partitions[pi] = intersection
                    new_idx = len(partitions)
                    partitions.append(difference)

                    for s in intersection:
                        state_to_part[s] = pi
                    for s in difference:
                        state_to_part[s] = new_idx

                    if pi in worklist:
                        worklist.append(new_idx)
                    else:
                        # Add the smaller half as the new splitter
                        if len(intersection) <= len(difference):
                            worklist.append(pi)
                        else:
                            worklist.append(new_idx)

        # 4. Build the minimized DFA.
        # Canonical representative for each partition: pick one (smallest id).
        part_rep: dict[int, int] = {}
        for idx, part in enumerate(partitions):
            if part:
                part_rep[idx] = min(part)

        # Assign new contiguous ids, making sure initial state gets 0.
        initial_part = state_to_part[self.initial]
        used_parts = sorted(
            {idx for idx, part in enumerate(partitions) if part},
            key=lambda i: (i != initial_part, i),
        )
        part_to_new: dict[int, int] = {}
        for new_id, old_idx in enumerate(used_parts):
            part_to_new[old_idx] = new_id

        new_transitions: dict[int, dict[str, int]] = {}
        new_accepting: set[int] = set()

        for old_idx in used_parts:
            new_id = part_to_new[old_idx]
            rep = part_rep[old_idx]
            row: dict[str, int] = {}
            for a in self.alphabet:
                t = self.transitions[rep][a]
                row[a] = part_to_new[state_to_part[t]]
            new_transitions[new_id] = row
            if rep in self.accepting:
                new_accepting.add(new_id)

        return DFA(
            num_states=len(used_parts),
            alphabet=list(self.alphabet),
            transitions=new_transitions,
            initial=0,
            accepting=new_accepting,
        )

    # ------------------------------------------------------------------
    # Graphviz DOT export
    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the DFA."""
        lines: list[str] = [
            "digraph EAD_DFA {",
            "    rankdir=LR;",
            '    node [shape=circle, fontname="Courier", fontsize=10];',
            '    edge [fontname="Courier", fontsize=10];',
            "",
            '    _start [shape=point, width=0.15];',
            f"    _start -> {self.initial};",
            "",
        ]

        dead_states = {s for s in range(self.num_states) if self._is_dead(s)}

        for s in range(self.num_states):
            shape = "doublecircle" if s in self.accepting else "circle"
            label = str(s)
            parts = [f"shape={shape}", f'label="{label}"']
            if s in dead_states:
                parts.append("style=filled")
                parts.append(f'fillcolor="{self._DEAD_FILL}"')
            lines.append(f"    {s} [{', '.join(parts)}];")

        lines.append("")

        for s in range(self.num_states):
            row = self.transitions.get(s, {})
            # Group targets -> set of symbols for compact multi-label edges
            target_syms: dict[int, list[str]] = {}
            for a in self.alphabet:
                t = row.get(a)
                if t is not None:
                    target_syms.setdefault(t, []).append(a)

            for idx, (t, syms) in enumerate(sorted(target_syms.items())):
                label = ",".join(syms)
                esc = label.replace("\\", "\\\\").replace('"', '\\"')
                col = self._EDGE_COLORS[idx % len(self._EDGE_COLORS)]
                lines.append(
                    f'    {s} -> {t} [label="{esc}", color={col}, fontcolor={col}];'
                )

        lines.append("}")
        return "\n".join(lines)

    def to_svg(self, path: str | Path) -> None:
        """Render the DFA to an SVG file using Graphviz ``dot``."""
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
# Step 1: Aho-Corasick automaton over F ∪ S
# ======================================================================

_F_TAG = "F"
_S_TAG = "S"


@dataclass
class _ACNode:
    id: int
    children: dict[str, int] = field(default_factory=dict)
    fail: int = 0
    dict_suffix: int = -1
    output: set[str] = field(default_factory=set)  # subset of {"F", "S"}


class _AhoCorasick:
    """Aho-Corasick automaton with explicit goto table."""

    def __init__(self, alphabet: list[str]) -> None:
        self.alphabet = alphabet
        self.nodes: list[_ACNode] = [_ACNode(id=0)]
        self.goto: dict[int, dict[str, int]] = {}

    # 1a: trie insertion
    def _insert(self, pattern: str, tag: str) -> None:
        cur = 0
        for ch in pattern:
            nxt = self.nodes[cur].children.get(ch)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes.append(_ACNode(id=nxt))
                self.nodes[cur].children[ch] = nxt
            cur = nxt
        self.nodes[cur].output.add(tag)

    # 1b + 1c: failure links and dictionary suffix links via BFS
    def _build_failures(self) -> None:
        root = self.nodes[0]
        root.fail = 0
        root.dict_suffix = -1

        queue: deque[int] = deque()
        for child_id in root.children.values():
            self.nodes[child_id].fail = 0
            self.nodes[child_id].dict_suffix = (
                0 if self.nodes[0].output else -1
            )
            queue.append(child_id)

        while queue:
            uid = queue.popleft()
            u = self.nodes[uid]
            for ch, vid in u.children.items():
                v = self.nodes[vid]
                f = u.fail
                while f != 0 and ch not in self.nodes[f].children:
                    f = self.nodes[f].fail
                v.fail = self.nodes[f].children.get(ch, 0)
                if v.fail == vid:
                    v.fail = 0

                # dict_suffix: nearest ancestor via fail with non-empty output
                if self.nodes[v.fail].output:
                    v.dict_suffix = v.fail
                else:
                    v.dict_suffix = self.nodes[v.fail].dict_suffix

                queue.append(vid)

    # 1d: explicit goto table for every (state, symbol) pair
    def _build_goto_table(self) -> None:
        for node in self.nodes:
            row: dict[str, int] = {}
            for a in self.alphabet:
                if a in node.children:
                    row[a] = node.children[a]
                else:
                    f = node.id
                    while f != 0 and a not in self.nodes[f].children:
                        f = self.nodes[f].fail
                    row[a] = self.nodes[f].children.get(a, 0)
            self.goto[node.id] = row

    def build(self, factors: set[str], suffixes: set[str]) -> None:
        for p in factors:
            if p:
                self._insert(p, _F_TAG)
        for p in suffixes:
            if p:
                self._insert(p, _S_TAG)
        self._build_failures()
        self._build_goto_table()


# ======================================================================
# Step 2: Mark states (has_factor / has_suffix)
# ======================================================================

def _mark_states(ac: _AhoCorasick) -> tuple[list[bool], list[bool]]:
    n = len(ac.nodes)
    has_factor = [False] * n
    has_suffix = [False] * n

    for node in ac.nodes:
        q = node.id
        # check node itself
        if _F_TAG in node.output:
            has_factor[q] = True
        if _S_TAG in node.output:
            has_suffix[q] = True
        # walk dict-suffix chain
        ds = node.dict_suffix
        while ds >= 0:
            dsn = ac.nodes[ds]
            if _F_TAG in dsn.output:
                has_factor[q] = True
            if _S_TAG in dsn.output:
                has_suffix[q] = True
            ds = dsn.dict_suffix

    return has_factor, has_suffix


# ======================================================================
# Step 3: Dead state and factor propagation
# ======================================================================

def _apply_dead_state(
    ac: _AhoCorasick,
    has_factor: list[bool],
) -> int:
    """Add a dead state to the AC goto table, redirect factor-marked targets.

    Returns the dead state id.
    """
    dead = len(ac.nodes)
    ac.goto[dead] = {a: dead for a in ac.alphabet}

    for q in list(ac.goto):
        if q == dead:
            continue
        row = ac.goto[q]
        for a in ac.alphabet:
            t = row[a]
            if t < len(has_factor) and has_factor[t]:
                row[a] = dead

    return dead


# ======================================================================
# Step 4: Prefix DFA and product construction
# ======================================================================

@dataclass
class _PrefixTrieNode:
    id: int
    children: dict[str, int] = field(default_factory=dict)
    is_end: bool = False


class _PrefixDFA:
    """Small DFA for checking forbidden prefixes.

    States: trie nodes + accept_id (absorbing) + dead_id (absorbing).
    """

    def __init__(self, prefixes: set[str], alphabet: list[str]) -> None:
        self.alphabet = alphabet
        self.nodes: list[_PrefixTrieNode] = [_PrefixTrieNode(id=0)]
        self._build_trie(prefixes)

        self.accept_id = len(self.nodes)
        self.dead_id = self.accept_id + 1

        self.goto: dict[int, dict[str, int]] = {}
        self._build_goto()

    def _build_trie(self, prefixes: set[str]) -> None:
        for p in prefixes:
            cur = 0
            for ch in p:
                nxt = self.nodes[cur].children.get(ch)
                if nxt is None:
                    nxt = len(self.nodes)
                    self.nodes.append(_PrefixTrieNode(id=nxt))
                    self.nodes[cur].children[ch] = nxt
                cur = nxt
            self.nodes[cur].is_end = True

    def _build_goto(self) -> None:
        # absorbing states
        self.goto[self.accept_id] = {a: self.accept_id for a in self.alphabet}
        self.goto[self.dead_id] = {a: self.dead_id for a in self.alphabet}

        # handle empty-string prefix: root itself is_end => initial state is dead
        # (any word starting with "" is forbidden, i.e. everything is forbidden)
        for node in self.nodes:
            if node.is_end:
                # This node completes a forbidden prefix; it should act dead.
                # But transitions *to* this node are handled by the parent.
                # A node marked is_end should never be entered — the parent
                # redirects to dead_id.  However, if the root is_end (empty
                # prefix in P), we handle it specially in the product.
                pass

            row: dict[str, int] = {}
            for a in self.alphabet:
                child = node.children.get(a)
                if child is not None:
                    if self.nodes[child].is_end:
                        row[a] = self.dead_id
                    else:
                        row[a] = child
                else:
                    row[a] = self.accept_id
            self.goto[node.id] = row


def _build_product(
    ac_goto: dict[int, dict[str, int]],
    ac_dead: int,
    has_suffix: list[bool],
    prefix_dfa: _PrefixDFA,
    alphabet: list[str],
) -> DFA:
    """BFS over the product (AC x PrefixDFA), assign integer IDs."""

    # If empty prefix "" is in P, the root of the prefix trie is_end,
    # meaning every string is forbidden => language is empty.
    root_is_dead = prefix_dfa.nodes[0].is_end

    initial_ac = 0
    initial_p = prefix_dfa.dead_id if root_is_dead else 0

    initial_pair = (initial_ac, initial_p)
    pair_to_id: dict[tuple[int, int], int] = {initial_pair: 0}
    id_to_pair: list[tuple[int, int]] = [initial_pair]

    transitions: dict[int, dict[str, int]] = {}
    queue: deque[int] = deque([0])

    while queue:
        sid = queue.popleft()
        q_ac, q_p = id_to_pair[sid]
        row: dict[str, int] = {}
        for a in alphabet:
            nxt_ac = ac_goto[q_ac][a]
            nxt_p = prefix_dfa.goto[q_p][a]
            nxt_pair = (nxt_ac, nxt_p)
            if nxt_pair not in pair_to_id:
                nid = len(id_to_pair)
                pair_to_id[nxt_pair] = nid
                id_to_pair.append(nxt_pair)
                queue.append(nid)
            row[a] = pair_to_id[nxt_pair]
        transitions[sid] = row

    # Step 5: accepting states
    accepting: set[int] = set()
    for sid, (q_ac, q_p) in enumerate(id_to_pair):
        if q_ac == ac_dead:
            continue
        if q_ac < len(has_suffix) and has_suffix[q_ac]:
            continue
        if q_p == prefix_dfa.dead_id:
            continue
        accepting.add(sid)

    return DFA(
        num_states=len(id_to_pair),
        alphabet=alphabet,
        transitions=transitions,
        initial=0,
        accepting=accepting,
    )


# ======================================================================
# Public API
# ======================================================================

def build_ead_dfa(
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
    alphabet: set[str],
) -> DFA:
    """Build a complete DFA accepting the language defined by the EAD.

    Returns a DFA A such that L(A) = { w in Sigma* | w has no prefix in P,
    no factor in F, no suffix in S }.
    """
    alpha_list = sorted(alphabet)

    # Step 1: Aho-Corasick over F ∪ S
    ac = _AhoCorasick(alpha_list)
    ac.build(factors, suffixes)

    # Step 2: mark states
    has_factor, has_suffix = _mark_states(ac)

    # Step 3: dead state + factor propagation
    ac_dead = _apply_dead_state(ac, has_factor)

    # Step 4: prefix DFA
    pdfa = _PrefixDFA(prefixes, alpha_list)

    # Steps 4c + 5 + 6: product construction with accepting states
    return _build_product(ac.goto, ac_dead, has_suffix, pdfa, alpha_list)


# ======================================================================
# Shortest characteristic factors (Algorithm 1, Janousek & Plachy 2025)
# ======================================================================

TaggedSubset = frozenset[tuple[int, bool]]


@dataclass
class CharacteristicFactors:
    """All six types of shortest characteristic factors of a DFA."""
    SFF: set[str]
    SFP: set[str]
    SFS: set[str]
    SFW: set[str]
    SAS: set[str]
    SAW: set[str]
    template_num_states: int = 0


def build_template_automaton(
    dfa: DFA,
    initial_states: set[TaggedSubset],
) -> tuple[
    set[TaggedSubset],
    dict[TaggedSubset, dict[str, TaggedSubset]],
    set[TaggedSubset],
    set[TaggedSubset],
]:
    """Tagged-subset BFS (Algorithm 1).

    Returns (states, adj, full_subset_states, has_initial_flag_states).
    ``adj`` maps each source state to {symbol: destination}.
    The language of each (initial, final-set) combination is the *reverse* of
    the corresponding shortest characteristic factor set.
    """
    n = dfa.num_states
    all_states_set = set(range(n))
    useless = dfa.useless_states()
    q0 = dfa.initial

    # Precompute reverse-transition: for each (target, symbol) -> list of sources
    rev: dict[tuple[int, str], list[int]] = {}
    for q in range(n):
        for a in dfa.alphabet:
            t = dfa.transitions[q][a]
            rev.setdefault((t, a), []).append(q)

    states: set[TaggedSubset] = set()
    adj: dict[TaggedSubset, dict[str, TaggedSubset]] = {}
    full_subset: set[TaggedSubset] = set()   # Q_hat_Q: dom = Q, >= 1 True
    has_initial: set[TaggedSubset] = set()   # Q_hat_0: dom != Q, (q0,True) in S

    queue: deque[TaggedSubset] = deque()
    for s0 in initial_states:
        states.add(s0)
        if any(flag for _, flag in s0):
            queue.append(s0)

    while queue:
        q_hat = queue.popleft()
        q_hat_dict = dict(q_hat)
        q_hat_keys = set(q_hat_dict)

        # Full-subset check (Q_hat_Q): dom(S) = Q
        if q_hat_keys == all_states_set:
            full_subset.add(q_hat)
            # Full-subset states have no outgoing transitions (shortest cutoff)
            continue

        # Non-full check (Q_hat_0): dom(S) != Q and (q0, True) in S
        if q_hat_dict.get(q0, False):
            has_initial.add(q_hat)

        edges: dict[str, TaggedSubset] = {}
        for a in dfa.alphabet:
            items: list[tuple[int, bool]] = []
            for target_q, flag in q_hat_dict.items():
                for src in rev.get((target_q, a), []):
                    new_flag = flag and (src not in useless)
                    items.append((src, new_flag))

            # Deduplicate: if a source appears multiple times (via different
            # targets), the flag is True if ANY path gives True.
            merged: dict[int, bool] = {}
            for src, flag in items:
                merged[src] = merged.get(src, False) or flag

            if not any(merged.values()):
                continue

            q_hat_prime = frozenset(merged.items())
            edges[a] = q_hat_prime

            if q_hat_prime not in states:
                states.add(q_hat_prime)
                queue.append(q_hat_prime)

        if edges:
            adj[q_hat] = edges

    return states, adj, full_subset, has_initial


def make_initial_states(
    dfa: DFA,
) -> tuple[TaggedSubset, TaggedSubset, TaggedSubset]:
    """Compute the three initial tagged subsets.

    Returns (q_hat_U, q_hat_nf, q_hat_f).
    """
    useless = dfa.useless_states()
    finals = dfa.accepting
    useful_non_final = set(range(dfa.num_states)) - finals - useless

    q_hat_u = frozenset((q, True) for q in useless)
    q_hat_nf = frozenset(
        [(q, True) for q in useful_non_final]
        + [(q, False) for q in useless]
    )
    q_hat_f = frozenset(
        [(q, True) for q in finals]
        + [(q, False) for q in useless]
    )
    return q_hat_u, q_hat_nf, q_hat_f


def _extract_words(
    initial: TaggedSubset,
    final_states: set[TaggedSubset],
    adj: dict[TaggedSubset, dict[str, TaggedSubset]],
    max_depth: int | None = None,
) -> set[str]:
    """DFS enumeration of all accepted strings, then reverse each one.

    The template automaton's language is the *reverse* of the characteristic
    factors, so we reverse each path label to get the actual factor.
    """
    results: set[str] = set()

    def dfs(state: TaggedSubset, path: list[str], depth: int) -> None:
        if state in final_states:
            results.add("".join(reversed(path)))
            return
        if max_depth is not None and depth >= max_depth:
            return
        for sym, dst in adj.get(state, {}).items():
            path.append(sym)
            dfs(dst, path, depth + 1)
            path.pop()

    dfs(initial, [], 0)
    return results


def _format_tagged_subset(ts: TaggedSubset) -> str:
    """Human-readable label for a tagged subset: {q⁺, q⁻, ...}."""
    parts = []
    for q, flag in sorted(ts):
        parts.append(f"{q}{'⁺' if flag else '⁻'}")
    return "{" + ",".join(parts) + "}"


_INIT_STYLES: dict[str, tuple[str, str]] = {
    "q̂_U":  ("#fff3cd", "gold3"),
    "q̂_nf": ("#f8d7da", "firebrick"),
    "q̂_f":  ("#e2d9f3", "purple"),
}


def template_automaton_to_dot(
    states: set[TaggedSubset],
    adj: dict[TaggedSubset, dict[str, TaggedSubset]],
    named_initials: dict[str, TaggedSubset],
    full_subset: set[TaggedSubset],
    has_initial: set[TaggedSubset],
    alphabet: list[str],
) -> str:
    """Return a Graphviz DOT representation of the template automaton.

    *named_initials* maps a human-readable name to each initial tagged
    subset, e.g. ``{"q̂_U": q_hat_u, "q̂_nf": q_hat_nf}``.

    Node shapes / colors:
      - q̂_U  initial: yellow fill, gold border   (SFF, SFP)
      - q̂_nf initial: pink fill, red border      (SFS, SFW)
      - q̂_f  initial: lavender fill, purple border
      - full_subset (Q̂_Q) terminal: double octagon, light blue
      - has_initial (Q̂_0) terminal: double circle, light green
      - ordinary states: plain circle
    Superscript ⁺/⁻ on each DFA-state id indicates the boolean tag.
    """
    state_list = sorted(states, key=lambda s: sorted(s))
    ts_to_id: dict[TaggedSubset, int] = {ts: i for i, ts in enumerate(state_list)}

    initial_set = set(named_initials.values())
    ts_to_init_name: dict[TaggedSubset, str] = {
        ts: name for name, ts in named_initials.items()
    }

    edge_colors = ("black", "darkred", "darkgreen", "darkblue", "darkorange")

    lines: list[str] = [
        "digraph TemplateAutomaton {",
        "    rankdir=LR;",
        '    node [fontname="Courier", fontsize=9];',
        '    edge [fontname="Courier", fontsize=10];',
        "",
    ]

    for name, ts in named_initials.items():
        if ts in ts_to_id:
            sid = ts_to_id[ts]
            _, border_col = _INIT_STYLES.get(name, ("#eeeeee", "black"))
            lines.append(
                f'    _start_{sid} [shape=point, width=0.15, '
                f'color={border_col}];'
            )
            lines.append(
                f'    _start_{sid} -> {sid} [color={border_col}];'
            )
    lines.append("")

    for ts, sid in ts_to_id.items():
        label = _format_tagged_subset(ts)
        esc_label = label.replace("\\", "\\\\").replace('"', '\\"')

        if ts in full_subset:
            attrs = (
                f'shape=doubleoctagon, label="{esc_label}", '
                f'style=filled, fillcolor="#cce5ff"'
            )
        elif ts in has_initial:
            attrs = (
                f'shape=doublecircle, label="{esc_label}", '
                f'style=filled, fillcolor="#d4edda"'
            )
        elif ts in initial_set:
            name = ts_to_init_name[ts]
            fill, border = _INIT_STYLES.get(name, ("#eeeeee", "black"))
            attrs = (
                f'shape=circle, label="{name}\\n{esc_label}", '
                f'style="filled,bold", fillcolor="{fill}", '
                f'color="{border}", penwidth=2'
            )
        else:
            attrs = f'shape=circle, label="{esc_label}"'
        lines.append(f"    {sid} [{attrs}];")

    lines.append("")

    for ts, sid in ts_to_id.items():
        edges = adj.get(ts, {})
        target_syms: dict[int, list[str]] = {}
        for a in alphabet:
            if a in edges:
                tid = ts_to_id[edges[a]]
                target_syms.setdefault(tid, []).append(a)
        for idx, (tid, syms) in enumerate(sorted(target_syms.items())):
            lbl = ",".join(syms)
            esc = lbl.replace("\\", "\\\\").replace('"', '\\"')
            col = edge_colors[idx % len(edge_colors)]
            lines.append(
                f'    {sid} -> {tid} [label="{esc}", color={col}, fontcolor={col}];'
            )

    lines.append("}")
    return "\n".join(lines)


def template_automaton_to_svg(
    states: set[TaggedSubset],
    adj: dict[TaggedSubset, dict[str, TaggedSubset]],
    named_initials: dict[str, TaggedSubset],
    full_subset: set[TaggedSubset],
    has_initial: set[TaggedSubset],
    alphabet: list[str],
    path: str | Path,
) -> None:
    """Render the template automaton to SVG via Graphviz ``dot``."""
    dot_src = template_automaton_to_dot(
        states, adj, named_initials, full_subset, has_initial, alphabet,
    )
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
            check=True, capture_output=True, text=True,
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


def template_automaton_to_dfa(
    states: set[TaggedSubset],
    adj: dict[TaggedSubset, dict[str, TaggedSubset]],
    initial_states: set[TaggedSubset],
    full_subset: set[TaggedSubset],
    has_initial: set[TaggedSubset],
    alphabet: list[str],
) -> DFA:
    """Convert the template automaton (tagged-subset BFS) into a plain DFA.

    Uses a virtual initial state with epsilon-transitions to all actual
    initial states (unioned into a single start via BFS renumbering).
    Accepting = full_subset | has_initial (both terminal families).
    """
    state_list = sorted(states, key=lambda s: sorted(s))
    ts_to_id: dict[TaggedSubset, int] = {}
    for i, ts in enumerate(state_list):
        ts_to_id[ts] = i

    n = len(state_list)
    sink = n
    virtual_init = n + 1
    total = n + 2

    transitions: dict[int, dict[str, int]] = {}
    accepting: set[int] = set()

    for ts, sid in ts_to_id.items():
        row: dict[str, int] = {}
        edges = adj.get(ts, {})
        for a in alphabet:
            row[a] = ts_to_id[edges[a]] if a in edges else sink
        transitions[sid] = row
        if ts in full_subset or ts in has_initial:
            accepting.add(sid)

    transitions[sink] = {a: sink for a in alphabet}

    init_targets = [ts_to_id[s] for s in initial_states if s in ts_to_id]
    if len(init_targets) == 1:
        actual_initial = init_targets[0]
        total -= 1
    else:
        actual_initial = virtual_init
        row = {}
        for a in alphabet:
            reachable: set[int] = set()
            for t in init_targets:
                reachable.add(transitions[t].get(a, sink))
            row[a] = next(iter(reachable)) if len(reachable) == 1 else sink
        transitions[virtual_init] = row
        if any(t in accepting for t in init_targets):
            accepting.add(virtual_init)

    return DFA(
        num_states=total,
        alphabet=alphabet,
        transitions=transitions,
        initial=actual_initial,
        accepting=accepting,
    )


def extract_characteristic_factors(
    dfa: DFA,
    max_depth: int | None = None,
) -> CharacteristicFactors:
    """Extract all six shortest characteristic factor sets from a DFA.

    The DFA should be complete (total transition function). For local DFAs
    all sets are finite and extraction always terminates. For non-local DFAs,
    pass *max_depth* to bound the search.

    The returned ``CharacteristicFactors.template_num_states`` records the
    number of states in the intermediate template automaton (tagged-subset
    construction) — the potential source of exponential blowup.
    """
    q_hat_u, q_hat_nf, q_hat_f = make_initial_states(dfa)

    inits: set[TaggedSubset] = set()
    if q_hat_u:
        inits.add(q_hat_u)
    if q_hat_nf:
        inits.add(q_hat_nf)
    if q_hat_f:
        inits.add(q_hat_f)

    ta_states, adj, full_subset, has_initial = build_template_automaton(
        dfa, inits,
    )

    config: dict[str, tuple[TaggedSubset, set[TaggedSubset]]] = {
        "SFF": (q_hat_u, full_subset),
        "SFP": (q_hat_u, has_initial),
        "SFS": (q_hat_nf, full_subset),
        "SFW": (q_hat_nf, has_initial),
        "SAS": (q_hat_f, full_subset),
        "SAW": (q_hat_f, has_initial),
    }

    results: dict[str, set[str]] = {}
    for name, (init, finals) in config.items():
        if not init:
            results[name] = set()
        else:
            results[name] = _extract_words(init, finals, adj, max_depth)

    return CharacteristicFactors(**results, template_num_states=len(ta_states))


# ======================================================================
# Brute-force oracle for testing
# ======================================================================

def _brute_accepts(
    w: str,
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
) -> bool:
    """Set-theoretic definition: True iff w has no forbidden prefix/factor/suffix."""
    for p in prefixes:
        if w[: len(p)] == p:
            return False
    for f in factors:
        if f and f in w:
            return False
        if not f:
            return False
    for s in suffixes:
        if w[len(w) - len(s):] == s:
            return False
    return True


def _dfa_accepts(dfa: DFA, w: str) -> bool:
    state = dfa.initial
    for ch in w:
        state = dfa.transitions[state][ch]
    return state in dfa.accepting


def _enumerate_strings(alphabet: list[str], max_len: int):
    yield ""
    for length in range(1, max_len + 1):
        for combo in iterproduct(alphabet, repeat=length):
            yield "".join(combo)


def verify(
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
    alphabet: set[str],
    max_len: int = 8,
) -> bool:
    dfa = build_ead_dfa(prefixes, factors, suffixes, alphabet)
    mini = dfa.minimize()
    alpha_list = sorted(alphabet)
    ok = True
    for w in _enumerate_strings(alpha_list, max_len):
        expected = _brute_accepts(w, prefixes, factors, suffixes)
        got = _dfa_accepts(dfa, w)
        got_min = _dfa_accepts(mini, w)
        if expected != got:
            print(f"  MISMATCH (raw)  w={w!r}  expected={expected}  got={got}")
            ok = False
        if expected != got_min:
            print(f"  MISMATCH (min)  w={w!r}  expected={expected}  got={got_min}")
            ok = False
    return ok


# ======================================================================
# Brute-force verification of characteristic factors
# ======================================================================

def _is_forbidden_factor(w: str, dfa: DFA) -> bool:
    """True iff for every state q, reading w from q ends in a useless state."""
    useless = dfa.useless_states()
    for q in range(dfa.num_states):
        s = q
        for ch in w:
            s = dfa.transitions[s][ch]
        if s not in useless:
            return False
    return True


def _is_forbidden_suffix(w: str, dfa: DFA) -> bool:
    """True iff for every state q, reading w from q does not reach an accepting state."""
    for q in range(dfa.num_states):
        s = q
        for ch in w:
            s = dfa.transitions[s][ch]
        if s in dfa.accepting:
            return False
    return True


def _is_forbidden_prefix(w: str, dfa: DFA) -> bool:
    """True iff reading w from q0 ends in a useless state."""
    useless = dfa.useless_states()
    s = dfa.initial
    for ch in w:
        s = dfa.transitions[s][ch]
    return s in useless


def _is_allowed_suffix(w: str, dfa: DFA) -> bool:
    """True iff for every state q, reading w from q either ends in a final
    state or a useless state, and at least one q ends in a final state."""
    useless = dfa.useless_states()
    has_final = False
    for q in range(dfa.num_states):
        s = q
        for ch in w:
            s = dfa.transitions[s][ch]
        if s in dfa.accepting:
            has_final = True
        elif s not in useless:
            return False
    return has_final


def _all_factors(w: str) -> set[str]:
    return {w[i:j] for i in range(len(w)) for j in range(i, len(w) + 1)}


def _all_proper_factors(w: str) -> set[str]:
    return _all_factors(w) - {w}


def _all_proper_suffixes(w: str) -> set[str]:
    return {w[i:] for i in range(1, len(w) + 1)}


def _all_proper_prefixes(w: str) -> set[str]:
    return {w[:i] for i in range(len(w))}


def verify_characteristic_factors(
    dfa: DFA,
    cf: CharacteristicFactors,
    max_len: int = 7,
) -> bool:
    """Brute-force check that extracted factors satisfy their definitions."""
    alpha_list = dfa.alphabet
    ok = True

    for w in cf.SFF:
        if not _is_forbidden_factor(w, dfa):
            print(f"  SFF fail: {w!r} is not a forbidden factor")
            ok = False
        for u in _all_proper_factors(w):
            if u and _is_forbidden_factor(u, dfa):
                print(f"  SFF fail: {w!r} has proper factor {u!r} that is also forbidden")
                ok = False

    for w in cf.SFS:
        if not _is_forbidden_suffix(w, dfa):
            print(f"  SFS fail: {w!r} is not a forbidden suffix")
            ok = False
        if _is_forbidden_factor(w, dfa):
            print(f"  SFS fail: {w!r} is also a forbidden factor (should be in SFF instead)")
            ok = False
        for u in _all_proper_suffixes(w):
            if u and _is_forbidden_suffix(u, dfa) and not _is_forbidden_factor(u, dfa):
                print(f"  SFS fail: {w!r} has proper suffix {u!r} that is also a forbidden suffix")
                ok = False

    for w in cf.SFP:
        if not _is_forbidden_prefix(w, dfa):
            print(f"  SFP fail: {w!r} is not a forbidden prefix")
            ok = False
        if _is_forbidden_factor(w, dfa):
            print(f"  SFP fail: {w!r} is also a forbidden factor (should be in SFF instead)")
            ok = False
        for u in _all_proper_prefixes(w):
            if u and _is_forbidden_prefix(u, dfa) and not _is_forbidden_factor(u, dfa):
                print(f"  SFP fail: {w!r} has proper prefix {u!r} that is also a forbidden prefix")
                ok = False

    for w in cf.SAS:
        if not _is_allowed_suffix(w, dfa):
            print(f"  SAS fail: {w!r} is not an allowed suffix")
            ok = False
        for u in _all_proper_suffixes(w):
            if u and _is_allowed_suffix(u, dfa):
                print(f"  SAS fail: {w!r} has proper suffix {u!r} that is also an allowed suffix")
                ok = False

    # Completeness via Proposition 1: the six sets must fully characterize L(A).
    for w in _enumerate_strings(alpha_list, max_len):
        dfa_result = _dfa_accepts(dfa, w)

        has_ff = any(f in w for f in cf.SFF if f)
        has_fp = any(w[:len(p)] == p for p in cf.SFP)
        has_fs = any(w[len(w) - len(s):] == s for s in cf.SFS if s)
        is_sfw = w in cf.SFW

        prop1 = not is_sfw and not has_fp and not has_ff and not has_fs

        if dfa_result != prop1:
            print(
                f"  Prop1 fail: w={w!r} dfa={dfa_result} prop1={prop1} "
                f"sfw={is_sfw} fp={has_fp} ff={has_ff} fs={has_fs}"
            )
            ok = False

    return ok


def _dfa_isomorphic(a: DFA, b: DFA) -> bool:
    """Check if two minimal DFAs are isomorphic (same language)."""
    if a.num_states != b.num_states:
        return False
    if sorted(a.alphabet) != sorted(b.alphabet):
        return False

    mapping: dict[int, int] = {a.initial: b.initial}
    queue: deque[int] = deque([a.initial])

    while queue:
        sa = queue.popleft()
        sb = mapping[sa]
        if (sa in a.accepting) != (sb in b.accepting):
            return False
        for sym in a.alphabet:
            ta = a.transitions[sa][sym]
            tb = b.transitions[sb][sym]
            if ta in mapping:
                if mapping[ta] != tb:
                    return False
            else:
                mapping[ta] = tb
                queue.append(ta)
    return True


# ======================================================================
# Demo / self-test
# ======================================================================

if __name__ == "__main__":
    cases: list[tuple[str, set[str], set[str], set[str], set[str]]] = [
        ("empty EAD", set(), set(), set(), {"a", "b"}),
        ("factors only", set(), {"ab", "ba"}, set(), {"a", "b"}),
        ("prefixes only", {"ab", "ba"}, set(), set(), {"a", "b"}),
        ("suffixes only", set(), set(), {"ab", "ba"}, {"a", "b"}),
        (
            "mixed P+F+S",
            {"aa"},
            {"bb"},
            {"ab"},
            {"a", "b"},
        ),
        (
            "ternary mixed",
            {"ab"},
            {"cc"},
            {"ba"},
            {"a", "b", "c"},
        ),
        ("single-char factor", set(), {"a"}, set(), {"a", "b"}),
        ("single-char prefix", {"a"}, set(), set(), {"a", "b"}),
        ("single-char suffix", set(), set(), {"a"}, {"a", "b"}),
        ("pattern in both F and S", set(), {"ab"}, {"ab"}, {"a", "b"}),
        ("prefix is also factor", {"ab"}, {"ab"}, set(), {"a", "b"}),
        (
            "overlapping patterns",
            {"ab"},
            {"ba"},
            {"aa"},
            {"a", "b"},
        ),
        (
            "long patterns",
            {"aab"},
            {"bab"},
            {"abb"},
            {"a", "b"},
        ),
        ("all singletons forbidden", {"a", "b"}, set(), set(), {"a", "b"}),
        (
            "nested prefixes",
            {"a", "ab", "abc"},
            set(),
            set(),
            {"a", "b", "c"},
        ),
    ]

    # --- EAD DFA tests ---
    print("=" * 60)
    print("EAD DFA construction tests")
    print("=" * 60)
    all_ok = True
    for name, tp, tf, ts, ta in cases:
        print(f"Test: {name}  P={tp} F={tf} S={ts} A={sorted(ta)}")
        ok = verify(tp, tf, ts, ta, max_len=7)
        if ok:
            print("  OK")
        else:
            all_ok = False
        print()

    if all_ok:
        print("All EAD DFA tests passed.\n")
    else:
        print("SOME EAD DFA TESTS FAILED.\n")

    # --- Characteristic factors tests ---
    print("=" * 60)
    print("Characteristic factors extraction tests")
    print("=" * 60)
    cf_ok = True
    for name, tp, tf, ts, ta in cases:
        dfa = build_ead_dfa(tp, tf, ts, ta).minimize()
        cf = extract_characteristic_factors(dfa)
        print(f"Test: {name}  ({dfa.num_states} states)")
        print(f"  SFF={sorted(cf.SFF)}  SFP={sorted(cf.SFP)}  SFS={sorted(cf.SFS)}")
        print(f"  SFW={sorted(cf.SFW)}  SAS={sorted(cf.SAS)}  SAW={sorted(cf.SAW)}")
        ok = verify_characteristic_factors(dfa, cf, max_len=7)
        if ok:
            print("  OK")
        else:
            cf_ok = False
        print()

    if cf_ok:
        print("All characteristic factors tests passed.\n")
    else:
        print("SOME CHARACTERISTIC FACTORS TESTS FAILED.\n")

    # --- Round-trip test: EAD -> DFA -> minimize -> extract -> rebuild -> compare ---
    print("=" * 60)
    print("Round-trip tests (EAD -> extract -> rebuild)")
    print("=" * 60)
    rt_ok = True
    for name, tp, tf, ts, ta in cases:
        original = build_ead_dfa(tp, tf, ts, ta).minimize()
        cf = extract_characteristic_factors(original)
        rebuilt = build_ead_dfa(cf.SFP, cf.SFF, cf.SFS, ta).minimize()

        iso = _dfa_isomorphic(original, rebuilt)
        status = "OK" if iso else "FAIL"
        print(f"  {status}  {name}  ({original.num_states} -> {rebuilt.num_states} states)")
        if not iso:
            rt_ok = False
    print()

    if rt_ok:
        print("All round-trip tests passed.\n")
    else:
        print("SOME ROUND-TRIP TESTS FAILED.\n")

    # --- Visual demo ---
    demo_p, demo_f, demo_s, demo_a = {"aa"}, {"bb"}, {"ab"}, {"a", "b"}
    demo = build_ead_dfa(demo_p, demo_f, demo_s, demo_a)
    mini = demo.minimize()
    cf = extract_characteristic_factors(mini)
    print(f"--- Visual demo  P={demo_p} F={demo_f} S={demo_s} ---")
    print(f"  DFA: {demo.num_states} states -> minimized: {mini.num_states} states")
    print(f"  SFF={sorted(cf.SFF)}  SFP={sorted(cf.SFP)}  SFS={sorted(cf.SFS)}")
    print(f"  SFW={sorted(cf.SFW)}  SAS={sorted(cf.SAS)}  SAW={sorted(cf.SAW)}")
    print()
    print("--- DOT (minimized) ---")
    print(mini.to_dot())

    try:
        mini.to_svg("ead_dfa_min.svg")
        print("\nSVG written to ead_dfa_min.svg")
    except RuntimeError as e:
        print(f"\nSkipping SVG: {e}")

    # Final summary
    if all_ok and cf_ok and rt_ok:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")
