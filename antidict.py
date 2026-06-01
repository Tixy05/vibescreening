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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extad import ExtAD, SLTL


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

    def is_dead(self, state: int) -> bool:
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

        dead_states = {s for s in range(self.num_states) if self.is_dead(s)}

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
    ead: "ExtAD",
) -> DFA:
    """Build a complete DFA accepting the language defined by the EAD.

    Returns a DFA A such that L(A) = { w in Sigma* | w has no prefix in P,
    no factor in F, no suffix in S }.
    """
    alpha_list = sorted(ead.alphabet)

    # Step 1: Aho-Corasick over F ∪ S
    ac = _AhoCorasick(alpha_list)
    ac.build(ead.factors, ead.suffixes)

    # Step 2: mark states
    has_factor, has_suffix = _mark_states(ac)

    # Step 3: dead state + factor propagation
    ac_dead = _apply_dead_state(ac, has_factor)

    # Step 4: prefix DFA
    pdfa = _PrefixDFA(ead.prefixes, alpha_list)

    # Steps 4c + 5 + 6: product construction with accepting states
    return _build_product(ac.goto, ac_dead, has_suffix, pdfa, alpha_list)


def _build_finite_word_dfa(words: set[str], alphabet: list[str]) -> DFA:
    """Build a DFA accepting exactly the finite set ``words``."""
    trie: list[dict[str, int]] = [{}]
    accepting: set[int] = set()

    for word in words:
        state = 0
        for ch in word:
            nxt = trie[state].get(ch)
            if nxt is None:
                nxt = len(trie)
                trie[state][ch] = nxt
                trie.append({})
            state = nxt
        accepting.add(state)

    dead_state = len(trie)
    transitions: dict[int, dict[str, int]] = {}
    for state, row_map in enumerate(trie):
        row: dict[str, int] = {}
        for ch in alphabet:
            row[ch] = row_map.get(ch, dead_state)
        transitions[state] = row
    transitions[dead_state] = {ch: dead_state for ch in alphabet}

    return DFA(
        num_states=len(trie) + 1,
        alphabet=list(alphabet),
        transitions=transitions,
        initial=0,
        accepting=accepting,
    )


def _subtract_finite_language(base: DFA, forbidden: DFA) -> DFA:
    """Return ``L(base) \\ L(forbidden)`` for complete DFAs on the same alphabet."""
    if base.alphabet != forbidden.alphabet:
        raise ValueError("Both DFAs must use the same alphabet.")

    start = (base.initial, forbidden.initial)
    pair_to_id: dict[tuple[int, int], int] = {start: 0}
    id_to_pair: list[tuple[int, int]] = [start]
    queue: deque[int] = deque([0])
    transitions: dict[int, dict[str, int]] = {}
    accepting: set[int] = set()

    while queue:
        sid = queue.popleft()
        left, right = id_to_pair[sid]

        if left in base.accepting and right not in forbidden.accepting:
            accepting.add(sid)

        row: dict[str, int] = {}
        for ch in base.alphabet:
            nxt = (base.transitions[left][ch], forbidden.transitions[right][ch])
            nid = pair_to_id.get(nxt)
            if nid is None:
                nid = len(id_to_pair)
                pair_to_id[nxt] = nid
                id_to_pair.append(nxt)
                queue.append(nid)
            row[ch] = nid
        transitions[sid] = row

    return DFA(
        num_states=len(id_to_pair),
        alphabet=list(base.alphabet),
        transitions=transitions,
        initial=0,
        accepting=accepting,
    )


def build_sltl_dfa(
    sltl: "SLTL",
) -> DFA:
    """Build a DFA from the full forbidden-side SLTL characteristic factors."""
    from extad import ExtAD

    ead = ExtAD(
        sltl.prefixes,
        sltl.factors,
        sltl.suffixes,
        dense_alphabet=sltl.dense_alphabet,
        alphabet=sltl.alphabet,
    )
    dfa = build_ead_dfa(ead)
    if not sltl.sfw:
        return dfa

    forbidden_words = _build_finite_word_dfa(set(sltl.sfw), sorted(sltl.alphabet))
    return _subtract_finite_language(dfa, forbidden_words)


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


@dataclass
class TemplateAutomaton:
    """Tagged-subset automaton from Algorithm 1 for a minimized DFA."""

    dfa: DFA
    states: set[TaggedSubset] = field(init=False)
    adj: dict[TaggedSubset, dict[str, TaggedSubset]] = field(init=False)
    full_subset: set[TaggedSubset] = field(init=False)
    has_initial: set[TaggedSubset] = field(init=False)
    q_hat_u: TaggedSubset = field(init=False)
    q_hat_nf: TaggedSubset = field(init=False)
    q_hat_f: TaggedSubset = field(init=False)

    def __post_init__(self) -> None:
        self.q_hat_u, self.q_hat_nf, self.q_hat_f = self._make_initial_states()
        self.states, self.adj, self.full_subset, self.has_initial = self._build()

    @classmethod
    def from_dfa(cls, dfa: DFA) -> TemplateAutomaton:
        """Construct the template automaton directly from a DFA."""
        return cls(dfa)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def named_initials(self) -> dict[str, TaggedSubset]:
        named: dict[str, TaggedSubset] = {}
        if self.q_hat_u:
            named["q_hat_U"] = self.q_hat_u
        if self.q_hat_nf:
            named["q_hat_nf"] = self.q_hat_nf
        if self.q_hat_f:
            named["q_hat_f"] = self.q_hat_f
        return named

    def _make_initial_states(self) -> tuple[TaggedSubset, TaggedSubset, TaggedSubset]:
        """Compute the three initial tagged subsets from the source DFA."""
        useless = self.dfa.useless_states()
        finals = self.dfa.accepting
        useful_non_final = set(range(self.dfa.num_states)) - finals - useless

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

    def _initial_states(self) -> set[TaggedSubset]:
        return set(self.named_initials.values())

    def _build(
        self,
    ) -> tuple[
        set[TaggedSubset],
        dict[TaggedSubset, dict[str, TaggedSubset]],
        set[TaggedSubset],
        set[TaggedSubset],
    ]:
        """Run the tagged-subset BFS from Algorithm 1."""
        n = self.dfa.num_states
        all_states_set = set(range(n))
        useless = self.dfa.useless_states()
        q0 = self.dfa.initial

        rev: dict[tuple[int, str], list[int]] = {}
        for q in range(n):
            for a in self.dfa.alphabet:
                t = self.dfa.transitions[q][a]
                rev.setdefault((t, a), []).append(q)

        states: set[TaggedSubset] = set()
        adj: dict[TaggedSubset, dict[str, TaggedSubset]] = {}
        full_subset: set[TaggedSubset] = set()
        has_initial: set[TaggedSubset] = set()

        queue: deque[TaggedSubset] = deque()
        for s0 in self._initial_states():
            states.add(s0)
            if any(flag for _, flag in s0):
                queue.append(s0)

        while queue:
            q_hat = queue.popleft()
            q_hat_dict = dict(q_hat)
            q_hat_keys = set(q_hat_dict)

            if q_hat_keys == all_states_set:
                full_subset.add(q_hat)
                continue

            if q_hat_dict.get(q0, False):
                has_initial.add(q_hat)

            edges: dict[str, TaggedSubset] = {}
            for a in self.dfa.alphabet:
                items: list[tuple[int, bool]] = []
                for target_q, flag in q_hat_dict.items():
                    for src in rev.get((target_q, a), []):
                        new_flag = flag and (src not in useless)
                        items.append((src, new_flag))

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

    def _extract_words(
        self,
        initial: TaggedSubset,
        final_states: set[TaggedSubset],
        max_depth: int | None = None,
    ) -> set[str]:
        """Enumerate accepted words and reverse them back to factors."""
        if not initial:
            return set()

        results: set[str] = set()

        def dfs(state: TaggedSubset, path: list[str], depth: int) -> None:
            if state in final_states:
                results.add("".join(reversed(path)))
                return
            if max_depth is not None and depth >= max_depth:
                return
            for sym, dst in self.adj.get(state, {}).items():
                path.append(sym)
                dfs(dst, path, depth + 1)
                path.pop()

        dfs(initial, [], 0)
        return results

    def extract_characteristic_factors(
        self,
        max_depth: int | None = None,
    ) -> CharacteristicFactors:
        """Extract all six shortest characteristic factor sets."""
        config: dict[str, tuple[TaggedSubset, set[TaggedSubset]]] = {
            "SFF": (self.q_hat_u, self.full_subset),
            "SFP": (self.q_hat_u, self.has_initial),
            "SFS": (self.q_hat_nf, self.full_subset),
            "SFW": (self.q_hat_nf, self.has_initial),
            "SAS": (self.q_hat_f, self.full_subset),
            "SAW": (self.q_hat_f, self.has_initial),
        }

        results: dict[str, set[str]] = {}
        for name, (initial, finals) in config.items():
            results[name] = self._extract_words(initial, finals, max_depth)

        return CharacteristicFactors(
            **results,
            template_num_states=self.num_states,
        )

    def _empty_language_dfa(self) -> DFA:
        return DFA(
            num_states=1,
            alphabet=list(self.dfa.alphabet),
            transitions={0: {a: 0 for a in self.dfa.alphabet}},
            initial=0,
            accepting=set(),
        )

    def to_dfa(
        self,
        initial: TaggedSubset,
        terminal_states: set[TaggedSubset],
    ) -> DFA:
        """Convert one shortest-language projection of the template automaton to a DFA."""
        if not initial or initial not in self.states:
            return self._empty_language_dfa()

        state_list = sorted(self.states, key=lambda s: sorted(s))
        ts_to_id = {ts: i for i, ts in enumerate(state_list)}
        sink = len(state_list)

        transitions: dict[int, dict[str, int]] = {}
        accepting: set[int] = set()

        for ts, sid in ts_to_id.items():
            if ts in terminal_states:
                accepting.add(sid)
                transitions[sid] = {a: sink for a in self.dfa.alphabet}
                continue

            edges = self.adj.get(ts, {})
            transitions[sid] = {
                a: ts_to_id[edges[a]] if a in edges else sink
                for a in self.dfa.alphabet
            }

        transitions[sink] = {a: sink for a in self.dfa.alphabet}

        return DFA(
            num_states=len(state_list) + 1,
            alphabet=list(self.dfa.alphabet),
            transitions=transitions,
            initial=ts_to_id[initial],
            accepting=accepting,
        )

    def to_sff_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_u, self.full_subset)

    def to_sfp_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_u, self.has_initial)

    def to_sfs_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_nf, self.full_subset)

    def to_sfw_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_nf, self.has_initial)

    def to_sas_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_f, self.full_subset)

    def to_saw_dfa(self) -> DFA:
        return self.to_dfa(self.q_hat_f, self.has_initial)

    def to_dot(self) -> str:
        """Return a paper-style Graphviz DOT representation of the template automaton."""
        state_list = sorted(self.states, key=_template_state_sort_key)
        ts_to_id: dict[TaggedSubset, int] = {ts: i for i, ts in enumerate(state_list)}
        by_rank: dict[int, list[TaggedSubset]] = {}
        for ts in state_list:
            by_rank.setdefault(_positive_count(ts), []).append(ts)

        lines: list[str] = [
            "digraph TemplateAutomaton {",
            "    rankdir=LR;",
            "    splines=true;",
            "    outputorder=edgesfirst;",
            "    nodesep=0.55;",
            "    ranksep=0.75;",
            '    graph [fontname="Times-Italic"];',
            '    node [shape=circle, fontname="Times-Roman", fontsize=18, width=0.62, height=0.62, fixedsize=false];',
            '    edge [fontname="Times-Italic", fontsize=16, arrowsize=0.8];',
            "",
        ]

        rank_keys = sorted(by_rank)
        for idx, rank in enumerate(rank_keys):
            anchor = f"_rank_{rank}"
            lines.append(f'    {anchor} [shape=point, width=0, label="", style=invis];')
            if idx:
                prev_anchor = f"_rank_{rank_keys[idx - 1]}"
                lines.append(f"    {prev_anchor} -> {anchor} [style=invis, weight=100];")
        lines.append("")

        for name, ts in self.named_initials.items():
            if ts in ts_to_id:
                sid = ts_to_id[ts]
                family_a, family_b = _INIT_FAMILY_LABELS[name]
                label_name = f"_label_{sid}"
                label_html = (
                    '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="0">'
                    '<TR><TD>'
                    f'<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="1" CELLPADDING="4"><TR><TD>{family_a}</TD></TR></TABLE>'
                    '</TD></TR>'
                    '<TR><TD>'
                    f'<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="1" CELLPADDING="4" STYLE="dashed"><TR><TD>{family_b}</TD></TR></TABLE>'
                    '</TD></TR>'
                    "</TABLE>>"
                )
                lines.append(
                    f"    {label_name} [shape=plain, margin=0, label={label_html}];"
                )
                lines.append(
                    f'    {label_name} -> {sid} [color=black, fontcolor=black, penwidth=1.1, minlen=1];'
                )
        lines.append("")

        for rank, states_in_rank in sorted(by_rank.items()):
            members = "; ".join([f"_rank_{rank}"] + [str(ts_to_id[ts]) for ts in states_in_rank])
            lines.append(f"    {{ rank=same; {members}; }}")
        lines.append("")

        for ts, sid in ts_to_id.items():
            label = _dot_label_for_tagged_subset(ts)
            if ts in self.full_subset:
                attrs = f"label={label}, peripheries=2, penwidth=1.2"
            elif ts in self.has_initial:
                attrs = f'label={label}, peripheries=2, style=dashed, penwidth=1.2'
            else:
                attrs = f"label={label}"
            lines.append(f"    {sid} [{attrs}];")

        lines.append("")

        for ts, sid in ts_to_id.items():
            edges = self.adj.get(ts, {})
            target_syms: dict[int, list[str]] = {}
            for a in self.dfa.alphabet:
                if a in edges:
                    tid = ts_to_id[edges[a]]
                    target_syms.setdefault(tid, []).append(a)
            for idx, (tid, syms) in enumerate(sorted(target_syms.items())):
                lbl = ",".join(syms)
                esc = lbl.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(
                    f'    {sid} -> {tid} [label="{esc}", color="black", fontcolor="black", penwidth=1.0];'
                )

        lines.append("}")
        return "\n".join(lines)

    def to_svg(self, path: str | Path) -> None:
        """Render the template automaton to an SVG file using Graphviz ``dot``."""
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


def _positive_count(ts: TaggedSubset) -> int:
    return sum(1 for _, flag in ts if flag)


def _template_state_sort_key(ts: TaggedSubset) -> tuple[int, int, list[tuple[int, bool]]]:
    return (_positive_count(ts), len(ts), sorted(ts))


def _dot_label_for_tagged_subset(ts: TaggedSubset) -> str:
    positive = [str(q) for q, flag in sorted(ts) if flag]
    negative = [str(q) for q, flag in sorted(ts) if not flag]
    main = "u" if not positive else f"u[{','.join(positive)}]"
    if not negative:
        return f'"{main}"'
    negative_line = ",".join(f"{q}-" for q in negative)
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD>{main}</TD></TR>'
        f'<TR><TD><FONT COLOR="gray55" POINT-SIZE="11">{negative_line}</FONT></TD></TR>'
        "</TABLE>>"
    )


_INIT_FAMILY_LABELS: dict[str, tuple[str, str]] = {
    "q_hat_U": ("SFF", "SFP"),
    "q_hat_nf": ("SFS", "SFW"),
    "q_hat_f": ("SAS", "SAW"),
}

_SLTL_TEMPLATE_FAMILIES: dict[str, tuple[str, str]] = {
    "SFP": ("q_hat_u", "has_initial"),
    "SFF": ("q_hat_u", "full_subset"),
    "SFS": ("q_hat_nf", "full_subset"),
    "SFW": ("q_hat_nf", "has_initial"),
}


def template_accepts_word(
    template: TemplateAutomaton,
    word: str,
    family: str,
) -> bool:
    """True iff *word* is accepted by the template projection for *family*.

    Paths follow the same orientation as :meth:`TemplateAutomaton._extract_words`
    (symbols of *word* are read right-to-left on template edges).
    """
    if family not in _SLTL_TEMPLATE_FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_SLTL_TEMPLATE_FAMILIES)}")

    initial_attr, finals_attr = _SLTL_TEMPLATE_FAMILIES[family]
    initial: TaggedSubset = getattr(template, initial_attr)
    finals: set[TaggedSubset] = getattr(template, finals_attr)

    if not initial or initial not in template.states:
        return False

    state = initial
    for ch in reversed(word):
        edges = template.adj.get(state)
        if edges is None or ch not in edges:
            return False
        state = edges[ch]
    return state in finals


def intersect_sltl(left: "SLTL", right: "SLTL") -> "SLTL":
    """Return a normalized SLTL for ``L(left) ∩ L(right)`` (union of forbidden sets)."""
    from extad import SLTL

    if left.dense_alphabet != right.dense_alphabet:
        raise ValueError("both SLTLs must use the same dense_alphabet setting")

    alphabet = left.alphabet | right.alphabet
    merged = SLTL(
        left.prefixes | right.prefixes,
        left.factors | right.factors,
        left.suffixes | right.suffixes,
        left.sfw | right.sfw,
        dense_alphabet=left.dense_alphabet,
        alphabet=alphabet,
    )
    return merged.normalize()


def sff_subset(sff_sub: set[str], sff_sup: set[str]) -> bool:
    """SFF clause for ``L(sub) ⊆ L(sup)`` (section 7).

    Each forbidden factor ``f`` of the including language (``sff_sup``) must have
    some ``f'`` in ``sff_sub`` that occurs as a substring of ``f``.
    """
    return all(
        any(f_prime in f for f_prime in sff_sub)
        for f in sff_sup
    )


def _pattern_contains_factor(pattern: str, factors: set[str]) -> bool:
    return any(u and u in pattern for u in factors)


def _prefix_covered_by_sub(p: str, sub: "SLTL") -> bool:
    """``p`` from ``SFP(sup)`` is a forbidden prefix in ``L(sub)``."""
    return any(p.startswith(p_sub) for p_sub in sub.prefixes) or _pattern_contains_factor(
        p, sub.factors
    )


def _suffix_covered_by_sub(s: str, sub: "SLTL") -> bool:
    """``s`` from ``SFS(sup)`` is a forbidden suffix in ``L(sub)``."""
    return any(s.endswith(s_sub) for s_sub in sub.suffixes) or _pattern_contains_factor(
        s, sub.factors
    )


def _word_forbidden_in_sltl(word: str, sltl: "SLTL") -> bool:
    return not _dfa_accepts(sltl.build_dfa(), word)


def sltl_is_subset(sub: "SLTL", sup: "SLTL") -> bool:
    """True iff ``L(sub) ⊆ L(sup)`` for SLTLs in normal form (section 7).

    Equivalently ``L(sup) ⊇ L(sub)``: every shortest forbidden pattern of *sup*
    is already enforced by *sub*'s prefixes, factors, suffixes, or whole words.
    """
    if not sup.alphabet <= sub.alphabet:
        return False

    for p in sup.prefixes:
        if not _prefix_covered_by_sub(p, sub):
            return False

    if not sff_subset(sub.factors, sup.factors):
        return False

    for s in sup.suffixes:
        if not _suffix_covered_by_sub(s, sub):
            return False

    for w in sup.sfw:
        if not _word_forbidden_in_sltl(w, sub):
            return False

    return True


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
    template = TemplateAutomaton.from_dfa(dfa)
    return template.extract_characteristic_factors(max_depth=max_depth)


# ======================================================================
# Brute-force oracle for testing
# ======================================================================

def _brute_sltl_accepts(w: str, sltl: "SLTL") -> bool:
    """Set-theoretic SLTL acceptance (Proposition 1 style)."""
    if w in sltl.sfw:
        return False
    for p in sltl.prefixes:
        if w[: len(p)] == p:
            return False
    for f in sltl.factors:
        if f and f in w:
            return False
        if not f:
            return False
    for s in sltl.suffixes:
        if w[len(w) - len(s) :] == s:
            return False
    return True


def _brute_accepts(
    w: str,
    ead: "ExtAD",
) -> bool:
    """Set-theoretic definition: True iff w has no forbidden prefix/factor/suffix."""
    for p in ead.prefixes:
        if w[: len(p)] == p:
            return False
    for f in ead.factors:
        if f and f in w:
            return False
        if not f:
            return False
    for s in ead.suffixes:
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
    ead: "ExtAD",
    max_len: int = 8,
) -> bool:
    dfa = build_ead_dfa(ead)
    mini = dfa.minimize()
    alpha_list = sorted(ead.alphabet)
    ok = True
    for w in _enumerate_strings(alpha_list, max_len):
        expected = _brute_accepts(w, ead)
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
    from extad import ExtAD

    cases: list[tuple[str, ExtAD]] = [
        ("empty EAD", ExtAD(set(), set(), set(), alphabet={"a", "b"})),
        ("factors only", ExtAD(set(), {"ab", "ba"}, set(), alphabet={"a", "b"})),
        ("prefixes only", ExtAD({"ab", "ba"}, set(), set(), alphabet={"a", "b"})),
        ("suffixes only", ExtAD(set(), set(), {"ab", "ba"}, alphabet={"a", "b"})),
        (
            "mixed P+F+S",
            ExtAD({"aa"}, {"bb"}, {"ab"}, alphabet={"a", "b"}),
        ),
        (
            "ternary mixed",
            ExtAD({"ab"}, {"cc"}, {"ba"}, alphabet={"a", "b", "c"}),
        ),
        ("single-char factor", ExtAD(set(), {"a"}, set(), alphabet={"a", "b"})),
        ("single-char prefix", ExtAD({"a"}, set(), set(), alphabet={"a", "b"})),
        ("single-char suffix", ExtAD(set(), set(), {"a"}, alphabet={"a", "b"})),
        ("pattern in both F and S", ExtAD(set(), {"ab"}, {"ab"}, alphabet={"a", "b"})),
        ("prefix is also factor", ExtAD({"ab"}, {"ab"}, set(), alphabet={"a", "b"})),
        (
            "overlapping patterns",
            ExtAD({"ab"}, {"ba"}, {"aa"}, alphabet={"a", "b"}),
        ),
        (
            "long patterns",
            ExtAD({"aab"}, {"bab"}, {"abb"}, alphabet={"a", "b"}),
        ),
        ("all singletons forbidden", ExtAD({"a", "b"}, set(), set(), alphabet={"a", "b"})),
        (
            "nested prefixes",
            ExtAD({"a", "ab", "abc"}, set(), set(), alphabet={"a", "b", "c"}),
        ),
    ]

    # --- EAD DFA tests ---
    print("=" * 60)
    print("EAD DFA construction tests")
    print("=" * 60)
    all_ok = True
    for name, ead in cases:
        print(
            f"Test: {name}  P={ead.prefixes} F={ead.factors} "
            f"S={ead.suffixes} A={sorted(ead.alphabet)}"
        )
        ok = verify(ead, max_len=7)
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
    for name, ead in cases:
        dfa = build_ead_dfa(ead).minimize()
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
    for name, ead in cases:
        original = build_ead_dfa(ead).minimize()
        cf = extract_characteristic_factors(original)
        rebuilt_ead = ExtAD(cf.SFP, cf.SFF, cf.SFS, alphabet=ead.alphabet)
        rebuilt = build_ead_dfa(rebuilt_ead).minimize()

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
    demo_ead = ExtAD({"aa"}, {"bb"}, {"ab"}, alphabet={"a", "b"})
    demo = build_ead_dfa(demo_ead)
    mini = demo.minimize()
    cf = extract_characteristic_factors(mini)
    print(
        f"--- Visual demo  P={demo_ead.prefixes} "
        f"F={demo_ead.factors} S={demo_ead.suffixes} ---"
    )
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
