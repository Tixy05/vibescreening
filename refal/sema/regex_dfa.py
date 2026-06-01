"""Build complete DFAs from SimpleRegex and annotated unions."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# antidict lives at repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from antidict import DFA

from .abstractizer import (
    AbstractEVar,
    AbstractSvar,
    AbstractTvar,
    AbstractedRegexedElement,
)
from .common import Alternation, Alphabet, AnyStarred, Letter, SimpleRegex
from .k_subwords import KSubword


@dataclass(frozen=True)
class BiClass:
    """Quotient class: annotation subset and terminal label."""

    label: str
    indices: frozenset[int]  # 1-based subword indices; empty for b0


def terminal_alphabet_for_step(step_index: int, bi_labels: list[str]) -> list[str]:
    """Γ_k: first step {b,s}, later {s} ∪ {b_i}."""
    if step_index == 0:
        return ["b", "s"]
    return ["s", *bi_labels]


def simple_regex_from_subword(
    subword: KSubword, terminal_letters: list[str],
) -> SimpleRegex:
    """Map k-subword to regex over *terminal_letters* (§4 morphism)."""
    t_alt = Alternation(tuple(Letter(x) for x in terminal_letters))
    parts: list[Alternation | AnyStarred] = []
    for elem in subword.elements:
        if isinstance(elem, AbstractEVar):
            parts.append(AnyStarred())
        elif isinstance(elem, AbstractTvar):
            parts.append(t_alt)
        elif isinstance(elem, AbstractSvar):
            if "s" not in terminal_letters:
                raise ValueError("s not in terminal alphabet")
            parts.append(Alternation((Letter("s"),)))
        elif isinstance(elem, AbstractedRegexedElement):
            parts.append(elem.alternation)
        else:
            raise TypeError(f"Unexpected element in k-subword: {elem!r}")
    alpha = Alphabet({Letter(x) for x in terminal_letters})
    return SimpleRegex(tuple(parts), alpha)


def _letters_from_regex(regex: SimpleRegex) -> list[str]:
    if isinstance(regex.alphabet, Alphabet):
        names = {L.name for L in regex.alphabet.letters}
    else:
        names = set(regex.alphabet)
    for piece in regex.concats:
        if isinstance(piece, Alternation):
            for L in piece.alternatives:
                names.add(L.name)
    return sorted(names)


def _fresh(
    nfa: dict[int, dict[str, set[int]]],
    eps: dict[int, set[int]],
) -> int:
    s = len(nfa)
    nfa[s] = {}
    eps[s] = set()
    return s


def _nfa_from_simple_regex(
    regex: SimpleRegex,
) -> tuple[dict[int, dict[str, set[int]]], dict[int, set[int]], set[int], list[str]]:
    """Epsilon NFA: (nfa_delta, epsilon, accepting, alphabet)."""
    alphabet = _letters_from_regex(regex)
    nfa: dict[int, dict[str, set[int]]] = {}
    eps: dict[int, set[int]] = {}
    cur = _fresh(nfa, eps)

    for piece in regex.concats:
        if isinstance(piece, AnyStarred):
            exit_s = _fresh(nfa, eps)
            for a in alphabet:
                nfa[cur].setdefault(a, set()).add(cur)
                nfa[cur].setdefault(a, set()).add(exit_s)
            eps[cur].add(exit_s)
            cur = exit_s
        elif isinstance(piece, Alternation):
            exit_s = _fresh(nfa, eps)
            for letter in piece.alternatives:
                nfa[cur].setdefault(letter.name, set()).add(exit_s)
            cur = exit_s
        else:
            raise TypeError(piece)

    accepting = {cur}
    return nfa, eps, accepting, alphabet


def _epsilon_closure(states: frozenset[int], eps: dict[int, set[int]]) -> frozenset[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for t in eps.get(s, ()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


def _determinize_and_complete(
    nfa: dict[int, dict[str, set[int]]],
    eps: dict[int, set[int]],
    nfa_accepting: set[int],
    alphabet: list[str],
) -> DFA:
    """Subset construction + complete trap state."""
    start = _epsilon_closure(frozenset({0}), eps)
    state_map: dict[frozenset[int], int] = {start: 0}
    id_to_set: list[frozenset[int]] = [start]
    trans: dict[int, dict[str, int]] = {}
    accepting: set[int] = set()
    queue: deque[int] = deque([0])

    while queue:
        sid = queue.popleft()
        nfa_set = id_to_set[sid]
        if nfa_set & nfa_accepting:
            accepting.add(sid)
        row: dict[str, int] = {}
        for a in alphabet:
            target: set[int] = set()
            for q in nfa_set:
                for t in nfa.get(q, {}).get(a, ()):
                    target.add(t)
            if not target:
                row[a] = -1
            else:
                key = _epsilon_closure(frozenset(target), eps)
                if key not in state_map:
                    state_map[key] = len(id_to_set)
                    id_to_set.append(key)
                    queue.append(state_map[key])
                row[a] = state_map[key]
        trans[sid] = row

    trap = len(id_to_set)
    trap_row = {a: trap for a in alphabet}
    trans[trap] = trap_row
    for sid in range(trap):
        for a in alphabet:
            if trans[sid][a] == -1:
                trans[sid][a] = trap

    return DFA(
        num_states=trap + 1,
        alphabet=list(alphabet),
        transitions=trans,
        initial=0,
        accepting=accepting,
    )


def simple_regex_to_dfa(regex: SimpleRegex) -> DFA:
    nfa, eps, acc, alphabet = _nfa_from_simple_regex(regex)
    if not alphabet:
        alphabet = ["b", "s"]
    return _determinize_and_complete(nfa, eps, acc, alphabet)


def annotated_union(dfas: list[tuple[DFA, int]]) -> tuple[DFA, dict[int, frozenset[int]]]:
    """Union of DFAs; state annotations are 1-based indices of accepted subwords."""
    if not dfas:
        raise ValueError("empty union")
    alphabet = list(dfas[0][0].alphabet)
    for dfa, _ in dfas[1:]:
        if dfa.alphabet != alphabet:
            raise ValueError("alphabet mismatch in union")

    start = tuple(dfa.initial for dfa, _ in dfas)
    state_map: dict[tuple[int, ...], int] = {start: 0}
    id_to_tuple: list[tuple[int, ...]] = [start]
    trans: dict[int, dict[str, int]] = {}
    annotations: dict[int, frozenset[int]] = {}
    queue: deque[int] = deque([0])

    while queue:
        sid = queue.popleft()
        tup = id_to_tuple[sid]
        ann: set[int] = set()
        for (dfa, idx), q in zip(dfas, tup):
            if q in dfa.accepting:
                ann.add(idx)
        annotations[sid] = frozenset(ann)

        row: dict[str, int] = {}
        for a in alphabet:
            nxt = tuple(dfa.transitions[q][a] for (dfa, _), q in zip(dfas, tup))
            if nxt not in state_map:
                state_map[nxt] = len(id_to_tuple)
                id_to_tuple.append(nxt)
                queue.append(state_map[nxt])
            row[a] = state_map[nxt]
        trans[sid] = row

    return (
        DFA(
            num_states=len(id_to_tuple),
            alphabet=alphabet,
            transitions=trans,
            initial=0,
            accepting={s for s, a in annotations.items() if a},
        ),
        annotations,
    )


def quotient_to_bi_classes(
    annotations: dict[int, frozenset[int]],
) -> list[BiClass]:
    """Map distinct annotations to quotient labels.

    Empty annotation ∅ maps to ``b0`` (reject / trap bucket). Non-empty
    annotations always use ``b1``, ``b2``, … — never ``b0``.
    """
    classes: list[BiClass] = []
    if any(not ann for ann in annotations.values()):
        classes.append(BiClass(label="b0", indices=frozenset()))
    for i, ann in enumerate(
        sorted(
            {a for a in annotations.values() if a},
            key=lambda x: (len(x), tuple(sorted(x))),
        ),
        start=1,
    ):
        classes.append(BiClass(label=f"b{i}", indices=ann))
    return classes


def union_accepts_all(dfa: DFA, max_len: int = 6) -> bool:
    """True if *dfa* accepts every word over its alphabet up to *max_len*."""
    from itertools import product as iterproduct

    alpha = dfa.alphabet
    for length in range(max_len + 1):
        for combo in iterproduct(alpha, repeat=length):
            w = "".join(combo)
            state = dfa.initial
            for ch in w:
                state = dfa.transitions[state][ch]
            if state not in dfa.accepting:
                return False
    return True


def bi_alternation_for_index(
    index: int, classes: list[BiClass],
) -> list[str]:
    """All b_j such that index ∈ annotation of class j."""
    return [c.label for c in classes if index in c.indices]


def build_step_dfa(
    subwords: list[KSubword],
    terminal_letters: list[str],
) -> tuple[DFA, dict[int, frozenset[int]], list[BiClass]]:
    """Build union DFA for all subwords and quotient annotations."""
    dfas: list[tuple[DFA, int]] = []
    for i, sw in enumerate(subwords, start=1):
        regex = simple_regex_from_subword(sw, terminal_letters)
        dfas.append((simple_regex_to_dfa(regex), i))
    union, annotations = annotated_union(dfas)
    classes = quotient_to_bi_classes(annotations)
    return union, annotations, classes
