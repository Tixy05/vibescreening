"""Fuzz-test DFA construction for Extended Antidictionaries.

Generates random EADs (P, F, S) over a given alphabet, builds the DFA
(both raw and minimized), then exhaustively checks all strings up to a
certain length against the brute-force set-theoretic definition.
"""

from __future__ import annotations

import random
import sys
from itertools import product as iterproduct

from antidict import build_ead_dfa, DFA


def brute_accepts(
    w: str,
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
) -> bool:
    for p in prefixes:
        if w[:len(p)] == p:
            return False
    for f in factors:
        if not f:
            return False
        if f in w:
            return False
    for s in suffixes:
        if w[len(w) - len(s):] == s:
            return False
    return True


def dfa_accepts(dfa: DFA, w: str) -> bool:
    state = dfa.initial
    for ch in w:
        state = dfa.transitions[state][ch]
    return state in dfa.accepting


def enumerate_strings(alphabet: list[str], max_len: int):
    yield ""
    for length in range(1, max_len + 1):
        for combo in iterproduct(alphabet, repeat=length):
            yield "".join(combo)


def random_word(alphabet: list[str], max_len: int) -> str:
    k = random.randint(1, max_len)
    return "".join(random.choice(alphabet) for _ in range(k))


def random_word_set(alphabet: list[str], max_count: int, max_word_len: int) -> set[str]:
    n = random.randint(0, max_count)
    return {random_word(alphabet, max_word_len) for _ in range(n)}


def fuzz_one(
    alphabet: list[str],
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
    check_len: int,
) -> bool:
    dfa = build_ead_dfa(prefixes, factors, suffixes, set(alphabet))
    mini = dfa.minimize()

    ok = True
    for w in enumerate_strings(alphabet, check_len):
        expected = brute_accepts(w, prefixes, factors, suffixes)
        got_raw = dfa_accepts(dfa, w)
        got_min = dfa_accepts(mini, w)

        if expected != got_raw:
            print(
                f"  MISMATCH (raw)  w={w!r:12s}  expected={expected}  got={got_raw}"
                f"  |  P={prefixes}  F={factors}  S={suffixes}"
            )
            ok = False
        if expected != got_min:
            print(
                f"  MISMATCH (min)  w={w!r:12s}  expected={expected}  got={got_min}"
                f"  |  P={prefixes}  F={factors}  S={suffixes}"
            )
            ok = False
    return ok


def main() -> None:
    seed = random.randrange(2**32)
    random.seed(seed)

    num_rounds = 5000
    alpha_options = [
        ["a", "b"],
        ["a", "b", "c"],
        ["0", "1"],
        ["x", "y", "z"],
    ]
    max_set_size = 4
    max_word_len = 4
    check_len = 7
    d_len = 10

    print(f"EAD DFA fuzzer  seed={seed}  rounds={num_rounds}  check_len={random.randint(check_len, check_len + d_len)}")
    print()

    failures = 0
    for i in range(1, num_rounds + 1):
        alphabet = random.choice(alpha_options)
        P = random_word_set(alphabet, max_set_size, max_word_len)
        F = random_word_set(alphabet, max_set_size, max_word_len)
        S = random_word_set(alphabet, max_set_size, max_word_len)

        # empty string in F means "everything is forbidden" — keep it interesting
        # but don't filter it out; the DFA should handle it correctly
        if not fuzz_one(alphabet, P, F, S, check_len):
            failures += 1
            print(f"  ^ round {i}  alphabet={alphabet}")
            print()

        if i % 500 == 0:
            print(f"  ... {i}/{num_rounds} rounds done, {failures} failure(s) so far")

    print()
    if failures:
        print(f"DONE — {failures} FAILING round(s) out of {num_rounds}.")
        sys.exit(1)
    else:
        print(f"DONE — all {num_rounds} rounds passed.")


if __name__ == "__main__":
    main()
