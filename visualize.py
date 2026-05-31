from typing import Iterable

import pandas as pd

from antidict import *
from extad import ExtAD

def visualize(
    prefixes: Iterable[str], factors: Iterable[str], suffixes: Iterable[str],
    path: str,
    dense_alphabet: bool = True,
    log: bool = True,
) -> None:
    ead = ExtAD.from_strings(
        prefixes,
        factors,
        suffixes,
        dense_alphabet=dense_alphabet,
    )

    dfa = build_ead_dfa(ead).minimize()
    dfa.to_svg(path + "_ead_dfa.svg")
    ta = TemplateAutomaton.from_dfa(dfa)
    ta.to_svg(path + "_template_automaton.svg")

    if log:
        print("--------------------------------")
        print(ead)
        print(ead.normalize())
        print("--------------------------------")


def visualize_dfa(dfa: DFA, path: str, *, minimize: bool = True) -> None:
    """Render a (hard-coded) complete DFA and its template automaton to SVG."""
    if minimize:
        dfa = dfa.minimize()
    dfa.to_svg(path + "_dfa.svg")
    TemplateAutomaton.from_dfa(dfa).to_svg(path + "_template_automaton.svg")


def build_powerset_membership_dfa(n: int) -> DFA:
    """Length-2 membership DFA: (i, m) accepts iff element i is in subset coded by m."""
    initial, trap = 0, n + 2
    sink = n + 1
    alphabet = [str(i) for i in range(1, 2**n + 1)]

    def mask_positions(m: int) -> set[int]:
        x = m - 1
        return {i for i in range(n) if (x >> i) & 1}

    transitions: dict[int, dict[str, int]] = {}
    trap_row = {a: trap for a in alphabet}
    transitions[trap] = trap_row
    transitions[sink] = {a: trap for a in alphabet}

    init_row = dict(trap_row)
    for i in range(1, n + 1):
        init_row[str(i)] = i
    transitions[initial] = init_row

    for i in range(1, n + 1):
        row = dict(trap_row)
        for m in range(1, 2**n + 1):
            if (i - 1) in mask_positions(m):
                row[str(m)] = sink
        transitions[i] = row

    return DFA(
        num_states=n + 3,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial,
        accepting={sink},
    )


powerset_rows = []
for n in range(1, 13):
    d = build_powerset_membership_dfa(n)
    t = TemplateAutomaton.from_dfa(d)
    powerset_rows.append({
        "n": n,
        "dfa_states": d.num_states,
        "template_states": t.num_states,
        "template_minus_2n": t.num_states - 2**n,
    })
powerset_df = pd.DataFrame(powerset_rows)
print(powerset_df.to_string(index=False))
visualize(
    ["ab", "bb"], ["ab"], ["aa", "ab"],
    "visulas/aa_ab_bb",
)

infixes = ["ac", "cb", "abc"]
suffixes = ["a", "bc", "b"]
visualize(
    [], infixes  + ["bcc"], suffixes,
    "visulas/tiny_diff/with_bcc",
)
visualize(
    [], infixes + ["bccc"], suffixes,
    "visulas/tiny_diff/with_bccc",
)


# infixes = ["abcc", "abcb", "bcaa", "bcac", "ba", "bb"]
# suffixes = ["b", "bca", "bc"]

infixes = ["abc", "bb", "ac"]
suffixes = ["ab", "aba", "aa"]
visualize(
    [], infixes, suffixes,
    "visulas/ab_ac",
)

# ead = ExtAD.from_strings(
#     [], ["aaab"], ["a"],
# )
# dfa = build_ead_dfa(ead).minimize()
# dfa.to_svg("visulas/BAD_ead_dfa.svg")
# ta = TemplateAutomaton.from_dfa(dfa)
# cf = ta.extract_characteristic_factors()
# # print(cf)
# ta.to_svg("visulas/BAD_template_automaton.svg")
# ead = ead.normalize()
ps = []
ss = ["s"]
fs = ["sssb"]
ead = ExtAD.from_strings(ps, fs, ss)
ead = ead.normalize()
print(ead)