from typing import Iterable

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