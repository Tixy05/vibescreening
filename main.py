"""Extended Antidictionary (EAD) class with normalization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import itertools
from typing import Iterable

from antidict import (
    DFA,
    build_ead_dfa,
    build_template_automaton,
    extract_characteristic_factors,
    make_initial_states,
    template_automaton_to_svg,
)


@dataclass(frozen=True)
class ExtAD:
    """An Extended Antidictionary defined by three finite sets of strings:
    forbidden prefixes, forbidden factors, and forbidden suffixes.

    The language it defines is:
      L = { w in Sigma* | w has no prefix in P, no factor in F, no suffix in S }
    """

    prefixes: frozenset[str] = field(default_factory=frozenset)
    factors: frozenset[str] = field(default_factory=frozenset)
    suffixes: frozenset[str] = field(default_factory=frozenset)

    def __str__(self) -> str:
        p = "{" + ", ".join(sorted(self.prefixes)) + "}"
        f = "{" + ", ".join(sorted(self.factors)) + "}"
        s = "{" + ", ".join(sorted(self.suffixes)) + "}"
        return f"⟨\n\t{p},\n\t{f},\n\t{s}\n⟩"

    def normalize(self, alphabet: set[str]) -> ExtAD:
        """Compute the canonical (normal-form) EAD for the same language.

        Pipeline: build DFA -> minimize -> extract shortest characteristic
        factors (SFP, SFF, SFS) -> return new ExtAD.

        Two EADs define the same language iff they produce the same normalized
        ExtAD through this process.
        """
        dfa = build_ead_dfa(
            set(self.prefixes), set(self.factors), set(self.suffixes), alphabet,
        ).minimize()
        cf = extract_characteristic_factors(dfa)
        return ExtAD(
            prefixes=frozenset(cf.SFP),
            factors=frozenset(cf.SFF),
            suffixes=frozenset(cf.SFS),
        )

    def normalization_state_blowup(
        self, alphabet: set[str], *, visualize: bool = False,
    ) -> tuple[int, int]:
        """Compare template automaton size vs minimized EAD DFA size.

        Normalization pipeline:
          1. build_ead_dfa(P, F, S) → minimize → DFA_min
          2. build_template_automaton on DFA_min → template automaton (TA)
          3. extract characteristic factors from TA

        The template automaton (tagged-subset construction) is the real
        intermediate structure that can blow up exponentially.

        A = template automaton (intermediate)
        B = minimized DFA of the EAD language

        Returns (|TA|, |DFA_min|).
        """
        P, F, S = set(self.prefixes), set(self.factors), set(self.suffixes)

        dfa_min = build_ead_dfa(P, F, S, alphabet).minimize()
        cf = extract_characteristic_factors(dfa_min)

        states_ta = cf.template_num_states
        states_min = dfa_min.num_states

        ratio = states_ta / states_min if states_min else float("inf")
        print(f"DFA_min (minimized EAD DFA):        {states_min} states")
        print(f"Template automaton (tagged-subset):  {states_ta} states")
        print(f"Blowup ratio TA / DFA_min:           {ratio:.2f}x")

        if visualize:
            dfa_min.to_svg("dfa_min.svg")
            print("SVG written: dfa_min.svg")

            q_hat_u, q_hat_nf, q_hat_f = make_initial_states(dfa_min)
            named = {}
            if q_hat_u:
                named["q̂_U"] = q_hat_u
            if q_hat_nf:
                named["q̂_nf"] = q_hat_nf
            if q_hat_f:
                named["q̂_f"] = q_hat_f
            ta_states, adj, full_sub, has_init = build_template_automaton(
                dfa_min, set(named.values()),
            )
            template_automaton_to_svg(
                ta_states, adj, named, full_sub, has_init,
                dfa_min.alphabet, "template_automaton.svg",
            )
            print("SVG written: template_automaton.svg")

        return states_ta, states_min


def visualize_ad(prefixes: Iterable[str], infixes: Iterable[str], suffixes: Iterable[str]) -> ExtAD:
    ...


if __name__ == "__main__":
    alphabet = {"a", "b", "c"}
    infix_length = 3

    # infixes = (
    #     "".join(p) for p in itertools.product(sorted(alphabet), repeat=infix_length)
    # )
    # suffixes = ["ab"]
    # ccb - a -- cc
    #     - b -/
    infixes = ["cca", "ccc", "cbc", "baa", "bba", "bab", "bbb", "aca", "acb", "bca", "bcb"]
    suffixes = ["c", "cb", "cba", "cbb"]


    # alphabet = {"a", "b", "c", "d"}
    # infixes = ["abb", "bca", "bcc", "cbb", "cba", "bac", "baa"]
    # suffixes = ["ab", "ba", "cb", "bc"]

    ad = ExtAD(prefixes=frozenset(), factors=frozenset(infixes), suffixes=frozenset(suffixes))
    print(ad)

    print("\n" + "=" * 60)
    print("Normalization state blowup + visualization")
    print("=" * 60)
    ad.normalization_state_blowup(alphabet, visualize=True)

