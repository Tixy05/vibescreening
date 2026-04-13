"""Extended Antidictionary (EAD) class with normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from antidict import (
    TemplateAutomaton,
    build_ead_dfa,
    extract_characteristic_factors,
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
          2. TemplateAutomaton(DFA_min) → template automaton (TA)
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

            template = TemplateAutomaton.from_dfa(dfa_min)
            try:
                template.to_svg("template_automaton.svg")
                print("SVG written: template_automaton.svg")
            except NotImplementedError as exc:
                print(f"Skipping template automaton SVG: {exc}")

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

