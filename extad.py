
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from antidict import DFA


@dataclass(slots=True)
class ExtAD:
    """Extended antidictionary with an attached alphabet."""

    prefixes: set[str] = field(init=False)
    factors: set[str] = field(init=False)
    suffixes: set[str] = field(init=False)
    alphabet: set[str] = field(init=False)
    dense_alphabet: bool = field(init=False)

    def __init__(
        self,
        prefixes: Iterable[str] = (),
        factors: Iterable[str] = (),
        suffixes: Iterable[str] = (),
        *,
        dense_alphabet: bool = True,
        alphabet: Iterable[str] | None = None,
    ) -> None:
        self.prefixes = set(prefixes)
        self.factors = set(factors)
        self.suffixes = set(suffixes)
        self.dense_alphabet = dense_alphabet
        self.alphabet = (
            set(alphabet)
            if alphabet is not None
            else self._compute_alphabet(dense_alphabet)
        )

    @classmethod
    def from_strings(
        cls,
        prefixes: Iterable[str] = (),
        factors: Iterable[str] = (),
        suffixes: Iterable[str] = (),
        *,
        dense_alphabet: bool = True,
        alphabet: Iterable[str] | None = None,
    ) -> "ExtAD":
        return cls(
            prefixes,
            factors,
            suffixes,
            dense_alphabet=dense_alphabet,
            alphabet=alphabet,
        )

    def _compute_alphabet(self, dense_alphabet: bool) -> set[str]:
        alphabet = set(
            "".join(self.prefixes)
            + "".join(self.factors)
            + "".join(self.suffixes)
        )
        if not dense_alphabet:
            if alphabet:
                alphabet.add(chr(ord(max(alphabet)) + 1))
            else:
                alphabet.add("a")
        return alphabet

    def __str__(self) -> str:
        p = "{" + ", ".join(sorted(self.prefixes)) + "}"
        f = "{" + ", ".join(sorted(self.factors)) + "}"
        s = "{" + ", ".join(sorted(self.suffixes)) + "}"
        a = "{" + ", ".join(sorted(self.alphabet)) + "}"
        return f"ExtAD(prefixes={p}, factors={f}, suffixes={s}, alphabet={a})"

    def build_dfa(self) -> "DFA":
        from antidict import build_ead_dfa

        return build_ead_dfa(self)

    def normalize(self) -> "ExtAD":
        from antidict import extract_characteristic_factors

        dfa = self.build_dfa().minimize()
        cf = extract_characteristic_factors(dfa)
        return ExtAD(
            cf.SFP,
            cf.SFF,
            cf.SFS,
            dense_alphabet=self.dense_alphabet,
            alphabet=self.alphabet,
        )

    def normalization_state_blowup(
        self,
        *,
        visualize: bool = False,
    ) -> tuple[int, int]:
        from antidict import TemplateAutomaton, extract_characteristic_factors

        dfa_min = self.build_dfa().minimize()
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
            template.to_svg("template_automaton.svg")
            print("SVG written: template_automaton.svg")

        return states_ta, states_min


class SLTL(ExtAD):
    """An ExtAD enriched with shortest forbidden words."""

    sfw: set[str] = field(init=False)

    def __init__(
        self,
        prefixes: Iterable[str] = (),
        factors: Iterable[str] = (),
        suffixes: Iterable[str] = (),
        sfw: Iterable[str] = (),
        *,
        dense_alphabet: bool = True,
        alphabet: Iterable[str] | None = None,
    ) -> None:
        self.sfw = set(sfw)
        super().__init__(
            prefixes,
            factors,
            suffixes,
            dense_alphabet=dense_alphabet,
            alphabet=alphabet,
        )

    @classmethod
    def from_strings(
        cls,
        prefixes: Iterable[str] = (),
        factors: Iterable[str] = (),
        suffixes: Iterable[str] = (),
        sfw: Iterable[str] = (),
        *,
        dense_alphabet: bool = True,
        alphabet: Iterable[str] | None = None,
    ) -> "SLTL":
        return cls(
            prefixes,
            factors,
            suffixes,
            sfw,
            dense_alphabet=dense_alphabet,
            alphabet=alphabet,
        )

    def _compute_alphabet(self, dense_alphabet: bool) -> set[str]:
        alphabet = super()._compute_alphabet(True)
        alphabet |= set("".join(self.sfw))
        if not dense_alphabet:
            if alphabet:
                alphabet.add(chr(ord(max(alphabet)) + 1))
            else:
                alphabet.add("a")
        return alphabet

    def __str__(self) -> str:
        p = "{" + ", ".join(sorted(self.prefixes)) + "}"
        f = "{" + ", ".join(sorted(self.factors)) + "}"
        s = "{" + ", ".join(sorted(self.suffixes)) + "}"
        w = "{" + ", ".join(sorted(self.sfw)) + "}"
        a = "{" + ", ".join(sorted(self.alphabet)) + "}"
        return (
            f"SLTL(prefixes={p}, factors={f}, suffixes={s}, "
            f"sfw={w}, alphabet={a})"
        )

    def build_dfa(self) -> "DFA":
        from antidict import build_sltl_dfa

        return build_sltl_dfa(self)

    def normalize(self) -> "SLTL":
        from antidict import extract_characteristic_factors

        dfa = self.build_dfa().minimize()
        cf = extract_characteristic_factors(dfa)
        return SLTL(
            cf.SFP,
            cf.SFF,
            cf.SFS,
            cf.SFW,
            dense_alphabet=self.dense_alphabet,
            alphabet=self.alphabet,
        )