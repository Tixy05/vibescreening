import unittest
from itertools import product as iterproduct
from pathlib import Path
import tempfile

from antidict import (
    DFA,
    TemplateAutomaton,
    build_ead_dfa,
    extract_characteristic_factors,
    verify_characteristic_factors,
)


Case = tuple[str, set[str], set[str], set[str], set[str]]


def dfa_accepts(template_dfa: DFA, word: str) -> bool:
    state: int = template_dfa.initial
    for ch in word:
        state = template_dfa.transitions[state][ch]
    return state in template_dfa.accepting


def enumerate_strings(alphabet: list[str], max_len: int):
    yield ""
    for length in range(1, max_len + 1):
        for combo in iterproduct(alphabet, repeat=length):
            yield "".join(combo)


CASES: list[Case] = [
    ("empty EAD", set(), set(), set(), {"a", "b"}),
    ("factors only", set(), {"ab", "ba"}, set(), {"a", "b"}),
    ("prefixes only", {"ab", "ba"}, set(), set(), {"a", "b"}),
    ("suffixes only", set(), set(), {"ab", "ba"}, {"a", "b"}),
    ("mixed P+F+S", {"aa"}, {"bb"}, {"ab"}, {"a", "b"}),
    ("ternary mixed", {"ab"}, {"cc"}, {"ba"}, {"a", "b", "c"}),
    ("single-char factor", set(), {"a"}, set(), {"a", "b"}),
    ("single-char prefix", {"a"}, set(), set(), {"a", "b"}),
    ("single-char suffix", set(), set(), {"a"}, {"a", "b"}),
    ("pattern in both F and S", set(), {"ab"}, {"ab"}, {"a", "b"}),
    ("prefix is also factor", {"ab"}, {"ab"}, set(), {"a", "b"}),
    ("overlapping patterns", {"ab"}, {"ba"}, {"aa"}, {"a", "b"}),
    ("long patterns", {"aab"}, {"bab"}, {"abb"}, {"a", "b"}),
    ("all singletons forbidden", {"a", "b"}, set(), set(), {"a", "b"}),
    ("nested prefixes", {"a", "ab", "abc"}, set(), set(), {"a", "b", "c"}),
]


class TemplateAutomatonTests(unittest.TestCase):
    def test_characteristic_factor_extraction_matches_definitions(self) -> None:
        for name, prefixes, factors, suffixes, alphabet in CASES:
            with self.subTest(name=name):
                dfa = build_ead_dfa(prefixes, factors, suffixes, alphabet).minimize()
                template = TemplateAutomaton.from_dfa(dfa)

                via_template = template.extract_characteristic_factors()
                via_function = extract_characteristic_factors(dfa)

                self.assertEqual(via_template, via_function)
                self.assertEqual(via_template.template_num_states, template.num_states)
                self.assertTrue(verify_characteristic_factors(dfa, via_template, max_len=7))

    def test_projected_dfas_accept_reversed_characteristic_sets(self) -> None:
        projection_builders = {
            "SFF": TemplateAutomaton.to_sff_dfa,
            "SFP": TemplateAutomaton.to_sfp_dfa,
            "SFS": TemplateAutomaton.to_sfs_dfa,
            "SFW": TemplateAutomaton.to_sfw_dfa,
            "SAS": TemplateAutomaton.to_sas_dfa,
            "SAW": TemplateAutomaton.to_saw_dfa,
        }

        for name, prefixes, factors, suffixes, alphabet in CASES:
            with self.subTest(name=name):
                dfa = build_ead_dfa(prefixes, factors, suffixes, alphabet).minimize()
                template = TemplateAutomaton.from_dfa(dfa)
                cf = template.extract_characteristic_factors()

                for family, builder in projection_builders.items():
                    with self.subTest(name=name, family=family):
                        projected = builder(template)
                        expected = {"".join(reversed(word)) for word in getattr(cf, family)}
                        horizon = max((len(word) for word in expected), default=0) + 1
                        observed = {
                            word
                            for word in enumerate_strings(projected.alphabet, horizon)
                            if dfa_accepts(projected, word)
                        }
                        self.assertEqual(observed, expected)

    def test_dot_and_svg_rendering(self) -> None:
        dfa = build_ead_dfa({"aa"}, {"bb"}, {"ab"}, {"a", "b"}).minimize()
        template = TemplateAutomaton.from_dfa(dfa)

        dot = template.to_dot()
        self.assertIn("digraph TemplateAutomaton", dot)
        self.assertIn("SFF", dot)
        self.assertIn("SFP", dot)
        self.assertIn("style=dashed", dot)
        self.assertIn("peripheries=2", dot)

        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "template.svg"
            template.to_svg(svg_path)
            self.assertTrue(svg_path.exists())
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
