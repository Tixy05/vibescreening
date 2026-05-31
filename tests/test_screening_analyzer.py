"""Tests for cascade encoding (screening analyzer)."""

from __future__ import annotations

import unittest
from pathlib import Path

from refal.parser.parser import build_ast_from_string
from refal.sema.abstractizer import (
    AbstractEVar,
    AbstractPattern,
    AbstractSvar,
    AbstractTvar,
    AbstractedRegexedElement,
)
from refal.sema.k_subwords import (
    KSubword,
    extract_k_subwords_at_level,
    extract_k_subwords_function,
    max_paren_depth_pattern,
)
from refal.sema.regex_dfa import (
    annotated_union,
    build_step_dfa,
    quotient_to_bi_classes,
    simple_regex_from_subword,
    simple_regex_to_dfa,
)
from refal.sema.screening_analyzer import encode_function

_LISTING3 = """\
f {
    t.1 (e.1 (s.1) s.2) = ;
    ((t.1) e.1 s.2) s.3 = ;
}
"""

_USER_DEPTH2 = """\
h {
    e.1 (e.2 (e.3 t.4) (s.5) (s.6)) = ;
}
"""

_EMPTY_PLUS_ETA = """\
g {
    = ;
    e.1 t.2 e.3 = ;
}
"""


def _patterns(source: str) -> list[AbstractPattern]:
    prog = build_ast_from_string(source)
    return [AbstractPattern.from_concrete(r.pattern) for r in prog.definitions[0].rules]


class TestParenDepth(unittest.TestCase):
    def test_listing3_depth(self) -> None:
        pats = _patterns(_LISTING3)
        self.assertEqual(max_paren_depth_pattern(pats[0]), 2)
        self.assertEqual(max_paren_depth_pattern(pats[1]), 2)

    def test_user_example_depth(self) -> None:
        pats = _patterns(_USER_DEPTH2)
        self.assertEqual(max_paren_depth_pattern(pats[0]), 2)


class TestKSubwords(unittest.TestCase):
    def test_level2_listing3(self) -> None:
        pats = _patterns(_LISTING3)
        subwords, _ = extract_k_subwords_function(pats, 2)
        shapes = {str(s) for s in subwords}
        self.assertEqual(shapes, {"s", "t"})

    def test_level2_user_flat_groups(self) -> None:
        pats = _patterns(_USER_DEPTH2)
        subwords, _ = extract_k_subwords_function(pats, 2)
        shapes = {str(s) for s in subwords}
        self.assertIn("et", shapes)
        self.assertIn("s", shapes)

    def test_level1_listing3_after_depth2_replace(self) -> None:
        from refal.sema.abstractizer import Alternation, Letter, replace_at_path
        from refal.sema.k_subwords import extract_k_subwords_at_level

        pats = _patterns(_LISTING3)
        _, occs0 = extract_k_subwords_at_level(pats[0], 2)
        repl = AbstractedRegexedElement(Alternation((Letter("b1"),)))
        p0 = replace_at_path(pats[0], occs0[0].path, repl)
        subwords, _ = extract_k_subwords_at_level(p0, 1)
        shapes = {str(s) for s in subwords}
        self.assertIn("e[b1]s", shapes)


class TestRegexDfa(unittest.TestCase):
    def test_accepts_s_not_b(self) -> None:
        sw = KSubword((AbstractSvar(),))
        dfa = simple_regex_to_dfa(simple_regex_from_subword(sw, ["b", "s"]))
        self.assertTrue(_accepts(dfa, "s"))
        self.assertFalse(_accepts(dfa, "b"))

    def test_t_accepts_b_or_s(self) -> None:
        sw = KSubword((AbstractTvar(),))
        dfa = simple_regex_to_dfa(simple_regex_from_subword(sw, ["b", "s"]))
        self.assertTrue(_accepts(dfa, "b"))
        self.assertTrue(_accepts(dfa, "s"))


def _accepts(dfa, word: str) -> bool:
    state = dfa.initial
    for ch in word:
        state = dfa.transitions[state][ch]
    return state in dfa.accepting


class TestQuotientLabels(unittest.TestCase):
    def test_non_empty_annotations_start_at_b1(self) -> None:
        classes = quotient_to_bi_classes({0: frozenset({1}), 1: frozenset({2})})
        labels = [(c.label, set(c.indices)) for c in classes]
        self.assertEqual(labels, [("b1", {1}), ("b2", {2})])
        self.assertNotIn("b0", [c.label for c in classes])

    def test_empty_annotation_gets_b0(self) -> None:
        classes = quotient_to_bi_classes(
            {0: frozenset({1, 2}), 1: frozenset({2}), 2: frozenset()}
        )
        labels = [(c.label, set(c.indices)) for c in classes]
        self.assertEqual(labels[0], ("b0", set()))
        self.assertCountEqual(
            labels[1:],
            [("b1", {2}), ("b2", {1, 2})],
        )


class TestEmptyPattern(unittest.TestCase):
    def test_empty_and_non_empty_union_covers_alphabet(self) -> None:
        empty = KSubword(())
        ete = KSubword((AbstractEVar(), AbstractTvar(), AbstractEVar()))
        dfas = [
            (simple_regex_to_dfa(simple_regex_from_subword(empty, ["b", "s"])), 1),
            (simple_regex_to_dfa(simple_regex_from_subword(ete, ["b", "s"])), 2),
        ]
        union, ann = annotated_union(dfas)
        for w in ("", "s", "b", "sb", "bs", "ss", "bb"):
            self.assertTrue(_accepts(union, w), f"union should accept {w!r}")
        classes = quotient_to_bi_classes(ann)
        self.assertNotIn("b0", [c.label for c in classes])
        self.assertEqual(
            [(c.label, set(c.indices)) for c in classes],
            [("b1", {1}), ("b2", {2})],
        )

    def test_g_function_no_b0(self) -> None:
        prog = build_ast_from_string(_EMPTY_PLUS_ETA)
        fn = prog.definitions[0]
        result = encode_function(fn)
        self.assertEqual(len(result.steps), 1)
        step = result.steps[0]
        self.assertEqual(step.level, 0)
        self.assertEqual([str(s) for s in step.subwords], ["", "ete"])
        labels = [c.label for c in step.bi_classes]
        self.assertNotIn("b0", labels)
        self.assertEqual(
            [(c.label, set(c.indices)) for c in step.bi_classes],
            [("b1", {1}), ("b2", {2})],
        )

    def test_g_writes_visualization(self) -> None:
        prog = build_ast_from_string(_EMPTY_PLUS_ETA)
        fn = prog.definitions[0]
        out = Path("output/screening_test")
        encode_function(fn, out_dir=out)
        self.assertTrue((out / "g_depth0.dot").exists())
        if _has_dot():
            self.assertTrue((out / "g_depth0.svg").exists())


class TestEncodeFunction(unittest.TestCase):
    def test_listing3_steps(self) -> None:
        prog = build_ast_from_string(_LISTING3)
        fn = prog.definitions[0]
        result = encode_function(fn)
        levels = [s.level for s in result.steps]
        self.assertEqual(levels, [2, 1, 0])

    def test_listing3_writes_svg(self) -> None:
        prog = build_ast_from_string(_LISTING3)
        fn = prog.definitions[0]
        out = Path("output/screening_test")
        result = encode_function(fn, out_dir=out)
        self.assertEqual(len(result.steps), 3)
        for level in (2, 1, 0):
            svg = out / f"f_depth{level}.svg"
            dot = out / f"f_depth{level}.dot"
            self.assertTrue(dot.exists(), f"missing {dot}")
            if _has_dot():
                self.assertTrue(svg.exists(), f"missing {svg}")


def _has_dot() -> bool:
    import shutil
    return shutil.which("dot") is not None


if __name__ == "__main__":
    unittest.main()
