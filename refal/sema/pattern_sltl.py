"""Anti-language SLTL extraction for screening (main.pdf §6.2)."""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extad import SLTL

from .abstractizer import (
    AbstractEVar,
    AbstractPattern,
    AbstractPatternElement,
    AbstractParenthesizedPattern,
    AbstractedRegexedElement,
)
from .common import SimpleRegex
from .k_subwords import KSubword
from .regex_dfa import simple_regex_from_subword, simple_regex_to_dfa


class FormKind(enum.StrEnum):
    EPE = "ePe"
    EP = "eP"
    PE = "Pe"
    P = "P"


@dataclass(frozen=True)
class FormSplit:
    kind: FormKind
    lead: int
    trail: int


def _has_top_level_evar(elements: list[AbstractPatternElement]) -> bool:
    return any(isinstance(e, AbstractEVar) for e in elements)


def detect_form(ap: AbstractPattern) -> FormSplit | None:
    """Detect SLTL family; reject if P contains a top-level e-var."""
    children = ap.children
    lead = 0
    while lead < len(children) and isinstance(children[lead], AbstractEVar):
        lead += 1
    trail = 0
    while trail < len(children) - lead and isinstance(
        children[-1 - trail], AbstractEVar
    ):
        trail += 1
    mid_end = len(children) - trail if trail else len(children)
    p_part = children[lead:mid_end]
    if _has_top_level_evar(p_part):
        return None
    if lead and trail:
        return FormSplit(FormKind.EPE, lead, trail)
    if lead:
        return FormSplit(FormKind.EP, lead, trail)
    if trail:
        return FormSplit(FormKind.PE, lead, trail)
    return FormSplit(FormKind.P, lead, trail)


def alphabet_from_pattern(ap: AbstractPattern) -> set[str]:
    letters = {"b", "s"}

    def walk(elements: list[AbstractPatternElement]) -> None:
        for elem in elements:
            if isinstance(elem, AbstractedRegexedElement):
                for letter in elem.alternation.alternatives:
                    letters.add(letter.name)
            elif isinstance(elem, AbstractParenthesizedPattern):
                walk(elem.children)

    walk(ap.children)
    return letters


def extract_rplus(
    original: AbstractPattern,
    encoded: AbstractPattern,
    form: FormSplit,
) -> KSubword:
    mid_end = (
        len(encoded.children) - form.trail if form.trail else len(encoded.children)
    )
    return KSubword(tuple(encoded.children[form.lead:mid_end]))


def _rplus_to_simple_regex(
    rplus: KSubword, terminal_letters: list[str],
) -> SimpleRegex:
    return simple_regex_from_subword(rplus, terminal_letters)


def rplus_language_words(
    regex: SimpleRegex,
    *,
    max_syms: int = 6,
) -> set[str]:
    dfa = simple_regex_to_dfa(regex)
    symbols = list(dfa.alphabet)
    words: set[str] = set()

    def dfs(state: int, built: str, depth: int) -> None:
        if depth > max_syms:
            return
        if state in dfa.accepting:
            words.add(built)
        for sym in symbols:
            dfs(dfa.transitions[state][sym], built + sym, depth + 1)

    dfs(dfa.initial, "", 0)
    return words


def _dense_symbol_map(symbols: set[str]) -> dict[str, str]:
    """Map terminal names (including multi-char ``b1``) to single-character symbols."""
    singles = sorted(s for s in symbols if len(s) == 1)
    multi = sorted(s for s in symbols if len(s) > 1)
    mapping: dict[str, str] = {s: s for s in singles}
    used = set(singles)
    code = ord("A")
    for sym in multi:
        while chr(code) in used:
            code += 1
        mapping[sym] = chr(code)
        used.add(chr(code))
        code += 1
    return mapping


def _remap_word(word: str, mapping: dict[str, str]) -> str:
    symbols = sorted(mapping.keys(), key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(word):
        for sym in symbols:
            if word.startswith(sym, i):
                out.append(mapping[sym])
                i += len(sym)
                break
        else:
            raise ValueError(f"cannot tokenize {word!r} with {sorted(mapping)}")
    return "".join(out)


def _remap_words(words: set[str], mapping: dict[str, str]) -> set[str]:
    return {_remap_word(w, mapping) for w in words}


def _dense_alphabet(alphabet: set[str]) -> tuple[set[str], dict[str, str]]:
    mapping = _dense_symbol_map(alphabet)
    return set(mapping.values()), mapping


def universe_sltl(alphabet: set[str] | None = None) -> SLTL:
    alpha = alphabet or {"b", "s"}
    return SLTL(
        prefixes=set(),
        factors=set(),
        suffixes=set(),
        sfw=set(),
        alphabet=alpha,
    ).normalize()


def align_alphabet(sltl: SLTL, other: SLTL) -> SLTL:
    alpha = sltl.alphabet | other.alphabet
    return SLTL(
        sltl.prefixes,
        sltl.factors,
        sltl.suffixes,
        sltl.sfw,
        dense_alphabet=sltl.dense_alphabet,
        alphabet=alpha,
    ).normalize()


def anti_sltl_from_form(
    form: FormSplit,
    rplus: KSubword,
    alphabet: set[str],
    *,
    dense_map: dict[str, str] | None = None,
) -> SLTL | None:
    """Build normalized anti-language SLTL per §6.2 table."""
    terminals = sorted(alphabet)
    regex = _rplus_to_simple_regex(rplus, terminals)
    words = rplus_language_words(regex)
    if not words and form.kind != FormKind.P:
        return None

    if dense_map is None:
        dense_alpha, mapping = _dense_alphabet(alphabet)
    else:
        mapping = dense_map
        dense_alpha = set(mapping.values())
    dense_words = _remap_words(words, mapping)

    match form.kind:
        case FormKind.EPE:
            raw = SLTL(factors=dense_words, alphabet=dense_alpha)
        case FormKind.EP:
            raw = SLTL(suffixes=dense_words, alphabet=dense_alpha)
        case FormKind.PE:
            raw = SLTL(prefixes=dense_words, alphabet=dense_alpha)
        case FormKind.P:
            raw = SLTL(sfw=dense_words if dense_words else {""}, alphabet=dense_alpha)
    return raw.normalize()


@dataclass(frozen=True)
class ESegmentSplit:
    """Top-level split on e-vars: optional leading/trailing e and non-e blocks."""

    segments: tuple[tuple[AbstractPatternElement, ...], ...]
    has_leading_e: bool
    has_trailing_e: bool


def split_on_top_level_e(ap: AbstractPattern) -> ESegmentSplit:
    """Split a flat abstract pattern into blocks separated by top-level e-vars."""
    segments: list[tuple[AbstractPatternElement, ...]] = []
    current: list[AbstractPatternElement] = []
    for elem in ap.children:
        if isinstance(elem, AbstractEVar):
            if current:
                segments.append(tuple(current))
                current = []
        else:
            current.append(elem)
    if current:
        segments.append(tuple(current))
    has_leading = bool(ap.children and isinstance(ap.children[0], AbstractEVar))
    has_trailing = bool(ap.children and isinstance(ap.children[-1], AbstractEVar))
    return ESegmentSplit(tuple(segments), has_leading, has_trailing)


def _encode_rule_in_function(
    shell_ap: AbstractPattern,
    peer_patterns: list[AbstractPattern],
    rule_index: int,
) -> AbstractPattern:
    from .screening_analyzer import encode_patterns

    patterns = [AbstractPattern(list(p.children)) for p in peer_patterns]
    patterns[rule_index] = shell_ap
    result = encode_patterns(patterns, name="_")
    return result.final_patterns[rule_index]


def _segment_shell_pattern(split: ESegmentSplit, index: int) -> AbstractPattern:
    """Build the e-minor shell eAe / ePe / Pe pattern for one segment."""
    seg = list(split.segments[index])
    n = len(split.segments)
    children: list[AbstractPatternElement] = []
    if index > 0 or split.has_leading_e:
        children.append(AbstractEVar())
    children.extend(seg)
    if index < n - 1:
        children.append(AbstractEVar())
    elif split.has_trailing_e:
        children.append(AbstractEVar())
    return AbstractPattern(children)


def _merge_sltl_fields(
    prefixes: set[str],
    factors: set[str],
    suffixes: set[str],
    sfw: set[str],
    part: SLTL,
) -> None:
    prefixes |= part.prefixes
    factors |= part.factors
    suffixes |= part.suffixes
    sfw |= part.sfw


def _less_restrictive_sltl(left: SLTL, right: SLTL) -> SLTL:
    """Return the anti-SLTL whose language is larger (less restrictive)."""
    from antidict import sltl_is_subset

    if sltl_is_subset(left, right):
        return right
    if sltl_is_subset(right, left):
        return left
    return right


def anti_sltl_approximate(
    ap: AbstractPattern,
    dense_map: dict[str, str],
    *,
    peer_patterns: list[AbstractPattern],
    rule_index: int,
) -> SLTL:
    """Approximate anti-SLTL for non-SLTL shapes via ⋂ eAe, eBe, … (§6.2 extension)."""
    split = split_on_top_level_e(ap)
    dense_alpha = set(dense_map.values())

    prefixes: set[str] = set()
    factors: set[str] = set()
    suffixes: set[str] = set()
    sfw: set[str] = set()

    if not split.segments:
        if not split.has_leading_e and not split.has_trailing_e:
            sfw = {""}
    else:
        for i in range(len(split.segments)):
            shell = _segment_shell_pattern(split, i)
            encoded = _encode_rule_in_function(shell, peer_patterns, rule_index)
            shell_form = detect_form(shell)
            if shell_form is None:
                continue
            rplus = KSubword(tuple(encoded.children))
            alpha = alphabet_from_pattern(encoded)
            part = anti_sltl_from_form(
                shell_form, rplus, alpha, dense_map=dense_map,
            )
            if part is not None:
                _merge_sltl_fields(prefixes, factors, suffixes, sfw, part)

        full_encoded = _encode_rule_in_function(ap, peer_patterns, rule_index)
        full_form = detect_form(ap)
        if full_form is not None:
            rplus = KSubword(tuple(full_encoded.children))
            alpha = alphabet_from_pattern(full_encoded)
            part = anti_sltl_from_form(
                full_form, rplus, alpha, dense_map=dense_map,
            )
            if part is not None:
                _merge_sltl_fields(prefixes, factors, suffixes, sfw, part)

    raw = SLTL(
        prefixes=prefixes,
        factors=factors,
        suffixes=suffixes,
        sfw=sfw,
        alphabet=dense_alpha,
    )
    return raw.normalize()


def anti_sltl_from_final_pattern(
    final_ap: AbstractPattern,
    dense_map: dict[str, str],
    *,
    abstract_ap: AbstractPattern | None = None,
    peer_patterns: list[AbstractPattern] | None = None,
    rule_index: int = 0,
) -> SLTL | None:
    """Build anti-SLTL from a cascade-encoded level-0 pattern."""
    form = detect_form(final_ap)
    if form is None:
        if abstract_ap is None or peer_patterns is None:
            return None
        return anti_sltl_approximate(
            abstract_ap,
            dense_map,
            peer_patterns=peer_patterns,
            rule_index=rule_index,
        )

    exact = anti_sltl_from_form(
        form,
        KSubword(tuple(final_ap.children)),
        alphabet_from_pattern(final_ap),
        dense_map=dense_map,
    )
    if exact is None:
        return None

    if abstract_ap is not None and detect_form(abstract_ap) is None:
        if peer_patterns is None:
            return exact
        approx = anti_sltl_approximate(
            abstract_ap,
            dense_map,
            peer_patterns=peer_patterns,
            rule_index=rule_index,
        )
        return _less_restrictive_sltl(approx, exact)

    return exact


def function_encoding_alphabet(final_patterns: list[AbstractPattern]) -> dict[str, str]:
    """Single dense symbol map shared by all rules in a function."""
    letters: set[str] = set()
    for ap in final_patterns:
        letters |= alphabet_from_pattern(ap)
    return _dense_symbol_map(letters)


def anti_sltl_from_encoded(
    original: AbstractPattern,
    encoded: AbstractPattern | None = None,
) -> SLTL | None:
    """Build anti-SLTL from *original*; encodes the P segment only."""
    form = detect_form(original)
    if form is None:
        return None
    mid_end = (
        len(original.children) - form.trail if form.trail else len(original.children)
    )
    p_elements = original.children[form.lead:mid_end]
    if p_elements:
        from .screening_analyzer import encode_abstract_pattern

        p_encoded = encode_abstract_pattern(AbstractPattern(p_elements))
    else:
        p_encoded = AbstractPattern([])
    rplus = KSubword(tuple(p_encoded.children))
    alpha = alphabet_from_pattern(p_encoded)
    return anti_sltl_from_form(form, rplus, alpha)
