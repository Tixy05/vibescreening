"""Anti-language SLTL extraction for screening (main.pdf §6.2)."""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from itertools import product
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
    AbstractSvar,
    AbstractTvar,
    AbstractedRegexedElement,
)
from .common import Alternation, AnyStarred, SimpleRegex
from .k_subwords import KSubword
from .regex_dfa import simple_regex_from_subword
from .trace_log import trace_step


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
    """Letters appearing in a flat encoded pattern: always ``s``, plus ``b_i`` from alternations.

    Base ``b`` is not seeded here — it exists only on the first cascade step
    (``terminal_alphabet_for_step(0)``); level-0 patterns use quotient labels ``b1``, …
    """
    letters = {"s"}

    def walk(elements: list[AbstractPatternElement]) -> None:
        for elem in elements:
            if isinstance(elem, AbstractedRegexedElement):
                for letter in elem.alternation.alternatives:
                    letters.add(letter.name)
            elif isinstance(elem, AbstractParenthesizedPattern):
                walk(elem.children)

    walk(ap.children)
    return letters


def pattern_has_tvar(ap: AbstractPattern) -> bool:
    """True if the flat pattern contains a top-level ``t`` (incl. inside parens)."""

    def walk(elements: list[AbstractPatternElement]) -> bool:
        for elem in elements:
            if isinstance(elem, AbstractTvar):
                return True
            if isinstance(elem, AbstractParenthesizedPattern) and walk(elem.children):
                return True
        return False

    return walk(ap.children)


def format_flat_pattern(ap: AbstractPattern) -> str:
    """Human-readable flat abstract pattern (level-0 shape)."""
    pieces: list[str] = []
    for elem in ap.children:
        match elem:
            case AbstractEVar():
                pieces.append("e")
            case AbstractSvar():
                pieces.append("s")
            case AbstractTvar():
                pieces.append("t")
            case AbstractedRegexedElement():
                alts = "|".join(letter.name for letter in elem.alternation.alternatives)
                pieces.append(f"({alts})")
            case AbstractParenthesizedPattern():
                inner = format_flat_pattern(AbstractPattern(list(elem.children)))
                pieces.append(f"({inner})")
            case _:
                pieces.append(str(elem))
    return " ".join(pieces)


def format_simple_regex(regex: SimpleRegex, terminal_letters: list[str]) -> str:
    """Regex string: ``(b|s)*``, ``(b1|b2)``, concatenated (§4 morphism)."""
    e_union = "|".join(sorted(terminal_letters))
    parts: list[str] = []
    for piece in regex.concats:
        if isinstance(piece, AnyStarred):
            parts.append(f"({e_union})*")
        elif isinstance(piece, Alternation):
            parts.append(
                "(" + "|".join(letter.name for letter in piece.alternatives) + ")"
            )
    return "".join(parts)


def flat_pattern_to_regex(
    ap: AbstractPattern,
    *,
    terminal_letters: list[str] | None = None,
) -> str:
    """Regular expression for a level-0 (flat) encoded pattern."""
    terminals = terminal_letters or sorted(alphabet_from_pattern(ap))
    regex = simple_regex_from_subword(KSubword(tuple(ap.children)), terminals)
    return format_simple_regex(regex, terminals)


def format_sltl_nf(sltl: SLTL) -> str:
    """Normalized SLTL antidictionary components."""
    return (
        f"SFP={sorted(sltl.prefixes)} "
        f"SFF={sorted(sltl.factors)} "
        f"SFS={sorted(sltl.suffixes)} "
        f"SFW={sorted(sltl.sfw)}"
    )


def format_dense_map(dense_map: dict[str, str]) -> str:
    """Sparse cascade alphabet → single-character SLTL alphabet."""
    return ", ".join(f"{sparse}→{dense}" for sparse, dense in sorted(dense_map.items()))


@dataclass(frozen=True)
class Level0PatternLog:
    index: int
    tag: str
    pattern: str
    regex: str
    sltl_nf: str
    sltl_note: str = ""


def level0_pattern_logs(
    final_patterns: list[AbstractPattern],
    dense_map: dict[str, str],
    *,
    encoding: object | None = None,
    abstract_patterns: list[AbstractPattern] | None = None,
    peer_patterns: list[AbstractPattern] | None = None,
) -> tuple[Level0PatternLog, ...]:
    entries: list[Level0PatternLog] = []
    for i, enc in enumerate(final_patterns):
        form = detect_form(enc)
        tag = f"SLTL {form.kind}" if form is not None else "non-SLTL"
        abstract_ap = (
            abstract_patterns[i] if abstract_patterns is not None else None
        )
        peers = peer_patterns if peer_patterns is not None else None
        anti = anti_sltl_from_final_pattern(
            enc,
            dense_map,
            encoding=encoding,
            abstract_ap=abstract_ap,
            peer_patterns=peers,
            rule_index=i,
        )
        if anti is None:
            sltl_nf = "(not built)"
            note = ""
        else:
            sltl_nf = format_sltl_nf(anti)
            note = ""
        entries.append(
            Level0PatternLog(
                index=i,
                tag=tag,
                pattern=format_flat_pattern(enc),
                regex=flat_pattern_to_regex(
                    enc,
                    terminal_letters=_flat_regex_terminals(encoding, final_patterns),
                ),
                sltl_nf=sltl_nf,
                sltl_note=note,
            )
        )
    return tuple(entries)


def trace_level0_patterns(
    function_name: str,
    rows: tuple[Level0PatternLog, ...],
) -> None:
    """Log level-0 pattern, regex, and anti-SLTL normal form (trace.log)."""
    trace_step(f"screen {function_name}: level-0 patterns")
    for row in rows:
        trace_step(f"  rule {row.index} [{row.tag}] pattern: {row.pattern}")
        trace_step(f"    regex: {row.regex}")
        trace_step(f"    SLTL: {row.sltl_nf}")


def extract_rplus(
    original: AbstractPattern,
    encoded: AbstractPattern,
    form: FormSplit,
) -> KSubword:
    mid_end = (
        len(encoded.children) - form.trail if form.trail else len(encoded.children)
    )
    return KSubword(tuple(encoded.children[form.lead:mid_end]))


def _letter_set_for_rplus_block(
    elem: AbstractPatternElement,
    terminal_letters: list[str],
    *,
    t_alphabet: frozenset[str] | None = None,
) -> set[str]:
    """Set(Alt) for one block in r+ (§6.2 / tmp.md W(Alt₁)×…×W(Altₙ))."""
    match elem:
        case AbstractedRegexedElement():
            return {letter.name for letter in elem.alternation.alternatives}
        case AbstractSvar():
            return {"s"} if "s" in terminal_letters else set()
        case AbstractTvar():
            if t_alphabet is not None:
                return set(t_alphabet)
            return set(terminal_letters)
        case AbstractEVar():
            raise ValueError("e-var inside P block")
        case AbstractParenthesizedPattern():
            raise ValueError("unflattened parenthesis in r+ block")
        case _:
            raise TypeError(f"unexpected r+ element: {elem!r}")


def rplus_forbidden_words(
    rplus: KSubword,
    terminal_letters: list[str],
    *,
    t_alphabet: frozenset[str] | None = None,
) -> set[str]:
    """W(r+) = Set(Alt₁) × … × Set(Altₙ) over concatenated blocks."""
    block_sets: list[set[str]] = [
        _letter_set_for_rplus_block(elem, terminal_letters, t_alphabet=t_alphabet)
        for elem in rplus.elements
    ]
    if not block_sets:
        return {""}
    return {"".join(letters) for letters in product(*block_sets)}


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
    alpha = alphabet or {"s"}
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


def _anti_sltl_e_only(
    terminal_letters: list[str],
    dense_map: dict[str, str],
    dense_alpha: set[str],
) -> SLTL:
    """Top-level ``e`` only (empty Φ): SFF = all monograms, SFW = {ε}.

    Skips ``normalize()``: characteristic-factor extraction would collapse
    monogram factors when only ``""`` is listed as forbidden word.
    """
    monograms = set(dense_alpha)
    trace_step(
        f"anti e-only: SFF={sorted(monograms)} SFW={{''}}"
    )
    return SLTL(factors=monograms, sfw={""}, alphabet=dense_alpha)


def anti_sltl_from_form(
    form: FormSplit,
    rplus: KSubword,
    alphabet: set[str],
    *,
    dense_map: dict[str, str] | None = None,
    t_alphabet: frozenset[str] | None = None,
) -> SLTL | None:
    """Build normalized anti-language SLTL per §6.2 table."""
    terminals = sorted(alphabet)
    if dense_map is None:
        dense_alpha, mapping = _dense_alphabet(alphabet)
    else:
        mapping = dense_map
        dense_alpha = set(mapping.values())

    if form.kind == FormKind.EP and not rplus.elements:
        return _anti_sltl_e_only(terminals, mapping, dense_alpha)

    trace_step(
        f"anti form {form.kind}: r+ product "
        f"({len(rplus.elements)} block(s), |Γ|={len(terminals)})"
    )
    words = rplus_forbidden_words(rplus, terminals, t_alphabet=t_alphabet)
    if not words and form.kind != FormKind.P:
        return None

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
    trace_step(f"anti form {form.kind}: normalize ({len(dense_words)} word(s))")
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

    trace_step("less_restrictive_sltl: compare left ⊆ right")
    if sltl_is_subset(left, right):
        return right
    trace_step("less_restrictive_sltl: compare right ⊆ left")
    if sltl_is_subset(right, left):
        return left
    return right


def anti_sltl_approximate(
    ap: AbstractPattern,
    dense_map: dict[str, str],
    *,
    peer_patterns: list[AbstractPattern],
    rule_index: int,
    encoding: object | None = None,
    t_alphabet: frozenset[str] | None = None,
) -> SLTL:
    """Approximate anti-SLTL for non-SLTL shapes via ⋂ eAe, eBe, … (§6.2 extension)."""
    if t_alphabet is None and encoding is not None:
        from .screening_analyzer import function_t_alphabet

        t_alphabet = function_t_alphabet(encoding)
    split = split_on_top_level_e(ap)
    trace_step(
        f"anti approx rule {rule_index}: {len(split.segments)} segment(s), "
        f"lead={split.has_leading_e} trail={split.has_trailing_e}"
    )
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
            trace_step(f"anti approx rule {rule_index}: segment {i} shell encode")
            shell = _segment_shell_pattern(split, i)
            encoded = _encode_rule_in_function(shell, peer_patterns, rule_index)
            shell_form = detect_form(shell)
            if shell_form is None:
                trace_step(f"anti approx rule {rule_index}: segment {i} skip (no form)")
                continue
            rplus = extract_rplus(shell, encoded, shell_form)
            alpha = alphabet_from_pattern(encoded)
            trace_step(
                f"anti approx rule {rule_index}: segment {i} anti_sltl_from_form"
            )
            part = anti_sltl_from_form(
                shell_form,
                rplus,
                alpha,
                dense_map=dense_map,
                t_alphabet=t_alphabet,
            )
            if part is not None:
                _merge_sltl_fields(prefixes, factors, suffixes, sfw, part)

        trace_step(f"anti approx rule {rule_index}: full pattern encode")
        full_encoded = _encode_rule_in_function(ap, peer_patterns, rule_index)
        full_form = detect_form(ap)
        if full_form is not None:
            rplus = extract_rplus(ap, full_encoded, full_form)
            alpha = alphabet_from_pattern(full_encoded)
            trace_step(f"anti approx rule {rule_index}: full anti_sltl_from_form")
            part = anti_sltl_from_form(
                full_form,
                rplus,
                alpha,
                dense_map=dense_map,
                t_alphabet=t_alphabet,
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
    trace_step(f"anti approx rule {rule_index}: final normalize")
    return raw.normalize()


def anti_sltl_from_final_pattern(
    final_ap: AbstractPattern,
    dense_map: dict[str, str],
    *,
    encoding: object | None = None,
    abstract_ap: AbstractPattern | None = None,
    peer_patterns: list[AbstractPattern] | None = None,
    rule_index: int = 0,
) -> SLTL | None:
    """Build anti-SLTL from a cascade-encoded level-0 pattern."""
    t_alphabet: frozenset[str] | None = None
    if encoding is not None:
        from .screening_analyzer import function_t_alphabet

        t_alphabet = function_t_alphabet(encoding)
    form = detect_form(final_ap)
    if form is None:
        if abstract_ap is None or peer_patterns is None:
            return None
        trace_step(f"anti rule {rule_index}: approximate (encoded form is None)")
        return anti_sltl_approximate(
            abstract_ap,
            dense_map,
            peer_patterns=peer_patterns,
            rule_index=rule_index,
            encoding=encoding,
            t_alphabet=t_alphabet,
        )

    trace_step(f"anti rule {rule_index}: exact from encoded ({form.kind})")
    rplus = extract_rplus(final_ap, final_ap, form)
    exact = anti_sltl_from_form(
        form,
        rplus,
        alphabet_from_pattern(final_ap),
        dense_map=dense_map,
        t_alphabet=t_alphabet,
    )
    if exact is None:
        return None

    if abstract_ap is not None and detect_form(abstract_ap) is None:
        if peer_patterns is None:
            return exact
        trace_step(f"anti rule {rule_index}: merge with approximate")
        approx = anti_sltl_approximate(
            abstract_ap,
            dense_map,
            peer_patterns=peer_patterns,
            rule_index=rule_index,
            encoding=encoding,
            t_alphabet=t_alphabet,
        )
        trace_step(f"anti rule {rule_index}: less_restrictive_sltl")
        return _less_restrictive_sltl(approx, exact)

    return exact


def _flat_regex_terminals(encoding: object | None, final_patterns: list[AbstractPattern]) -> list[str]:
    if encoding is None:
        from .screening_analyzer import sorted_terminal_alphabet

        return sorted_terminal_alphabet({"b", "s"})
    from .screening_analyzer import flat_terminal_alphabet

    return flat_terminal_alphabet(encoding, final_patterns)


def function_encoding_alphabet(
    final_patterns: list[AbstractPattern],
    encoding: object | None = None,
) -> dict[str, str]:
    """Single dense symbol map shared by all rules in a function."""
    letters: set[str] = set()
    for ap in final_patterns:
        letters |= alphabet_from_pattern(ap)
    if encoding is not None and any(pattern_has_tvar(ap) for ap in final_patterns):
        from .screening_analyzer import function_t_alphabet

        letters |= function_t_alphabet(encoding)
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
    shell_ap = AbstractPattern(list(p_elements))
    if p_elements:
        from .screening_analyzer import encode_patterns, function_t_alphabet

        enc_result = encode_patterns([shell_ap], name="_")
        p_encoded = enc_result.final_patterns[0]
        t_alpha = (
            function_t_alphabet(enc_result)
            if pattern_has_tvar(shell_ap)
            else None
        )
    else:
        p_encoded = AbstractPattern([])
        t_alpha = None
    rplus = KSubword(tuple(p_encoded.children))
    alpha = alphabet_from_pattern(p_encoded)
    if t_alpha is not None:
        alpha |= set(t_alpha)
    dense_map = _dense_symbol_map(alpha)
    return anti_sltl_from_form(
        form, rplus, alpha, dense_map=dense_map, t_alphabet=t_alpha,
    )
