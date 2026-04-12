"""
Multi-string suffix structure **without** concatenating words into one buffer.

Each ``add(word)`` appends the string to ``words`` and inserts **only** that word's
suffixes into a compressed Patricia-style tree. Edge labels are plain ``str`` slices
(substrings of that word at insert time), not indices into a global array.

``Node.word_indices`` is the union of word indices that contribute a **suffix** in the
subtree (suffix ends at a leaf; internal nodes aggregate from children after each add).

Each ``add(word)`` stores the original ``word`` in ``words`` but inserts suffixes of
``word + end_marker`` (same single-character marker for every word) so no suffix is a
prefix of another within the same logical string—e.g. for ``bb`` and ``ba``, the suffix
``b`` of ``bb`` appears as an edge labeled the marker from the ``b`` node.

Internal nodes get ``suffix_link`` to the locus of ``path_from_root(node)[1:]``; Graphviz
output draws these as dashed ``suffix`` edges.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Default terminator appended on insert (BLACK CIRCLE U+25CF). Not validated against
# ``alphabet_size`` so UTF-8 markers work with a typical byte-sized alphabet for words.
_DEFAULT_END_MARKER = "\u25cf"


@dataclass
class Edge:
    """Outgoing edge from a parent ``Node``; label is non-empty."""

    label: str
    target: "Node"


class Node:
    __slots__ = ("parent", "children", "word_indices", "suffix_link")

    def __init__(self, parent: Optional["Node"] = None) -> None:
        self.parent = parent
        self.children: Dict[str, Edge] = {}
        self.word_indices: Set[int] = set()
        self.suffix_link: Optional[Node] = None


def iter_nodes(root: Node) -> List[Node]:
    out: List[Node] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        n = stack.pop()
        k = id(n)
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
        for e in n.children.values():
            stack.append(e.target)
    return out


def _dot_escape_label(s: str) -> str:
    out: List[str] = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif o >= 32:
            out.append(ch)
        else:
            out.append(f"\\x{o:02x}")
    return "".join(out)


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class SuffixTree:
    def __init__(self, alphabet_size: int = 256, end_marker: str = _DEFAULT_END_MARKER) -> None:
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2")
        if len(end_marker) != 1:
            raise ValueError("end_marker must be a single character")
        self.alphabet_size = alphabet_size
        self.end_marker = end_marker
        self.words: List[str] = []
        self.root = Node(parent=None)

    def _validate_word(self, word: str) -> None:
        if self.end_marker in word:
            raise ValueError(
                f"Word must not contain the end-of-string marker {self.end_marker!r}"
            )
        for c in word:
            if ord(c) >= self.alphabet_size:
                raise ValueError(
                    f"Character {c!r} (code {ord(c)}) must be < alphabet_size {self.alphabet_size}"
                )

    def _new_node(self, parent: Optional[Node]) -> Node:
        return Node(parent=parent)

    def _insert_at(self, node: Node, word_idx: int, remaining: str) -> None:
        """Insert suffix string ``remaining`` (non-empty or empty) under ``node``."""
        if not remaining:
            node.word_indices.add(word_idx)
            return

        c0 = remaining[0]
        if c0 not in node.children:
            leaf = self._new_node(node)
            leaf.word_indices.add(word_idx)
            node.children[c0] = Edge(remaining, leaf)
            return

        edge = node.children[c0]
        L = edge.label
        k = _lcp_len(L, remaining)

        if k == len(L):
            self._insert_at(edge.target, word_idx, remaining[k:])
            return

        if k == len(remaining):
            # Split: remaining is a proper prefix of L
            mid = self._new_node(node)
            mid.word_indices.add(word_idx)
            old_target = edge.target
            tail_l = L[k:]
            assert tail_l
            edge.label = remaining
            edge.target = mid
            mid.children[tail_l[0]] = Edge(tail_l, old_target)
            old_target.parent = mid
            return

        # Split with a fork: common prefix L[:k], then two branches
        mid = self._new_node(node)
        old_target = edge.target
        left_rest = L[k:]
        right_rest = remaining[k:]
        assert left_rest and right_rest
        common = L[:k]
        edge.label = common
        edge.target = mid
        mid.children[left_rest[0]] = Edge(left_rest, old_target)
        old_target.parent = mid
        leaf = self._new_node(mid)
        leaf.word_indices.add(word_idx)
        mid.children[right_rest[0]] = Edge(right_rest, leaf)

    def _pull_up_word_indices(self, n: Node) -> Set[int]:
        acc: Set[int] = set(n.word_indices)
        for e in n.children.values():
            acc |= self._pull_up_word_indices(e.target)
        n.word_indices = acc
        return acc

    def add(self, word: str) -> None:
        if not word:
            return
        self._validate_word(word)
        j = len(self.words)
        self.words.append(word)
        marked = word + self.end_marker
        for start in range(len(marked)):
            self._insert_at(self.root, j, marked[start:])
        self._pull_up_word_indices(self.root)
        self._recompute_suffix_links()

    def _recompute_suffix_links(self) -> None:
        """
        For each explicit internal node with path label P from the root, set ``suffix_link``
        to the locus of the proper suffix P[1:] (standard suffix-tree convention).
        """
        for n in iter_nodes(self.root):
            n.suffix_link = None
        for n in iter_nodes(self.root):
            if n is self.root or not n.children:
                continue
            p = self.path_from_root(n)
            if len(p) <= 1:
                n.suffix_link = self.root
            else:
                t = self.locus_node(p[1:])
                n.suffix_link = t if t is not None else self.root

    def path_from_root(self, node: Node) -> str:
        """Concatenation of edge labels from root to ``node`` (empty for root)."""
        if node is self.root:
            return ""
        parts: List[str] = []
        cur = node
        while cur.parent is not None:
            p = cur.parent
            found: Optional[str] = None
            for e in p.children.values():
                if e.target is cur:
                    found = e.label
                    break
            assert found is not None
            parts.append(found)
            cur = p
        parts.reverse()
        return "".join(parts)

    def locus_node(self, sub: str) -> Optional[Node]:
        """
        Explicit node at the locus of ``sub`` after spelling it from the root.
        If ``sub`` ends inside a compressed edge, returns that edge's **target** node.
        """
        if not sub:
            return self.root
        node = self.root
        i = 0
        n = len(sub)
        while i < n:
            c = sub[i]
            if c not in node.children:
                return None
            edge = node.children[c]
            L = edge.label
            j = 0
            while j < len(L):
                if i == n:
                    return edge.target
                if L[j] != sub[i]:
                    return None
                j += 1
                i += 1
            node = edge.target
        return node

    def contains(self, query: str) -> bool:
        if not query:
            return True
        node = self.root
        i = 0
        n = len(query)
        while i < n:
            c = query[i]
            if c not in node.children:
                return False
            edge = node.children[c]
            L = edge.label
            j = 0
            while j < len(L):
                if i == n:
                    return True
                if L[j] != query[i]:
                    return False
                j += 1
                i += 1
            node = edge.target
        return True

    def to_dot(self) -> str:
        lines: List[str] = [
            "digraph SuffixTree {",
            '  graph [rankdir=LR];',
            '  node [shape=box, style=filled, fillcolor="#f0f0f0", fontname="DejaVu Sans Mono"];',
            '  edge [fontname="DejaVu Sans", fontsize=11, arrowsize=0.8];',
        ]
        nid = 0
        node_ids: dict[int, str] = {}

        def visit(node: Node) -> str:
            nonlocal nid
            key = id(node)
            if key in node_ids:
                return node_ids[key]
            node_id = f"n{nid}"
            nid += 1
            node_ids[key] = node_id
            wi = sorted(node.word_indices)
            wi_txt = ",".join(str(x) for x in wi) if wi else ""
            box_lbl = f"{node_id}\\n{{{wi_txt}}}" if wi_txt else node_id
            lines.append(f'  {node_id} [label="{box_lbl}"];')
            for key_ch in sorted(node.children.keys()):
                edge = node.children[key_ch]
                edge_lbl = _dot_escape_label(edge.label)
                cid = visit(edge.target)
                lines.append(f'  {node_id} -> {cid} [label="{edge_lbl}"];')
            return node_id

        visit(self.root)
        for n in iter_nodes(self.root):
            sl = n.suffix_link
            if sl is None or sl is n:
                continue
            a = node_ids[id(n)]
            b = node_ids[id(sl)]
            lines.append(
                f"  {a} -> {b} [style=dashed, color=\"#888888\", "
                f"arrowsize=0.6, constraint=false];"
            )
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_svg(self, path: str) -> None:
        dot = shutil.which("dot")
        if dot is None:
            raise FileNotFoundError(
                "Graphviz `dot` not found in PATH; install graphviz to use to_svg()"
            )
        proc = subprocess.run(
            [dot, "-Tsvg", "-o", path],
            input=self.to_dot(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"dot failed ({proc.returncode}): {err}")