import itertools

from suffix_automaton import SuffixAutomaton
from completed_sa import CompletedAutomaton
from suffix_tree import SuffixTree


def build_adjacency(ca: CompletedAutomaton) -> dict[int, set[int]]:
    """Directed adjacency list from the CSA, excluding self-loops."""
    adj: dict[int, set[int]] = {}
    for st in ca.states:
        neighbors: set[int] = set()
        for targets in st.trans.values():
            for tgt in targets:
                if tgt != st.id:
                    neighbors.add(tgt)
        adj[st.id] = neighbors
    return adj


def _scc_of(adj: dict[int, set[int]], nodes: set[int], s: int) -> set[int] | None:
    """SCC containing *s* in the subgraph induced by *nodes*,
    or None if it has fewer than 2 vertices."""
    forward: set[int] = set()
    stack = [s]
    while stack:
        v = stack.pop()
        if v in forward:
            continue
        forward.add(v)
        for w in adj.get(v, set()):
            if w in nodes and w not in forward:
                stack.append(w)

    rev: dict[int, set[int]] = {v: set() for v in nodes}
    for v in nodes:
        for w in adj.get(v, set()):
            if w in nodes:
                rev[w].add(v)

    backward: set[int] = set()
    stack = [s]
    while stack:
        v = stack.pop()
        if v in backward:
            continue
        backward.add(v)
        for w in rev.get(v, set()):
            if w not in backward:
                stack.append(w)

    scc = forward & backward
    return scc if len(scc) >= 2 else None


def find_all_cycles(ca: CompletedAutomaton) -> list[list[int]]:
    """Find all elementary cycles in the CSA (excluding self-loops).

    Uses Johnson's algorithm. Each cycle is returned as a list of state ids;
    the last state implicitly transitions back to the first.
    """
    adj = build_adjacency(ca)
    nodes = sorted(adj.keys())
    cycles: list[list[int]] = []

    for start_idx in range(len(nodes)):
        s = nodes[start_idx]
        subgraph_nodes = set(nodes[start_idx:])

        scc_or_none = _scc_of(adj, subgraph_nodes, s)
        if scc_or_none is None:
            continue
        scc: set[int] = scc_or_none

        blocked: set[int] = set()
        block_map: dict[int, set[int]] = {v: set() for v in scc}

        def unblock(u: int) -> None:
            blocked.discard(u)
            for w in list(block_map[u]):
                block_map[u].discard(w)
                if w in blocked:
                    unblock(w)

        def circuit(v: int, path: list[int]) -> bool:
            found = False
            path.append(v)
            blocked.add(v)

            for w in adj.get(v, set()):
                if w not in scc:
                    continue
                if w == s:
                    cycles.append(list(path))
                    found = True
                elif w not in blocked:
                    if circuit(w, path):
                        found = True

            if found:
                unblock(v)
            else:
                for w in adj.get(v, set()):
                    if w in scc:
                        block_map[w].add(v)

            path.pop()
            return found

        circuit(s, [])

    return cycles


def build_edge_labels(ca: CompletedAutomaton) -> dict[tuple[int, int], set[str]]:
    """Map every (src, tgt) pair to the set of characters that label it."""
    labels: dict[tuple[int, int], set[str]] = {}
    for st in ca.states:
        for ch, targets in st.trans.items():
            for tgt in targets:
                if tgt != st.id:
                    labels.setdefault((st.id, tgt), set()).add(ch)
    return labels


def cycle_labels(
    cycle: list[int],
    edge_labels: dict[tuple[int, int], set[str]],
) -> list[tuple[int, int, set[str]]]:
    """Return labelled edges for a cycle.

    Each element is ``(src, tgt, {chars})`` for consecutive states in the
    cycle (including the closing edge from the last state back to the first).
    """
    result: list[tuple[int, int, set[str]]] = []
    for i in range(len(cycle)):
        src = cycle[i]
        tgt = cycle[(i + 1) % len(cycle)]
        result.append((src, tgt, edge_labels.get((src, tgt), set())))
    return result


def print_cycles(
    cycles: list[list[int]],
    ca: CompletedAutomaton,
) -> None:
    """Pretty-print every cycle with its edge labels."""
    edge_labels = build_edge_labels(ca)
    print(f"Found {len(cycles)} elementary cycle(s) (excluding self-loops):")
    for i, cyc in enumerate(cycles, 1):
        labelled = cycle_labels(cyc, edge_labels)
        parts: list[str] = []
        for src, tgt, chars in labelled:
            lbl = ",".join(sorted(chars)) if chars else "?"
            parts.append(f"{src} --{lbl}--> {tgt}")
        print(f"  {i}. {' | '.join(parts)}")


def reaches_leaf(tree: SuffixTree, word: str) -> bool:
    """Check whether *word* exactly matches a stored suffix in *tree*.

    Returns True iff the traversal consumes every character of *word* and
    the only remaining part of the current edge (or the only child of the
    landing node) is the end-marker leading to a leaf.
    """
    node = tree.root
    i, n = 0, len(word)
    while i < n:
        c = word[i]
        if c not in node.children:
            return False
        edge = node.children[c]
        label = edge.label
        j = 0
        while j < len(label) and i < n:
            if label[j] != word[i]:
                return False
            j += 1
            i += 1
        if j < len(label):
            return label[j:] == tree.end_marker
        node = edge.target
    em = tree.end_marker
    if em in node.children:
        return not node.children[em].target.children
    return False


def analyze_cycles(infixes: list[str], suffixes: list[str]) -> None:
    """Build a CSA from *infixes*, a suffix tree from reversed *suffixes*,
    then verify that no cyclic permutation of any cycle-word (reversed)
    reaches a leaf in the suffix tree."""
    alphabet: set[str] = {ch for w in infixes for ch in w}

    sa = SuffixAutomaton()
    for w in infixes:
        sa.add_word(w)
    ca = CompletedAutomaton(sa, alphabet)

    st = SuffixTree()
    for s in suffixes:
        st.add(s[::-1])

    cycles = find_all_cycles(ca)
    edge_labels = build_edge_labels(ca)

    print(f"Infixes    : {infixes}")
    print(f"Suffixes   : {suffixes}")
    print(f"Alphabet   : {sorted(alphabet)}")
    print(f"CSA states : {ca.num_states}")
    print(f"Cycles     : {len(cycles)}")
    print()

    any_violation = False
    for cyc_idx, cyc in enumerate(cycles, 1):
        labelled = cycle_labels(cyc, edge_labels)
        label_sets = [sorted(chars) for _, _, chars in labelled]

        for word_tuple in itertools.product(*label_sets):
            word = "".join(word_tuple)
            k = len(word)
            for rot in range(k):
                perm = word[rot:] + word[:rot]
                rev = perm[::-1]
                if reaches_leaf(st, rev):
                    any_violation = True
                    cyc_str = " -> ".join(str(s) for s in cyc) + f" -> {cyc[0]}"
                    print(
                        f"  VIOLATION  cycle {cyc_idx} [{cyc_str}]  "
                        f"word={word!r}  perm={perm!r}  rev={rev!r}"
                    )

    if not any_violation:
        print("  No violations: no reversed cyclic permutation reaches a leaf.")


if __name__ == "__main__":
    # infixes = ["caa", "cac", "abb", "aba", "bcc", "bcb"]
    # suffixes = ["abc", "bca", "cab"]
    infixes = ["ba", "bc", "abb", "cbb"]
    suffixes = ["ab", "cb"]

    analyze_cycles(infixes, suffixes)