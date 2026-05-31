from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Alternation:
    alternatives: list[Letter]

    def __str__(self) -> str:
        return "[" + " ".join(str(a) for a in self.alternatives) + "]"


@dataclass
class AnyStarred:
    def __str__(self) -> str:
        return ".*"


@dataclass
class AnyLetter:
    def __str__(self) -> str:
        return "."


@dataclass
class SimpleRegex:
    """Regex that are ( "(a | b | ...)" | ".*" | ".")*"""
    concats: list[Alternation | AnyStarred | AnyLetter]
    alphabet: Alphabet

    def __str__(self) -> str:
        return " ".join(str(c) for c in self.concats)

    # def from_flat_abstract_pattern(self, ap: AbstractPattern) -> "SimpleRegex":
    #     from morphism import AbstractEVar, AbstractParenthesizedPattern, AbstractTvar

    #     result: list[Alternation | AnyStarred | AnyLetter] = []
    #     for elem in ap.children:
    #         if isinstance(elem, AbstractParenthesizedPattern):
    #             raise ValueError("No parenthesis allowed here!")
    #         elif isinstance(elem, AbstractEVar):
    #             result.append(AnyStarred())
    #         elif isinstance(elem, AbstractTvar):
    #             result.append(Alternation(list(self.alphabet.letters)))
    #             result.append(AnyStarred())
    #     return SimpleRegex(result, self.alphabet)


@dataclass(frozen=True)
class Letter:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Alphabet:
    letters: set[Letter]