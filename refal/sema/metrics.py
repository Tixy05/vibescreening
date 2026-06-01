"""Metrics for screening analysis logs."""

from __future__ import annotations

from dataclasses import dataclass

from extad import SLTL


@dataclass(frozen=True)
class SltlAntidictStats:
    prefixes: int
    factors: int
    suffixes: int
    sfw: int

    @property
    def total(self) -> int:
        return self.prefixes + self.factors + self.suffixes + self.sfw


def sltl_antidict_stats(sltl: SLTL) -> SltlAntidictStats:
    """Counts in normalized SLTL antidictionary (SFP, SFF, SFS, SFW)."""
    return SltlAntidictStats(
        prefixes=len(sltl.prefixes),
        factors=len(sltl.factors),
        suffixes=len(sltl.suffixes),
        sfw=len(sltl.sfw),
    )
