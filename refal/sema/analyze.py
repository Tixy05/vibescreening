"""Program-level screening analysis entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from refal.parser.parser import build_ast_from_file, build_ast_from_string

from .screening import FunctionScreening, screen_function


@dataclass
class ProgramScreeningReport:
    functions: list[FunctionScreening] = field(default_factory=list)

    def screened_by_function(self) -> dict[str, list[int]]:
        return {f.name: f.screened_indices for f in self.functions}


def analyze_source(source: str) -> ProgramScreeningReport:
    program = build_ast_from_string(source)
    return ProgramScreeningReport(
        functions=[screen_function(defn) for defn in program.definitions]
    )


def analyze_file(path: str | Path) -> ProgramScreeningReport:
    program = build_ast_from_file(str(path))
    return ProgramScreeningReport(
        functions=[screen_function(defn) for defn in program.definitions]
    )


def _print_report(report: ProgramScreeningReport) -> None:
    for fn in report.functions:
        print(f"Function {fn.name}:")
        for rule in fn.rules:
            flags = []
            if rule.discarded:
                flags.append("discarded")
            elif rule.good:
                flags.append("good")
            else:
                flags.append("bad (no constraint)")
            if rule.approximated:
                flags.append("approx")
            status = "SCREENED" if rule.screened else "ok"
            print(
                f"  rule {rule.index}: {status} "
                f"({', '.join(flags)}) — {rule.reason}"
            )
        screened = fn.screened_indices
        if screened:
            print(f"  screened rules: {screened}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refal screening analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Path to a .ref source file")
    group.add_argument("--string", dest="source", help="Refal source text")
    args = parser.parse_args(argv)

    if args.source is not None:
        report = analyze_source(args.source)
    else:
        report = analyze_file(args.file)

    _print_report(report)


if __name__ == "__main__":
    main()
