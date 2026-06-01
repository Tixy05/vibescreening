"""Program-level screening analysis entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from refal.parser.parser import build_ast_from_file, build_ast_from_string

from .file_log import log_root_for_source, write_function_log
from .screening import FunctionScreening, screen_function
from .trace_log import IncrementalTrace, trace_scope


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


def analyze_file(
    path: str | Path,
    *,
    log_dir: Path | None = Path("log"),
) -> ProgramScreeningReport:
    source = Path(path)
    program = build_ast_from_file(str(source))
    write_logs = log_dir is not None
    log_root = log_root_for_source(source, log_base=log_dir) if write_logs else None

    functions: list[FunctionScreening] = []
    program_trace: IncrementalTrace | None = None
    if write_logs and log_root is not None:
        program_trace = IncrementalTrace(log_root / "trace.log")
        program_trace.step(f"analyze {source.name}: {len(program.definitions)} function(s)")

    try:
        for defn in program.definitions:
            func_dir = log_root / defn.name if log_root is not None else None
            fn_trace = (
                IncrementalTrace(func_dir / "trace.log") if func_dir is not None else None
            )
            if program_trace is not None:
                program_trace.step(f"function {defn.name} start")
            with trace_scope(fn_trace):
                fn = screen_function(defn, collect_log=write_logs)
            if write_logs and fn.log is not None and func_dir is not None:
                write_function_log(func_dir, fn.log)
            if program_trace is not None:
                program_trace.step(f"function {defn.name} done")
            functions.append(fn)
    finally:
        if program_trace is not None:
            program_trace.step("analyze finished")
            program_trace.close()

    return ProgramScreeningReport(functions=functions)


def _print_report(report: ProgramScreeningReport) -> None:
    for fn in report.functions:
        print(f"Function {fn.name}:")
        for rule in fn.rules:
            flags: list[str] = []
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
        report = analyze_file(args.file, log_dir=Path("log"))

    _print_report(report)


if __name__ == "__main__":
    main()
