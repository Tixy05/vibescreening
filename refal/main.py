from refal.parser.parser import build_ast_from_file
from refal.sema.abstractizer import *
# from os import listdir

if __name__ == "__main__":
    rewrite_map: dict[AbstractPatternElement, AbstractPatternElement] = {
        AbstractEVar(): AbstractedRegexedElement(
            Alternation([Letter("s"), Letter("b")])
            )
    }

    program = build_ast_from_file("refal/test_programs/tiny.ref")
    functions = program.definitions
    for f in functions:
        print(f)
        for rule in f.rules:
            p = rule.pattern
            ap = AbstractPattern.from_concrete(p)
            rewritten = rewrite(ap, rewrite_map)
            print(rewritten)
