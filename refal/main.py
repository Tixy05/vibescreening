from refal.parser.parser import build_ast_from_file
from refal.sema.abstractizer import *
from os import listdir

if __name__ == "__main__":
    program = build_ast_from_file("refal/test_programs/tiny.ref")
    print(program.pretty())