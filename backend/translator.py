import ast
from z3 import *

class PythonToZ3Translator(ast.NodeVisitor):
    def __init__(self):
        self.vars = {}
        self.local_vars = {}

    def translate(self, code, shared_vars=None):
        tree = ast.parse(code)
        func_def = tree.body[0]
        for arg in func_def.args.args:
            name = arg.arg
            self.vars[name] = shared_vars[name] if shared_vars and name in shared_vars else Int(name)
        for stmt in func_def.body:
            if isinstance(stmt, ast.Return):
                return self.visit(stmt.value)
        return None

    def visit_Name(self, node): return self.vars.get(node.id) or self.local_vars.get(node.id)
    def visit_BinOp(self, node):
        l, r = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.Div): return l / r
        raise NotImplementedError
