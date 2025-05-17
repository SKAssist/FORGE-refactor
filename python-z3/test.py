import ast
from z3 import *

class PythonToZ3Translator(ast.NodeVisitor):
    def __init__(self):
        self.vars = {}
        self.solver = Solver()
    
    def translate(self, func_code):
        tree = ast.parse(func_code)
        # Assume single function, single return
        func_def = tree.body[0]
        # Map arguments to Z3 Int variables
        for arg in func_def.args.args:
            self.vars[arg.arg] = Int(arg.arg)
        # Translate function body (assume single return)
        return_stmt = func_def.body[-1]
        if not isinstance(return_stmt, ast.Return):
            raise NotImplementedError("Only functions with a return statement supported.")
        z3_expr = self.visit(return_stmt.value)
        return z3_expr
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        else:
            raise NotImplementedError(f"Operator {type(node.op)} not supported.")
    
    def visit_Name(self, node):
        if node.id in self.vars:
            return self.vars[node.id]
        else:
            raise Exception(f"Variable {node.id} not defined.")
    
    def visit_Constant(self, node):
        return node.value
    
    def visit_Num(self, node):  # for python <3.8
        return node.n
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        else:
            raise NotImplementedError(f"Unary operator {type(node.op)} not supported.")

# Usage example
def example():
    func1 = """
def add(x, y):
    return x + y
"""
    func2 = """
def add_refactor(x, y):
    return y + x
"""
    translator1 = PythonToZ3Translator()
    expr1 = translator1.translate(func1)

    translator2 = PythonToZ3Translator()
    expr2 = translator2.translate(func2)

    s = Solver()
    x, y = Ints('x y')
    # Check equivalence: expr1 == expr2 for all x,y
    s.add(expr1 != expr2)
    print("Checking equivalence...")
    if s.check() == sat:
        print("Functions NOT equivalent.")
        print("Counterexample:", s.model())
    else:
        print("Functions are equivalent.")

if __name__ == "__main__":
    example()
