import ast
from z3 import *
from crosshair.core_and_libs import analyze_function

class PythonToZ3Translator(ast.NodeVisitor):
    def __init__(self):
        self.vars = {}
        self.solver = Solver()
    
    def translate(self, func_code, shared_vars=None):
        tree = ast.parse(func_code)
        func_def = tree.body[0]
        for arg in func_def.args.args:
            if shared_vars and arg.arg in shared_vars:
                self.vars[arg.arg] = shared_vars[arg.arg]
            else:
                self.vars[arg.arg] = Int(arg.arg)
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
    
    def visit_Num(self, node):
        return node.n
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        else:
            raise NotImplementedError(f"Unary operator {type(node.op)} not supported.")

# Helper to exec function code string and extract function by name
def get_function_from_code(code: str, func_name: str):
    loc = {}
    exec(code, {}, loc)
    return loc[func_name]

# CrossHair equivalence check using analyze_function
def crosshair_equiv_check(func1_code, func2_code, func1_name, func2_name):
    f1 = get_function_from_code(func1_code, func1_name)
    f2 = get_function_from_code(func2_code, func2_name)

    # We create a wrapper function that asserts equality of outputs
    def equiv_func(x: int, y: int) -> bool:
        return f1(x, y) == f2(x, y)

    # Run CrossHair analysis on the equivalence function
    issues = analyze_function(equiv_func)
    if issues:
        print("CrossHair found counterexamples:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("CrossHair found no counterexamples, functions are likely equivalent.")
        return True

# Usage example integrating both Z3 and CrossHair checks
def example():
    func1 = """
def add(x, y):
    return x * y
"""
    func2 = """
def add_refactor(a, b):
    return b * a
"""

    translator1 = PythonToZ3Translator()
    translator2 = PythonToZ3Translator()
    shared_vars = {'x': Int('x'), 'y': Int('y')}

    expr1 = translator1.translate(func1, shared_vars)
    expr2 = translator2.translate(func2, {'a': shared_vars['x'], 'b': shared_vars['y']})

    s = Solver()
    s.add(expr1 != expr2)
    print("Checking equivalence with Z3...")
    if s.check() == sat:
        print("Functions NOT equivalent according to Z3.")
        print("Counterexample:", s.model())
    else:
        print("Functions are equivalent according to Z3.")
    
    print("\nChecking equivalence with CrossHair...")
    crosshair_equiv_check(func1, func2, "add", "add_refactor")

if __name__ == "__main__":
    example()


# def foo(x,y):
#     if x :
#         pass
#     if x:
#         print(x)
#     if x != 0:
#         if x > 0 and x < 0:
#             return
        
# foo(x,y)
# -> foo(x)
    