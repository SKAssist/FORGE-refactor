import ast

class ASTAnalyzer:
    @staticmethod
    def extract_function_signature(code: str):
        tree = ast.parse(code)
        func = tree.body[0]
        name = func.name
        args = [arg.arg for arg in func.args.args]
        returns = ast.unparse(func.returns) if func.returns else None
        return name, args, returns

    @staticmethod
    def analyze_code_complexity(code: str):
        tree = ast.parse(code)
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.metrics

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.metrics = {
            "cyclomatic_complexity": 1,
            "num_operations": 0,
            "max_nesting_depth": 0,
            "current_depth": 0
        }

    def visit_If(self, node):
        self.metrics["cyclomatic_complexity"] += 1
        self.metrics["current_depth"] += 1
        self.metrics["max_nesting_depth"] = max(self.metrics["max_nesting_depth"], self.metrics["current_depth"])
        self.generic_visit(node)
        self.metrics["current_depth"] -= 1

    def visit_For(self, node): self.visit_If(node)
    def visit_While(self, node): self.visit_If(node)
    def visit_BinOp(self, node):
        self.metrics["num_operations"] += 1
        self.generic_visit(node)
