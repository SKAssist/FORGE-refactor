import ast
import inspect
import json
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from crosshair.core_and_libs import analyze_function
from z3 import *

# ===============================================
# LLM Integration Components
# ===============================================

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    # Add more as needed

class LLMConfig:
    provider: LLMProvider
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.2
    max_tokens: int = 2000
    api_url: Optional[str] = None

class LLMInterface:
    """Interface for interacting with LLM APIs"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self.config.provider == LLMProvider.OPENAI:
            try:
                import openai
                openai.api_key = self.config.api_key
                self.client = openai.OpenAI()
            except ImportError:
                raise ImportError("OpenAI Python package not found. Install with: pip install openai")
                
        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Anthropic Python package not found. Install with: pip install anthropic")
        
        elif self.config.provider == LLMProvider.OLLAMA:
            try:
                import requests
                self.client = None  # Ollama doesn't need a client object
                self.requests = requests
                if not self.config.api_url:
                    self.config.api_url = "http://localhost:11434/api/generate"
            except ImportError:
                raise ImportError("Requests Python package not found. Install with: pip install requests")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def generate_code(self, prompt: str) -> str:
        """Generate code using the configured LLM"""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
                
            elif self.config.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.config.provider == LLMProvider.OLLAMA:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                headers = {"Content-Type": "application/json"}
                response = self.requests.post(self.config.api_url, json=payload, headers=headers)
                response_json = response.json()
                return response_json["response"]
                
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""

    def analyze_verification_results(self, original_code: str, refactored_code: str, 
                                    verification_results: Dict) -> str:
        """Process verification results to provide insights and improvement suggestions"""
        
        prompt = f"""
        I need help analyzing the results of code equivalence verification.
        
        Original code:
        ```python
        {original_code}
        ```
        
        Refactored code:
        ```python
        {refactored_code}
        ```
        
        Verification results:
        {json.dumps(verification_results, indent=2)}
        
        Please analyze these results and provide:
        1. An explanation of why the verification failed (if it did)
        2. Specific suggestions for fixing the refactored code
        3. Any edge cases or assumptions that might need consideration
        
        Return your response in a clear, structured format.
        """
        
        return self.generate_code(prompt)

# ===============================================
# AST Analysis and Z3 Translation
# ===============================================

class ASTAnalyzer:
    """Advanced AST analysis for Python code"""
    
    @staticmethod
    def extract_function_signature(code: str) -> Tuple[str, List[str], str]:
        """Extract function name, parameters and return type hint if available"""
        tree = ast.parse(code)
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Code does not contain a function definition")
            
        func_def = tree.body[0]
        func_name = func_def.name
        
        # Extract parameter names
        params = []
        for arg in func_def.args.args:
            params.append(arg.arg)
            
        # Extract return type annotation if available
        return_type = ""
        if func_def.returns:
            return_type = ast.unparse(func_def.returns)
            
        return func_name, params, return_type
    
    @staticmethod
    def analyze_code_complexity(code: str) -> Dict[str, int]:
        """Calculate various complexity metrics for the code"""
        tree = ast.parse(code)
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.metrics

class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate code complexity metrics"""
    
    def __init__(self):
        self.metrics = {
            "cyclomatic_complexity": 1,  # Base complexity of 1
            "num_operations": 0,
            "max_nesting_depth": 0,
            "current_depth": 0,
        }
    
    def visit_If(self, node):
        self.metrics["cyclomatic_complexity"] += 1
        self.metrics["current_depth"] += 1
        self.metrics["max_nesting_depth"] = max(
            self.metrics["max_nesting_depth"], 
            self.metrics["current_depth"]
        )
        super().generic_visit(node)
        self.metrics["current_depth"] -= 1
    
    def visit_For(self, node):
        self.metrics["cyclomatic_complexity"] += 1
        self.metrics["current_depth"] += 1
        self.metrics["max_nesting_depth"] = max(
            self.metrics["max_nesting_depth"], 
            self.metrics["current_depth"]
        )
        super().generic_visit(node)
        self.metrics["current_depth"] -= 1
    
    def visit_While(self, node):
        self.metrics["cyclomatic_complexity"] += 1
        self.metrics["current_depth"] += 1
        self.metrics["max_nesting_depth"] = max(
            self.metrics["max_nesting_depth"], 
            self.metrics["current_depth"]
        )
        super().generic_visit(node)
        self.metrics["current_depth"] -= 1
    
    def visit_BinOp(self, node):
        self.metrics["num_operations"] += 1
        super().generic_visit(node)

class PythonToZ3Translator(ast.NodeVisitor):
    """Enhanced translator from Python AST to Z3 expressions"""
    
    def __init__(self):
        self.vars = {}
        self.solver = Solver()
        self.local_vars = {}
        self.functions = {}
        self.z3_types = {
            'int': Int,
            'float': Real,
            'bool': Bool,
        }
    
    def register_function(self, func_name, z3_func):
        """Register a Python function with its Z3 equivalent"""
        self.functions[func_name] = z3_func
    
    def translate(self, func_code, shared_vars=None, type_hints=None):
        """Translate a Python function to Z3 expressions"""
        tree = ast.parse(func_code)
        func_def = tree.body[0]
        
        # Process arguments
        for arg in func_def.args.args:
            arg_name = arg.arg
            
            # Use shared vars if provided
            if shared_vars and arg_name in shared_vars:
                self.vars[arg_name] = shared_vars[arg_name]
                continue
                
            # Use type hints if provided
            if type_hints and arg_name in type_hints:
                z3_type = self.z3_types.get(type_hints[arg_name], Int)
                self.vars[arg_name] = z3_type(arg_name)
            else:
                # Default to Int
                self.vars[arg_name] = Int(arg_name)
        
        # Process function body
        result = None
        for stmt in func_def.body:
            if isinstance(stmt, ast.Return):
                result = self.visit(stmt.value)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.local_vars[target.id] = self.visit(stmt.value)
            elif isinstance(stmt, ast.If):
                # Handle if statements (partially - simple cases only)
                condition = self.visit(stmt.test)
                then_body = None
                else_body = None
                
                # Process 'then' branch
                for then_stmt in stmt.body:
                    if isinstance(then_stmt, ast.Return):
                        then_body = self.visit(then_stmt.value)
                
                # Process 'else' branch if it exists
                if stmt.orelse:
                    for else_stmt in stmt.orelse:
                        if isinstance(else_stmt, ast.Return):
                            else_body = self.visit(else_stmt.value)
                
                if then_body is not None and else_body is not None:
                    result = If(condition, then_body, else_body)
        
        return result
    
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
        elif isinstance(node.op, ast.Mod):
            return left % right
        elif isinstance(node.op, ast.Pow):
            # Note: Z3 doesn't directly support power, this is for simple integer cases
            if isinstance(right, int) and right >= 0:
                result = 1
                for _ in range(right):
                    result = result * left
                return result
            else:
                raise NotImplementedError("Only simple integer powers supported")
        else:
            raise NotImplementedError(f"Operator {type(node.op)} not supported")
    
    def visit_BoolOp(self, node):
        values = [self.visit(val) for val in node.values]
        
        if isinstance(node.op, ast.And):
            result = values[0]
            for val in values[1:]:
                result = And(result, val)
            return result
        elif isinstance(node.op, ast.Or):
            result = values[0]
            for val in values[1:]:
                result = Or(result, val)
            return result
        else:
            raise NotImplementedError(f"Boolean operator {type(node.op)} not supported")
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        
        # Handle multiple comparisons (e.g., a < b < c)
        result = None
        for op, right_node in zip(node.ops, node.comparators):
            right = self.visit(right_node)
            
            if isinstance(op, ast.Eq):
                comp = left == right
            elif isinstance(op, ast.NotEq):
                comp = left != right
            elif isinstance(op, ast.Lt):
                comp = left < right
            elif isinstance(op, ast.LtE):
                comp = left <= right
            elif isinstance(op, ast.Gt):
                comp = left > right
            elif isinstance(op, ast.GtE):
                comp = left >= right
            else:
                raise NotImplementedError(f"Comparison operator {type(op)} not supported")
            
            if result is None:
                result = comp
            else:
                result = And(result, comp)
            
            # For chained comparisons, the left side becomes the right side
            left = right
            
        return result
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.Not):
            return Not(operand)
        else:
            raise NotImplementedError(f"Unary operator {type(node.op)} not supported")
    
    def visit_Name(self, node):
        if node.id in self.local_vars:
            return self.local_vars[node.id]
        elif node.id in self.vars:
            return self.vars[node.id]
        else:
            raise Exception(f"Variable {node.id} not defined")
    
    def visit_Constant(self, node):
        return node.value
    
    def visit_Num(self, node):  # For Python < 3.8 compatibility
        return node.n
    
    def visit_Call(self, node):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        
        if func_name in self.functions:
            args = [self.visit(arg) for arg in node.args]
            return self.functions[func_name](*args)
        else:
            raise NotImplementedError(f"Function call to {func_name} not supported")
    
    def visit_IfExp(self, node):
        # Handle ternary expressions: x if cond else y
        condition = self.visit(node.test)
        then_branch = self.visit(node.body)
        else_branch = self.visit(node.orelse)
        
        return If(condition, then_branch, else_branch)

    def visit_Subscript(self, node):
        # Simple array/list access - very limited support
        var_name = node.value.id if isinstance(node.value, ast.Name) else None
        
        if not var_name or var_name not in self.vars:
            raise NotImplementedError("Only simple variable subscripts supported")
            
        if isinstance(node.slice, ast.Index):  # Python < 3.9
            index = self.visit(node.slice.value)
        else:  # Python >= 3.9
            index = self.visit(node.slice)
            
        # This assumes the variable is a Z3 array
        return self.vars[var_name][index]

# ===============================================
# CrossHair Integration
# ===============================================

class CrossHairVerifier:
    """Enhanced integration with CrossHair for function verification"""
    
    @staticmethod
    def generate_property_function(func1, func2, param_names, additional_conditions=None):
        """Generate a property function that CrossHair can verify"""
        
        def property_func(*args):
            # Apply additional conditions if provided
            if additional_conditions:
                for condition in additional_conditions:
                    if not condition(*args):
                        return True  # Skip verification for inputs not meeting conditions
            
            return func1(*args) == func2(*args)
        
        # Set proper argument names for better CrossHair error messages
        property_func.__name__ = "verify_equivalence"
        
        # Create a new namespace for the exec
        namespace = {}
        
        # Create signature with proper parameter names
        param_str = ", ".join([f"{name}: Any" for name in param_names])
        exec(f"def property_wrapper({param_str}) -> bool:\n    return property_func({', '.join(param_names)})", 
            {"property_func": property_func, "Any": Any}, namespace)
        
        # Get the newly created function with proper signature
        property_wrapper = namespace["property_wrapper"]
        
        return property_wrapper
    
    @staticmethod
    def analyze_with_crosshair(property_func, max_iterations=100, timeout=10):
        """Run CrossHair analysis on the property function with customizable parameters"""
        # Check the parameters accepted by analyze_function
        import inspect
        params = inspect.signature(analyze_function).parameters
        
        # Build kwargs based on available parameters
        kwargs = {}
        if 'timeout' in params:
            kwargs['timeout'] = timeout
        if 'max_iterations' in params:
            kwargs['max_iterations'] = max_iterations
        # For newer versions of CrossHair that might use different parameter
        if 'max_iters' in params:
            kwargs['max_iters'] = max_iterations
        
        # Call analyze_function with only the parameters it accepts
        issues = analyze_function(property_func, **kwargs)
        
        results = []
        for issue in issues:
            # Extract relevant information from the issue
            results.append({
                "message": str(issue),
                "inputs": issue.conditions,
                "severity": issue.severity
            })
            
        return results

# ===============================================
# Z3 Integration
# ===============================================

class Z3Verifier:
    """Enhanced Z3-based verification for function equivalence"""
    
    @staticmethod
    def verify_equivalence(expr1, expr2, timeout_ms=5000):
        """Verify if two Z3 expressions are equivalent"""
        s = Solver()
        s.set("timeout", timeout_ms)
        s.add(expr1 != expr2)
        
        result = {
            "equivalent": False,
            "status": None,
            "counterexample": None,
            "timeout": False,
            "error": None
        }
        
        try:
            check_result = s.check()
            
            if check_result == sat:
                result["status"] = "sat"
                
                # Extract counterexample
                model = s.model()
                counterexample = {}
                for d in model.decls():
                    counterexample[d.name()] = model[d]
                
                result["counterexample"] = counterexample
                
            elif check_result == unsat:
                result["status"] = "unsat"
                result["equivalent"] = True
                
            else:  # unknown
                result["status"] = "unknown"
                result["timeout"] = True
                
        except Z3Exception as e:
            result["error"] = str(e)
            
        return result
    
    @staticmethod
    def verify_with_constraints(expr1, expr2, constraints, timeout_ms=5000):
        """Verify equivalence with additional constraints on the inputs"""
        s = Solver()
        s.set("timeout", timeout_ms)
        
        # Add constraints
        for constraint in constraints:
            s.add(constraint)
        
        # Check equivalence under constraints
        s.add(expr1 != expr2)
        
        result = {
            "equivalent": False,
            "status": None,
            "counterexample": None,
            "timeout": False,
            "error": None,
            "with_constraints": True
        }
        
        try:
            check_result = s.check()
            
            if check_result == sat:
                result["status"] = "sat"
                model = s.model()
                counterexample = {}
                for d in model.decls():
                    counterexample[d.name()] = model[d]
                result["counterexample"] = counterexample
                
            elif check_result == unsat:
                result["status"] = "unsat"
                result["equivalent"] = True
                
            else:
                result["status"] = "unknown"
                result["timeout"] = True
                
        except Z3Exception as e:
            result["error"] = str(e)
            
        return result

# ===============================================
# Core Refactoring and Verification Engine
# ===============================================

class RefactoringEngine:
    """Main engine for code refactoring and verification"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = LLMInterface(llm_config) if llm_config else None
    
    def generate_refactored_code(self, original_code: str, 
                                refactoring_strategy: str = "optimize_for_readability") -> str:
        """Generate refactored version of the code using LLM"""
        if not self.llm:
            raise ValueError("LLM not configured. Please provide LLM configuration.")
        
        # Analyze the original code to extract relevant information
        func_name, params, return_type = ASTAnalyzer.extract_function_signature(original_code)
        complexity_metrics = ASTAnalyzer.analyze_code_complexity(original_code)
        
        # Build a detailed prompt for the LLM
        prompt = f"""
        Refactor the following Python function using the strategy: {refactoring_strategy}
        
        Original function:
        ```python
        {original_code}
        ```
        
        Function complexity metrics:
        - Cyclomatic complexity: {complexity_metrics["cyclomatic_complexity"]}
        - Number of operations: {complexity_metrics["num_operations"]}
        - Maximum nesting depth: {complexity_metrics["max_nesting_depth"]}
        
        I need you to refactor this function while preserving its exact behavior.
        The refactored function should:
        1. Have the same input-output behavior for all valid inputs
        2. Have improved {refactoring_strategy} characteristics
        3. Be functionally equivalent to the original
        
        Improve clarity, efficiency, and maintainability:
            - Remove redundancy
            - Simplify logic
            - Decompose complex logic into helpers
            - Improve naming
            - Ensure functionality is preserved
        Return only the Python code for the refactored function without any explanation.
        """
        
        refactored_code = self.llm.generate_code(prompt)
        
        # Clean up the code to remove any markdown code blocks
        refactored_code = re.sub(r'```python|```', '', refactored_code).strip()
        
        return refactored_code
    
    def generate_verification_constraints(self, original_code: str, refactored_code: str) -> List[str]:
        """Generate additional verification constraints using LLM"""
        if not self.llm:
            raise ValueError("LLM not configured. Please provide LLM configuration.")
        
        prompt = f"""
        Analyze the following functions and generate verification constraints to ensure their equivalence.
        
        Original function:
        ```python
        {original_code}
        ```
        
        Refactored function:
        ```python
        {refactored_code}
        ```
        
        Please generate 2-4 logical constraints that would be important to verify for these functions.
        Express each constraint in natural language. Focus on edge cases, input domain restrictions, or invariants.
        
        Format your response as a JSON list of strings, each representing a constraint.
        """
        
        constraints_text = self.llm.generate_code(prompt)
        
        # Extract JSON list from response
        try:
            # Find JSON pattern in the text
            json_match = re.search(r'\[.*\]', constraints_text, re.DOTALL)
            if json_match:
                constraints_text = json_match.group(0)
            
            constraints = json.loads(constraints_text)
            return constraints
        except Exception as e:
            print(f"Error parsing constraints: {e}")
            return []
    
    def verify_refactoring(self, original_code: str, refactored_code: str) -> Dict:
        """Verify the equivalence of original and refactored code"""
        results = {
            "z3_verification": None,
            "crosshair_verification": None,
            "equivalent": False,
            "error": None
        }
        
        try:
            # Extract function information
            original_func_name, original_params, _ = ASTAnalyzer.extract_function_signature(original_code)
            refactored_func_name, refactored_params, _ = ASTAnalyzer.extract_function_signature(refactored_code)
            
            # Execute both function definitions
            original_func = get_function_from_code(original_code, original_func_name)
            refactored_func = get_function_from_code(refactored_code, refactored_func_name)
            
            # Z3 verification
            translator1 = PythonToZ3Translator()
            translator2 = PythonToZ3Translator()
            
            # Create shared symbolic variables
            shared_vars = {}
            for param in original_params:
                shared_vars[param] = Int(param)
            
            # Create parameter mapping between original and refactored
            param_mapping = {}
            if len(original_params) == len(refactored_params):
                param_mapping = dict(zip(refactored_params, [shared_vars[p] for p in original_params]))
            
            # Translate functions to Z3 expressions
            expr1 = translator1.translate(original_code, shared_vars)
            expr2 = translator2.translate(refactored_code, param_mapping)
            
            # Verify with Z3
            z3_result = Z3Verifier.verify_equivalence(expr1, expr2)
            results["z3_verification"] = z3_result
            
            # Generate additional constraints if LLM is available
            constraints = []
            if self.llm:
                constraint_texts = self.generate_verification_constraints(original_code, refactored_code)
                if constraint_texts:
                    print("Generated constraints:")
                    for c in constraint_texts:
                        print(f"- {c}")
            
            # CrossHair verification
            property_func = CrossHairVerifier.generate_property_function(
                original_func, 
                refactored_func,
                original_params
            )
            
            crosshair_results = CrossHairVerifier.analyze_with_crosshair(property_func)
            results["crosshair_verification"] = crosshair_results
            
            # Determine overall equivalence
            results["equivalent"] = (
                z3_result["equivalent"] and 
                len(crosshair_results) == 0
            )
            
        except Exception as e:
            results["error"] = str(e)
            traceback.print_exc()
            
        return results
    
    def refactor_with_feedback_loop(self, original_code: str, 
                                  refactoring_strategy: str = "optimize_for_readability",
                                  max_iterations: int = 3) -> Dict:
        """Refactor code and verify in a feedback loop until equivalence is achieved"""
        if not self.llm:
            raise ValueError("LLM not configured. Please provide LLM configuration.")
        
        iterations = []
        refactored_code = None
        final_verification = None
        equivalent = False
        
        for i in range(max_iterations):
            print(f"\nIteration {i+1}/{max_iterations}")
            
            # Generate refactored code (or use feedback to improve previous attempt)
            if i == 0 or refactored_code is None:
                refactored_code = self.generate_refactored_code(original_code, refactoring_strategy)
            else:
                # Use verification results to improve the refactoring
                analysis = self.llm.analyze_verification_results(
                    original_code, 
                    refactored_code, 
                    final_verification
                )
                
                print(f"Feedback from LLM:\n{analysis}\n")
                
                # Generate improved refactoring based on feedback
                prompt = f"""
                Original code:
                ```python
                {original_code}
                ```
                
                Previous refactored code with issues:
                ```python
                {refactored_code}
                ```
                
                Analysis of verification issues:
                {analysis}
                
                Please provide a corrected version of the refactored code that fixes these issues.
                Return only the Python code without any explanation.
                """
                
                refactored_code = self.llm.generate_code(prompt)
                refactored_code = re.sub(r'```python|```', '', refactored_code).strip()
            
            print(f"Generated refactored code:")
            print(refactored_code)
            
            # Verify the refactoring
            verification_results = self.verify_refactoring(original_code, refactored_code)
            final_verification = verification_results
            equivalent = verification_results["equivalent"]
            
            # Store iteration results
            iterations.append({
                "iteration": i+1,
                "refactored_code": refactored_code,
                "verification_results": verification_results
            })
            
            # Check if we've achieved equivalence
            if equivalent:
                print(f"Equivalence achieved at iteration {i+1}!")
                break
            else:
                print(f"Verification failed. Attempting another iteration.")
        
        # Return final results
        return {
            "original_code": original_code,
            "final_refactored_code": refactored_code,
            "equivalent": equivalent,
            "iterations": iterations,
            "iterations_performed": len(iterations),
            "max_iterations": max_iterations,
            "refactoring_strategy": refactoring_strategy
        }

# ===============================================
# Helper Functions
# ===============================================

def get_function_from_code(code: str, func_name: str):
    """Execute function code string and extract function by name"""
    loc = {}
    exec(code, {}, loc)
    return loc[func_name]

def extract_code_block(text: str) -> str:
    """Extract code from markdown-style code blocks"""
    pattern = r'```(?:python)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ===============================================
# Usage and Examples
# ===============================================

def simple_example():
    """Simple example using only Z3 and CrossHair without LLM integration"""
    func1 = """
def add(x, y):
    return x + y
"""
    func2 = """
def add_refactored(a, b):
    return a + b
"""

    translator1 = PythonToZ3Translator()
    translator2 = PythonToZ3Translator()
    shared_vars = {'x': Int('x'), 'y': Int('y')}

    expr1 = translator1.translate(func1, shared_vars)
    expr2 = translator2.translate(func2, {'a': shared_vars['x'], 'b': shared_vars['y']})

    # Z3 verification
    z3_result = Z3Verifier.verify_equivalence(expr1, expr2)
    print("Z3 Verification Result:", z3_result)
    
    # CrossHair verification
    f1 = get_function_from_code(func1, "add")
    f2 = get_function_from_code(func2, "add_refactored")
    
    property_func = CrossHairVerifier.generate_property_function(
        f1, f2, ["x", "y"]
    )
    
    crosshair_results = CrossHairVerifier.analyze_with_crosshair(property_func)
    print("\nCrossHair Verification Results:", crosshair_results)

def complex_example_with_llm():
    """Example with LLM integration for refactoring and feedback loop"""
    # Configure LLM using Ollama
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama4",  # or whatever model you're using with Ollama
        temperature=0.2,
        api_url="http://localhost:11434/api/generate"  # adjust this URL if needed
    )
    
    engine = RefactoringEngine(llm_config)
    
    # Example function to refactor
    original_code = """
def calculate_statistics(numbers):
    total = 0
    for num in numbers:
        total += num
    mean = total / len(numbers)
    
    variance = 0
    for num in numbers:
        variance += (num - mean) ** 2
    variance = variance / len(numbers)
    
    std_dev = variance ** 0.5
    
    return mean, variance, std_dev
"""
    
    results = engine.refactor_with_feedback_loop(
        original_code=original_code,
        refactoring_strategy="optimize_for_performance",
        max_iterations=3
    )
    
    print("\nFinal Results:")
    print(f"Equivalent: {results['equivalent']}")
    print(f"Iterations performed: {results['iterations_performed']}")
    print("\nFinal refactored code:")
    print(results["final_refactored_code"])

# Using your existing query_llama3 function directly
def integrate_with_existing_llm_function(original_code: str, max_iterations: int = 3):
    """Using the existing query_llama3 function with our refactoring system"""
    import requests  # ensure requests is imported
    
    # Define the Ollama API URL (update as needed)
    OLLAMA_API_URL = "http://localhost:11434/api/generate" 
    
    def query_llama3(prompt: str) -> str:
        payload = {
            "model": "llama4",
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
        response_json = response.json()
        
        return response_json["response"]
    
    # Create a simple wrapper class that uses your existing function
    class LlamaAdapter:
        def generate_code(self, prompt: str) -> str:
            return query_llama3(prompt)
        
        def analyze_verification_results(self, original_code: str, refactored_code: str, 
                                        verification_results: Dict) -> str:
            prompt = f"""
            I need help analyzing the results of code equivalence verification.
            
            Original code:
            ```python
            {original_code}
            ```
            
            Refactored code:
            ```python
            {refactored_code}
            ```
            
            Verification results:
            {json.dumps(verification_results, indent=2)}
            
            Please analyze these results and provide:
            1. An explanation of why the verification failed (if it did)
            2. Specific suggestions for fixing the refactored code
            3. Any edge cases or assumptions that might need consideration
            
            Return your response in a clear, structured format.
            """
            
            return query_llama3(prompt)
    
    # Create a modified RefactoringEngine that uses our adapter
    class CustomRefactoringEngine(RefactoringEngine):
        def __init__(self):
            self.llm = LlamaAdapter()
    
    # Use the custom engine
    engine = CustomRefactoringEngine()
    results = engine.refactor_with_feedback_loop(
        original_code=original_code,
        refactoring_strategy="optimize_for_readability",
        max_iterations=max_iterations
    )
    
    return results
    