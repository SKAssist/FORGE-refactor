import ast
import inspect
import json
import re
import subprocess
import time
import sys
import traceback
import json
import sys
import math
import requests
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("verification_system")

from crosshair.core_and_libs import analyze_function
from z3 import *
OLLAMA_API_URL = "http://localhost:11434/api/generate"


from mistralai import Mistral

api_key = "shZl9BS91mQ08w7NX4FpGlifFyiMH5fj"
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def call_mistral(prompt):
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content

def query_llama3(prompt: str) -> str:
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
    response_json = response.json()
    
    return response_json["response"]

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
       pass
    
    
    
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
        # return call_mistral(prompt)
        
        return query_llama3(prompt)

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
    @staticmethod
    def extract_all_functions(code: str) -> Dict[str, ast.FunctionDef]:
        """Extract all function definitions from code"""
        tree = ast.parse(code)
        functions = {}
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
                
        return functions

    @staticmethod
    def detect_entry_point(functions: Dict[str, ast.FunctionDef], original_func_name: str) -> str:
        """Determine which function is the main entry point"""
        # If original function name exists in refactored code, that's our entry point
        if original_func_name in functions:
            return original_func_name
            
        # Otherwise, heuristic: the entry point is likely the function that isn't called by others
        called_functions = set()
        for func_name, func_def in functions.items():
            visitor = FunctionCallVisitor()
            visitor.visit(func_def)
            called_functions.update(visitor.called_functions)
            
        # Functions that aren't called by other functions
        candidate_entry_points = set(functions.keys()) - called_functions
        
        if len(candidate_entry_points) == 1:
            return next(iter(candidate_entry_points))
        elif len(candidate_entry_points) > 1:
            # If multiple candidates, return the one with the most similar signature to original
            # For simplicity, return the first one
            return next(iter(candidate_entry_points))
        else:
            # All functions are called somewhere, might be recursive
            # Default to the function with the same name as original, or the first one
            return original_func_name if original_func_name in functions else next(iter(functions.keys()))

class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to find function calls"""
    
    def __init__(self):
        self.called_functions = set()
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
        self.generic_visit(node)

class RuntimeEquivalenceTester:
    """Test function equivalence through runtime execution"""
    
    @staticmethod
    def test_equivalence(original_code: str, refactored_code: str, num_tests: int = 100) -> Dict:
        """Test equivalence by running both functions with the same inputs"""
        # Extract function information
        try:
            original_functions = ASTAnalyzer.extract_all_functions(original_code)
            refactored_functions = ASTAnalyzer.extract_all_functions(refactored_code)
            
            original_func_name = next(iter(original_functions.keys()))
            refactored_entry_point = ASTAnalyzer.detect_entry_point(
                refactored_functions, original_func_name)
            
            # Extract parameter info
            original_params = [arg.arg for arg in original_functions[original_func_name].args.args]
            
            # Dynamically execute the code to get actual function objects
            original_namespace = {}
            refactored_namespace = {}
            
            exec(original_code, original_namespace)
            exec(refactored_code, refactored_namespace)
            
            original_func = original_namespace[original_func_name]
            refactored_func = refactored_namespace[refactored_entry_point]
            
            # Generate test cases
            test_cases = RuntimeEquivalenceTester.generate_test_cases(original_params, num_tests)
            
            # Run tests
            results = {
                "equivalent": True,
                "tests_run": num_tests,
                "failures": []
            }
            
            for test_case in test_cases:
                try:
                    original_result = original_func(*test_case)
                    refactored_result = refactored_func(*test_case)
                    
                    if original_result != refactored_result:
                        results["equivalent"] = False
                        results["failures"].append({
                            "inputs": test_case,
                            "original_output": str(original_result),
                            "refactored_output": str(refactored_result)
                        })
                except Exception as e:
                    results["equivalent"] = False
                    results["failures"].append({
                        "inputs": test_case,
                        "error": str(e)
                    })
            
            return results
        except Exception as e:
            return {
                "equivalent": False,
                "error": str(e),
                "tests_run": 0,
                "failures": []
            }
    
    @staticmethod
    def generate_test_cases(param_names, num_tests):
        """Generate test cases for the given parameters"""
        import random
        
        test_cases = []
        for _ in range(num_tests):
            # Basic test cases for common parameter types
            test_case = []
            for param in param_names:
                # Simple heuristic based on parameter name
                if param.lower() in ('n', 'num', 'number', 'count', 'size', 'length'):
                    # Likely an integer
                    test_case.append(random.randint(-100, 100))
                elif param.lower() in ('x', 'y', 'z', 'value', 'val'):
                    # Could be integer or float
                    if random.random() < 0.5:
                        test_case.append(random.randint(-100, 100))
                    else:
                        test_case.append(random.uniform(-100.0, 100.0))
                elif param.lower() in ('list', 'array', 'items', 'elements', 'numbers', 'values', 'data'):
                    # Likely a list
                    list_length = random.randint(0, 10)
                    test_case.append([random.randint(-100, 100) for _ in range(list_length)])
                elif param.lower() in ('dict', 'map', 'hash', 'table'):
                    # Likely a dictionary
                    dict_size = random.randint(0, 5)
                    test_case.append({f"key{i}": random.randint(-100, 100) for i in range(dict_size)})
                elif param.lower() in ('flag', 'check', 'is_', 'has_', 'enable', 'disable'):
                    # Likely a boolean
                    test_case.append(random.choice([True, False]))
                elif param.lower() in ('name', 'path', 'file', 'str', 'string'):
                    # Likely a string
                    choices = ["", "test", "hello", "world", "python", "a" * random.randint(1, 10)]
                    test_case.append(random.choice(choices))
                else:
                    # Default to integer
                    test_case.append(random.randint(-100, 100))
            
            test_cases.append(test_case)
        
        return test_cases
    
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
        self.llm = LLMInterface(None)
    
    def generate_refactored_code(self, original_code: str, refactoring_strategy: str = "optimize_for_readability") -> str:
        """Generate refactored version of the code using Llama"""
        
        # Extract function information
        try:
            func_name, params, return_type = ASTAnalyzer.extract_function_signature(original_code)
            complexity_metrics = ASTAnalyzer.analyze_code_complexity(original_code)
        except Exception as e:
            print(f"Error analyzing code: {e}")
            complexity_metrics = {"cyclomatic_complexity": "unknown", "num_operations": "unknown", "max_nesting_depth": "unknown"}
        
        # Build prompt based on refactoring strategy
        if refactoring_strategy == "optimize_for_readability":
            prompt = f"""
            Refactor the following code to improve clarity and readability:
            
            - Extract complex logic into properly named helper functions
            - Improve variable names for better understanding
            - Simplify complex conditionals and operations
            - Ensure the main function keeps the same name and parameters
            - Keep the exact same behavior for all inputs
            - Remove all redundant variables and lines of code
            - Change function signature if necessary
            
            Original code:
            ```python
            {original_code}
            ```
            
            Return only the Python code for the refactored function without any explanation.
            """
        else:
            # Default prompt
            prompt = f"""
            Refactor the following Python function using the strategy: {refactoring_strategy}
            
            Original function:
            ```python
            {original_code}
            ```
            
            I need you to refactor this function while preserving its exact behavior.
            The refactored function should:
            1. Have the same input-output behavior for all valid inputs
            2. Have improved {refactoring_strategy} characteristics
            3. Be functionally equivalent to the original
            
            Improve clarity, efficiency, and maintainability:
            - Remove redundancy
            - Simplify logic
            - DO NOT DECOMPOSE THE FUNCTION INTO MULTIPLE
            - Improve naming
            - Ensure functionality is preserved
            
            Return only the Python code for the refactored function without any explanation.
            """
        
        refactored_code = query_llama3(prompt)
        # refactored_code = call_mistral(prompt)
        
        # Clean up the code to remove any markdown code blocks
        refactored_code = extract_code_block(refactored_code)

        return refactored_code
    
    def generate_verification_constraints(self, original_code: str, refactored_code: str) -> List[str]:
        """Generate additional verification constraints using LLM"""
       
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
        # constraints_text = call_mistral(prompt)
        constraints_text = query_llama3(prompt)
        
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
        """Verify the equivalence of original and refactored code with detailed logging"""
        logger.info("Starting verification process")
        
        results = {
            "z3_verification": None,
            "crosshair_verification": None,
            "runtime_verification": None,
            "equivalent": False,
            "error": None,
            "verification_steps": []  # Track which steps were actually performed
        }
        
        try:
            # Check if code has multiple functions
            logger.info("Extracting functions from code")
            original_functions = ASTAnalyzer.extract_all_functions(original_code)
            refactored_functions = ASTAnalyzer.extract_all_functions(refactored_code)
            
            logger.info(f"Found {len(original_functions)} functions in original code")
            logger.info(f"Found {len(refactored_functions)} functions in refactored code")
            
            has_helper_functions = len(refactored_functions) > 1
            if has_helper_functions:
                logger.info("Detected helper functions in refactored code")
            
            # Runtime testing (works for both single and multiple function code)
            logger.info("Starting runtime verification")
            try:
                start_time = time.time()
                runtime_results = RuntimeEquivalenceTester.test_equivalence(
                    original_code, refactored_code, num_tests=50)
                end_time = time.time()
                
                logger.info(f"Runtime verification completed in {end_time - start_time:.2f} seconds")
                logger.info(f"Runtime verification result: equivalent={runtime_results.get('equivalent', False)}")
                
                if not runtime_results.get('equivalent', False) and runtime_results.get('failures'):
                    logger.info(f"Runtime verification failures: {len(runtime_results['failures'])}")
                    logger.info(f"First failure example: {runtime_results['failures'][0] if runtime_results['failures'] else 'None'}")
                
                results["runtime_verification"] = runtime_results
                results["verification_steps"].append("runtime")
                
                # For simple code without helpers, continue with other verification methods
                if not has_helper_functions:
                    logger.info("Code has no helper functions, proceeding with Z3 and CrossHair verification")
                    
                    try:
                        # Get function signatures for Z3 and CrossHair
                        logger.info("Extracting function signatures")
                        original_func_name, original_params, _ = ASTAnalyzer.extract_function_signature(original_code)
                        refactored_func_name, refactored_params, _ = ASTAnalyzer.extract_function_signature(refactored_code)
                        
                        logger.info(f"Original function: {original_func_name} with params {original_params}")
                        logger.info(f"Refactored function: {refactored_func_name} with params {refactored_params}")
                        
                        # Execute both function definitions
                        logger.info("Loading functions from code")
                        original_func = get_function_from_code(original_code, original_func_name)
                        refactored_func = get_function_from_code(refactored_code, refactored_func_name)
                        
                        # Z3 verification
                        logger.info("Starting Z3 verification")
                        try:
                            start_time = time.time()
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
                            
                            logger.info("Translating functions to Z3 expressions")
                            expr1 = translator1.translate(original_code, shared_vars)
                            expr2 = translator2.translate(refactored_code, param_mapping)
                            
                            logger.info("Verifying with Z3")
                            z3_result = Z3Verifier.verify_equivalence(expr1, expr2)
                            end_time = time.time()
                            
                            logger.info(f"Z3 verification completed in {end_time - start_time:.2f} seconds")
                            logger.info(f"Z3 verification result: {z3_result}")
                            
                            results["z3_verification"] = z3_result
                            results["verification_steps"].append("z3")
                        except Exception as e:
                            logger.error(f"Z3 verification error: {e}")
                            logger.error(f"Z3 verification traceback: {traceback.format_exc()}")
                            results["z3_verification"] = {"error": str(e)}
                        
                        # CrossHair verification
                        logger.info("Starting CrossHair verification")
                        try:
                            start_time = time.time()
                            logger.info("Generating property function for CrossHair")
                            property_func = CrossHairVerifier.generate_property_function(
                                original_func, 
                                refactored_func,
                                original_params
                            )
                            
                            logger.info("Running CrossHair analysis")
                            crosshair_results = CrossHairVerifier.analyze_with_crosshair(property_func)
                            end_time = time.time()
                            
                            logger.info(f"CrossHair verification completed in {end_time - start_time:.2f} seconds")
                            logger.info(f"CrossHair verification found {len(crosshair_results)} issues")
                            
                            if crosshair_results:
                                logger.info(f"First CrossHair issue: {crosshair_results[0]}")
                            
                            results["crosshair_verification"] = crosshair_results
                            results["verification_steps"].append("crosshair")
                        except Exception as e:
                            logger.error(f"CrossHair verification error: {e}")
                            logger.error(f"CrossHair verification traceback: {traceback.format_exc()}")
                            results["crosshair_verification"] = {"error": str(e)}
                    except Exception as e:
                        logger.error(f"Function-specific verification error: {e}")
                        logger.error(f"Function verification traceback: {traceback.format_exc()}")
                        # If more specific verification methods fail, rely on runtime verification
                
                # Determine overall equivalence based on available results
                logger.info("Determining overall equivalence")
                
                if has_helper_functions:
                    # For code with helper functions, rely primarily on runtime testing
                    logger.info("Code has helper functions, relying on runtime verification only")
                    results["equivalent"] = runtime_results.get("equivalent", False)
                else:
                    # For simple code, consider all methods
                    z3_equivalent = results["z3_verification"].get("equivalent", False) if isinstance(results["z3_verification"], dict) else False
                    crosshair_failed = len(results["crosshair_verification"]) > 0 if isinstance(results["crosshair_verification"], list) else True
                    runtime_equivalent = runtime_results.get("equivalent", False)
                    
                    logger.info(f"Z3 verification says equivalent: {z3_equivalent}")
                    logger.info(f"CrossHair verification found issues: {crosshair_failed}")
                    logger.info(f"Runtime testing says equivalent: {runtime_equivalent}")
                    
                    # If Z3 verification passed OR runtime testing passed and CrossHair didn't find issues
                    results["equivalent"] = z3_equivalent or (runtime_equivalent and not crosshair_failed)
                    logger.info(f"Final equivalence determination: {results['equivalent']}")
            except Exception as e:
                logger.error(f"Runtime verification error: {e}")
                logger.error(f"Runtime verification traceback: {traceback.format_exc()}")
                results["runtime_verification"] = {"error": str(e)}
                
                # Try to fall back to other methods if runtime testing fails
                if not has_helper_functions:
                    logger.info("Runtime verification failed, falling back to other methods")
                    z3_equivalent = results["z3_verification"].get("equivalent", False) if isinstance(results["z3_verification"], dict) else False
                    crosshair_failed = len(results["crosshair_verification"]) > 0 if isinstance(results["crosshair_verification"], list) else True
                    results["equivalent"] = z3_equivalent and not crosshair_failed
                    logger.info(f"Fallback equivalence determination: {results['equivalent']}")
        except Exception as e:
            logger.error(f"Overall verification error: {e}")
            logger.error(f"Verification traceback: {traceback.format_exc()}")
            results["error"] = str(e)
        
        logger.info(f"Verification completed. Steps performed: {results['verification_steps']}")
        logger.info(f"Final result: {results['equivalent']}")
        
        return results
    
    def refactor_with_feedback_loop(self, original_code: str, 
                                  refactoring_strategy: str = "optimize_for_readability",
                                  max_iterations: int = 3) -> Dict:
        """Refactor code and verify in a feedback loop until equivalence is achieved"""
        
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
                # refactored_code = call_mistral(prompt)
                refactored_code = query_llama3(prompt)
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
    # First try to find a standard markdown code block
    pattern = r'```(?:python)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no standard code block found, try to handle the case where
    # only opening ``` exists without closing ```
    pattern = r'```(?:python)?\s*(.*?)$'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no opening markers found at all, just strip and return the text
    return text.strip()

