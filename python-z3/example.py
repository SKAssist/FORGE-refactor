import json
import sys
import requests
from typing import Dict, List

# Import the refactoring system - adjust path as needed
from refactor_verification_system import (
    CrossHairVerifier, Z3Verifier, PythonToZ3Translator,
    ASTAnalyzer, extract_code_block, get_function_from_code
)

# Define your Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

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

class RefactorWithLlama:
    def __init__(self):
        pass
    
    def generate_refactored_code(self, original_code: str, refactoring_strategy: str = "optimize_for_readability") -> str:
        """Generate refactored version of the code using Llama"""
        print("Generating Code")
        # Extract function information
        try:
            func_name, params, return_type = ASTAnalyzer.extract_function_signature(original_code)
            complexity_metrics = ASTAnalyzer.analyze_code_complexity(original_code)
        except Exception as e:
            print(f"Error analyzing code: {e}")
            complexity_metrics = {"cyclomatic_complexity": "unknown", "num_operations": "unknown", "max_nesting_depth": "unknown"}
        
        # Build prompt for the LLM
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
        
        Return only the Python code for the refactored function without any explanation.
        """
        
        refactored_code = query_llama3(prompt)
        
        # Clean up the code to remove any markdown code blocks
        refactored_code = extract_code_block(refactored_code)
    
        return refactored_code
    
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
            from z3 import Int
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
            import traceback
            results["error"] = str(e)
            traceback.print_exc()
            
        return results
    
    def analyze_verification_results(self, original_code: str, refactored_code: str, 
                                   verification_results: Dict) -> str:
        """Use LLM to analyze verification results and provide improvement suggestions"""
        
        prompt = f"""
        Analyze the results of code equivalence verification.
        
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
                analysis = self.analyze_verification_results(
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
                
                refactored_code = query_llama3(prompt)
                refactored_code = extract_code_block(refactored_code)
            
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

# def main():
#     # Example original code to refactor
#     original_code = """
# def is_prime(n):
#     if n <= 1:
#         return False
#     if n <= 3:
#         return True
#     if n % 2 == 0 or n % 3 == 0:
#         return False
#     i = 5
#     while i * i <= n:
#         if n % i == 0 or n % (i + 2) == 0:
#             return False
#         i += 6
#     return True
# """

#     refactor_engine = RefactorWithLlama()
#     results = refactor_engine.refactor_with_feedback_loop(
#         original_code=original_code,
#         refactoring_strategy="optimize_for_readability",
#         max_iterations=3
#     )
    
#     print("\n--- Final Results ---")
#     print(f"Equivalent: {results['equivalent']}")
#     print(f"Iterations performed: {results['iterations_performed']}")
#     print("\nOriginal code:")
#     print(results["original_code"])
#     print("\nFinal refactored code:")
#     print(results["final_refactored_code"])

def main():
    # A more complex example to refactor
    original_code = """
def process_data(data, thresholds=None, filters=None, transform_func=None):
    
    if not data:
        return {"error": "No data provided", "status": "failed"}
    
    # Initialize defaults
    if thresholds is None:
        thresholds = {"min": float('-inf'), "max": float('inf')}
    if filters is None:
        filters = []
    
    # Filter based on thresholds
    filtered_data = []
    for value in data:
        if thresholds["min"] <= value <= thresholds["max"]:
            valid = True
            
            # Apply additional filters
            for filter_func in filters:
                if not filter_func(value):
                    valid = False
                    break
            
            if valid:
                filtered_data.append(value)
    
    # Apply transformations if provided
    if transform_func and filtered_data:
        transformed_data = []
        for value in filtered_data:
            try:
                transformed_value = transform_func(value)
                transformed_data.append(transformed_value)
            except Exception as e:
                # Skip values that cause transformation errors
                continue
        filtered_data = transformed_data
    
    # Calculate statistics
    if not filtered_data:
        return {
            "status": "success", 
            "filtered_data": [], 
            "count": 0,
            "statistics": None
        }
    
    sum_values = 0
    min_value = float('inf')
    max_value = float('-inf')
    
    for value in filtered_data:
        sum_values += value
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value
    
    avg_value = sum_values / len(filtered_data)
    
    # Calculate variance and standard deviation
    variance = 0
    for value in filtered_data:
        variance += (value - avg_value) ** 2
    variance = variance / len(filtered_data)
    std_dev = variance ** 0.5
    
    return {
        "status": "success",
        "filtered_data": filtered_data,
        "count": len(filtered_data),
        "statistics": {
            "min": min_value,
            "max": max_value,
            "avg": avg_value,
            "sum": sum_values,
            "variance": variance,
            "std_dev": std_dev
        }
    }
"""

    refactor_engine = RefactorWithLlama()
    results = refactor_engine.refactor_with_feedback_loop(
        original_code=original_code,
        refactoring_strategy="optimize_for_readability",
        max_iterations=3
    )
    
    print("\n--- Final Results ---")
    print(f"Equivalent: {results['equivalent']}")
    print(f"Iterations performed: {results['iterations_performed']}")
    print("\nOriginal code:")
    print(results["original_code"])
    print("\nFinal refactored code:")
    print(results["final_refactored_code"])

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()