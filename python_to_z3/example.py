import json
import sys
import requests
from typing import Dict, List

# Import the refactoring system - adjust path as needed
from refactor_verification_system import (
    CrossHairVerifier, Z3Verifier, PythonToZ3Translator,
    ASTAnalyzer, extract_code_block, get_function_from_code, RefactoringEngine
)



def main():
    # Example original code to refactor
    original_code = """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
"""

    refactor_engine = RefactoringEngine()
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

# def main():
#     # A more complex example to refactor
#     original_code = """
# def process_data(data, thresholds=None, filters=None, transform_func=None):
    
#     if not data:
#         return {"error": "No data provided", "status": "failed"}
    
#     # Initialize defaults
#     if thresholds is None:
#         thresholds = {"min": float('-inf'), "max": float('inf')}
#     if filters is None:
#         filters = []
    
#     # Filter based on thresholds
#     filtered_data = []
#     for value in data:
#         if thresholds["min"] <= value <= thresholds["max"]:
#             valid = True
            
#             # Apply additional filters
#             for filter_func in filters:
#                 if not filter_func(value):
#                     valid = False
#                     break
            
#             if valid:
#                 filtered_data.append(value)
    
#     # Apply transformations if provided
#     if transform_func and filtered_data:
#         transformed_data = []
#         for value in filtered_data:
#             try:
#                 transformed_value = transform_func(value)
#                 transformed_data.append(transformed_value)
#             except Exception as e:
#                 # Skip values that cause transformation errors
#                 continue
#         filtered_data = transformed_data
    
#     # Calculate statistics
#     if not filtered_data:
#         return {
#             "status": "success", 
#             "filtered_data": [], 
#             "count": 0,
#             "statistics": None
#         }
    
#     sum_values = 0
#     min_value = float('inf')
#     max_value = float('-inf')
    
#     for value in filtered_data:
#         sum_values += value
#         if value < min_value:
#             min_value = value
#         if value > max_value:
#             max_value = value
    
#     avg_value = sum_values / len(filtered_data)
    
#     # Calculate variance and standard deviation
#     variance = 0
#     for value in filtered_data:
#         variance += (value - avg_value) ** 2
#     variance = variance / len(filtered_data)
#     std_dev = variance ** 0.5
    
#     return {
#         "status": "success",
#         "filtered_data": filtered_data,
#         "count": len(filtered_data),
#         "statistics": {
#             "min": min_value,
#             "max": max_value,
#             "avg": avg_value,
#             "sum": sum_values,
#             "variance": variance,
#             "std_dev": std_dev
#         }
#     }
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

if __name__ == "__main__":
    main()
