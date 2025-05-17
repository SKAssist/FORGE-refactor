import requests
from z3 import *

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

def clean_code(code_str: str) -> str:
    if code_str.startswith("```") and code_str.endswith("```"):
        code_str = code_str.strip("`\n ")
    return code_str

def clean_code_with_llm(code_str: str) -> str:
    prompt = f"""
The following is a Python code snippet that may include markdown formatting like triple backticks or inconsistent indentation.
Please clean it up by removing any markdown formatting, fixing indentation, and returning only the valid Python code without explanations or comments.
No non-Python text or words or lines or comments. 
Code snippet:
{code_str}

Cleaned Python code:
"""
    return query_llama3(prompt)


class PythonToZ3Translator:
    def translate(self, python_code: str) -> str:
        prompt = f"""
Translate the following Python function into equivalent Z3 Python API code that models its behavior and returns the output expression.
Define all necessary variables as Z3 symbolic variables.

Only provide the Python code using Z3 API, no explanations or imports. 


Python function:
{python_code}

Z3 Python code:
"""
        z3_code = query_llama3(prompt)
        return z3_code

def check_equivalence(func1: str, func2: str) -> None:
    translator1 = PythonToZ3Translator()
    translator2 = PythonToZ3Translator()

    expr1_code = translator1.translate(func1)
    expr2_code = translator2.translate(func2)
    # print(expr1_code)
    # print(expr2_code)
    expr1_code = clean_code(expr1_code)
    expr2_code = clean_code(expr2_code)
    print(expr1_code)
    print(expr2_code)
    s = Solver()

    # Prepare a dictionary to use as local environment for exec
    local_env = {}

    # Execute the Z3 code to get output expressions
    # We expect the Z3 code to assign the output to a variable named `output`
    exec(expr1_code, globals(), local_env)
    expr1 = local_env.get("output")
    exec(expr2_code, globals(), local_env)
    expr2 = local_env.get("output")

    if expr1 is None or expr2 is None:
        print("Failed to parse Z3 expressions from translation.")
        return

    s.add(expr1 != expr2)
    print("Checking equivalence...")
    if s.check() == sat:
        print("Functions NOT equivalent.")
        print("Counterexample:", s.model())
    else:
        print("Functions are equivalent.")

def example():
    func1 = """
def add(x, y):
    return x + y
"""
    func2 = """
def add_refactor(x, y):
    return y + x
"""
    check_equivalence(func1, func2)

if __name__ == "__main__":
    example()
