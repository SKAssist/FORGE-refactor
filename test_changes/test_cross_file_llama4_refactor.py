import os
import re
import sys
import requests
import difflib
from pathlib import Path
from refactor_verification import (
    CrossHairVerifier, Z3Verifier, PythonToZ3Translator,
    ASTAnalyzer, extract_code_block, get_function_from_code, RefactoringEngine
)


# Configuration
TARGET_DIR = 'test-project'
SEED_FILENAME = 'dummy_prime.py'
SEED_PATH = os.path.join(TARGET_DIR, SEED_FILENAME)
EXTENSIONS = {'.py'}
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3'

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

def call_llama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload)
        res.raise_for_status()
        return res.json()["response"].strip()
    except Exception as e:
        print(f"❌ Error calling LLaMA: {e}")
        return ""


def strip_fences(text: str) -> str:
    if text.startswith(""):
        lines = text.strip().split("\n")
        if lines[0].startswith("") and lines[-1].startswith(""):
            return "\n".join(lines[1:-1])
    return text.strip()


def refactor_seed(seed_code: str) -> str:
    prompt = f"""Refactor the following Python function to improve clarity, structure, and maintainability.

- Break the function into smaller, well-named helper functions that each handle one concern (e.g., validation, filtering, transformation, or output)
- Eliminate redundancy and simplify logic
- Improve variable and function names
- Preserve all functionality
- Do not include explanations or comments—just return the refactored code

Code:
{seed_code}
"""
    return strip_fences(call_llama(prompt))


def refactor_call_context(seed_before: str, seed_after: str, usage_block: str) -> str:
    prompt = f"""You are refactoring the following block of code that calls a function which was just refactored.

Here is the original function before the change:
{seed_before}

Here is the refactored version of the function:
{seed_after}

Now update the following block of code so it uses the new function correctly.

⚠️ IMPORTANT:
- The refactored function now performs its own input validation and filtering.
- Do NOT duplicate that logic in the callsite.
- Adjust the function name and arguments as needed.
- Preserve the original intent and functionality.

Block:
{usage_block}

Return only the updated code, with no explanation.
"""
    return strip_fences(call_llama(prompt))


def refactor_with_engine(original_code):
    refactor_engine = RefactoringEngine()
    results = refactor_engine.refactor_with_feedback_loop(
        original_code=original_code,
        refactoring_strategy="optimize_for_readability",
        max_iterations=3
    )
    return results["final_refactored_code"]

def extract_function_name(code: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def find_function_usage_blocks(code: str, func_name: str, context_window: int = 1):
    lines = code.splitlines()
    pattern = re.compile(rf'\b{func_name}\s*\(.*?\)')
    blocks = []

    for i, line in enumerate(lines):
        if pattern.search(line):
            start = max(i - context_window, 0)
            end = min(i + context_window + 1, len(lines))
            snippet = "\n".join(lines[start:end]).strip()
            blocks.append((snippet, start, end))

    return blocks


def replace_blocks(content: str, new_blocks: list, positions: list) -> str:
    lines = content.splitlines()
    for (start, end), new_block in zip(positions, new_blocks):
        lines[start:end] = new_block.strip().splitlines()
    return "\n".join(lines)


def print_diff(old: str, new: str, filename: str):
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f'{filename} (before)',
        tofile=f'{filename} (after)',
        lineterm=''
    )
    print("\n".join(diff))


def process_files(func_name: str, seed_before: str, seed_after: str):
    changed_files = 0
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if Path(file).suffix not in EXTENSIONS:
                continue

            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()

            if file == SEED_FILENAME:
                # Overwrite the seed file
                print(f"\n Overwriting seed file: {SEED_FILENAME}")
                print_diff(content, seed_after, file)
                with open(path, 'w') as f:
                    f.write(seed_after)
                changed_files += 1
                continue

            blocks_with_pos = find_function_usage_blocks(content, func_name)
            if not blocks_with_pos:
                print(f" {file}: No calls to `{func_name}` found.")
                continue

            new_blocks = []
            positions = []

            for block, start, end in blocks_with_pos:
                refactored = refactor_call_context(seed_before, seed_after, block)
                if refactored and refactored.strip() != block.strip():
                    print(f"\n🔧 Refactored block in {file}:")
                    print_diff(block, refactored, file)
                    new_blocks.append(refactored)
                    positions.append((start, end))
                else:
                    print(f"⚠️  Block unchanged in {file}.\n")
                    new_blocks.append(block)
                    positions.append((start, end))

            new_content = replace_blocks(content, new_blocks, positions)
            if new_content != content:
                with open(path, 'w') as f:
                    f.write(new_content)
                changed_files += 1

    print(f"\n✅ Done. Refactored {changed_files} file(s).")


def main():
    if not os.path.exists(SEED_PATH):
        print(f"❌ Seed file not found: {SEED_PATH}")
        return

    with open(SEED_PATH, 'r') as f:
        seed_before = f.read().strip()

    if not seed_before:
        print("❌ Seed snippet is empty.")
        return

    print("📤 Sending seed snippet to LLaMA 4...")
    # seed_after = refactor_seed(seed_before)
    seed_after = refactor_with_engine(seed_before)

    if not seed_after:
        print("❌ Refactor failed.")
        return

    print("✅ Refactor received. Showing diff with original seed...")
    print_diff(seed_before, seed_after, SEED_FILENAME)

    # Overwrite the seed file
    with open(SEED_PATH, 'w') as f:
        f.write(seed_after)
    print(f"💾 Overwrote {SEED_FILENAME} with refactored version.\n")

    func_name = extract_function_name(seed_before)
    if not func_name:
        print("❌ Could not determine function name.")
        return

    print(f"🔍 Searching for usages of `{func_name}` in files...\n")
    process_files(func_name, seed_before, seed_after)


if __name__ == "__main__":
    main()