# refactor_snippet_workspace.py

import os
import re
import requests
import difflib
import sys
from pathlib import Path

# Configuration
EXTENSIONS = {'.py'}
MODEL_NAME = 'llama3'
OLLAMA_URL = 'http://localhost:11434/api/generate'


def call_llama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload)
    return res.json()["response"].strip()


def strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.strip().split("\n")
        if lines[0].startswith("```") and lines[-1].startswith("```"):
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


def process_files(func_name: str, seed_before: str, seed_after: str, workspace_root: str, seed_path: str):
    changed_files = 0
    for root, _, files in os.walk(workspace_root):
        for file in files:
            if Path(file).suffix not in EXTENSIONS:
                continue

            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                content = f.read()

            if os.path.abspath(full_path) == os.path.abspath(seed_path):
                print(f"\n💾 Overwriting seed file: {file}")
                print_diff(content, seed_after, file)
                with open(full_path, 'w') as f:
                    f.write(seed_after)
                changed_files += 1
                continue

            blocks_with_pos = find_function_usage_blocks(content, func_name)
            if not blocks_with_pos:
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
                    new_blocks.append(block)
                    positions.append((start, end))

            new_content = replace_blocks(content, new_blocks, positions)
            if new_content != content:
                with open(full_path, 'w') as f:
                    f.write(new_content)
                changed_files += 1

    print(f"\n✅ Done. Refactored {changed_files} file(s).")


def main():
    if len(sys.argv) < 3:
        print("Usage: python refactor_snippet_workspace.py <seed_path> <workspace_path>")
        return

    seed_path = sys.argv[1]
    workspace = sys.argv[2]

    with open(seed_path, 'r') as f:
        seed_before = f.read().strip()

    print("📤 Sending seed snippet to LLaMA...")
    seed_after = refactor_seed(seed_before)

    if not seed_after.strip():
        print("❌ Refactor failed.")
        return

    print("✅ Refactor received:")
    print_diff(seed_before, seed_after, Path(seed_path).name)

    func_name = extract_function_name(seed_before)
    if not func_name:
        print("❌ Could not extract function name.")
        return

    process_files(func_name, seed_before, seed_after, workspace, seed_path)


if __name__ == "__main__":
    main()
