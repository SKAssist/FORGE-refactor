# FORGE: Formally-verified Optimized Refactoring Guided by Equivalence

![FORGE Banner](https://via.placeholder.com/1200x300)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![VS Code Extension](https://img.shields.io/badge/VSCode-Extension-blue)](https://marketplace.visualstudio.com/items?itemName=FORGE.forge-refactor)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Meta Llama 3 Hackathon Winner](https://img.shields.io/badge/Meta%20Llama%203-Hackathon%20Winner-green)](https://github.com/FORGE-refactor)

FORGE is a VSCode extension that automatically refactors your Python code using Llama 3, then mathematically verifies the refactored code's equivalence to the original. Say goodbye to bugs introduced during refactoring and AI-generated code bloat.

## ✨ Features

- **Guaranteed Equivalence**: Formal verification ensures refactored code behaves exactly like the original  
- **Clean Refactoring**: Intelligent improvements without unnecessary abstractions or bloat  
- **Multi-Method Verification**: Z3 theorem proving, runtime testing, and property-based testing  
- **Feedback Loop**: When verification fails, FORGE iteratively refines the solution  
- **VSCode Integration**: Simple right-click to refactor selected code  
- **Cross-File References**: Automatically updates function calls across your project  

## 🚀 Installation

Install directly from the VSCode Marketplace:

1. Open VSCode  
2. Go to Extensions (Ctrl+Shift+X)  
3. Search for "FORGE Refactor"  
4. Click Install  

Or install via the command line:

```bash
code --install-extension FORGE.forge-refactor
