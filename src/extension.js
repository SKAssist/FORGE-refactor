"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const child_process_1 = require("child_process");
function activate(context) {
    const disposable = vscode.commands.registerCommand('forge.refactorSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor.');
            return;
        }
        const selection = editor.selection;
        const selectedCode = editor.document.getText(selection);
        if (!selectedCode.trim()) {
            vscode.window.showWarningMessage('Select a Python function to refactor.');
            return;
        }
        // Save selection to snippet_input.py
        const projectDir = path.join(__dirname, '..', 'test-project');
        const seedPath = path.join(projectDir, 'snippet_input.py');
        try {
            if (!fs.existsSync(projectDir)) {
                fs.mkdirSync(projectDir);
            }
            fs.writeFileSync(seedPath, selectedCode, 'utf8');
            vscode.window.showInformationMessage('📤 Saved selection to snippet_input.py');
            // Run your Python backend
            (0, child_process_1.exec)('python3 backend/refactor.py', (error, stdout, stderr) => {
                if (error) {
                    vscode.window.showErrorMessage(`❌ Error: ${error.message}`);
                    return;
                }
                if (stderr) {
                    console.error(stderr);
                }
                vscode.window.showInformationMessage('✅ Refactor complete.');
                vscode.window.showInformationMessage('Check changes in test-project directory.');
                console.log(stdout);
            });
        }
        catch (err) {
            vscode.window.showErrorMessage('❌ Failed to write or execute: ' + err.message);
        }
    });
    context.subscriptions.push(disposable);
}
function deactivate() { }
