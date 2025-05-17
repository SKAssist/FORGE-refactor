import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';

export function activate(context: vscode.ExtensionContext) {
	const disposable = vscode.commands.registerCommand('forge.refactorSelection', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			vscode.window.showInformationMessage('No active editor.');
			return;
		}

		const selection = editor.selection;
		const selectedText = editor.document.getText(selection);
		if (!selectedText.trim()) {
			vscode.window.showInformationMessage('Please highlight a code snippet to refactor.');
			return;
		}

		const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
		if (!workspaceFolder) {
			vscode.window.showErrorMessage('Workspace not found.');
			return;
		}

		const workspacePath = workspaceFolder.uri.fsPath;
		const seedPath = path.join(workspacePath, 'test-project', 'snippet_input.py');
		const scriptPath = path.join(context.extensionPath, 'refactor_snippet_workspace.py');

		// Write highlighted code to snippet file
		fs.mkdirSync(path.dirname(seedPath), { recursive: true });
		fs.writeFileSync(seedPath, selectedText);

		const command = `python3 "${scriptPath}" "${seedPath}" "${workspacePath}"`;
		vscode.window.showInformationMessage('🔧 Refactoring with LLaMA...');

		exec(command, (err, stdout, stderr) => {
			if (err) {
				vscode.window.showErrorMessage(`❌ Refactor failed: ${stderr}`);
				return;
			}

			fs.readFile(seedPath, 'utf8', (err, data) => {
				if (err) {
					vscode.window.showErrorMessage('❌ Failed to read refactored snippet.');
					return;
				}
				editor.edit(editBuilder => {
					editBuilder.replace(selection, data);
				});
				vscode.window.showInformationMessage('✅ Snippet refactored and applied.');
				console.log(stdout);
			});
		});
	});

	context.subscriptions.push(disposable);
}

export function deactivate() {}
