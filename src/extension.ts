import * as vscode from 'vscode';

// Command 1: Refactor with formal verification
async function refactorWithVerification() {
	vscode.window.showInformationMessage('🔍 Refactor with verification triggered!');
	// Future: Add Ollama logic here
}

// Command 2: Cross-file code change
async function crossFileChange() {
	vscode.window.showInformationMessage('📂 Cross-file change triggered!');
	// Future: Add multi-file edit logic here
}

// Extension activation
export function activate(context: vscode.ExtensionContext) {
	console.log('Congratulations, your extension "forge" is now active!');

	// Original Hello World command
	const helloDisposable = vscode.commands.registerCommand('forge.helloWorld', () => {
		vscode.window.showInformationMessage('Hello World from FORGE!');
	});
	// New command: Refactor with verification
	const refactorDisposable = vscode.commands.registerCommand('forge.refactorWithVerification', refactorWithVerification);

	// New command: Multi-file code change
	const crossFileDisposable = vscode.commands.registerCommand('forge.crossFileChange', crossFileChange);

	// Register all commands
	context.subscriptions.push(helloDisposable, refactorDisposable, crossFileDisposable);
}

// Extension deactivation
export function deactivate() {}
	