/**
 * @file crossFileChange.ts
 * @description Defines the "Cross-file Code Change" command. This command will later scan and
 * refactor across multiple files using a planned approach (planned for future implementation).
 *
 * @dependencies
 * - ../agents/ollamaAgent (planned)
 * - ../utils/fileUtils (planned)
 * - ../core/changePlanner (planned)
 */

import * as vscode from 'vscode';
// Planned: import { queryOllama } from '../agents/ollamaAgent';
// Planned: import { verifyEquivalence } from '../utils/verification';

export async function refactorWithVerification() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showErrorMessage('❌ No active editor found.');
    return;
  }

  const document = editor.document;
  const originalCode = document.getText();

  if (!originalCode || originalCode.trim() === '') {
    vscode.window.showWarningMessage('⚠️ File is empty or contains only whitespace.');
    return;
  }

  vscode.window.showInformationMessage('🚀 FORGE: Starting refactor with verification...');

  // --- Placeholder: Replace with actual Ollama call
  const refactoredCode = `// TODO: This is a placeholder refactor\n${originalCode}`;

  // --- Placeholder: Replace with actual semantic equivalence check
  const isVerified = true;

  if (!isVerified) {
    vscode.window.showWarningMessage('⚠️ Refactor failed semantic equivalence check. Aborting changes.');
    return;
  }

  const fullRange = new vscode.Range(
    document.positionAt(0),
    document.positionAt(originalCode.length)
  );

  const edit = new vscode.WorkspaceEdit();
  edit.replace(document.uri, fullRange, refactoredCode);

  const success = await vscode.workspace.applyEdit(edit);

  if (success) {
    vscode.window.showInformationMessage('✅ Refactor complete and verified.');
  } else {
    vscode.window.showErrorMessage('❌ Failed to apply verified refactor.');
  }
}

import * as vscode from 'vscode';

/**
 * @file crossFileChange.ts
 * @description Scans and logs all workspace source files.
 */

export async function crossFileChange() {
  const output = vscode.window.createOutputChannel('FORGE: CrossFileChange');
  output.show(true);
  output.appendLine('🔍 FORGE is scanning files...');

  try {
    const files = await vscode.workspace.findFiles('**/*.{js,ts,py,cpp}', '**/node_modules/**', 100);

    if (files.length === 0) {
      vscode.window.showWarningMessage('⚠️ No files found in workspace.');
      return;
    }

    for (const file of files) {
      const doc = await vscode.workspace.openTextDocument(file);
      const fileName = file.fsPath.split('/').pop();
      const preview = doc.getText().split('\n').slice(0, 2).join(' ');
      output.appendLine(`📄 ${fileName}: ${preview}`);
    }

    vscode.window.showInformationMessage(`✅ Scanned ${files.length} files.`);
  } catch (err) {
    vscode.window.showErrorMessage('❌ Error during cross-file scan.');
    output.appendLine(`❌ ${err}`);
  }
}
