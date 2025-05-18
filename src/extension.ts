import * as vscode from 'vscode';
import { exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand(
    'llamaRefactor.refactorSelection',
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage('❌ No active editor found.');
        return;
      }

      const selection = editor.document.getText(editor.selection);
      if (!selection.trim()) {
        vscode.window.showErrorMessage('❌ No code selected.');
        return;
      }

      const documentPath = editor.document.uri.fsPath;
      const targetDir = path.dirname(documentPath);
      const scriptPath = '/home/ubuntu/FORGE-refactor/test_changes/test_cross_file_llama4_refactor.py';

      try {
        // Overwrite the file with the selected snippet
        fs.writeFileSync(documentPath, selection);
        await editor.document.save();

        // Call the Python script with both the seed file and target directory
        exec(`python3 "${scriptPath}" "${documentPath}" "${targetDir}"`, (error, stdout, stderr) => {
          if (error) {
            vscode.window.showErrorMessage(`❌ Error: ${(error as Error).message}`);
            console.error(stderr);
            return;
          }

          vscode.window.showInformationMessage('✅ LLaMA refactor complete.');
          console.log(stdout);
          if (stderr) console.error(stderr);
        });
      } catch (err) {
        vscode.window.showErrorMessage(`❌ Failed to process selection: ${(err as Error).message}`);
      }
    }
  );

  context.subscriptions.push(disposable);
}

export function deactivate() {}
