import * as vscode from 'vscode';
import { exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('LLaMA Refactor');

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
      const tempSeedPath = path.join(os.tmpdir(), 'snippet_input.py');
      const scriptPath = '/home/ubuntu/FORGE-refactor/test_changes/test_simple';

      try {
        // 🧠 Write selection to temp seed file
        fs.writeFileSync(tempSeedPath, selection);
        vscode.window.showInformationMessage('📤 Sending seed snippet to LLaMA...');
        outputChannel.appendLine(`[info] Seed snippet written to ${tempSeedPath}`);
        outputChannel.appendLine(`[info] Target directory: ${targetDir}`);

        // 🚀 Execute refactor script
        exec(`python3 "${scriptPath}" "${tempSeedPath}" "${targetDir}"`, (error, stdout, stderr) => {
          if (error) {
            vscode.window.showErrorMessage(`❌ Error during LLaMA refactor.`);
            outputChannel.appendLine(`[error] ${error.message}`);
            if (stderr) outputChannel.appendLine(`[stderr] ${stderr}`);
            outputChannel.show(true);
            return;
          }

          // 🧾 Notify success and log output
          vscode.window.showInformationMessage('✅ LLaMA refactor complete!');
          outputChannel.appendLine(`[success] Refactor complete.\n--- stdout ---\n${stdout}`);
          if (stderr.trim()) {
            vscode.window.showWarningMessage('⚠️ LLaMA completed with warnings.');
            outputChannel.appendLine(`[warning] stderr:\n${stderr}`);
          }

          // ✅ Show output panel with all logs
          outputChannel.show(true);

          // 🔍 Additional visual guidance
          vscode.window.showInformationMessage('📂 Check the Output panel for file-by-file logs.');
        });

        vscode.window.showInformationMessage('⚙️ Refactor process started...');
      } catch (err) {
        vscode.window.showErrorMessage(`❌ Failed to prepare refactor: ${(err as Error).message}`);
        outputChannel.appendLine(`[crash] ${err}`);
        outputChannel.show(true);
      }
    }
  );

  context.subscriptions.push(disposable);
}

export function deactivate() {}
