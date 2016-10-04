import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.command.CommandProcessor;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by batman on 9/26/16.
 */
public class ReadCurrentFile extends AnAction {
    public void actionPerformed(AnActionEvent e) {
        // Reloading file comes from tutorial
        if (!reloadCurrentFile(e)) return;

        TestForm.main(null);
    }

    private boolean reloadCurrentFile(AnActionEvent e) {
        final Project project = e.getProject();
        if (project == null) {
            return false;
        }
        Editor editor = FileEditorManager.getInstance(project).getSelectedTextEditor();
        if (editor == null) {
            return false;
        }

        return setLoadedFile(project, editor.getDocument(), loadFile(FileDocumentManager.getInstance().getFile(editor
                .getDocument())));
    }

    private boolean setLoadedFile(final Project project, final Document document, final String contents) {
        if (project == null || document == null || contents == null) {
            return false;
        }

        final Runnable readRunner = new Runnable() {
            @Override
            public void run() {
                document.setText(contents);
            }
        };
        ApplicationManager.getApplication().invokeLater(new Runnable() {
            @Override
            public void run() {
                CommandProcessor.getInstance().executeCommand(project, new Runnable() {
                    @Override
                    public void run() {
                        ApplicationManager.getApplication().runWriteAction(readRunner);
                    }
                }, "DiskRead", null);
            }
        });
        return true;
    }

    private String loadFile(final VirtualFile virtualFile) {
        if (virtualFile == null) {
            return null;
        }
        String contents;
        try {
            BufferedReader br = new BufferedReader(new FileReader(virtualFile.getPath()));
            String currentLine;
            StringBuilder stringBuilder = new StringBuilder();
            while ((currentLine = br.readLine()) != null) {
                stringBuilder.append(currentLine);
                stringBuilder.append("\n");
            }
            contents = stringBuilder.toString();
        } catch (IOException e) {
            return null;
        }
        return contents;
    }
}
