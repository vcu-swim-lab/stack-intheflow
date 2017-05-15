package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.editor.event.DocumentEvent;
import com.intellij.openapi.editor.event.DocumentListener;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.fileEditor.FileEditorManagerEvent;
import com.intellij.openapi.fileEditor.FileEditorManagerListener;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.util.messages.MessageBusConnection;
import org.jetbrains.annotations.NotNull;

/**
 * Created by chase on 5/15/17.
 */
public class DocumentListenerComponent implements ProjectComponent {

    private final Project project;
    private MessageBusConnection connection;

    public DocumentListenerComponent(Project project) {
        this.project = project;
    }

    @Override
    public void projectOpened() {
        // Subscribe to compiler output
        if (project != null) {
            connection = project.getMessageBus().connect();
            connection.subscribe(FileEditorManagerListener.FILE_EDITOR_MANAGER, new FileEditorManagerListener() {

                @Override
                public void fileOpened(@NotNull FileEditorManager fileEditorManager, @NotNull VirtualFile virtualFile) {
                    FileDocumentManager.getInstance().getDocument(virtualFile).addDocumentListener(new DocumentListener() {
                        @Override
                        public void beforeDocumentChange(DocumentEvent documentEvent) {

                        }

                        @Override
                        public void documentChanged(DocumentEvent documentEvent) {
                            if (documentEvent.getOldLength() > documentEvent.getNewLength()) {
                                System.out.println("DELETE -\n Old: " + documentEvent.getOldFragment() + "\n New: " + documentEvent.getNewFragment());
                            } else {
                                System.out.println("INSERT -\n Old: " + documentEvent.getOldFragment() + "\n New: " + documentEvent.getNewFragment());
                            }
                            
                        }
                    });
                }

                @Override
                public void fileClosed(@NotNull FileEditorManager fileEditorManager, @NotNull VirtualFile virtualFile) {

                }

                @Override
                public void selectionChanged(@NotNull FileEditorManagerEvent fileEditorManagerEvent) {

                }
            });
        }
    }

    @Override
    public void projectClosed() {

    }

    @Override
    public void initComponent() {

    }

    @Override
    public void disposeComponent() {
        connection.disconnect();
    }

    @NotNull
    @Override
    public String getComponentName() {
        return "DocumentListenerComponent";
    }
}
