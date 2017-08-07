package io.github.vcuswimlab.stackintheflow.controller.component.document;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.editor.event.DocumentEvent;
import com.intellij.openapi.editor.event.DocumentListener;
import com.intellij.openapi.editor.event.EditorMouseEvent;
import com.intellij.openapi.editor.event.EditorMouseListener;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.fileEditor.FileEditorManagerEvent;
import com.intellij.openapi.fileEditor.FileEditorManagerListener;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.DifficultyTrigger;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEvent;
import org.jetbrains.annotations.NotNull;

import java.awt.*;

/**
 * Created by chase on 5/15/17.
 */
public class DocumentListenerComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.DocumentListenerComponent";

    private final Project project;
    private MessageBusConnection connection;

    public DocumentListenerComponent(Project project) {
        this.project = project;
    }

    @Override
    public void projectOpened() {
        // Subscribe to editor events
        if (project != null) {
            connection = project.getMessageBus().connect();

            connection.subscribe(FileEditorManagerListener.FILE_EDITOR_MANAGER, new FileEditorManagerListener() {

                // Subscribe to edit events
                @Override
                public void fileOpened(@NotNull FileEditorManager fileEditorManager, @NotNull VirtualFile virtualFile) {

                    Editor editor = fileEditorManager.getSelectedTextEditor();

                    Document document = FileDocumentManager.getInstance().getDocument(virtualFile);

                    if (document != null) {
                        document.addDocumentListener(new DocumentListener() {

                            @Override
                            public void beforeDocumentChange(DocumentEvent documentEvent) {

                            }

                            @Override
                            public void documentChanged(DocumentEvent documentEvent) {

                                if (!project.isDisposed()) {
                                    long timeStamp = System.currentTimeMillis();

                                    CharSequence oldFragment = documentEvent.getOldFragment();
                                    CharSequence newFragment = documentEvent.getNewFragment();


                                    DifficultyTrigger publisher = project.getMessageBus().syncPublisher(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC);

                                    if (oldFragment.length() == 0 && newFragment.length() > 0) { //Event was an insert
                                        publisher.doEdit(new EditorEvent.Insert(newFragment.toString(), editor, timeStamp));
                                    } else if (oldFragment.length() > 0 && newFragment.length() == 0) { //Event was a delete
                                        publisher.doEdit(new EditorEvent.Delete(oldFragment.toString(), editor, timeStamp));
                                    }
                                }
                            }
                        });

                        if (editor != null) {

                            // Subscribe to scrolling events
                            editor.getScrollingModel().addVisibleAreaListener(visibleAreaEvent -> {

                                if (!project.isDisposed()) {
                                    DifficultyTrigger publisher = project.getMessageBus().syncPublisher(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC);

                                    Rectangle oldRectangle = visibleAreaEvent.getOldRectangle();
                                    Rectangle newRectangle = visibleAreaEvent.getNewRectangle();

                                    if (oldRectangle != null && oldRectangle.getCenterY() != newRectangle.getCenterY()) {
                                        publisher.doEdit(new EditorEvent.Scroll(fileEditorManager.getSelectedTextEditor(), System.currentTimeMillis()));
                                    }
                                }
                            });

                            // Subscribe to mouse events
                            editor.addEditorMouseListener(new EditorMouseListener() {

                                @Override
                                public void mousePressed(EditorMouseEvent editorMouseEvent) {

                                }

                                @Override
                                public void mouseClicked(EditorMouseEvent editorMouseEvent) {

                                    if (!project.isDisposed()) {
                                        DifficultyTrigger publisher = project.getMessageBus().syncPublisher(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC);

                                        publisher.doEdit(new EditorEvent.Click(fileEditorManager.getSelectedTextEditor(), System.currentTimeMillis()));
                                    }
                                }

                                @Override
                                public void mouseReleased(EditorMouseEvent editorMouseEvent) {

                                }

                                @Override
                                public void mouseEntered(EditorMouseEvent editorMouseEvent) {

                                }

                                @Override
                                public void mouseExited(EditorMouseEvent editorMouseEvent) {

                                }
                            });
                        }
                    }
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
        return COMPONENT_ID;
    }
}
