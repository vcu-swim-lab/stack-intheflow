package io.github.vcuswimlab.stackintheflow.model.difficulty.events;

import com.intellij.openapi.editor.Editor;

/**
 * Created by Chase on 5/23/2017.
 */
public abstract class EditorEvent {

    private Editor editor;
    private long timeStamp;

    private EditorEvent(Editor editor, long timeStamp) {
        this.editor = editor;
        this.timeStamp = timeStamp;
    }

    public long getTimeStamp() {
        return timeStamp;
    }

    public Editor getEditor() {
        return editor;
    }

    public abstract EditorEventType getType();

    @Override
    public String toString() {
        return "EditorEvent{" +
                "timeStamp=" + timeStamp +
                '}';
    }

    public static class Insert extends EditorEvent {

        private String insertedText;

        public Insert(String insertedText, Editor editor, long timeStamp) {
            super(editor, timeStamp);
            this.insertedText = insertedText;
        }

        @Override
        public EditorEventType getType() {
            return EditorEventType.INSERT;
        }

        public String getInsertedText() {
            return insertedText;
        }

        @Override
        public String toString() {
            return "Insert{" +
                    "timeStamp=" + getTimeStamp() +
                    ", insertedText='" + insertedText + '\'' +
                    '}';
        }
    }

    public static class Delete extends EditorEvent {

        private String deletedText;

        public Delete(String deletedText, Editor editor, long timeStamp) {
            super(editor, timeStamp);
            this.deletedText = deletedText;
        }

        @Override
        public EditorEventType getType() {
            return EditorEventType.DELETE;
        }

        public String getDeletedText() {
            return deletedText;
        }

        @Override
        public String toString() {
            return "Delete{" +
                    "timeStamp=" + getTimeStamp() +
                    ", deletedText='" + deletedText + '\'' +
                    '}';
        }
    }

    public static class Scroll extends EditorEvent {

        public Scroll(Editor editor, long timeStamp) {
            super(editor, timeStamp);
        }

        @Override
        public EditorEventType getType() {
            return EditorEventType.SCROLL;
        }

        @Override
        public String toString() {
            return "Scroll{" +
                    "timeStamp=" + getTimeStamp() +
                    '}';
        }
    }

    public static class Click extends EditorEvent {

        public Click(Editor editor, long timeStamp) {
            super(editor, timeStamp);
        }

        @Override
        public EditorEventType getType() {
            return EditorEventType.CLICK;
        }

        @Override
        public String toString() {
            return "Click{" +
                    "timeStamp=" + getTimeStamp() +
                    '}';
        }
    }
}
