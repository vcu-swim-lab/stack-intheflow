package io.github.vcuswimlab.stackintheflow.model.difficulty.events;

/**
 * Created by Chase on 5/23/2017.
 */
public abstract class EditorEvent {

    private long timeStamp;

    private EditorEvent(long timeStamp) {
        this.timeStamp = timeStamp;
    }

    public long getTimeStamp() {
        return timeStamp;
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

        public Insert(String insertedText, long timeStamp) {
            super(timeStamp);
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

        public Delete(String deletedText, long timeStamp) {
            super(timeStamp);
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

    public static class Update extends EditorEvent {

        private String oldText;
        private String newText;

        public Update(String oldText, String newText, long timeStamp) {
            super(timeStamp);
            this.oldText = oldText;
            this.newText = newText;
        }

        @Override
        public EditorEventType getType() {
            return EditorEventType.UPDATE;
        }

        public String getOldText() {
            return oldText;
        }

        public String getNewText() {
            return newText;
        }

        @Override
        public String toString() {
            return "Update{" +
                    "timeStamp=" + getTimeStamp() +
                    ", oldText='" + oldText + '\'' +
                    ", newText='" + newText + '\'' +
                    '}';
        }
    }
}
