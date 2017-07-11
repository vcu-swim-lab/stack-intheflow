package io.github.vcuswimlab.stackintheflow.model.difficulty.events;

/**
 * Created by Chase on 5/23/2017.
 */
public enum EditorEventType {

    INSERT("Insert"), DELETE("Delete"), SCROLL("Scroll"), CLICK("Click");

    private String string;

    EditorEventType(String string) {
        this.string = string;
    }

    @Override
    public String toString() {
        return string;
    }
}
