package io.github.vcuswimlab.stackintheflow.model.difficulty.events;

import com.intellij.util.messages.Topic;

/**
 * Created by Chase on 5/23/2017.
 */
public interface DifficultyTrigger {

    Topic<DifficultyTrigger> DIFFICULTY_TRIGGER_TOPIC = Topic.create("Difficulty Trigger", DifficultyTrigger.class);

    void doEdit(EditorEvent event);
}
