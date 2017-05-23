package io.github.vcuswimlab.stackintheflow.model.difficulty;

import com.intellij.util.messages.MessageBus;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.DifficultyTrigger;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEvent;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEventType;

import java.util.ArrayDeque;
import java.util.EnumMap;
import java.util.Map;
import java.util.Queue;

/**
 * Created by Chase on 5/23/2017.
 */
public class DifficultyModel {

    private final int MAX_QUEUE_SIZE = 25;

    private MessageBus messageBus;
    private MessageBusConnection connection;

    private Map<EditorEventType, Integer> eventCounts;

    private Queue<EditorEvent> eventQueue;

    public DifficultyModel(MessageBus messageBus) {
        this.messageBus = messageBus;
        connection = messageBus.connect();
        eventCounts = new EnumMap<>(EditorEventType.class);
        eventQueue = new ArrayDeque<>(MAX_QUEUE_SIZE);

        connection.subscribe(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC, event -> {
            System.out.println(event.toString());
            eventCounts.put(event.getType(), eventCounts.getOrDefault(event.getType(), 0) + 1);

            if (isFull()) {
                EditorEvent oldEvent = eventQueue.poll();
                eventCounts.put(oldEvent.getType(), eventCounts.getOrDefault(oldEvent.getType(), 1) - 1);
            }

            eventQueue.offer(event);
        });
    }

    public boolean isFull() {
        return eventQueue.size() == 25;
    }

    public double getRatio(EditorEventType eventType) {

        if (eventCounts.isEmpty()) {
            return 0;
        }

        return eventCounts.getOrDefault(eventType, 0) / (double) eventCounts.size();
    }
}
