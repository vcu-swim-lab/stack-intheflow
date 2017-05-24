package io.github.vcuswimlab.stackintheflow.model.difficulty;

import com.intellij.openapi.project.Project;
import com.intellij.util.messages.MessageBus;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.DifficultyTrigger;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEvent;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEventType;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI;

import java.util.ArrayDeque;
import java.util.EnumMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Created by Chase on 5/23/2017.
 */
public class DifficultyModel {

    private static final double DELETE_RATIO = .6;
    private static final int QUERY_DELAY = 30; //Delay in seconds
    private static final int INACTIVE_DELAY = 15; //Delay in minutes
    private final int MAX_QUEUE_SIZE = 25;

    private Project project;
    private MessageBus messageBus;
    private MessageBusConnection connection;

    private Map<EditorEventType, Integer> eventCounts;

    private Queue<EditorEvent> eventQueue;
    private State currentState;
    private ScheduledThreadPoolExecutor timer;
    private ScheduledFuture<?> inactiveTaskFuture;
    private SearchToolWindowGUI gui;

    public DifficultyModel(Project project) {
        this.project = project;
        messageBus = project.getMessageBus();
        connection = messageBus.connect();
        eventCounts = new EnumMap<>(EditorEventType.class);
        eventQueue = new ArrayDeque<>(MAX_QUEUE_SIZE);

        currentState = State.COLLECT;
        timer = new ScheduledThreadPoolExecutor(1);

        gui = project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI();

        connection.subscribe(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC, event -> {

            switch (currentState) {
                case PAUSE:
                    //Transition to collect state, queue event
                    currentState = State.COLLECT;
                    System.out.println("COLLECT!");
                case COLLECT:
                    eventCounts.put(event.getType(), eventCounts.getOrDefault(event.getType(), 0) + 1);

                    //Reset Inactive Task
                    if (inactiveTaskFuture != null) {
                        inactiveTaskFuture.cancel(true);
                    }
                    inactiveTaskFuture = timer.schedule(() -> {
                        currentState = State.PAUSE;
                        System.out.println("PAUSE!");
                    }, INACTIVE_DELAY, TimeUnit.MINUTES);

                    if (isFull()) {
                        EditorEvent oldEvent = eventQueue.poll();
                        eventCounts.put(oldEvent.getType(), eventCounts.getOrDefault(oldEvent.getType(), 1) - 1);

                        // If we have crossed the threshold, initiate a query and transition to query state
                        if (getRatio(EditorEventType.DELETE) >= DELETE_RATIO) {

                            //Fire query
                            String autoQuery = project.getComponent(TermStatComponent.class).generateQuery(event.getEditor());
                            project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().executeQuery(autoQuery);

                            System.out.println("QUERY!");
                            eventQueue.clear();
                            currentState = State.QUERY;

                            //After QUERY_DELAY seconds, transition to collect state
                            timer.schedule(() -> currentState = State.COLLECT, QUERY_DELAY, TimeUnit.SECONDS);

                            //Remove the inactive task as we no longer need it in this state
                            inactiveTaskFuture.cancel(true);
                        }
                    }

                    eventQueue.offer(event);
                    break;
                case QUERY:
                    //Do nothing
                    break;
            }
        });
    }

    private boolean isFull() {
        return eventQueue.size() == 25;
    }

    private double getRatio(EditorEventType eventType) {

        if (eventCounts.isEmpty()) {
            return 0;
        }

        return eventCounts.getOrDefault(eventType, 0) / (double) eventQueue.size();
    }

    private enum State {
        COLLECT, QUERY, PAUSE
    }
}
