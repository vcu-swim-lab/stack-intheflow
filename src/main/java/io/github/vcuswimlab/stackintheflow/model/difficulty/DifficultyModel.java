package io.github.vcuswimlab.stackintheflow.model.difficulty;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowManager;
import com.intellij.util.messages.MessageBus;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.DifficultyTrigger;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEvent;
import io.github.vcuswimlab.stackintheflow.model.difficulty.events.EditorEventType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Created by Chase on 5/23/2017.
 */
public class DifficultyModel {

    private static final int EVENT_DELAY = 1000; // Delay in milliseconds
    private static final double DELETE_RATIO = .6;
    private static final double NON_EDIT_RATIO = .6;
    private static final int QUERY_DELAY = 30; // Delay in seconds
    private static final int INACTIVE_DELAY = 15; // Delay in minutes
    private final int MAX_QUEUE_SIZE = 25;
    private Logger logger = LogManager.getLogger("ROLLING_FILE_APPENDER");


    private Project project;
    private MessageBus messageBus;
    private MessageBusConnection connection;

    private Map<EditorEventType, Integer> eventCounts;

    private Deque<EditorEvent> eventQueue;
    private State currentState;
    private ScheduledThreadPoolExecutor timer;
    private ScheduledFuture<?> inactiveTaskFuture;

    public DifficultyModel(Project project) {
        this.project = project;
        messageBus = project.getMessageBus();
        connection = messageBus.connect();
        eventCounts = new EnumMap<>(EditorEventType.class);
        eventQueue = new ArrayDeque<>(MAX_QUEUE_SIZE);

        currentState = State.COLLECT;
        timer = new ScheduledThreadPoolExecutor(1);

        connection.subscribe(DifficultyTrigger.DIFFICULTY_TRIGGER_TOPIC, event -> {

            // Check to limit the number of consecutive events in a short period of time.
            EditorEvent lastEvent = eventQueue.peekLast();
            if (lastEvent != null && lastEvent.getType() == event.getType()) {

                if (event.getTimeStamp() - lastEvent.getTimeStamp() < EVENT_DELAY) {
                    return;
                }
            }

            System.out.println(event);

            switch (currentState) {
                case PAUSE:
                    // Transition to collect state, queue event
                    currentState = State.COLLECT;
                case COLLECT:
                    eventCounts.put(event.getType(), eventCounts.getOrDefault(event.getType(), 0) + 1);

                    // Reset Inactive Task
                    if (inactiveTaskFuture != null) {
                        inactiveTaskFuture.cancel(true);
                    }
                    inactiveTaskFuture = timer.schedule(() -> currentState = State.PAUSE, INACTIVE_DELAY, TimeUnit.MINUTES);

                    if (isFull()) {
                        EditorEventType oldEventType = eventQueue.poll().getType();
                        eventCounts.put(oldEventType, eventCounts.getOrDefault(oldEventType, 1) - 1);

                        // If we have crossed the threshold, initiate a query and transition to query state
                        if (getRatio(EditorEventType.DELETE, EditorEventType.INSERT) >= DELETE_RATIO ||
                                getRatio(EditorEventType.INSERT) + getRatio(EditorEventType.DELETE) < NON_EDIT_RATIO) {
                            ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow("StackInTheFlow");

                            // Check to see if the tool window is visible before generating the query.
                            if (toolWindow.isVisible()) {
                                // Generate the autoQuery
                                String autoQuery = project.getComponent(TermStatComponent.class).generateQuery(event.getEditor());

                                //Logging the threshold and event counts

                                if (getRatio(EditorEventType.DELETE, EditorEventType.INSERT) >= DELETE_RATIO) {
                                    logger.info("{DeleteRatioAutoQuery: " + autoQuery + ", Scroll: "+ Integer.toString(eventCounts.getOrDefault(EditorEventType.SCROLL, 0)) + ", Click: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.CLICK, 0)) + ", Insert: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.INSERT, 0)) + ", Delete: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.DELETE, 0)) + "}");
                                }
                                if (getRatio(EditorEventType.INSERT) + getRatio(EditorEventType.DELETE) < NON_EDIT_RATIO){
                                    logger.info("{NonEditRatioAutoQuery: " + autoQuery + ", Scroll: "+ Integer.toString(eventCounts.getOrDefault(EditorEventType.SCROLL, 0)) + ", Click: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.CLICK, 0)) + ", Insert: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.INSERT, 0)) + ", Delete: " + Integer.toString(eventCounts.getOrDefault(EditorEventType.DELETE, 0)) + "}");
                                }

                                // Execute Search
                                project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().executeQuery(autoQuery, true);
                            }

                            eventQueue.clear();
                            eventCounts.clear();

                            currentState = State.QUERY;

                            // After QUERY_DELAY seconds, transition to collect state
                            timer.schedule(() -> currentState = State.COLLECT, QUERY_DELAY, TimeUnit.SECONDS);

                            // Remove the inactive task as we no longer need it in this state
                            inactiveTaskFuture.cancel(true);
                        }
                    }

                    eventQueue.offer(event);
                    break;
                case QUERY:
                    // Do nothing
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

    private double getRatio(EditorEventType num, EditorEventType denom) {

        int numCount = eventCounts.getOrDefault(num, 0);
        int denomCount = eventCounts.getOrDefault(denom, 0);

        if (denomCount == 0) {
            return 0;
        }

        return numCount / (double) (numCount + denomCount);
    }
    private enum State {
        COLLECT, QUERY, PAUSE
    }
}
