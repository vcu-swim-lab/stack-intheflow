package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.execution.filters.InputFilter;
import com.intellij.openapi.components.ProjectComponent;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * <h1>ConsoleErrorComponent</h1>
 * Created on: 5/31/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.ConsoleErrorComponent";

    private Map<InputFilter, Map<String, StringBuilder>> messageBuilder;

    @Override
    public void initComponent() {
        messageBuilder = new HashMap<>();
    }

    public void appendMessage(InputFilter console, String type, String line) {
        if (!messageBuilder.containsKey(console)) {
            messageBuilder.put(console, new HashMap<>());
            messageBuilder.get(console).put(type, new StringBuilder());
        }
        messageBuilder.get(console).get(type).append(line);
    }

    public Map<String, List<String>> getMessages(InputFilter console) {
        if (messageBuilder.containsKey(console)) {
            Map<String, List<String>> builtMessage = messageBuilder.get(console).keySet().stream()
                    .collect(Collectors.toMap(m -> m, m -> Collections.singletonList(messageBuilder.get(console).get(m).toString())));

            messageBuilder.remove(console);
            return builtMessage;
        } else {
            return null;
        }
    }

    @NotNull
    @Override
    public String getComponentName() {
        return COMPONENT_ID;
    }

    @Override
    public void projectOpened() {

    }

    @Override
    public void projectClosed() {

    }

    @Override
    public void disposeComponent() {

    }
}
