package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.execution.filters.InputFilter;
import com.intellij.openapi.components.ProjectComponent;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

/**
 * <h1>ConsoleErrorComponent</h1>
 * Created on: 5/31/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.ConsoleErrorComponent";

    private Map<InputFilter, StringBuilder> messageBuilder;

    @Override
    public void initComponent() {
        messageBuilder = new HashMap<>();
    }

    public void appendError(InputFilter console, String line) {
        if (!messageBuilder.containsKey(console)) {
            messageBuilder.put(console, new StringBuilder());
        }
        messageBuilder.get(console).append(line);
    }

    public String getError(InputFilter console) {
        if (messageBuilder.containsKey(console)) {
            String errorMessage = messageBuilder.get(console).toString();
            messageBuilder.remove(console);
            return errorMessage;
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
