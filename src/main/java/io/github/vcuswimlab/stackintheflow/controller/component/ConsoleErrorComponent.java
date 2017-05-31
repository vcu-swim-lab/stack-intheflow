package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.project.Project;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

/**
 * <h1>ConsoleErrorComponent</h1>
 * Created on: 5/31/2017
 *
 * @author Tyler John Haden
 */
public class ConsoleErrorComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.ConsoleErrorComponent";

    private StringBuilder errorMessage;

    @Override
    public void initComponent() {
        errorMessage = new StringBuilder();
    }

    public void clearError() {
        errorMessage.setLength(0);
    }

    public void appendError(String line) {
        errorMessage.append(line);
    }

    public String getError() {
        return errorMessage.toString();
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
