package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.execution.filters.ConsoleInputFilterProvider;
import com.intellij.execution.filters.InputFilter;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import org.jetbrains.annotations.NotNull;

import java.util.*;

/**
 * <h1>RuntimeErrorFilter</h1>
 * Project: stack-intheflow
 * Created on: 5/28/2017
 *
 * @author Tyler John Haden
 * @version 1.0
 */
public class RuntimeErrorFilter implements ConsoleInputFilterProvider {

    private static final StringBuilder errorMessageBuilder = new StringBuilder();

    private static final ConsoleViewContentType SYSTEM_OUTPUT = ConsoleViewContentType.SYSTEM_OUTPUT;
    private static final ConsoleViewContentType ERROR_OUTPUT = ConsoleViewContentType.ERROR_OUTPUT;
    private static final String END_STATEMENT = "Process finished with exit code";

    @NotNull
    @Override
    public InputFilter[] getDefaultFilters(@NotNull Project project) {
        errorMessageBuilder.setLength(0);

        InputFilter inputFilter = (s, consoleViewContentType) -> {

            if(consoleViewContentType.equals(ERROR_OUTPUT)) {
                errorMessageBuilder.append(s);

            } else if(consoleViewContentType.equals(SYSTEM_OUTPUT) && s.contains(END_STATEMENT)) {
                project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI()
                        .setConsoleError(Arrays.asList(errorMessageBuilder.toString()));
            }
            return null;
        };

        return new InputFilter[]{inputFilter};
    }
}
