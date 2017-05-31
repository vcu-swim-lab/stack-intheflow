package io.github.vcuswimlab.stackintheflow.controller.error;

import com.intellij.execution.filters.ConsoleInputFilterProvider;
import com.intellij.execution.filters.InputFilter;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.ConsoleErrorComponent;
import org.jetbrains.annotations.NotNull;

/**
 * <h1>RuntimeErrorFilterProvider</h1>
 * Created on: 5/30/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorFilterProvider implements ConsoleInputFilterProvider {
    @NotNull
    @Override
    public InputFilter[] getDefaultFilters(@NotNull Project project) {
        ConsoleErrorComponent consoleErrorComponent = project.getComponent(ConsoleErrorComponent.class);
        consoleErrorComponent.clearError();

        // s refers to a single line, this filter will be called for each line separately during execution
        InputFilter errorInputFilter = (s, consoleViewContentType) -> {
            if (consoleViewContentType.equals(ConsoleViewContentType.ERROR_OUTPUT)) {
                consoleErrorComponent.appendError(s);
            }
            return null;
        };

        return new InputFilter[]{errorInputFilter};
    }
}
