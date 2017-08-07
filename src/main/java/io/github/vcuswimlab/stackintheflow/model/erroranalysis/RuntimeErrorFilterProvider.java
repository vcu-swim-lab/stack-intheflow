package io.github.vcuswimlab.stackintheflow.model.erroranalysis;

import com.intellij.execution.filters.ConsoleInputFilterProvider;
import com.intellij.execution.filters.InputFilter;
import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.Pair;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.RuntimeErrorComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.regex.Pattern;

/**
 * <h1>RuntimeErrorFilterProvider</h1>
 * Created on: 5/30/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorFilterProvider implements ConsoleInputFilterProvider {

    private static Pattern endOfExecutionPattern = Pattern.compile("\nProcess finished with exit code \\d+\n");

    @NotNull
    @Override
    public InputFilter[] getDefaultFilters(@NotNull Project project) {
        return new InputFilter[]{new ConsoleErrorInputFilter(project)};
    }

    private class ConsoleErrorInputFilter implements InputFilter {

        private Project project;
        private RuntimeErrorComponent runtimeErrorComponent;

        private ConsoleErrorInputFilter(Project project) {
            this.project = project;
            this.runtimeErrorComponent = project.getComponent(RuntimeErrorComponent.class);
        }

        @Nullable
        @Override
        public List<Pair<String, ConsoleViewContentType>> applyFilter(String s, ConsoleViewContentType consoleViewContentType) {
            if(ServiceManager.getService(PersistSettingsComponent.class).runtimeErrorEnabled()) {
                if (consoleViewContentType.equals(ConsoleViewContentType.ERROR_OUTPUT)) {
                    // all ERROR_OUTPUT is a result of runtime error
                    runtimeErrorComponent.appendMessage(this, "ERROR", s);

//            } else if (consoleViewContentType.equals(ConsoleViewContentType.LOG_WARNING_OUTPUT)) {
//                runtimeErrorComponent.appendMessage(this, "WARNING", s);

                } else if (consoleViewContentType.equals(ConsoleViewContentType.SYSTEM_OUTPUT) &&
                        endOfExecutionPattern.matcher(s).matches()) {
                    // if SYSTEM_OUTPUT sends '\nProcess finished with exit code \d+\n', execution has completed

                    // get error messages from component, remove 'this' instance from hash map
                    ErrorMessage runtimeErrorMessage = runtimeErrorComponent.getMessages(this);

                    // if 'consoleErrorComponent.appendError()' was never called, null is returned
                    if (runtimeErrorMessage != null && runtimeErrorMessage.get(ErrorMessage.MessageType.ERROR).length != 0) {
                        List<String> parsedMessages = ErrorMessageParser.parseRuntimeError(runtimeErrorMessage, project);
                        project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().errorQuery(parsedMessages, false, "runtime");
                    }
                } else if (consoleViewContentType.equals(ConsoleViewContentType.SYSTEM_OUTPUT)){
                    runtimeErrorComponent.appendMessage(this, "INFORMATION", s);
                }
            }
            return null;
        }
    }
}
