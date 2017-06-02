package io.github.vcuswimlab.stackintheflow.controller.error;

import com.intellij.execution.ui.ConsoleView;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.project.Project;
import icons.StackInTheFlowIcons;
import io.github.vcuswimlab.stackintheflow.controller.component.ConsoleErrorComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI;

import java.util.Arrays;
import java.util.List;

/**
 * <h1>ConsoleQueryAction</h1>
 * Created on: 5/31/2017
 *
 * @author Tyler John Haden
 */
public class ConsoleQueryAction extends AnAction {

    public ConsoleQueryAction(){
        super("Query runtime error", null, StackInTheFlowIcons.ACTION_ICON);
    }

    @Override
    public void actionPerformed(AnActionEvent anActionEvent) {
        Project project = anActionEvent.getProject();
        String errorMessage = project.getComponent(ConsoleErrorComponent.class).getError();
        if(!errorMessage.isEmpty()) {
            List<String> parsedMessages = ErrorMessageParser.parseRuntimeError(errorMessage, project);
            project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().setConsoleError(parsedMessages);
        }
    }
}
