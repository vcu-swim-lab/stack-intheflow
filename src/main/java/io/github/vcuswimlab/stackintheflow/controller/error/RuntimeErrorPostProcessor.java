package io.github.vcuswimlab.stackintheflow.controller.error;

import com.intellij.execution.actions.ConsoleActionsPostProcessor;
import com.intellij.execution.filters.Filter;
import com.intellij.execution.ui.ConsoleView;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.IconLoader;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import io.github.vcuswimlab.stackintheflow.controller.error.ConsoleQueryAction;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <h1>RuntimeErrorPostProcessor</h1>
 * Created on: 5/30/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorPostProcessor extends ConsoleActionsPostProcessor {
    @NotNull
    @Override
    public AnAction[] postProcess(@NotNull ConsoleView console, @NotNull AnAction[] actions) {
        List<AnAction> anActions = new ArrayList<>();
        anActions.addAll(Arrays.asList(actions));
        anActions.add(new ConsoleQueryAction());
        return anActions.toArray(new AnAction[anActions.size()]);
    }
}
