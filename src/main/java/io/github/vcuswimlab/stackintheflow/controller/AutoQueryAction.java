package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowManager;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.view.JavaBridge;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI;

import java.util.List;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryAction extends AnAction {

    private List<String> compilerMessages;

    @Override
    public void update(final AnActionEvent e) {
        //Get required data keys
        final Project project = e.getData(CommonDataKeys.PROJECT);
        final Editor editor = e.getData(CommonDataKeys.EDITOR);
        //Set visibility only in case of existing project and editor
        e.getPresentation().setVisible((project != null && editor != null));
    }

    @Override
    public void actionPerformed(AnActionEvent e) {
        //Get required data keys
        final Project project = e.getData(CommonDataKeys.PROJECT);
        final Editor editor = e.getData(CommonDataKeys.EDITOR);

        //Generate the autoQuery
        String autoQuery = project.getComponent(TermStatComponent.class).generateQuery(editor);

        //Execute Search and Open Tool Window
        ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow("StackInTheFlow");
        SearchToolWindowGUI toolWindowGUI = project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI();
        toolWindowGUI.autoQuery(autoQuery, true, "action");
        toolWindow.activate(() -> {
        });
    }
}
