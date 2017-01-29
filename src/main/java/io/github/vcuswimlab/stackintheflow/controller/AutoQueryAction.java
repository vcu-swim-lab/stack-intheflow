package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.openapi.wm.ToolWindowManager;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowFactory;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryAction extends AnAction {

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

        final Document document = editor.getDocument();
        String text = document.getText();

        String autoQuery = AutoQueryGenerator.generateQuery(text);
        JerseyResponse response = QueryExecutor.executeQuery(autoQuery);

        //Populate tool window with autoQuery search results
        ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow("StackInTheFlow");
        SearchToolWindowFactory toolWindowFactory = SearchToolWindowFactory.getInstance();
        toolWindowFactory.updateList(response.getItems());
        toolWindow.activate(() -> {
        });
    }
}
