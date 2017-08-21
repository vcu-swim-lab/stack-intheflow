package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.tags.UserTagStatComponent;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;

public class SearchToolWindowFactory {
    private JPanel content;

    public SearchToolWindowGUI buildGUI(@NotNull ToolWindow toolWindow, Project project) {
        SearchToolWindowGUI windowGUI = new SearchToolWindowGUIBuilder()
                .setContent(content)
                .setProject(project)
                .setSearchModel(project.getComponent(UserTagStatComponent.class).getSearchModel()).build();
        ContentFactory contentFactory = ContentFactory.SERVICE.getInstance();
        Content windowContent = contentFactory.createContent(windowGUI.getContentPanel(), "", false);
        toolWindow.getContentManager().addContent(windowContent);
        return windowGUI;
    }
}
