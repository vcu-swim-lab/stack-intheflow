package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.tags.UserTagStatComponent;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;

public class SearchToolWindowFactory {
    private static final String NO_JAVAFX_FOUND_MESSAGE =
            "<html>" +
                "Your platform does not support JavaFX." +
                "<br />" +
                "Please follow the <a href=\"\">instructions</a> to install the dependencies." +
            "</html>";

    private JPanel content;

    public SearchToolWindowGUI buildGUI(@NotNull ToolWindow toolWindow, Project project) {
        SearchToolWindowGUI windowGUI = new SearchToolWindowGUIBuilder()
                .setContent(content)
                .setProject(project)
                .setSearchModel(project.getComponent(UserTagStatComponent.class).getSearchModel()).build();

        ContentFactory contentFactory = ContentFactory.SERVICE.getInstance();
        Content windowContent;

        // Check if javafx was found
        if(windowGUI != null) {
            windowContent = contentFactory.createContent(windowGUI.getContentPanel(), "", false);
        } else {
            windowContent = contentFactory.createContent(getNoJavaFXFoundPanel(), "", false);
        }

        toolWindow.getContentManager().addContent(windowContent);
        return windowGUI;
    }

    private static JPanel getNoJavaFXFoundPanel() {
        JPanel noJavaFXFoundPanel = new JPanel(new GridBagLayout());
        noJavaFXFoundPanel.add(new JLabel(NO_JAVAFX_FOUND_MESSAGE));
        return noJavaFXFoundPanel;
    }
}
