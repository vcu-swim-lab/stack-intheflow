package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.tags.UserTagStatComponent;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.event.HyperlinkEvent;
import java.awt.*;

public class SearchToolWindowFactory {
    private JPanel content;

    private static final String NO_JAVAFX_FOUND_MESSAGE =
            "<html>" +
                    "Your platform does not support JavaFX." +
                    "<br />" +
                    "Please follow these <a href=\"http://github.com/vcu-swim-lab/stack-intheflow\">instructions</a> to install the dependencies." +
            "</html>";

    public SearchToolWindowGUI buildGUI(@NotNull ToolWindow toolWindow, Project project) {

        SearchToolWindowGUI windowGUI = null;
        JPanel jPanel;
        if(isJavaFXAvailable()) {
            windowGUI = new SearchToolWindowGUI.SearchToolWindowGUIBuilder()
                    .setContent(content)
                    .setProject(project)
                    .setSearchModel(project.getComponent(UserTagStatComponent.class).getSearchModel()).build();
            jPanel = windowGUI.getContentPanel();
        } else {
            jPanel = getNoJavaFXFoundPanel();
        }

        ContentFactory contentFactory = ContentFactory.SERVICE.getInstance();
        Content windowContent = contentFactory.createContent(jPanel, "", false);
        toolWindow.getContentManager().addContent(windowContent);

        return windowGUI;
    }

    private static JPanel getNoJavaFXFoundPanel() {
        JPanel noJavaFXFoundPanel = new JPanel(new GridBagLayout());
        JEditorPane noJavaFXFoundPane = new JEditorPane();
        noJavaFXFoundPane.setContentType("text/html");
        noJavaFXFoundPane.setEditable(false);
        noJavaFXFoundPane.setOpaque(false);
        noJavaFXFoundPane.setText(NO_JAVAFX_FOUND_MESSAGE);
        noJavaFXFoundPane.addHyperlinkListener(e -> {
            if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
                BrowserLauncher.getInstance().browse(e.getDescription(), WebBrowserManager.getInstance().getFirstActiveBrowser());
            }
        });
        noJavaFXFoundPanel.add(noJavaFXFoundPane);
        return noJavaFXFoundPanel;
    }

    private static boolean isJavaFXAvailable() {
        try {
            return Class.forName("javafx.scene.web.WebView") != null;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
