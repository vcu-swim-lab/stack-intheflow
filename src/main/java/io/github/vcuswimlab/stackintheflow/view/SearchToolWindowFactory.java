package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.wm.ToolWindow;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;

public class SearchToolWindowFactory {

    private JButton searchButton;
    private JTextField searchBox;
    private JPanel content;
    private JList<Question> resultsList;
    private JScrollPane resultsScrollPane;
    private JPanel searchJPanel;
    private JEditorPane consoleErrorPane;

    public SearchToolWindowGUI buildGUI(@NotNull ToolWindow toolWindow) {
        SearchToolWindowGUI windowGUI = new SearchToolWindowGUI.SearchToolWindowGUIBuilder()
                .setContent(content)
                .setConsoleErrorPane(consoleErrorPane)
                .setResultsList(resultsList)
                .setResultsScrollPane(resultsScrollPane)
                .setSearchBox(searchBox)
                .setSearchButton(searchButton)
                .setSearchJPanel(searchJPanel).build();
        ContentFactory contentFactory = ContentFactory.SERVICE.getInstance();
        Content windowContent = contentFactory.createContent(windowGUI.getContentPanel(), "", false);
        toolWindow.getContentManager().addContent(windowContent);
        return windowGUI;
    }
}
