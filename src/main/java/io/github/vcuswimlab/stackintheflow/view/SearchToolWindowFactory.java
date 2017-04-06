package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.controller.QueryExecutor;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

public class SearchToolWindowFactory implements ToolWindowFactory {

    private static SearchToolWindowFactory instance;
    private JButton searchButton;
    private JTextField searchBox;
    private JPanel content;
    private JList<Question> resultsList;
    private JScrollPane resultsScrollPane;
    private JPanel searchJPanel;
    private JEditorPane consoleErrorPane;
    private ToolWindow toolWindow;
    private DefaultListModel<Question> questionListModel;

    private List<String> compilerMessages;

    public SearchToolWindowFactory() {
        //Hide the console error area until it is needed
        consoleErrorPane.setVisible(false);

        searchButton.addActionListener(e -> executeQuery(searchBox.getText()));
        searchBox.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                super.keyPressed(e);
                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                    searchButton.doClick();
                }
            }
        });
        resultsList.setListData(new Question[0]);
        questionListModel = new DefaultListModel<>();
        resultsList.setModel(questionListModel);
        QuestionRenderer renderer = new QuestionRenderer(resultsList);
        resultsList.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent componentEvent) {
                renderer.setWidth(resultsList.getWidth());
            }
        });
        resultsList.setCellRenderer(renderer);
        resultsList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent evt) {
                JList<String> list = (JList<String>)evt.getSource();
                if (evt.getClickCount() == 1 && evt.getButton() == MouseEvent.BUTTON1) {
                    if (handleSingleClick(evt, list)) return;
                }

                if (evt.getClickCount() == 2 && evt.getButton() == MouseEvent.BUTTON1) {
                    handleDoubleClick(evt, list);
                }
            }

            private void handleDoubleClick(MouseEvent evt, JList<String> list) {
                int index = list.locationToIndex(evt.getPoint());
                if (index < 0 || index >= questionListModel.size()) {
                    return;
                }

                openBrowser(questionListModel.get(index).getLink());
            }

            private boolean handleSingleClick(MouseEvent evt, JList<String> list) {
                int index = list.locationToIndex(evt.getPoint());
                if (index < 0 || index >= questionListModel.size()) {
                    return false;
                }

                questionListModel.get(index).toggleExpanded();
                refreshListView();
                return true;
            }
        });
        instance = this;

        consoleErrorPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                setSearchBoxContent(compilerMessages.get(0));
                executeQuery(compilerMessages.get(0));
                consoleErrorPane.setVisible(false);
            }
        });
    }

    private void refreshListView() {
        updateList(listModelToList());
    }

    @NotNull
    private List<Question> listModelToList() {
        List<Question> questions = new ArrayList<>();
        for(int i = 0; i < questionListModel.size(); i++) {
            questions.add(questionListModel.get(i));
        }
        return questions;
    }

    public static SearchToolWindowFactory getInstance() {
        return instance;
    }

    @Override
    public void createToolWindowContent(@NotNull Project project, @NotNull ToolWindow toolWindow) {
        this.toolWindow = toolWindow;
        ContentFactory contentFactory = ContentFactory.SERVICE.getInstance();
        Content windowContent = contentFactory.createContent(content, "", false);
        toolWindow.getContentManager().addContent(windowContent);
    }

    private void executeQuery(String query) {
        JerseyResponse jerseyResponse = QueryExecutor.executeQuery(query);
        List<Question> questionList = jerseyResponse.getItems();
        updateList(questionList);
    }

    public void updateList(List<Question> elements) {
        if(elements == null) {
            return;
        }

        questionListModel.clear();
        for (Question element : elements) {
            questionListModel.addElement(element);
        }

        if (elements.isEmpty()) {
            questionListModel.addElement(new Question(null, "Sorry, your search returned no results :(", "", "", "http://www.stackoverflow.com"));
        }
    }

    public void setSearchBoxContent(String content) {
        searchBox.setText(content);
    }

    public void setConsoleError(List<String> compilerMessages) {
        this.compilerMessages = compilerMessages;

        if (!compilerMessages.isEmpty()) {
            HTMLEditorKit kit = new HTMLEditorKit();
            HTMLDocument doc = new HTMLDocument();
            consoleErrorPane.setEditorKit(kit);
            consoleErrorPane.setDocument(doc);
            String fontStartBlockLink = "<font color=\""+ EditorFonts.getHyperlinkColorHex() +"\">";
            String fontStartBlockDefault = "<font color=\""+ EditorFonts.getPrimaryFontColorHex() +"\">";

            try {
                kit.insertHTML(doc, 0, fontStartBlockDefault + "search for: " + "</font><a href=\"\"><u>" +
                                fontStartBlockLink + compilerMessages.get(0) +
                                "</font></u></a>",
                        0, 0, null);
            } catch (Exception e) {
                e.printStackTrace();
            }
            consoleErrorPane.setVisible(true);
        } else {
            consoleErrorPane.setVisible(false);
        }
    }

    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
