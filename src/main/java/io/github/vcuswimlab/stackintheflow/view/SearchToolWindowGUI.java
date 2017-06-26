package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.sun.javafx.application.PlatformImpl;
import io.github.vcuswimlab.stackintheflow.controller.QueryExecutor;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Question;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;
import javafx.collections.ObservableList;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import netscape.javascript.JSObject;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.stream.Collectors;

/**
 * Created by chase on 5/18/17.
 */
public class SearchToolWindowGUI {
    private JPanel content;
    private DefaultListModel<Question> questionListModel;

    private PersonalSearchModel searchModel;

    private List<String> compilerMessages;

    private ScheduledThreadPoolExecutor timer;

    private Stage stage;
    private WebView webView;
    private JFXPanel jfxPanel;
    private WebEngine engine;

    private SearchToolWindowGUI(JPanel content,
                                PersonalSearchModel searchModel) {
        this.content = content;
        this.searchModel = searchModel;

        compilerMessages = new ArrayList<>();

        timer = new ScheduledThreadPoolExecutor(1);

        initComponents();
        //addListeners();
    }

    private void initComponents(){
        jfxPanel = new JFXPanel();
        createScene();

        content.setLayout(new BorderLayout());
        content.add(jfxPanel, BorderLayout.CENTER);
    }

    private void createScene(){
        PlatformImpl.startup(() -> {
            stage = new Stage();

            stage.setTitle("Stack in the Flow");
            stage.setResizable(true);

            Group root = new Group();
            Scene scene = new Scene(root,80,20);
            stage.setScene(scene);

            // Set up the embedded browser:
            webView = new WebView();
            engine = webView.getEngine();

            String htmlFileURL = SearchToolWindowGUI.class.getResource("SearchToolWindow.html").toExternalForm();
            engine.load(htmlFileURL);

            JSObject jsobj = (JSObject) engine.executeScript("window");
            jsobj.setMember("JavaBridge", new JavaBridge());

            ObservableList<Node> children = root.getChildren();
            children.add(webView);

            jfxPanel.setScene(scene);
        });
    }

    /*
    private void addListeners() {

        consoleErrorPane.setVisible(false);

        searchButton.addActionListener(e -> executeQuery(searchBox.getText(), false));
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
                JList<String> list = (JList<String>) evt.getSource();
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

                Question question = questionListModel.get(index);

                // If they are expanding the question, increase the tags in the search model
                if (!question.isExpanded()) {
                    searchModel.increaseTags(question.getTags());
                }

                question.toggleExpanded();
                refreshListView();
                return true;
            }
        });

        consoleErrorPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                executeQuery(compilerMessages.get(0), true);
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
        for (int i = 0; i < questionListModel.size(); i++) {
            questions.add(questionListModel.get(i));
        }
        return questions;
    } */

    public void executeQuery(String query, boolean backoff) {
    /*
        Future<List<Question>> questionListFuture = timer.submit(() -> {
            String searchQuery = query;

            JerseyResponse jerseyResponse = QueryExecutor.executeQuery(searchQuery);
            List<Question> questionList = jerseyResponse.getItems();

            if (backoff && questionList.isEmpty()) {

                Deque<String> queryStack = new ArrayDeque<>();
                queryStack.addAll(Arrays.asList(searchQuery.split("\\s")));

                while (questionList.isEmpty()) {
                    queryStack.pop();
                    searchQuery = queryStack.stream().collect(Collectors.joining(" "));
                    jerseyResponse = QueryExecutor.executeQuery(searchQuery);
                    questionList = jerseyResponse.getItems();
                }
            }

            setSearchBoxContent(searchQuery);
            return searchModel.rankQuesitonList(questionList);
        });

        try {
            updateList(questionListFuture.get());
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        */
    }
    /*
    private void updateList(List<Question> elements) {
        if (elements == null) {
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

    private void setSearchBoxContent(String content) {
        searchBox.setText(content);
    }
    */
    public void setConsoleError(List<String> compilerMessages) {
        /*
        this.compilerMessages = compilerMessages;

        if (!compilerMessages.isEmpty()) {
            HTMLEditorKit kit = new HTMLEditorKit();
            HTMLDocument doc = new HTMLDocument();
            consoleErrorPane.setEditorKit(kit);
            consoleErrorPane.setDocument(doc);
            String fontStartBlockLink = "<font color=\"" + EditorFonts.getHyperlinkColorHex() + "\">";
            String fontStartBlockDefault = "<font color=\"" + EditorFonts.getPrimaryFontColorHex() + "\">";

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
        } */
    }

    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }

    public JPanel getContentPanel() {
        return content;
    }

    public static class SearchToolWindowGUIBuilder {
        private JPanel content;
        private PersonalSearchModel searchModel;

        public SearchToolWindowGUIBuilder setContent(JPanel content) {
            this.content = content;
            return this;
        }


        public SearchToolWindowGUIBuilder setSearchModel(PersonalSearchModel searchModel) {
            this.searchModel = searchModel;
            return this;
        }

        public SearchToolWindowGUI build() {
            return new SearchToolWindowGUI(
                    content,
                    searchModel
            );
        }
    }
}
