package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.ide.ui.LafManager;
import com.intellij.openapi.project.Project;
import com.intellij.util.ui.UIUtil;
import io.github.vcuswimlab.stackintheflow.controller.Logging;
import io.github.vcuswimlab.stackintheflow.controller.QueryExecutor;
import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Question;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;
import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.util.Pair;
import netscape.javascript.JSObject;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chase on 5/18/17.
 */
public class SearchToolWindowGUI {
    private JPanel content;
    private Logging logger = new Logging() ;
    private Project project;

    private PersonalSearchModel searchModel;

    private WebView webView;
    private JFXPanel jfxPanel;
    private WebEngine engine;
    private JSObject window;
    private JavaBridge bridge;

    private SearchToolWindowGUI(JPanel content, Project project,
                                PersonalSearchModel searchModel) {
        this.content = content;
        this.project = project;
        this.searchModel = searchModel;
        bridge = new JavaBridge(this);
        initComponents();
    }

    private void initComponents(){
        LafManager.getInstance().addLafManagerListener(source -> updateUISettings());

        jfxPanel = new JFXPanel();
        createScene();
        content.setLayout(new BorderLayout());
        content.add(jfxPanel, BorderLayout.CENTER);
        Platform.setImplicitExit(false);
    }

    private void createScene(){
        Platform.runLater(() -> {
            StackPane root = new StackPane();
            Scene scene = new Scene(root);
            webView = new WebView();
            engine = webView.getEngine();

            String htmlFileURL = this.getClass().getClassLoader().getResource("SearchToolWindow.html").toExternalForm();
            engine.load(htmlFileURL);

            engine.getLoadWorker().stateProperty().addListener((ov, oldState, newState) -> {
                if(newState == Worker.State.SUCCEEDED) {
                    window = (JSObject) engine.executeScript("window");
                    window.setMember("JavaBridge", bridge);
                    window.call("initialize");
                    updateUISettings();
                }
            });
            root.getChildren().add(webView);

            jfxPanel.setScene(scene);
        });
    }

    public void updateUISettings(){
        Platform.runLater(() -> {
            boolean isDark = UIUtil.isUnderDarcula();
            window.call("updateUISettings", isDark);
        });
    }

    public void autoQuery(String query, boolean backoff, String reasoning){ //reasoning is either "action" or "difficulty"
        Platform.runLater(() -> {
            window.call("autoSearch", query, backoff, reasoning);
            window.call("updateUISearchType", "Relevance");
        });
    }

    public void errorQuery(List<String> parsedMessages, boolean backoff, String reasoning){ //Reasoning is either "runtime" or "compiler"
        Platform.runLater(() -> {
            Pair<String, List<Question>> questionListPair = retrieveResults(parsedMessages.get(1), "", backoff, JerseyGet.SortType.RELEVANCE);
            if(questionListPair.getValue().isEmpty()) {
                questionListPair = retrieveResults(parsedMessages.get(0), "", backoff, JerseyGet.SortType.RELEVANCE);
            }
            window.call("reset");
            window.call("resetSearchTags");
            window.call("showAutoQueryIcon", reasoning);
            window.call("updateUISearchType", "Relevance");
            window.call("setSearchBox", questionListPair.getKey());
            window.call("addCurrentQueryToHistory");
            window.call("logQuery", reasoning);
            updateQuestionList(questionListPair.getValue());
        });
    }

    public void executeQuery(String query, String tags, boolean backoff, JerseyGet.SortType sortType, boolean addToQueryHistory, String reasoning) {
        Platform.runLater(() -> {
            Pair<String, List<Question>> questionListPair = retrieveResults(query, tags, backoff, sortType);
            window.call("setSearchBox", questionListPair.getKey());
            window.call("logQuery", reasoning);
            if(addToQueryHistory) {
                window.call("addCurrentQueryToHistory");
            }
            updateQuestionList(questionListPair.getValue());
        });
    }

    private Pair<String, List<Question>> retrieveResults(String query, String tags, boolean backoff, JerseyGet.SortType sortType) {
        String searchQuery = query;
        JerseyResponse jerseyResponse = QueryExecutor.executeQuery(searchQuery + " " + tags, sortType);
        List<Question> questionList = jerseyResponse.getItems();

        if(backoff && questionList.isEmpty()) {
            Deque<String> queryStack = new ArrayDeque<>();
            queryStack.addAll(Arrays.asList(searchQuery.split("\\s")));

            while (questionList.isEmpty() && queryStack.size() > 1) {
                queryStack.pop();
                searchQuery = queryStack.stream().collect(Collectors.joining(" "));
                jerseyResponse = QueryExecutor.executeQuery(searchQuery + " " + tags);
                questionList = jerseyResponse.getItems();
            }

        }

        if(sortType.equals(JerseyGet.SortType.RELEVANCE)) {
            questionList = searchModel.rankQuestionList(questionList);
        }

        return new Pair<>(searchQuery, questionList);
    }

    private void updateQuestionList(List<Question> questions) {
        for(Question question : questions){
            window.call("getQuestion", question.getTitle(), question.getBody(), question.getTags().toArray(), question.getLink());
        }
        window.call("displayQuestions");
        window.call("generateListeners");
    }

    public void log(String message){
        logger.info(message);
    }

    public void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }

    public JPanel getContentPanel() {
        return content;
    }

    public Project getProject() {
        return project;
    }

    public static class SearchToolWindowGUIBuilder {
        private JPanel content;
        private PersonalSearchModel searchModel;
        private Project project;

        public SearchToolWindowGUIBuilder setContent(JPanel content) {
            this.content = content;
            return this;
        }

        public SearchToolWindowGUIBuilder setSearchModel(PersonalSearchModel searchModel) {
            this.searchModel = searchModel;
            return this;
        }

        public SearchToolWindowGUIBuilder setProject(Project project) {
            this.project = project;
            return this;
        }

        public SearchToolWindowGUI build() {
            return new SearchToolWindowGUI(
                    content,
                    project,
                    searchModel
            );
        }
    }
}
