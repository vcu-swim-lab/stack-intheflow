package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.ide.ui.LafManager;
import com.intellij.openapi.project.Project;
import com.intellij.util.ui.UIUtil;
import io.github.vcuswimlab.stackintheflow.controller.Logging;
import io.github.vcuswimlab.stackintheflow.controller.QueryExecutor;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;
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
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chase on 5/18/17.
 */

/**
 * The UI is written as an HTML page that is displayed in the JavaFX's WebView component, which allows HTML pages to be rendered in a Java environment.
 * <p>
 * Some notes on the UI:
 * <p>
 * There are a lot of messaging between JavaScript and Java because no one language has all the information necessary. This is the most complex interaction
 * in the UI code, as often times there is a chain of information being passed back and forth between the two languages. JavaFX supports this interaction
 * through two means. First is the JSObject window object, which, when initialized, allows the Java code to call a JavaScript function. The other is
 * a JavaBridge, which needs to be manually created in Java, and allows the JavaScript to call a Java method. These two objects allow for all communication
 * between the two languages.
 * <p>
 * However, a pitfall is often synchronization. ALL UI CODE MUST BE IN THE JAVAFX PLATFORM THREAD. This is an asynchronous thread, which means that a lot of
 * synchronization issues can arise. For example, a JavaScript calls method A in Java, which does its work in the Platform thread. Method A updates property X in
 * the JavaScript based on some condition. In the JavaScript file, right after calling method A, you access property X. Property X will not be updated when you access it,
 * because the Java code has not run yet. A way to get around this is in method A, call function B after updating property X, and in function B is where you access
 * property X. This ensures that property X has been updated before it is accessed. This is why there are long chains of communication back and forth in this code.
 * <p>
 * The most prevalent example is the chain of events for querying. Let's take, for example, a difficulty detection autoquery. In the backend, it is recognized that
 * the user is "stuck." The Difficulty Detection component then calls the autoQuery method in this class. This does not, however, directly execute the query, because
 * there is pre-processing that needs to be done in JS first. autoQuery(...) calls autoSearch(...) in the JS, which does some pre-processing and then calls autoQuery(...) in
 * JavaBridge. This method then calls executeQuery(...) in this class, which actually executes the query. However, we're not done, since the JS still needs some information,
 * so functions like setSearchBox(...) and addAutoQueryIcon(...) are called, which finally ends the chain.
 */

public class SearchToolWindowGUI {
    private JPanel content;
    private Logging logger;
    private Project project;

    private PersonalSearchModel searchModel;

    private WebView webView;
    private JFXPanel jfxPanel;
    private WebEngine engine;

    //These need to be instance objects or they will be garbage collected.
    private JSObject window; //Object to interact with JS. Use window.call(...) to call a JS function
    private JavaBridge bridge; //Object to interact with JS. In JS, use JavaBridge.<method>(...) to call a Java method

    private SearchToolWindowGUI(JPanel content, Project project,
                                PersonalSearchModel searchModel) {
        this.content = content;
        this.project = project;
        this.logger = new Logging(project);
        this.searchModel = searchModel;
        bridge = new JavaBridge(this);
        initComponents();

    }

    /**
     * Initialize the components of JavaFX to create the UI
     * This method is called in the constructor for SearchToolWindowGUI
     */
    private void initComponents() {
        LafManager.getInstance().addLafManagerListener(source -> updateUISettings()); //Listener for when the user changes IntelliJ color theme

        jfxPanel = new JFXPanel();
        createScene();
        content.setLayout(new BorderLayout());
        content.add(jfxPanel, BorderLayout.CENTER);

        //Chase, I'm sure you know what this line does...
        Platform.setImplicitExit(false); //See issue #90
    }

    /**
     * Creates and loads the HTML file in the proper JavaFX components.
     * This method is called in initComponents()
     */
    private void createScene(){
        Platform.runLater(() -> {
            StackPane root = new StackPane();
            Scene scene = new Scene(root);
            webView = new WebView();
            engine = webView.getEngine();

            String htmlFileURL = this.getClass().getClassLoader().getResource("SearchToolWindow.html").toExternalForm();
            engine.load(htmlFileURL);

            //Listener to make sure it is properly loaded
            engine.getLoadWorker().stateProperty().addListener((ov, oldState, newState) -> {
                if(newState == Worker.State.SUCCEEDED) {
                    window = (JSObject) engine.executeScript("window"); //Init window object
                    window.setMember("JavaBridge", bridge);
                    window.call("initialize"); //Initalizes Javascript logic. (Replacement for $(document).ready(...))
                    updateUISettings();
                }
            });
            root.getChildren().add(webView);

            jfxPanel.setScene(scene);
        });
    }

    /**
     * Update the color of the UI to match the color theme of the IntelliJ IDE
     */
    public void updateUISettings(){
        Platform.runLater(() -> {
            boolean isDark = UIUtil.isUnderDarcula();
            window.call("updateUISettings", isDark);
        });
    }

    /**
     * This method is called when the program generates an autoquery. This does not include queries based on error messages
     * This method is called from backend Components indicating that an autoquery is necessary
     *
     * @param query     - the query that was generated from the backend
     * @param backoff   - whether or not to backoff query words
     * @param reasoning - either "action", which is when the user requests an autoquery, or "difficulty", which is when the user is "stuck"
     */
    public void autoQuery(String query, boolean backoff, String reasoning){ //reasoning is either "action" or "difficulty"
        Platform.runLater(() -> {
            window.call("autoSearch", query, backoff, reasoning);
            window.call("updateUISearchType", "Relevance");
        });
    }

    /**
     * This method is called when a query is generated for an error message.
     * This method is called from the Compile and Runtime error listener components.
     *
     * @param parsedMessages - a List of two possible error messages
     * @param backoff        - whether or not to backoff
     * @param reasoning      - either "runtime" or "compiler" for their respective type of error messages
     */
    public void errorQuery(List<String> parsedMessages, boolean backoff, String reasoning) {
        if (project.getComponent(ToolWindowComponent.class).toolWindowIsVisible()) {
            Platform.runLater(() -> {
                //There are two parsed error messages. Try the second one first, if that gives nothing, then try the first one.
                Pair<String, List<Question>> questionListPair = retrieveResults(parsedMessages.get(1), "", backoff, JerseyGet.SortType.RELEVANCE);
                if (questionListPair.getValue().isEmpty()) {
                    questionListPair = retrieveResults(parsedMessages.get(0), "", backoff, JerseyGet.SortType.RELEVANCE);
                }
                //For this logic, we can't send a message to JS first, so all the stuff done in JS query functions needs to be done here.
                window.call("reset");
                window.call("resetSearchTags");
                window.call("showAutoQueryIcon", reasoning);
                window.call("updateUISearchType", "Relevance");
                window.call("setSearchBox", questionListPair.getKey());
                window.call("addCurrentQueryToHistory");
                window.call("logQuery", reasoning);
                updateQuestionList(questionListPair.getValue());
            });
        } else {
            log("\"QueryEventType\": \"" + reasoning + "\", \"Visible\": \"false\"}");
        }
    }

    /**
     * Executes the actual query and returns the results to JS
     * @param query - string to query
     * @param tags - tags to query
     * @param backoff - whether or not to backoff
     * @param sortType - how to sort the results - relevance, votes, newest, active
     * @param addToQueryHistory - whether or not this query should be added to the history
     * @param reasoning - why this query is being executed - manual, action, difficulty, compiler, runtime
     */
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

    /**
     * Issues a search request to the API
     * @return a Pair holding the final query string and the list of questions that was returned
     */
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

        //Only do personalization ranking when sorting by relevance
        if(sortType.equals(JerseyGet.SortType.RELEVANCE)) {
            questionList = searchModel.rankQuestionList(questionList);
        }

        return new Pair<>(searchQuery, questionList);
    }

    /**
     * Sends the questions results to JavaScript to be displayed
     * @param questions - the list of Questions that was returned from the API
     */
    private void updateQuestionList(List<Question> questions) {
        for (Question question : questions) { //Send each question individually and decompose it into the relevant information
            window.call("getQuestion", question.getTitle(), question.getBody(), question.getTags().toArray(), question.getLink());
        }
        window.call("displayQuestions");
        window.call("generateListeners");
    }

    public void updateSearchModel(Collection<String> tags, int amount) {
        searchModel.increaseTags(tags, amount);
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
