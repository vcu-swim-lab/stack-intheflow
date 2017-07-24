package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.editor.colors.EditorColorsManager;
import com.intellij.util.ui.UIUtil;
import com.sun.javafx.application.PlatformImpl;
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
import javafx.stage.Stage;
import netscape.javascript.JSObject;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import javax.swing.*;
import java.awt.*;
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
    private Logger logger = LogManager.getLogger("ROLLING_FILE_APPENDER");


    private PersonalSearchModel searchModel;

    private List<String> compilerMessages;

    private ScheduledThreadPoolExecutor timer;

    private Stage stage;
    private WebView webView;
    private JFXPanel jfxPanel;
    private WebEngine engine;
    private JSObject window;
    private JavaBridge bridge;
    private EditorColorsManager editorColorsManager;

    private SearchToolWindowGUI(JPanel content,
                                PersonalSearchModel searchModel) {
        this.editorColorsManager = EditorColorsManager.getInstance();
        this.content = content;
        this.searchModel = searchModel;

        compilerMessages = new ArrayList<>();

        timer = new ScheduledThreadPoolExecutor(1);
        bridge = new JavaBridge(this);
        initComponents();
    }

    private void initComponents(){
        ApplicationManager.getApplication().getMessageBus().connect().subscribe(EditorColorsManager.TOPIC, scheme -> {
            updateUISettings();
        });

        jfxPanel = new JFXPanel();
        createScene();
        content.setLayout(new BorderLayout());
        content.add(jfxPanel, BorderLayout.CENTER);
        Platform.setImplicitExit(false);
    }

    private void createScene(){
        PlatformImpl.startup(() -> {
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

    /*
    private void addListeners() {

        consoleErrorPane.setVisible(false);

        searchButton.addActionListener(e -> executeQuery(searchBox.getText(), false));

        //Logging search query's

        searchButton.addActionListener(e -> logger.info("{SearchQuery: " + searchBox.getText() + "}"));
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

                //Logging question browser

                logger.info("{QuestionBrowserListRank: " + index + ", Title: " + questionListModel.get(index).getTitle() + ", Tags: " + questionListModel.get(index).getTags() + "}");
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

                //Logging question expansion

                logger.info("{QuestionExpansionListRank: " + index + ", Title: " + questionListModel.get(index).getTitle() + ", Tags: " + questionListModel.get(index).getTags() + "}");

                refreshListView();
                return true;
            }
        });

        consoleErrorPane.addHyperlinkListener(e -> {
            if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
                setSearchBoxContent(e.getDescription());

                //Logging the error pane

                logger.info("{consoleErrorQuery: " + e.getDescription() + ", Rank: " + consoleErrorPane.getText().indexOf(e.getDescription()) + "}");

                executeQuery(e.getDescription(), false);
            }
        });
    }

    private void refreshListView() {
        updateList(listModelToList());
    }
 */

    public void autoQuery(String query, boolean backoff){
        Platform.runLater(() -> {
            window.call("autoSearch", query, backoff);
        });
    }

    public void executeQuery(String query, boolean backoff, String searchMethod) {
        Future<List<Question>> questionListFuture = timer.submit(() -> {
            String searchQuery = query;
            JerseyResponse jerseyResponse = QueryExecutor.executeQuery(searchQuery, JerseyGet.SortType.valueOf(searchMethod));
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
            //setSearchBoxContent(searchQuery);
            return searchModel.rankQuesitonList(questionList);
        });

        try {
            List<Question> questionList = questionListFuture.get();
            for(Question question : questionList){
                window.call("getQuestion", question.getTitle(), question.getBody(), question.getTags().toArray(), question.getLink());
            }
            engine.executeScript("displayQuestions()");

        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
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

            //Logging that an error has occured

            logger.info("{ConsoleError: An error has occured");

            // for each message in compilerMessages, build html link
            String consoleErrorHTML = compilerMessages.stream().map(message ->
                    "<font color=\"" + EditorFonts.getPrimaryFontColorHex() + "\">" +
                        "search for:&nbsp;&nbsp;" +
                    "</font>" +
                    "<font color=\"" + EditorFonts.getHyperlinkColorHex() + "\">" +
                        // href allows hyperlink listener to grab message
                        "<a href=\"" + message.replace("<", "&lt;").replace(">", "&gt;") + "\">" +
                            "<u>" +
                                message.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>") +
                            "</u>" +
                        "</a>" +
                    "</font>").collect(Collectors.joining("<br><br>"));

            try {
                kit.insertHTML(doc, 0, consoleErrorHTML, 0, 0, null);
                consoleErrorPane.setVisible(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            consoleErrorPane.setVisible(false);
        } */
    }

    public void openBrowser(String url) {
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

    public JavaBridge getBridge(){ return this.bridge;}

    public String rgbToHex(int r, int g, int b){ return String.format("#%02x%02x%02x", r, g, b); }

    public String colorToHex(Color color){
        return rgbToHex(color.getRed(), color.getGreen(), color.getBlue());
    }

    public static Color getSlightlyDarkerColor(Color c) {
        float[] hsl = Color.RGBtoHSB(c.getRed(), c.getGreen(), c.getBlue(), new float[3]);
        return new Color(Color.HSBtoRGB(hsl[0], hsl[1], hsl[2] - .04f > 0 ? hsl[2] - .04f : hsl[2]));
    }

    public static Color getSlightlyLighterColor(Color c) {
        float[] hsl = Color.RGBtoHSB(c.getRed(), c.getGreen(), c.getBlue(), new float[3]);
        return new Color(Color.HSBtoRGB(hsl[0], hsl[1], hsl[2] + .04f < 1 ? hsl[2] + .04f : hsl[2]));
    }
}
