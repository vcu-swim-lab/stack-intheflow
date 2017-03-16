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
                    // Primary Single-click detected.
                    int index = list.locationToIndex(evt.getPoint());

                    if (index < 0 || index >= questionListModel.size()) {
                        return;
                    }

                    Question question = questionListModel.get(index);

//                    int baseY = 0;
//                    for(int i = 0; i < index; i++) {
//                        baseY += renderer.getCellHeight(questionListModel.get(i).isExpanded());
//                    }
//                    baseY += renderer.getBodyHeight(question.isExpanded());
                    //TODO: Calculate Y bounds from question height!!

//                    int mouseY = (int)Math.round(evt.getPoint().getY());
//                    int bodyHeight = baseY;
//                    if(mouseY > bodyHeight && mouseY < bodyHeight + renderer.getTextHeight(1,0)) {
//                        double mouseX = Math.round(evt.getPoint().getX());
//                        String tags = question.getTagsAsFormattedString();
//                        int endX = 0;
//                        int tagIndex = 0;
//                        boolean inTags = false;
//                        for (String s : question.getTags()) {
//                            endX += renderer.getTextWidth("[" + s + "] ");
//                            if (mouseX < endX) {
//                                inTags = true;
//                                break;
//                            }
//                            tagIndex++;
//                        }
//
//                        if (inTags) {
//                            searchBox.setText(searchBox.getText() + " " + question.getTags().get(tagIndex));
//                        }
//                    } else {
                    question.toggleExpanded();
                    List<Question> questions = new ArrayList<Question>();
                    for(int i = 0; i < questionListModel.size(); i++) {
                        questions.add(questionListModel.get(i));
                    }
                    updateList(questions);
//                    }

//                    BufferedWriter writer = null;
//                    try {
//                        writer = new BufferedWriter(new FileWriter(new File("/home/batman/Desktop/MyOutputFileHere.txt")));
//                        writer.write(evt.getPoint().toString());
//                        writer.write("\nHello, World!!!");
//                        writer.write("\n" + mouseY + ", " + bodyHeight + ", " + (int)(bodyHeight + renderer
//                                .getTextHeight
//                                (1,0)));
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    } finally {
//                        try {
//                            writer.close();
//                        } catch (IOException e) {
//                            e.printStackTrace();
//                        }
//                    }
                }

                if (evt.getClickCount() == 2 && evt.getButton() == MouseEvent.BUTTON1) {
                    // Primary Double-click detected
                    int index = list.locationToIndex(evt.getPoint());

                    if (index < 0 || index >= questionListModel.size()) {
                        return;
                    }

                    openBrowser(questionListModel.get(index).getLink());
                }
            }
        });
        instance = this;
//        searchBox.setMinimumSize(new Dimension(1,1));
//        searchBox.setMaximumSize(new Dimension(10000,10000));
//        searchJPanel.setMinimumSize(new Dimension(1,1));
//        searchJPanel.setMaximumSize(new Dimension(10000,10000));
        consoleErrorPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                executeQuery(compilerMessages.get(0));
                consoleErrorPane.setVisible(false);
            }
        });
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

    //Stub method to be fleshed out
    private void executeQuery(String query) {

        JerseyResponse jerseyResponse = QueryExecutor.executeQuery(query);

        List<Question> questionList = jerseyResponse.getItems();
        updateList(questionList);

//        String bodyTest = "This is a really long question body. I'm making this mainly to test how the excerpt will " +
//                "do with an exceptionally long body, like in real questions. As a consequence, I have a whole lot of " +
//                "extra space to fill with not a lot to talk about. So... how's everyone doing? It's currently Monday," +
//                " November 28, at 9:51 AM at the time of writing. Thanksgiving happened, but not much to talk about " +
//                "for me. We just stayed home and ate a big meal with immediate family, no guests or anything else." +
//                " I hope you all had a little bit more happen for your Thanksgiving stories. Also, it's actually " +
//                "below freezing today this morning, at least where I am. How about that? Winter really is coming. " +
//                "I think I've done enough needless rambling. This test body seems to be more than long enough. I " +
//                "figure I may as well fill out the last sentence, just in case though.";
//        //Test code to populate list. Possibly use some variant of this in unit testing later?
//        Question[] questions = new Question[] {
//            new Question(new ArrayList<>(), bodyTest, bodyTest,  "Is every NP Hard problem computable?", "http://cs" +
//                    ".stackexchange" +
//                    ".com/questions/65655/is-every-np-hard-problem-computable"),
//            new Question(new ArrayList<>(), bodyTest, bodyTest,  "Relevant Stack Overflow Question!", "http://www.stackoverflow.com")
//        };
//        updateList(Arrays.asList(questions));
//        updateList(new ArrayList<Question>());
    }

    //TODO: This method should be unit tested either directly or indirectly once testing is set up.
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

        // It's great this can be done in one line; I'm not sure how to modify this for the new question handling
        // code however.
        //resultsList.setListData(elements.stream().map(Question::getTitle).collect(Collectors.toList()).toArray
        // (new String[elements.size()]));
    }

    public void setSearchBoxContent(String content) {
        searchBox.setText(content);
    }

    public void setConsoleError(List<String> compilerMessages) {

        this.compilerMessages = compilerMessages;

        HTMLEditorKit kit = new HTMLEditorKit();
        HTMLDocument doc = new HTMLDocument();
        consoleErrorPane.setEditorKit(kit);
        consoleErrorPane.setDocument(doc);

        try {
            kit.insertHTML(doc, 0, "Search For: " + "<a href=\"\"><u>" + compilerMessages.get(0) + "</u></a>", 0, 0, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        consoleErrorPane.setVisible(true);
    }

    //Stub method to be fleshed out
    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
