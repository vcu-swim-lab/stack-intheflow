package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.ui.components.JBList;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import com.intellij.uiDesigner.core.GridConstraints;
import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Query;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

public class SearchToolWindowFactory implements ToolWindowFactory {

    private JButton searchButton;
    private JTextField searchBox;
    private JPanel content;
    private JList<Question> list1;
    private ToolWindow toolWindow;
    private JerseyGet jerseyGet;
    private DefaultListModel<Question> questionListModel;
    private final String filter = "!-MOiNm40F1U019gR)UUjNV-IQScciBJZ0";

    public SearchToolWindowFactory() {
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
        jerseyGet = new JerseyGet();
        // TODO: Add list1 to scrollpane instead. Allows potentially infinite scrolling.
        list1.setListData(new Question[0]);
        questionListModel = new DefaultListModel<>();
        list1.setModel(questionListModel);
        list1.setCellRenderer(new QuestionRenderer());
        list1.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent evt) {
                JList<String> list = (JList<String>)evt.getSource();
                if (evt.getClickCount() == 2 && evt.getButton() == MouseEvent.BUTTON1) {
                    // Primary Double-click detected
                    int index = list.locationToIndex(evt.getPoint());
                    openBrowser(questionListModel.get(index).getLink());
                }
            }
        });
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
        Query q = new Query("stackoverflow")
                .set(Query.Component.Q, query)
                .set(Query.Component.FILTER, filter)
                .set(Query.Component.SORT, "relevance");

        JerseyResponse jerseyResponse = jerseyGet.executeQuery(q, JerseyGet.SearchType.ADVANCED);

        List<Question> questionList = jerseyResponse.getItems();
        updateList(questionList);

        //Test code to populate list. Possibly use some variant of this in unit testing later?
//        Question[] questions = new Question[] {
//            new Question(new ArrayList<>(), "QuestionBody", "I am asking a question that seems to be related to what " +
//                    "you're coding.",  "Is every NP Hard problem computable?", "http://cs" +
//                    ".stackexchange" +
//                    ".com/questions/65655/is-every-np-hard-problem-computable"),
//            new Question(new ArrayList<>(), "QuestionBody", "I am asking a question that seems to be related to what " +
//                    "you're coding.",  "Relevant Stack Overflow Question!", "http://www.stackoverflow.com")
//        };
//        updateList(Arrays.asList(questions));
    }

    //TODO: This method should be unit tested either directly or indirectly once testing is set up.
    private void updateList(List<Question> elements) {
        if(elements == null) {
            return;
        }

        questionListModel.clear();
        for (Question element : elements) {
            questionListModel.addElement(element);
        }

        // It's great this can be done in one line; I'm not sure how to modify this for the new question handling
        // code however.
        //list1.setListData(elements.stream().map(Question::getTitle).collect(Collectors.toList()).toArray
        // (new String[elements.size()]));
    }

    //Stub method to be fleshed out
    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
