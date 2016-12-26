package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Query;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.List;

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
    private void updateList(List<Question> elements) {
        if(elements == null) {
            return;
        }

        questionListModel.clear();
        for (Question element : elements) {
            questionListModel.addElement(element);
        }

        if(elements.size() == 0) {
            questionListModel.addElement(new Question(null, "Sorry, your search returned no results :(", "", "", "http://www.stackoverflow.com"));
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
