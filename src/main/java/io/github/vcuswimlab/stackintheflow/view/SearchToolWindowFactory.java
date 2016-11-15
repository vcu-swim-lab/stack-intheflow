package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
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

public class SearchToolWindowFactory implements ToolWindowFactory {

    private JButton searchButton;
    private JTextField searchBox;
    private JPanel content;
    private JList list1;
    private ToolWindow toolWindow;
    private JerseyGet jerseyGet;
    private Question[] currentQuestions;

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
        Query q = new Query(query)
                .set(Query.Component.SITE, "stackoverflow");

        JerseyResponse jerseyResponse = jerseyGet.executeQuery(q, JerseyGet.SearchType.EXCERPTS);

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
        currentQuestions = new Question[elements.size()];
        content.remove(list1);
        list1 = new JList();
        final DefaultListModel defaultListModel1 = new DefaultListModel();
        //TODO: This should be done with an iterator.
        Iterator<Question> it = elements.iterator();
        int index = 0;
        while(it.hasNext()) {
            Question current = it.next();
            defaultListModel1.addElement(questionDisplayName(current));
            currentQuestions[index] = current;
            index++;
        }
        list1.setModel(defaultListModel1);
        list1.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent evt) {
                JList list = (JList)evt.getSource();
                if (evt.getClickCount() == 2) {
                    // Double-click detected
                    int index = list.locationToIndex(evt.getPoint());
                    openBrowser(currentQuestions[index].getLink());
                }
            }
        });
        content.add(list1, new GridConstraints(2, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_WANT_GROW, null, new Dimension(150, 50), null, 0, false));
    }

    private String questionDisplayName(Question question) {
        // TODO: Later, we can set this up to do a fancier question display.
        return question.getTitle();
    }

    //Stub method to be fleshed out
    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
