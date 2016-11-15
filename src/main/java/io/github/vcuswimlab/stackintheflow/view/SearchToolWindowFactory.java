package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowser;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.ide.browsers.WebBrowserService;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import io.github.vcuswimlab.stackintheflow.representation.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import java.awt.*;
import java.awt.event.*;
import java.lang.reflect.Field;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

public class SearchToolWindowFactory implements ToolWindowFactory {

    private JButton searchButton;
    private JTextField searchBox;
    private JPanel content;
    private JList list1;
    private ToolWindow toolWindow;
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
        Messages.showMessageDialog("Query: " + query, "Query", null);

        Question[] questions = null;
        try {
            questions = new Question[] {
                new Question("Is every NP Hard problem computable?", "I am asking a question that seems to be " +
                        "related to what you're coding.", new URL("http://cs.stackexchange" +
                        ".com/questions/65655/is-every-np-hard-problem-computable"), new ArrayList<>()),
                new Question("Relevant Stack Overflow Question!", "I am asking a question that seems to be " +
                        "related to what you're coding.", new URL("http://www.stackoverflow.com"), new
                        ArrayList<>())
            };
        } catch(MalformedURLException e) {
            e.printStackTrace();
        }
        updateList(questions);
    }

    //TODO: This method should be unit tested either directly or indirectly once testing is set up.
    private void updateList(Question[] elements) {
        if(elements == null) {
            return;
        }
        currentQuestions = new Question[elements.length];
        content.remove(list1);
        list1 = new JList();
        final DefaultListModel defaultListModel1 = new DefaultListModel();
        for (int i = 0; i < elements.length; i++) {
            defaultListModel1.addElement(questionDisplayName(elements[i]));
            currentQuestions[i] = elements[i];
        }
        list1.setModel(defaultListModel1);
        list1.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent evt) {
                JList list = (JList)evt.getSource();
                if (evt.getClickCount() == 2) {
                    // Double-click detected
                    int index = list.locationToIndex(evt.getPoint());
                    openBrowser(currentQuestions[index].getLink().toString());
                }
            }
        });
        content.add(list1, new GridConstraints(2, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_WANT_GROW, null, new Dimension(150, 50), null, 0, false));

    }

    private String questionDisplayName(Question question) {
        // TODO: Later, we can set this up to do a fancier question display.
        return question.getName();
    }

    //Stub method to be fleshed out
    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
