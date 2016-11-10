package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.browsers.BrowserLauncher;
import com.intellij.ide.browsers.WebBrowserManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.ui.Messages;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowFactory;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;
import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.Query;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
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
    private JerseyGet jerseyGet;

    public SearchToolWindowFactory() {
        searchButton.addActionListener(e -> executeQuery(searchBox.getText()));
        searchBox.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                super.keyPressed(e);
                if(e.getKeyCode() == KeyEvent.VK_ENTER) {
                    searchButton.doClick();
                }
            }
        });
        list1.addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged(ListSelectionEvent listSelectionEvent) {
                //TODO: Technically, this isn't the correct place to open the browser. It works for testing purposes,
                // but it should be fixed before release.
                openBrowser("http://www.stackoverflow.com");
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
        Query q = new Query(query);
        jerseyGet.executeQuery(q);
        Messages.showMessageDialog("Query: " + query, "Query", null);

        try {
            setQuestion(0,new Question("Relevant Stack Overflow Question!", "I am asking a question that seems to be " +
                    "related to what you're coding.", new URL("http://www.stackoverflow.com"), new ArrayList<>()));
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
    }

    //TODO: This method should be unit tested either directly or indirectly once testing is set up.
    private void setQuestion(int index, Question question) {
        setElement(index, question.getName());
        //TODO: The final version probably won't just display the question name. How the question is displayed and
        // what information is part of that display should be subject to careful design consideration.
        //TODO: Need a way to store question URL so it activates when clicked. I think correctly implementing
        // setElement first would make this significantly simpler. However, if that's not possible for some reason, we
        // can maintain a second list containing the questions.
    }

    private void setElement(int index, String newValue) {
        //TODO: They certainly didn't intend for the JList to be edited with reflection. The correct solution is to
        // get IntelliJ to create the JList with a mutable ListModel. I haven't been able to find where to do that,
        // but if we figure it out, it would probably lead to much cleaner code for the entire question/list
        // interaction code.
        String str = ((String)list1.getModel().getElementAt(index));
        final Class<String> type = String.class;
        try {
            final Field valueField = type.getDeclaredField("value");
            valueField.setAccessible(true);
            valueField.set(str, newValue.toCharArray());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    //Stub method to be fleshed out
    private void openBrowser(String url) {
        BrowserLauncher.getInstance().browse(url, WebBrowserManager.getInstance().getFirstActiveBrowser());
    }
}
