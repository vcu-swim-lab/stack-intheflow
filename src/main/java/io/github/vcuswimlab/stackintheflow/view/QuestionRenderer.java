package io.github.vcuswimlab.stackintheflow.view;

import io.github.vcuswimlab.stackintheflow.model.Question;

import javax.swing.*;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;

/**
 * Created by batman on 11/16/16.
 */
public class QuestionRenderer extends JTextPane implements ListCellRenderer<Question> {
    // This constant is the magic dimension size that gives exactly 3 lines of wrapping with no issues. In a more
    // general implementaiton, we'd need to somehow calculate this from font size and formatting.
    private static final int DIMENSION_MAGIC = 65;
    private static final int DIMENSION_MAGIC2 = 124;
    public QuestionRenderer() {
        setOpaque(true);
        setEditable(false);
    }

    @Override
    public Component getListCellRendererComponent(JList<? extends Question> list, Question question, int index, boolean
            isSelected, boolean cellHasFocus) {
        String title = question.getTitle() + "\n";

        if(isSelected) {
            setBackground(list.getSelectionBackground());
            setForeground(list.getSelectionForeground());
        } else {
            setBackground(list.getBackground());
            setForeground(list.getForeground());
        }
        setText("");
        // TODO: Works for whatever original isExpanded value is, but does not work when updated. Update is
        // definitely detected, as changing body or the like will immediately update, but for some reason it isn't
        // performed.
        Dimension dim = question.isExpanded()
                ? new Dimension(DIMENSION_MAGIC2,DIMENSION_MAGIC2)
                : new Dimension(DIMENSION_MAGIC,DIMENSION_MAGIC);
        setMaximumSize(dim);
        setMinimumSize(dim);
        setPreferredSize(dim);
        setSize(dim);

        HTMLEditorKit kit = new HTMLEditorKit();
        HTMLDocument doc = new HTMLDocument();
        setEditorKit(kit);
        setDocument(doc);

        try
        {
            kit.insertHTML(doc, doc.getLength(), "<b>" + title, 0, 0, HTML.Tag.B);
            kit.insertHTML(doc, doc.getLength(), question.getBody(), 0, 0, null);
        } catch (Exception e) { e.printStackTrace(); }

        return this;
    }

    //TODO: Maybe move this into the question object itself?
    private String generateExcerpt(Question question, int maxLength) {
        //TODO: Probably want a more advanced excerpt generation system later. This works for now, however.
        String body = question.getBody().replaceAll("<.*?>", "");
        if(body.length() <= maxLength) {
            return body;
        }
        return body.substring(0, maxLength-3) + "...";
    }
}
