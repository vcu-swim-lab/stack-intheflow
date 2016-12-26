package io.github.vcuswimlab.stackintheflow.view;

import io.github.vcuswimlab.stackintheflow.model.Question;

import javax.swing.*;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;
import java.awt.*;

/**
 * Created by batman on 11/16/16.
 */
public class QuestionRenderer extends JTextPane implements ListCellRenderer<Question> {
    // This constant is the magic dimension size that gives exactly 3 lines of wrapping with no issues. In a more
    // general implementaiton, we'd need to somehow calculate this from font size and formatting.
    private static final int DIMENSION_MAGIC = 75;
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
        Dimension dim = new Dimension(DIMENSION_MAGIC,DIMENSION_MAGIC);
        setMaximumSize(dim);
        setMinimumSize(dim);
        setPreferredSize(dim);
        setSize(dim);

        StyledDocument doc = getStyledDocument();
        SimpleAttributeSet titleAttributes = new SimpleAttributeSet();
        StyleConstants.setBold(titleAttributes, true);

        try
        {
            doc.insertString(0, title, titleAttributes);
            doc.insertString(doc.getLength(), question.getBody(), null);
        } catch (Exception e) { e.printStackTrace(); }


        //TODO: JEditorPane supports HTML formatting natively, and can fix the bottom line clipping issue. Investigate!
//        TitledBorder focusBorder = new TitledBorder(BorderFactory.createEmptyBorder(),
//                title,0,0,getFont().deriveFont(getFont().getStyle() | Font.BOLD));
//        setText(generateExcerpt(question, question.getBody().length()));
//        setLineWrap(true);
//        setWrapStyleWord(true);
//        setRows(3);
//        setBorder(focusBorder);
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
