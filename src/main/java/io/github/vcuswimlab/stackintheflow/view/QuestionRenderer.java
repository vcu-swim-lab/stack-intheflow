package io.github.vcuswimlab.stackintheflow.view;

import io.github.vcuswimlab.stackintheflow.model.Question;

import javax.swing.*;
import javax.swing.text.BadLocationException;
import javax.swing.text.Utilities;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;
import java.util.List;

/**
 * Created by batman on 11/16/16.
 */
public class QuestionRenderer extends JTextPane implements ListCellRenderer<Question> {
    // This constant is the magic dimension size that gives exactly 3 lines of wrapping with no issues. In a more
    // general implementation, we'd need to somehow calculate this from font size and formatting.
//    private final int DIMENSION_MAGIC = getTextHeight(3, 1);
//    private final int DIMENSION_MAGIC2 = getTextHeight(7, 1);
//
    private static final int NORMAL_LINE_HEIGHT = 15;
    private static final int BOLD_LINE_HEIGHT = 20;

    public QuestionRenderer() {
        setOpaque(true);
        setEditable(false);
    }

    private int getLineCount() {
        int totalCharacters = getText().length();
        int lineCount = (totalCharacters == 0) ? 1 : 0;

        try {
            int offset = totalCharacters;
            while (offset > 0) {
                offset = Utilities.getRowStart(this, offset) - 1;
                lineCount++;
            }
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
        return lineCount;
    }

    private int getTextHeight(int normalLines, int boldLines) {
        //TODO: For some reason, this gives the wrong answer. Investigate!
//        Font font = getFont();
//        Font boldFont = font.deriveFont(Font.BOLD);
//        return getFontMetrics(font).getHeight() * normalLines + getFontMetrics(boldFont).getHeight() * boldLines;
        return NORMAL_LINE_HEIGHT * normalLines + BOLD_LINE_HEIGHT * boldLines;
    }

    @Override
    public Component getListCellRendererComponent(JList<? extends Question> list, Question question, int index, boolean
            isSelected, boolean cellHasFocus) {
        if(isSelected) {
            setBackground(list.getSelectionBackground());
            setForeground(list.getSelectionForeground());
        } else {
            setBackground(list.getBackground());
            setForeground(list.getForeground());
        }

        int maxLines = question.isExpanded() ? 7 : 3;
        int dimensionSize = getTextHeight(maxLines, 1);
        Dimension dim = new Dimension(dimensionSize,dimensionSize);
        setTextFromQuestion(question, dim);

        // Compact into space needed to fit.
//        int maxLines = question.isExpanded() ? 7 : 3;
//        int dimensionSize = getTextHeight(maxLines-1,1);//getTextHeight(Math.min(getLineCount(), maxLines), 1);
//        dim = new Dimension(dimensionSize, dimensionSize);
//
//        setTextFromQuestion(question, dim);

        return this;
    }

    private void setTextFromQuestion(Question question, Dimension dim) {
        String title = question.getTitle() + "\n";
        setText("");

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
            kit.insertHTML(doc, doc.getLength(), bodyProcessing(question.getBody()), 0, 0, null);
            kit.insertHTML(doc, doc.getLength(), formatTags(question.getTags()), 0, 0, null);
        } catch (Exception e) { e.printStackTrace(); }
    }

    private String bodyProcessing(String body) {
        return body.replaceAll("<code>","<i>").replaceAll("</code>","</i>");
    }

    private String formatTags(List<String> tags) {
        StringBuilder out = new StringBuilder();
        for(String str : tags) {
            out.append("[" + str + "] ");
        }
        return out.toString();
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
