package io.github.vcuswimlab.stackintheflow.view;


import com.intellij.openapi.editor.colors.EditorColorsManager;
import com.intellij.openapi.editor.colors.EditorColorsScheme;
import io.github.vcuswimlab.stackintheflow.model.Question;

import javax.swing.*;
import javax.swing.text.BadLocationException;
import javax.swing.text.Utilities;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;
import java.awt.event.MouseEvent;

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
    private static final int BOLD_LINE_HEIGHT = 18;
    private JTextField parentContent;
    private JComponent otherReferenceContent;

    public QuestionRenderer(JTextField parentContent, JComponent otherReferenceContent) {
        setOpaque(true);
        setEditable(false);
        this.parentContent = parentContent;
        this.otherReferenceContent = otherReferenceContent;
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

    public int getTextHeight(int normalLines, int boldLines) {
        Font font = getFont();
        Font boldFont = font.deriveFont(Font.BOLD);
        return (getFontMetrics(font).getHeight() - 3) * normalLines + (getFontMetrics(boldFont).getHeight
                ()-1) * boldLines;
        //return NORMAL_LINE_HEIGHT * normalLines + BOLD_LINE_HEIGHT * boldLines;
    }

    public int getTextWidth(String s) {
        return getFontMetrics(getFont()).stringWidth(s) - (int)Math.round(1.4*(double)s.length());
    }

    public int getBodyHeight(boolean isExpanded) {
        return getTextHeight(isExpanded ? 10 : 3, 1);
    }
    public int getCellHeight(boolean isExpanded) {
        return getBodyHeight(isExpanded) + getTextHeight(2,0);
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

        //int maxLines = question.isExpanded() ? 10 : 3;
        int dimensionSize = getBodyHeight(question.isExpanded());
        Dimension dim = new Dimension(parentContent.getWidth() + otherReferenceContent.getWidth(),dimensionSize);
        setTextFromQuestion(question, dim);

        // Compact into space needed to fit.
//        int maxLines = question.isExpanded() ? 7 : 3;
//        int dimensionSize = getTextHeight(maxLines-1,1);//getTextHeight(Math.min(getLineCount(), maxLines), 1);
//        dim = new Dimension(dimensionSize, dimensionSize);
//
//        setTextFromQuestion(question, dim);
        JTextPane test = new JTextPane();
        test.setOpaque(true);
        test.setEditable(false);
        if(isSelected) {
            test.setBackground(list.getSelectionBackground());
            test.setForeground(list.getSelectionForeground());
        } else {
            test.setBackground(list.getBackground());
            test.setForeground(list.getForeground());
        }
        test.setText("");
        Dimension d = new Dimension(parentContent.getWidth() + otherReferenceContent.getWidth(), getTextHeight(0,2));
        test.setMaximumSize(d);
        test.setMinimumSize(d);
        test.setPreferredSize(d);
        test.setSize(d);
        //test.setLocation(0,dimensionSize);

        HTMLEditorKit kit = new HTMLEditorKit();
        HTMLDocument doc = new HTMLDocument();
        test.setEditorKit(kit);
        test.setDocument(doc);

        String tagsString = question.getTagsAsFormattedString();
        try
        {
            kit.insertHTML(doc, doc.getLength(), "<font color=\"006BFF\">" + tagsString, 0, 0, HTML.Tag.FONT);
        } catch (Exception e) { e.printStackTrace(); }



        JPanel output = new JPanel(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridx = 0;
        c.gridy = 0;
        output.add(this,c);
        c.gridx = 0;
        c.gridy = 1;
        output.add(test, c);
        output.setPreferredSize(output.getPreferredSize());
        output.validate();

//        output.addMouseListener(new MouseAdapter() {
//            @Override
//            public void mouseClicked(MouseEvent mouseEvent) {
//                super.mouseClicked(mouseEvent);
//                 //if(mouseEvent.getClickCount() == 1 && mouseEvent.getButton() == MouseEvent.BUTTON1) {
//                    test.setCaretPosition(test.viewToModel(mouseEvent.getPoint()));
//                    parentContent.setText(parentContent.getText() + test.getCaretPosition());
//
//
//                //}
//            }
//        });

        return output;
    }

    public String getClickedWord(MouseEvent evt) {
        try {
            String wrd = null;
            int pt = viewToModel(evt.getPoint());
            int spt = Utilities.getWordStart(this, pt);
            int ept = Utilities.getWordEnd(this, pt);
            this.setSelectionStart(spt);
            this.setSelectionEnd(ept);
            wrd = this.getSelectedText();
            //System.out.println("TextPane word=" + wrd);
            return wrd;
        } catch(Exception e) {
            return null;
        }
    }

    private void setTextFromQuestion(Question question, Dimension dim) {
        EditorColorsScheme colorsScheme = EditorColorsManager.getInstance().getGlobalScheme();
        String textColor = convertColorToHex(colorsScheme.getDefaultForeground());

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
            kit.insertHTML(doc, doc.getLength(), "<font color=\"" + textColor + "\"><b>" + title + HTML.Tag.B, 0, 0, HTML.Tag.FONT);
            kit.insertHTML(doc, doc.getLength(), "<font color=\"" + textColor + "\">" + bodyProcessing(question.getBody())
                            + "</font>",
                    0, 0, null);
        } catch (Exception e) { e.printStackTrace(); }
    }

    private String bodyProcessing(String body) {

        EditorColorsScheme colorsScheme = EditorColorsManager.getInstance().getGlobalScheme();
        String codeFont = colorsScheme.getEditorFontName();

        return body.replaceAll("<code>", "<font face=\"" + codeFont + "\" color=\"FF6A00\">").replaceAll("</code>", "</font>");
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

    private String convertColorToHex(Color c) {
        return String.format("#%02x%02x%02x", c.getRed(), c.getGreen(), c.getBlue());
    }
}
