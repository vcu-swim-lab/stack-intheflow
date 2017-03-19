package io.github.vcuswimlab.stackintheflow.view;


import io.github.vcuswimlab.stackintheflow.model.Question;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;

/**
 *
 * Created by batman on 11/16/16.
 */
public class QuestionRenderer extends JTextPane implements ListCellRenderer<Question> {
    private int width;

    public QuestionRenderer(JComponent parentContent) {
        this.width = parentContent.getWidth();
    }

    @Override
    public Component getListCellRendererComponent(JList<? extends Question> list, Question question, int index, boolean
            isSelected, boolean cellHasFocus) {
        question.fixNulls();
        setFontSizeToEditorFontSize();

        Dimension dim = new Dimension(getCellWidth(),getBodyHeight(question.isExpanded()));
        HTMLEditorKit kit = new HTMLEditorKit();
        HTMLDocument doc = new HTMLDocument();

        initHTMLJTextPane(this, dim, kit, doc, list, isSelected);
        setTextFromQuestion(question, kit, doc);

        return buildOutputPane(this, buildTagsPane(list, question, isSelected));
    }

    public void setWidth(int width) {
        this.width = width;
    }

    private void setTextFromQuestion(Question question, HTMLEditorKit kit, HTMLDocument doc) {
        String textColor = EditorFonts.getPrimaryFontColorHex();
        String title = question.getTitle() + "\n";
        setText("");

        try
        {
            kit.insertHTML(doc, doc.getLength(), "<font color=\"" + textColor + "\"><b>" + title, 0, 0, HTML.Tag.FONT);
            kit.insertHTML(doc, doc.getLength(), "<font color=\"" + textColor + "\">" + bodyProcessing(question.getBody())
                            + "</font>",
                    0, 0, null);
        } catch (Exception e) { e.printStackTrace(); }
    }

    private String bodyProcessing(String body) {
        return body.replaceAll("<code>", "<font face=\"" + EditorFonts.getPrimaryFontName() + "\" color=\"FF6A00\">")
                .replaceAll
                ("</code>", "</font>");
    }

    private void initHTMLJTextPane(JTextPane pane, Dimension d, HTMLEditorKit kit, HTMLDocument doc, JList<? extends
            Question>
            list, boolean isSelected) {
        pane.setOpaque(true);
        pane.setEditable(false);
        if(isSelected) {
            pane.setBackground(list.getSelectionBackground());
            pane.setForeground(list.getSelectionForeground());
        } else {
            pane.setBackground(list.getBackground());
            pane.setForeground(list.getForeground());
        }
        pane.setText("");
        pane.setMaximumSize(d);
        pane.setMinimumSize(d);
        pane.setPreferredSize(d);
        pane.setSize(d);

        pane.setEditorKit(kit);
        pane.setDocument(doc);
    }

    @NotNull
    private JTextPane buildTagsPane(JList<? extends Question> list, Question question, boolean isSelected) {
        JTextPane test = new JTextPane();
        HTMLEditorKit kit = new HTMLEditorKit();
        HTMLDocument doc = new HTMLDocument();
        initHTMLJTextPane(test, new Dimension(getCellWidth(), getTextHeight(0,2)), kit, doc, list, isSelected);

        String tagsString = question.getTagsAsFormattedString();
        try
        {
            kit.insertHTML(doc, doc.getLength(), "<font color=\"006BFF\">" + tagsString, 0, 0, HTML.Tag.FONT);
        } catch (Exception e) { e.printStackTrace(); }
        return test;
    }

    @NotNull
    private JPanel buildOutputPane(Component mainPanel, JTextPane tagsPane) {
        JPanel output = new JPanel(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridx = 0;
        c.gridy = 0;
        output.add(mainPanel,c);
        c.gridx = 0;
        c.gridy = 1;
        output.add(tagsPane, c);
        output.setPreferredSize(output.getPreferredSize());
        output.validate();
        return output;
    }

    private void setFontSizeToEditorFontSize() {
        setFont(getFont().deriveFont(EditorFonts.getPrimaryFontSize()));
    }

    private int getCellWidth() {
        return width;
    }

    private int getTextHeight(int normalLines, int boldLines) {
        Font font = getFont();
        Font boldFont = font.deriveFont(Font.BOLD);
        return (getFontMetrics(font).getHeight() - 2) * normalLines + (getFontMetrics(boldFont).getHeight
                ()) * boldLines;
    }

    private int getBodyHeight(boolean isExpanded) {
        return getTextHeight(isExpanded ? 10 : 3, 1);
    }
}
