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
    private static final int NORMAL_LINE_HEIGHT = 15;
    private static final int BOLD_LINE_HEIGHT = 18;
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

    private String getFontStartBlock(String fontName, String fontColorHex) {
        return "<font face=\"" + fontName + "\" color=\""+ fontColorHex +"\">";
    }

    private String getFontStartBlockNoName(String fontColorHex) {
        return "<font color=\""+ fontColorHex +"\">";
    }

    private void setTextFromQuestion(Question question, HTMLEditorKit kit, HTMLDocument doc) {
        String textColor = EditorFonts.getPrimaryFontColorHex();
        String title = question.getTitle() + "\n";
        String fontName = getFont().getName();
        setText("");

        try
        {
            kit.insertHTML(doc, doc.getLength(), getFontStartBlockNoName(textColor) + "<b>" + title, 0,
                    0,
                    HTML.Tag.FONT);
            kit.insertHTML(doc, doc.getLength(), getFontStartBlockNoName(textColor) + bodyProcessing
                            (question.getBody(), EditorFonts.getCodeFontColorHex())
                            + "</font>",
                    0, 0, null);
        } catch (Exception e) { e.printStackTrace(); }
    }

    private String replaceHtmlTag(String html, String tagName, String startTagReplacement, String endTagReplacement) {
        String startTag = "<" + tagName + ">";
        String endTag = "</" + tagName+ ">";
        return html.replaceAll(startTag, startTagReplacement).replaceAll(endTag, endTagReplacement);
    }

    private String bodyProcessing(String body, String textColor) {
        String codeFont = EditorFonts.getPrimaryFontName();
        String fontStartBlock = getFontStartBlock(codeFont, textColor); //FF6A00
        String html = replaceHtmlTag(body, "code", fontStartBlock, "</font>");
        html = replaceHtmlTag(html, "blockquote", fontStartBlock, "</font>");
        html = replaceHtmlTag(html, "pre", "", "");

        return html;
    }

    private void initHTMLJTextPane(JTextPane pane, Dimension d, HTMLEditorKit kit, HTMLDocument doc, JList<? extends
            Question>
            list, boolean isSelected) {
        pane.setOpaque(true);
        pane.setEditable(false);
        if(isSelected) {
            //pane.setBackground(list.getSelectionBackground());
            //pane.setForeground(list.getSelectionForeground());
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
        //006BFF
        try
        {
            kit.insertHTML(doc, doc.getLength(), "<font color=\"" + EditorFonts.getHyperlinkColorHex() + "\">" +
                    tagsString, 0, 0, HTML.Tag
                    .FONT);
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
        //setFont(getFont().deriveFont(EditorFonts.getPrimaryFontSize()));
    }

    private int getCellWidth() {
        return width;
    }

    private int getTextHeight(int normalLines, int boldLines) {
        Font font = getFont();
        Font boldFont = font.deriveFont(Font.BOLD);
        return NORMAL_LINE_HEIGHT*normalLines + BOLD_LINE_HEIGHT*boldLines;
        //return (getFontMetrics(font).getHeight() - 3) * normalLines + (getFontMetrics(boldFont).getHeight
        //        ()-1) * boldLines;
    }

    private int getBodyHeight(boolean isExpanded) {
        return getTextHeight(isExpanded ? 10 : 3, 1);
    }
}
