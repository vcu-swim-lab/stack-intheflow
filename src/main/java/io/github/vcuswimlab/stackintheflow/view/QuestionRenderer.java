package io.github.vcuswimlab.stackintheflow.view;
import io.github.vcuswimlab.stackintheflow.model.Question;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;

/**
 * Created by batman on 11/16/16.
 */
public class QuestionRenderer extends JLabel implements ListCellRenderer<Question> {
    private static final int EXCERPT_LENGTH = 80;
    public QuestionRenderer() {
        setOpaque(true);
    }

    @Override
    public Component getListCellRendererComponent(JList<? extends Question> list, Question question, int index, boolean
            isSelected, boolean cellHasFocus) {
        String title = question.getTitle();

        if(isSelected) {
            setBackground(list.getSelectionBackground());
            setForeground(list.getSelectionForeground());
        } else {
            setBackground(list.getBackground());
            setForeground(list.getForeground());
        }
        TitledBorder focusBorder = new TitledBorder(LineBorder.createGrayLineBorder(),
                title);
        setText(generateExcerpt(question, EXCERPT_LENGTH));
        setBorder(focusBorder);
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
