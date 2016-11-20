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
        setText(question.getExcerpt());
        setBorder(focusBorder);
        return this;
    }
}
