package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.editor.colors.EditorColorsManager;
import com.intellij.openapi.editor.colors.EditorColorsScheme;

import java.awt.*;

/**
 *
 * Created by batman on 3/19/17.
 */
public class EditorFonts {
    private static EditorFonts instance;

    private EditorColorsScheme colorsScheme;
    private EditorFonts() {
        colorsScheme = EditorColorsManager.getInstance().getGlobalScheme();
    }

    private static EditorColorsScheme getInstanceScheme() {
        if(instance == null) {
            instance = new EditorFonts();
        }
        return instance.colorsScheme;
    }

    public static String getPrimaryFontName() {
        return getInstanceScheme().getEditorFontName();
    }

    public static String getPrimaryFontColorHex() {
        return convertColorToHex(getInstanceScheme().getDefaultForeground());
    }

    public static float getPrimaryFontSize() {
        return (float)getInstanceScheme().getEditorFontSize();
    }

    private static String convertColorToHex(Color c) {
        return String.format("#%02x%02x%02x", c.getRed(), c.getGreen(), c.getBlue());
    }

}
