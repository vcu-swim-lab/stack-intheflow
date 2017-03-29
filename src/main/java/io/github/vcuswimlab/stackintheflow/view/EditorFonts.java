package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.editor.colors.CodeInsightColors;
import com.intellij.openapi.editor.colors.EditorColorsManager;
import com.intellij.openapi.editor.colors.EditorColorsScheme;

import java.awt.*;

/**
 *
 * Created by batman on 3/19/17.
 *
 *
 * "ColorKey and TextAttributesKey are only keys that used to map a notions
 like "background color" or "field reference attributes" to a concrete
 colors and attributes being given a concrete color scheme. Actual
 character colors are being calculated during editor painting and aren't
 stored anywhere so you need either re-implement the editor drawing
 algorithm (which I hope you wouldn't do :) ) or learn how to zoom
 original editor component while it actually renders all the stuff."
 --
 Maxim Shafirov
 JetBrains Inc.
 *
 */
public class EditorFonts {
    private static EditorColorsScheme getInstanceScheme() {
        return EditorColorsManager.getInstance().getGlobalScheme();
    }

    public static String getPrimaryFontName() {
        return getInstanceScheme().getEditorFontName();
    }

    public static String getPrimaryFontColorHex()  {
        return convertColorToHex(getInstanceScheme().getDefaultForeground());
    }

    public static String getCodeFontColorHex() {
        return convertColorToHex(EditorColorsManager.getInstance().getGlobalScheme().getAttributes(CodeInsightColors
                .TODO_DEFAULT_ATTRIBUTES).getForegroundColor().brighter());
    }

    public static String getHyperlinkColorHex() {
        return convertColorToHex(EditorColorsManager.getInstance().getGlobalScheme().getAttributes(CodeInsightColors
                .HYPERLINK_ATTRIBUTES).getForegroundColor());
    }

    public static float getPrimaryFontSize() {
        return (float)getInstanceScheme().getEditorFontSize();
    }

    private static String convertColorToHex(Color c) {
        return String.format("#%02x%02x%02x", c.getRed(), c.getGreen(), c.getBlue());
    }

}
