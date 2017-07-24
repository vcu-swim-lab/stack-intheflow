package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.options.Configurable;
import com.intellij.openapi.options.ConfigurationException;
import org.jetbrains.annotations.Nls;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;

/**
 * <h1>SettingsConfigurable</h1>
 * Created on: 7/24/2017
 *
 * @author Tyler John Haden
 */
public class SettingsConfigurable implements Configurable {

    private static final String DISPLAY_NAME = "Stack-InTheFlow";

    @Nls
    @Override
    public String getDisplayName() {
        return DISPLAY_NAME;
    }

    @Nullable
    @Override
    public String getHelpTopic() {
        return null;
    }

    @Nullable
    @Override
    public JComponent createComponent() {
        return null;
    }

    @Override
    public boolean isModified() {
        return false;
    }

    @Override
    public void apply() throws ConfigurationException {

    }
}
