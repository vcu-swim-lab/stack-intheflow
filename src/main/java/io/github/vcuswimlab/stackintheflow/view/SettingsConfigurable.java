package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.options.Configurable;
import com.intellij.openapi.options.ConfigurationException;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent;
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
    private static PersistSettingsComponent settingsComponent;
    private static JTextArea fooTextArea;

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
        settingsComponent = ServiceManager.getService(PersistSettingsComponent.class);
        JPanel panel = new JPanel();
        fooTextArea = new JTextArea();
        fooTextArea.setText(settingsComponent.getFoo());
        panel.add(fooTextArea);
        return panel;
    }

    @Override
    public boolean isModified() {
        return !settingsComponent.getFoo().equals(fooTextArea.getText());
    }

    @Override
    public void apply() throws ConfigurationException {
        settingsComponent.setFoo(fooTextArea.getText());
    }
}
