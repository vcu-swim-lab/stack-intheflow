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
    private static UIState uiState;

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
        uiState = new UIState(settingsComponent.getAutoQuery());
        return uiState.getPanel();
    }

    @Override
    public boolean isModified() {
        return uiState.getAutoQuery() != settingsComponent.getAutoQuery();
    }

    @Override
    public void apply() throws ConfigurationException {
        settingsComponent.setAutoQuery(uiState.getAutoQuery());
    }

    private class UIState {
        JPanel panel;
        JCheckBox autoQueryCheckBox;

        UIState(boolean autoQueryState) {
            panel = new JPanel();
            autoQueryCheckBox = new JCheckBox("Allow auto query", autoQueryState);
            panel.add(autoQueryCheckBox);
        }

        JComponent getPanel() {
            return this.panel;
        }

        boolean getAutoQuery() {
            return this.autoQueryCheckBox.isSelected();
        }
    }
}
