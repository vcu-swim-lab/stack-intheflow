package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.options.Configurable;
import com.intellij.openapi.options.ConfigurationException;
import io.github.vcuswimlab.stackintheflow.controller.Logging;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent.SettingKey;
import org.jetbrains.annotations.Nls;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;
import java.util.Iterator;
import java.util.Map;

/**
 * <h1>SettingsConfigurable</h1>
 * Created on: 7/24/2017
 *
 * @author Tyler John Haden
 */
public class SettingsConfigurable implements Configurable {

    private static final String DISPLAY_NAME = "Stack-InTheFlow";
    private static PersistSettingsComponent persistSettingsComponent;
    private static SettingsGUI settingsGUI;
    private Logging logger = new Logging();

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

        settingsGUI = new SettingsGUI();
        persistSettingsComponent = ServiceManager.getService(PersistSettingsComponent.class);
        return settingsGUI.build(persistSettingsComponent.getSettingsMap());
    }

    @Override
    public void disposeUIResources() {
        settingsGUI = null;
    }

    @Override
    public boolean isModified() {
        Map<SettingKey, Boolean> guiState = settingsGUI.getGUIState();
        Map<SettingKey, Boolean> persistState = persistSettingsComponent.getSettingsMap();
        return !guiState.equals(persistState);
    }

    @Override
    public void apply() throws ConfigurationException {
        Map<SettingKey, Boolean> guiState = settingsGUI.getGUIState();
        persistSettingsComponent.updateSettings(guiState);
        StringBuilder sb = new StringBuilder();
        sb.append("\"" + "SettingsEventType" + "\"" + ":" + "\"" + "Changed" + "\"" + ", ");
        Iterator<Map.Entry<SettingKey, Boolean>> it = guiState.entrySet().iterator();
        while (it.hasNext()){

            Map.Entry<SettingKey, Boolean> pair = it.next();
            sb.append("\"" + pair.getKey().toString() + "\"" + ":" + "\"" + pair.getValue().toString() + "\"");

            if (it.hasNext()){
                sb.append(", ");
            }
        }
        sb.append("}");
        logger.info(sb.toString());
    }

    @Override
    public void reset() {
    }
}
