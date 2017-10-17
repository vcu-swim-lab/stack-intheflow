package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import com.intellij.util.PlatformUtils;
import com.intellij.util.xmlb.XmlSerializerUtil;
import org.jetbrains.annotations.Nullable;

import java.util.EnumMap;
import java.util.Map;

/**
 * <h1>SettingsComponent</h1>
 * Created on: 7/24/2017
 *
 * @author Tyler John Haden
 */
// http://www.jetbrains.org/intellij/sdk/docs/basics/persisting_state_of_components.html
@State(
        name = "SettingsState", storages = {
        @Storage("StackInTheFlow-settings.xml")
})
public class PersistSettingsComponent implements PersistentStateComponent<PersistSettingsComponent> {


    public Map<SettingKey, Boolean> settingsMap;

    public PersistSettingsComponent() {
        if (!PlatformUtils.isIntelliJ()) {
            noStateLoaded();
        }
    }

    @Override
    public void noStateLoaded() {
        settingsMap = new EnumMap<>(SettingKey.class);
        settingsMap.put(SettingKey.AUTO_QUERY, true);
        settingsMap.put(SettingKey.RUNTIME_ERROR, true);
        settingsMap.put(SettingKey.COMPILE_ERROR, true);
        settingsMap.put(SettingKey.DIFFICULTY, true);
        settingsMap.put(SettingKey.LOGGING, true);
    }

    public boolean autoQueryEnabled() {
        return this.settingsMap.get(SettingKey.AUTO_QUERY);
    }

    public boolean runtimeErrorEnabled() {
        if (this.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.settingsMap.get(SettingKey.RUNTIME_ERROR);
        } else {
            return false;
        }
    }

    public boolean compileErrorEnabled() {
        if (this.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.settingsMap.get(SettingKey.COMPILE_ERROR);
        } else {
            return false;
        }
    }

    public boolean difficultyEnabled() {
        if (this.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.settingsMap.get(SettingKey.DIFFICULTY);
        } else {
            return false;
        }
    }

    public boolean loggingEnabled() {
        if (this.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.settingsMap.get(SettingKey.LOGGING);
        } else {
            return false;
        }
    }

    public Map<SettingKey, Boolean> getSettingsMap() {
        return this.settingsMap;
    }

    public void updateSettings(SettingKey setting, boolean enabled) {
        this.settingsMap.put(setting, enabled);
    }

    public void updateSettings(Map<SettingKey, Boolean> updatedSettingsMap) {
        this.settingsMap.putAll(updatedSettingsMap);
    }

    @Nullable
    @Override
    public PersistSettingsComponent getState() {
        return this;
    }

    @Override
    public void loadState(PersistSettingsComponent state) {
        XmlSerializerUtil.copyBean(state, this);
    }

    public enum SettingKey {
        AUTO_QUERY,
        RUNTIME_ERROR,
        COMPILE_ERROR,
        DIFFICULTY,
        LOGGING
    }
}