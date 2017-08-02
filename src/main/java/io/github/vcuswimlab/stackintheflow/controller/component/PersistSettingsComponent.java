package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import org.jetbrains.annotations.Nullable;

import java.util.EnumMap;
import java.util.HashMap;

/**
 * <h1>SettingsComponent</h1>
 * Created on: 7/24/2017
 *
 * @author Tyler John Haden
 */

@State(
        name = "SettingsState", storages = {
        @Storage(
                id = "stack-overflow",
                file = "$APP_CONFIG$/stackoverflow-settings.xml")
})
public class PersistSettingsComponent implements PersistentStateComponent<PersistSettingsComponent.State> {

    public State state = new State();

    @Nullable
    @Override
    public State getState() {
        return state;
    }

    @Override
    public void loadState(State state) {
        this.state = state;
    }

    public boolean autoQueryEnabled() {
        return this.state.settingsMap.get(SettingKey.AUTO_QUERY);
    }

    public boolean runtimeErrorEnabled() {
        if(this.state.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.state.settingsMap.get(SettingKey.RUNTIME_ERROR);
        } else {
            return false;
        }
    }

    public boolean compileErrorEnabled() {
        if(this.state.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.state.settingsMap.get(SettingKey.COMPILE_ERROR);
        } else {
            return false;
        }
    }

    public boolean difficultyEnabled() {
        if(this.state.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.state.settingsMap.get(SettingKey.DIFFICULTY);
        } else {
            return false;
        }
    }

    public boolean loggingEnabled() {
        if(this.state.settingsMap.get(SettingKey.AUTO_QUERY)) {
            return this.state.settingsMap.get(SettingKey.LOGGING);
        } else {
            return false;
        }
    }

    public EnumMap<SettingKey, Boolean> getSettingsMap() {
        return this.state.settingsMap.clone();
    }

    public void updateSettings(SettingKey setting, boolean enabled) {
        this.state.settingsMap.put(setting, enabled);
    }

    public void updateSettings(EnumMap<SettingKey, Boolean> updatedSettingsMap) {
        this.state.settingsMap.putAll(updatedSettingsMap);
    }

    public static class State {
        EnumMap<SettingKey, Boolean> settingsMap;

        private State() {
            settingsMap = new EnumMap<>(SettingKey.class);
            settingsMap.put(SettingKey.AUTO_QUERY, true);
            settingsMap.put(SettingKey.RUNTIME_ERROR, true);
            settingsMap.put(SettingKey.COMPILE_ERROR, true);
            settingsMap.put(SettingKey.DIFFICULTY, true);
            settingsMap.put(SettingKey.LOGGING, true);
        }
    }

    public enum SettingKey {
        AUTO_QUERY,
        RUNTIME_ERROR,
        COMPILE_ERROR,
        DIFFICULTY,
        LOGGING
    }
}
