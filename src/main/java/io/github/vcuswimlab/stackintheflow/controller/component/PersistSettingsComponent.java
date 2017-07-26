package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import org.jetbrains.annotations.Nullable;

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
                file = "$APP_CONFIG$/stackoverflowpersist.xml")
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

    public boolean getAutoQuery() {
        return this.state.autoQuery;
    }

    public void setAutoQuery(boolean value) {
        this.state.autoQuery = value;
    }

    public static class State {
        public boolean autoQuery;
        public State() {
            this.autoQuery = true;
        }
    }
}
