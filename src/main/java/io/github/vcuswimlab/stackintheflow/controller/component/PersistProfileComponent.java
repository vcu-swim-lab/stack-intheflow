package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import org.jetbrains.annotations.Nullable;

import java.util.HashMap;
import java.util.Map;

/**
 * <h1>PersistProfileComponent</h1>
 * Created on: 8/1/2017
 *
 * @author Tyler John Haden
 */

@State(
        name = "ProfileState", storages = {
        @Storage(
                id = "stack-overflow",
                file = "$APP_CONFIG$/stackoverflow-profile.xml")
})
public class PersistProfileComponent implements PersistentStateComponent<PersistProfileComponent.State> {

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

    public Map<String, Integer> getUserStateMap() {
        return state.userStatMap;
    }

    public static class State {
        private Map<String, Integer> userStatMap;
        private State() {
            userStatMap = new HashMap<>();
        }
    }
}
