package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.PersistentStateComponent;
import com.intellij.openapi.components.State;
import com.intellij.openapi.components.Storage;
import com.intellij.util.xmlb.XmlSerializerUtil;
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
public class PersistProfileComponent implements PersistentStateComponent<PersistProfileComponent> {

    private Map<String, Integer> userStatMap;

    public Map<String, Integer> getUserStateMap() {
        return userStatMap;
    }

    @Override
    public void noStateLoaded() {
        userStatMap = new HashMap<>();
    }

    @Override
    public void loadState(PersistProfileComponent state) {
        XmlSerializerUtil.copyBean(state, this);
    }

    @Nullable
    @Override
    public PersistProfileComponent getState() {
        return this;
    }

}
