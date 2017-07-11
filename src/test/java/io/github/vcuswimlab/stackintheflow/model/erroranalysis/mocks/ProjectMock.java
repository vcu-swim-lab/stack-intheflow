package io.github.vcuswimlab.stackintheflow.model.erroranalysis.mocks;

import com.intellij.openapi.components.BaseComponent;
import com.intellij.openapi.extensions.ExtensionPointName;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.Condition;
import com.intellij.openapi.util.Key;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.util.messages.MessageBus;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.picocontainer.PicoContainer;

/**
 * <h1>ProjectMock</h1>
 * Created on: 7/4/2017
 *
 * @author Tyler John Haden
 */
public class ProjectMock implements Project {

    @Override
    public <T> T getComponent(@NotNull Class<T> aClass) {
        if(aClass.equals(RuntimeErrorComponentMock.class)) {
            try {
                return aClass.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                e.printStackTrace();
                return null;
            }
        } else {
            return null;
        }
    }

    @NotNull
    @Override
    public <T> T[] getComponents(@NotNull Class<T> aClass) {
        return null;
    }

    @NotNull
    @Override
    public String getName() {
        return null;
    }

    @Override
    public VirtualFile getBaseDir() {
        return null;
    }

    @Nullable
    @Override
    public String getBasePath() {
        return null;
    }

    @Nullable
    @Override
    public VirtualFile getProjectFile() {
        return null;
    }

    @Nullable
    @Override
    public String getProjectFilePath() {
        return null;
    }

    @Nullable
    @Override
    public String getPresentableUrl() {
        return null;
    }

    @Nullable
    @Override
    public VirtualFile getWorkspaceFile() {
        return null;
    }

    @NotNull
    @Override
    public String getLocationHash() {
        return null;
    }

    @Override
    public void save() {

    }

    @Override
    public boolean isOpen() {
        return false;
    }

    @Override
    public boolean isInitialized() {
        return false;
    }

    @Override
    public boolean isDefault() {
        return false;
    }

    @Override
    public BaseComponent getComponent(@NotNull String s) {
        return null;
    }

    @Override
    public <T> T getComponent(@NotNull Class<T> aClass, T t) {
        return null;
    }

    @Override
    public boolean hasComponent(@NotNull Class aClass) {
        return false;
    }

    @NotNull
    @Override
    public PicoContainer getPicoContainer() {
        return null;
    }

    @NotNull
    @Override
    public MessageBus getMessageBus() {
        return null;
    }

    @Override
    public boolean isDisposed() {
        return false;
    }

    @NotNull
    @Override
    public <T> T[] getExtensions(@NotNull ExtensionPointName<T> extensionPointName) {
        return null;
    }

    @NotNull
    @Override
    public Condition<?> getDisposed() {
        return null;
    }

    @Override
    public void dispose() {

    }

    @Nullable
    @Override
    public <T> T getUserData(@NotNull Key<T> key) {
        return null;
    }

    @Override
    public <T> void putUserData(@NotNull Key<T> key, @Nullable T t) {

    }
}
