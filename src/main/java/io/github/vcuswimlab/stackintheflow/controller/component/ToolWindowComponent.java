package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.ProjectComponent;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI;
import org.jetbrains.annotations.NotNull;

/**
 * Created by chase on 5/18/17.
 */
public class ToolWindowComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.ToolWindowComponent";
    private SearchToolWindowGUI searchToolWindowGUI;

    @Override
    public void projectOpened() {

    }

    @Override
    public void projectClosed() {

    }

    @Override
    public void initComponent() {

    }

    @Override
    public void disposeComponent() {

    }

    @NotNull
    @Override
    public String getComponentName() {
        return COMPONENT_ID;
    }

    public SearchToolWindowGUI getSearchToolWindowGUI() {
        return searchToolWindowGUI;
    }

    public void setSearchToolWindowGUI(SearchToolWindowGUI searchToolWindowGUI) {
        this.searchToolWindowGUI = searchToolWindowGUI;
    }
}
