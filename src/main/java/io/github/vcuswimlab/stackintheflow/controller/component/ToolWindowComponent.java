package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowAnchor;
import com.intellij.openapi.wm.ToolWindowManager;
import icons.StackInTheFlowIcons;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowFactory;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI;
import org.jetbrains.annotations.NotNull;

/**
 * Created by chase on 5/18/17.
 */
public class ToolWindowComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.ToolWindowComponent";
    private final Project project;
    private SearchToolWindowGUI searchToolWindowGUI;
    private ToolWindow toolWindow = null;

    public ToolWindowComponent(Project project) {
        this.project = project;
    }

    @Override
    public void projectOpened() {
        toolWindow = ToolWindowManager.getInstance(project).registerToolWindow("StackInTheFlow", false, ToolWindowAnchor.RIGHT);
        toolWindow.setIcon(StackInTheFlowIcons.TOOL_WINDOW_ICON);
        toolWindow.setAvailable(true, () -> {
            SearchToolWindowFactory windowFactory = new SearchToolWindowFactory();
            setSearchToolWindowGUI(windowFactory.buildGUI(toolWindow, project));
        });
    }

    public boolean toolWindowIsVisible(){
        if(toolWindow == null) {
            return false;
        } else {
            return toolWindow.isVisible();
        }
    }

    @Override
    public void projectClosed() {
        ToolWindowManager.getInstance(project).unregisterToolWindow("StackInTheFlow");
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
