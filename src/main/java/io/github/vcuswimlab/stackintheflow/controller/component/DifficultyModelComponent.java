package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.model.difficulty.DifficultyModel;
import org.jetbrains.annotations.NotNull;

/**
 * Created by Chase on 5/23/2017.
 */
public class DifficultyModelComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.DifficultyModelComponent";

    private Project project;
    private DifficultyModel difficultyModel;


    public DifficultyModelComponent(Project project) {
        this.project = project;
    }

    @Override
    public void projectOpened() {
        difficultyModel = new DifficultyModel(project);
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

    public DifficultyModel getDifficultyModel() {
        return difficultyModel;
    }
}
