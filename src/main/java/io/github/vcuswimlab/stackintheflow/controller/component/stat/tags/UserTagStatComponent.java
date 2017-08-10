package io.github.vcuswimlab.stackintheflow.controller.component.stat.tags;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistProfileComponent;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;
import org.jetbrains.annotations.NotNull;

import java.util.Map;

/**
 * Created by chase on 6/13/17.
 */
public class UserTagStatComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.UserTagStatComponent";

    private TagStatComponent tagStatComponent;
    private PersonalSearchModel searchModel;
    private Map<String, Integer> userStatMap;

    public UserTagStatComponent(TagStatComponent tagStatComponent, Project project) {
        this.tagStatComponent = tagStatComponent;
        this.userStatMap = ServiceManager.getService(project, PersistProfileComponent.class).getUserStatMap();
    }

    @Override
    public void projectOpened() {
        searchModel = new PersonalSearchModel(tagStatComponent, userStatMap);
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

    public PersonalSearchModel getSearchModel() {
        return searchModel;
    }
}
