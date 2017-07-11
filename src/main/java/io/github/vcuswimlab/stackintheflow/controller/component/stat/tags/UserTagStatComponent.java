package io.github.vcuswimlab.stackintheflow.controller.component.stat.tags;

import com.intellij.openapi.components.ProjectComponent;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;
import org.jetbrains.annotations.NotNull;

/**
 * Created by chase on 6/13/17.
 */
public class UserTagStatComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.UserTagStatComponent";

    private TagStatComponent tagStatComponent;
    private PersonalSearchModel searchModel;

    public UserTagStatComponent(TagStatComponent tagStatComponent) {
        this.tagStatComponent = tagStatComponent;
    }

    @Override
    public void projectOpened() {
        searchModel = new PersonalSearchModel(tagStatComponent);
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
