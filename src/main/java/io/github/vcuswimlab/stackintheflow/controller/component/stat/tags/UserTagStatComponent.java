package io.github.vcuswimlab.stackintheflow.controller.component.stat.tags;

import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.editor.Editor;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistProfileComponent;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;
import io.github.vcuswimlab.stackintheflow.model.L2H.L2HPredictor;
import org.jetbrains.annotations.NotNull;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

/**
 * Created by chase on 6/13/17.
 */
public class UserTagStatComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.UserTagStatComponent";

    private TagStatComponent tagStatComponent;
    private PersonalSearchModel searchModel;
	private L2HPredictor l2hPredictor;
    private Map<String, Integer> userStatMap;

    public UserTagStatComponent(TagStatComponent tagStatComponent, Project project) {
        this.tagStatComponent = tagStatComponent;
        this.userStatMap = ServiceManager.getService(project, PersistProfileComponent.class).getUserStatMap();
    }

	public boolean createInitialTagPredictions(String filePath) {
		try {
			l2hPredictor.computeL2HPredictions(filePath, "./temp_predictions_file.txt", "./temp_text_data_file.txt");
			int maxTags = 5;
			int countAmount = 5;
			List<String> tagNames = new ArrayList<>();
			List<Double> tagProbabilities = new ArrayList<>();
			l2hPredictor.computeMostLikelyTags("./temp_predictions_file.txt", tagNames, tagProbabilities, maxTags);

			List<String> tagToIncrement = new ArrayList<>();
			for(int i = 0; i < maxTags; i++) {
				int tagAmt = (int)(tagProbabilities.get(i)*countAmount + 0.5);
				if(tagAmt != 0) {
					tagToIncrement.clear();
					tagToIncrement.add(tagNames.get(i));
					searchModel.increaseTags(tagToIncrement, tagAmt);
				}
			}
		} catch(Exception e) {
			return false;
		}

		return true;
	}

    @Override
    public void projectOpened() {
        searchModel = new PersonalSearchModel(tagStatComponent, userStatMap);
		try {
			l2hPredictor = new L2HPredictor();
		} catch(IOException e) {
			// L2H could not be loaded, so don't add initial predictions
			l2hPredictor = null;
		}
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
