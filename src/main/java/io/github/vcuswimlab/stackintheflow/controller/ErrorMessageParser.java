package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Created by chase on 3/23/17.
 */
public class ErrorMessageParser {

    public static String parseError(String error, Project project) {
        TermStatComponent termStatComponent = project.getComponent(TermStatComponent.class);

        String cleanedError = error.replaceAll("[:;\"\']", "");

        //return cleanedError;
        return Arrays.stream(cleanedError.split("\\s")).filter(s -> termStatComponent.getTermStat(s).isPresent())
                .collect(Collectors.joining(" "));
    }

}
