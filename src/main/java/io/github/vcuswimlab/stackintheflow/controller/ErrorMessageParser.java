package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
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

    public static List<String> parseCompilerMessages(Map<String, List<String>> compilerMessages, Project project) {
        return compilerMessages.get("ERROR").stream().map(e -> ErrorMessageParser.parseError(e, project)).collect(Collectors.toList());
    }

}
