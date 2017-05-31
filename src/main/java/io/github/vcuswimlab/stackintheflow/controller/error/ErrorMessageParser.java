package io.github.vcuswimlab.stackintheflow.controller.error;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;

import java.util.*;
import java.util.regex.Pattern;
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

    public static List<String> parseCompilerError(Map<String, List<String>> compilerMessages, Project project) {
        return compilerMessages.get("ERROR").stream()
                .map(e -> ErrorMessageParser.parseError(e, project))
                .collect(Collectors.toList());
    }

    public static List<String> parseRuntimeError(String errorMessage, Project project) {
        return Collections.singletonList(parseError(errorMessage, project));
    }
}
