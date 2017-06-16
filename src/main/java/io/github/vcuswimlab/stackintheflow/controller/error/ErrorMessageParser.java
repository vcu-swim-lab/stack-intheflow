package io.github.vcuswimlab.stackintheflow.controller.error;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.ToolWindowComponent;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by chase on 3/23/17.
 */
public class ErrorMessageParser {

    private static Pattern javaVersionPattern = Pattern.compile("((javac?)|(jdk)) ?(v(ersion)?)? ?1.[1-9](.[\\d_]+)?", Pattern.CASE_INSENSITIVE);
    private static Pattern firstLinePattern = Pattern.compile("\\A.*");
    private static Pattern firstLineExceptionPatternGroup1 = Pattern.compile("\\A.*java\\.lang\\.(.*)[:\n]");

    public static String parseError(String error, Project project) {
        TermStatComponent termStatComponent = project.getComponent(TermStatComponent.class);

        String cleanedError = error.replaceAll("[:;\"\']", "");

        //return cleanedError;
        return Arrays.stream(cleanedError.split("\\s")).filter(s -> termStatComponent.getTermStat(s).isPresent())
                .collect(Collectors.joining(" "));
    }

    public static List<String> parseCompilerError(Map<String, List<String>> messages, Project project) {
        List<String> debugMessages = new ArrayList<>();

        debugMessages.addAll(findPattern(firstLinePattern, messages, "ERROR", "WARNING"));
        debugMessages.addAll(findPattern(javaVersionPattern, messages, "ERROR", "WARNING", "INFORMATION"));
        return debugMessages;

//        return compilerMessages.get("ERROR").stream()
//                .map(e -> ErrorMessageParser.parseError(e, project))
//                .collect(Collectors.toList());
    }

    public static List<String> parseRuntimeError(Map<String, List<String>> messages, Project project) {
        List<String> debugMessages = new ArrayList<>();

        debugMessages.addAll(findPattern(firstLineExceptionPatternGroup1, 1, messages, "ERROR"));
        debugMessages.addAll(findPattern(javaVersionPattern, messages, "ERROR", "WARNING", "INFORMATION"));
        return debugMessages;
//        return Collections.singletonList(parseError(errorMessage, project));
    }

    private static List<String> findPattern(Pattern pattern, Map<String, List<String>> messages, String... keys) {
        return findPattern(pattern, 0, messages, keys);
    }

    private static List<String> findPattern(Pattern pattern, int group, Map<String, List<String>> messages, String... keys) {
        List<String> matchedGroups = new ArrayList<>();
        for(String key : keys) {
            if(messages.containsKey(key)) {
                for(String message : messages.get(key)) {
                    Matcher messageMatcher = pattern.matcher(message);
                    while(messageMatcher.find()) {
                        matchedGroups.add(messageMatcher.group(group));
                    }
                }
            }
        }
        return matchedGroups;
    }
}
