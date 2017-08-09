package io.github.vcuswimlab.stackintheflow.model.erroranalysis;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by chase on 3/23/17.
 */
public class ErrorMessageParser {

    public static Pattern javaVersionPattern_6 = Pattern.compile("((javac?)|(jdk)) ?(v(ersion)?)? ?1.([1-9])(.[\\d_]+)?", Pattern.CASE_INSENSITIVE);
    public static Pattern javaLangExceptionPattern_1 = Pattern.compile("java\\.lang\\.([a-zA-Z]+(Exception|Bounds|Error))");
    public static Pattern javaIOExceptionPattern_1 = Pattern.compile("java\\.io\\.([a-zA-Z]+(Exception|Error))");

    private static int tokenLimit = 10;

    private static List<String> filterTerms(List<String> words, Project project) {
        TermStatComponent termStatComponent = project.getComponent(TermStatComponent.class);
        return words.stream().filter(s -> termStatComponent.getTermStat(s).isPresent()).collect(Collectors.toList());
    }

    public static List<String> parseCompilerError(ErrorMessage messages, Project project) {

        String[] error = messages.get(ErrorMessage.MessageType.ERROR);
        String[] warning = messages.get(ErrorMessage.MessageType.WARNING);
        String[] information = messages.get(ErrorMessage.MessageType.INFORMATION);

        Set<String> matchedKeywords = new LinkedHashSet<>();

        matchedKeywords.add("compile error");
        matchedKeywords.addAll(findPattern(javaLangExceptionPattern_1, 1, error));
        matchedKeywords.addAll(findPattern(javaIOExceptionPattern_1, 1, error));
        matchedKeywords.addAll(findPattern(javaVersionPattern_6, 6, error, warning, information)
                .stream().map(v -> "java " + v).collect(Collectors.toList()));

        for(String m : error) {
            for(String line : m.split("\n")) {
                String[] tokens = line.replaceAll("[:;\"\']", "").split("\n");
                List<String> filteredTokens = filterTerms(Arrays.asList(tokens), project);
                if(filteredTokens.size() >= tokenLimit - matchedKeywords.size()) {
                    matchedKeywords.addAll(filteredTokens.subList(0, tokenLimit - matchedKeywords.size()));
                    break;
                } else {
                    matchedKeywords.addAll(filteredTokens);
                }
            }
            if(matchedKeywords.size() >= tokenLimit){
                break;
            }
        }

        return Arrays.asList(matchedKeywords.stream().collect(Collectors.joining(" ")), parseFirstLine(error));
    }

    public static List<String> parseRuntimeError(ErrorMessage messages, Project project) {

        String[] error = messages.get(ErrorMessage.MessageType.ERROR);
        String[] warning = messages.get(ErrorMessage.MessageType.WARNING);
        String[] information = messages.get(ErrorMessage.MessageType.INFORMATION);

        Set<String> matchedKeywords = new LinkedHashSet<>();

        matchedKeywords.add("runtime error");
        matchedKeywords.addAll(findPattern(javaLangExceptionPattern_1, 1, error));
        matchedKeywords.addAll(findPattern(javaIOExceptionPattern_1, 1, error));
        matchedKeywords.addAll(findPattern(javaVersionPattern_6, 6, error, warning, information)
                .stream().map(v -> "java " + v).collect(Collectors.toList()));

        for(String m : error) {
            for(String line : m.split("\n")) {
                String[] tokens = line.replaceAll("[:;\"\']", "").split("\n");
                List<String> filteredTokens = filterTerms(Arrays.asList(tokens), project);
                if(filteredTokens.size() >= tokenLimit - matchedKeywords.size()) {
                    matchedKeywords.addAll(filteredTokens.subList(0, tokenLimit - matchedKeywords.size()));
                    break;
                } else {
                    matchedKeywords.addAll(filteredTokens);
                }
            }
            if(matchedKeywords.size() >= tokenLimit){
                break;
            }
        }

        return Arrays.asList(matchedKeywords.stream().collect(Collectors.joining(" ")), parseFirstLine(error));
    }

    public static List<String> findPattern(@NotNull Pattern pattern, int group, @NotNull String[]... textBlocks) {
        List<String> matchedGroups = new ArrayList<>();
        for(String[] textBlock : textBlocks) {
            for(String text : textBlock) {
                Matcher textMatcher = pattern.matcher(text);
                while(textMatcher.find()) {
                    matchedGroups.add(textMatcher.group(group));
                }
            }
        }
        return matchedGroups;
    }

    public static String parseFirstLine(@NotNull String[] message) {
        if(message.length == 0) {
            return null;
        }
        if(message[0].equals("\n")) {
            return "";
        }
        return message[0].split("\n")[0].replaceAll(":", "");
    }
}
