package io.github.vcuswimlab.stackintheflow.controller;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryGenerator {

    private static final Pattern importStatementPattern = Pattern.compile("import\\s+([\\w\\.]*?(\\w+));");

    public static String generateQuery(String editorText) {

        Matcher matcher = importStatementPattern.matcher(editorText);
        StringBuilder sb = new StringBuilder();

        while (matcher.find()) {
            sb.append(matcher.group(1));
        }

        return sb.toString();
    }
}
