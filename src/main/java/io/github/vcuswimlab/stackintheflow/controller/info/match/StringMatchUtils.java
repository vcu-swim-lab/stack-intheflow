package io.github.vcuswimlab.stackintheflow.controller.info.match;


import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class StringMatchUtils {

    public static final Pattern IMPORT_STATEMENT_PATTERN = Pattern.compile("import\\s+([\\w\\.]*?(\\w+));");
    public static final Pattern TERM_PATTERN = Pattern.compile("\\b([A-Z]\\w+)\\b");

    public static String removeComments(String s) {
        return s.replaceAll("(//.*?\\n)|(/\\*(.|\\n)*?\\*/)", "");
    }

    public static Set<String> extractImports(String s) {
        Matcher matcher = IMPORT_STATEMENT_PATTERN.matcher(s);

        Set<String> imports = new HashSet<>();

        while (matcher.find()) {
            imports.add(matcher.group(1));
        }

        return imports;
    }
}
