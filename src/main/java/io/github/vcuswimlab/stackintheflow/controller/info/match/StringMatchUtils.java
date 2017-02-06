package io.github.vcuswimlab.stackintheflow.controller.info.match;


import java.util.regex.Pattern;

public class StringMatchUtils {

    public static final Pattern IMPORT_STATEMENT_PATTERN = Pattern.compile("import\\s+([\\w\\.]*?(\\w+));");
    public static final Pattern TERM_PATTERN = Pattern.compile("\\b([A-Z]\\w+)\\b");

    public static String removeComments(String s) {
        return s.replaceAll("(//.*?\\n)|(/\\*(.|\\n)*?\\*/)", "");
    }
}
