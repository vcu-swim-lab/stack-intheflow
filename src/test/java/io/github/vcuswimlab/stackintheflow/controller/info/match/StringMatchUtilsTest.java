package io.github.vcuswimlab.stackintheflow.controller.info.match;

import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class StringMatchUtilsTest {

    @Test
    public void removeComments() {
        String singleLineCommentString = "test outside of commment // this is a single line comment\n";
        String multilineComment1 = "test outside of commment /* Multiline comment */";
        String multilineComment2 = "test outside of commment /** JavaDoc comment */";
        assertEquals(StringMatchUtils.removeComments(singleLineCommentString), "test outside of commment ");
        assertEquals(StringMatchUtils.removeComments(multilineComment1), "test outside of commment ");
        assertEquals(StringMatchUtils.removeComments(multilineComment2), "test outside of commment ");
    }

    @Test
    public void extractImports() {
        String testString = "\n" +
                "import java.util.regex.Matcher;\n" +
                "import java.util.regex.Pattern;\n" +
                "import java.util.Set;\n" +
                "public class StringMatchUtils {\n" +
                "    public static final Pattern IMPORT_STATEMENT_PATTERN = Pattern.compile(\"import\\\\s+([\\\\w\\\\.]*?(\\\\w+));\");\n" +
                "    public static final Pattern TERM_PATTERN = Pattern.compile(\"\\\\b([A-Z]\\\\w+)\\\\b\");\n";
        Set<String> imports = StringMatchUtils.extractImports(testString);
        assertEquals(3, imports.size());
        assertTrue(imports.contains("java.util.regex.Matcher"));
        assertTrue(imports.contains("java.util.regex.Pattern"));
        assertTrue(imports.contains("java.util.Set"));
    }
}