package io.github.vcuswimlab.stackintheflow.model.erroranalysis;

import org.junit.Test;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * <h1>ErrorMessageParserTest</h1>
 * Created on: 7/2/2017
 *
 * @author Tyler John Haden
 */
@RunWith(Theories.class)
public class ErrorMessageParserTest {

    @DataPoints
    public static RegexTestDataPoint[] regexTestData = new RegexTestDataPoint[]
            {
                    new RegexTestDataPoint(
                            new String[] {"(java.lang.IllegalArgumentException)here abcdjava 1.7 banana jdk1.8.6_1\n" +
                                    "java.io.IOError geography"},
                            new String[] {"7", "8"},
                            new String[] {"IllegalArgumentException"},
                            new String[] {"IOError"}),
                    new RegexTestDataPoint(
                            new String[] {"java.lang.AssertionError: Already disposed\n" +
                                    "            at com.intellij.openapi.components.impl.ComponentMa\n" +
                                    "            java.lang.StackOverflowErrorasdf"},
                            new String[] {},
                            new String[] {"AssertionError", "StackOverflowError"},
                            new String[] {}),
                    new RegexTestDataPoint(
                            new String[] {"jav1::jdk 1.3pyhtonjavac 1.5.9javac1.6"},
                            new String[] {"3", "5", "6"},
                            new String[] {},
                            new String[] {})
            };
    @DataPoints
    public static String[] textLines = new String[]
            {
                    "aiwugno\niwernf", "hfk ikdmn  dkj", "java is awesome\t\t  ", "\nasdf", "", "\n"
            };

    @Theory
    public void testFindPattern(RegexTestDataPoint regexTestDataPoint) {
        List<String> javaVersionsResult = ErrorMessageParser.findPattern(
                ErrorMessageParser.javaVersionPattern_6, 6, regexTestDataPoint.text);
        List<String> javaLangExceptionsResult = ErrorMessageParser.findPattern(
                ErrorMessageParser.javaLangExceptionPattern_1, 1, regexTestDataPoint.text);
        List<String> javaIOExceptionsResult = ErrorMessageParser.findPattern(
                ErrorMessageParser.javaIOExceptionPattern_1, 1, regexTestDataPoint.text);

        assertArrayEquals(Arrays.stream(regexTestDataPoint.javaVersions).sorted().toArray(),
                javaVersionsResult.stream().sorted().toArray());
        assertArrayEquals(Arrays.stream(regexTestDataPoint.javaLangExceptions).sorted().toArray(),
                javaLangExceptionsResult.stream().sorted().toArray());
        assertArrayEquals(Arrays.stream(regexTestDataPoint.javaIOExceptions).sorted().toArray(),
                javaIOExceptionsResult.stream().sorted().toArray());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFindPattern_NullText() {
        ErrorMessageParser.findPattern(null, 0, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFindPattern_NullPattern() {
        ErrorMessageParser.findPattern(null, 0, textLines);
    }

    @Theory
    public void testFindFirstLine(String lineOfText) {
        if(lineOfText.equals("\n")) {
            assertEquals("", ErrorMessageParser.parseFirstLine(new String[] {lineOfText}));
        } else {
            assertEquals(lineOfText.split("\n")[0], ErrorMessageParser.parseFirstLine(new String[] {lineOfText}));
        }
    }

    @Test
    public void testFindFirstLine_EmptyArray() {
        assertEquals(null, ErrorMessageParser.parseFirstLine(new String[0]));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFindFirstLine_NullArray() {
        assertEquals(null, ErrorMessageParser.parseFirstLine(null));
    }

    static class RegexTestDataPoint {
        String[] text;
        String[] javaVersions;
        String[] javaLangExceptions;
        String[] javaIOExceptions;
        RegexTestDataPoint(String[] text, String[] javaVersions, String[] javaLangExceptions, String[] javaIOExceptions) {
            this.text = text;
            this.javaVersions = javaVersions;
            this.javaLangExceptions = javaLangExceptions;
            this.javaIOExceptions = javaIOExceptions;
        }
    }
}
