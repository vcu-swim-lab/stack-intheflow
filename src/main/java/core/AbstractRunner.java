package core;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;

/**
 *
 * @author vietan
 */
public class AbstractRunner {

    public static final String TopWordFile = "top-words.txt";
    public static final String TopWordWithDocsFile = "top-words-with-docs.txt";
    public static final String TopicCoherenceFile = "topic-coherence.txt";
    protected static CommandLineParser parser;
    protected static Options options;
    protected static CommandLine cmd;
    protected static boolean verbose = true;
    protected static boolean debug = true;

    protected static void addOption(String optName, String optDesc) {
        options.addOption(OptionBuilder.withLongOpt(optName)
                .withDescription(optDesc)
                .hasArg()
                .withArgName(optName)
                .create());
    }

    public static void setVerbose(boolean v) {
        verbose = v;
    }

    public static void setDebug(boolean d) {
        debug = d;
    }

    public static void log(String msg) {
        System.out.print("[LOG] " + msg);
    }

    public static void logln(String msg) {
        System.out.println("[LOG] " + msg);
    }

    public static String getHelpString(String className) {
        return "java -cp 'dist/segan.jar' " + className + " -help";
    }

    public static String getCompletedTime() {
        DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
        Date dateobj = new Date();
        return df.format(dateobj);
    }
}
