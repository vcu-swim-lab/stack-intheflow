package util;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;

/**
 *
 * @author vietan
 */
public class CLIUtils {

    private static HelpFormatter formatter;

    static {
        formatter = new HelpFormatter();
    }

    public static double[] getDoubleArrayArgument(CommandLine cmd, String argName, 
            double[] defaultVals, String splitter) {
        if (cmd.hasOption(argName)) {
            String argVal = cmd.getOptionValue(argName);
            String[] sval = argVal.split(splitter);
            double[] vals = new double[sval.length];
            for (int i = 0; i < vals.length; i++) {
                vals[i] = Double.parseDouble(sval[i]);
            }
            return vals;
        }
        return defaultVals;
    }

    public static int[] getIntArrayArgument(CommandLine cmd, String argName, 
            int[] defaultVals, String splitter) {
        if (cmd.hasOption(argName)) {
            String argVal = cmd.getOptionValue(argName);
            String[] sval = argVal.split(splitter);
            int[] vals = new int[sval.length];
            for (int i = 0; i < vals.length; i++) {
                vals[i] = Integer.parseInt(sval[i]);
            }
            return vals;
        }
        return defaultVals;
    }

    public static double getDoubleArgument(CommandLine cmd, String argName, double defaultVal) {
        if (cmd.hasOption(argName)) {
            return Double.parseDouble(cmd.getOptionValue(argName));
        }
        return defaultVal;
    }

    public static String getStringArgument(CommandLine cmd, String argName, String defaultVal) {
        if (cmd.hasOption(argName)) {
            return cmd.getOptionValue(argName);
        }
        return defaultVal;
    }

    public static int getIntegerArgument(CommandLine cmd, String argName, int defaultVal) {
        if (cmd.hasOption(argName)) {
            return Integer.parseInt(cmd.getOptionValue(argName));
        }
        return defaultVal;
    }

    public static void printHelp(String str, Options options) {
        formatter.printHelp(str, options);
    }
}
