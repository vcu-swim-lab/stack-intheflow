package util.normalizer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import util.MiscUtils;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class ZNormalizer extends AbstractNormalizer {

    private final double mean;
    private final double stdev;

    public ZNormalizer(double[] data) {
        this.mean = StatUtils.mean(data);
        this.stdev = StatUtils.standardDeviation(data);
    }

    public ZNormalizer(double mean, double stdev) {
        this.mean = mean;
        this.stdev = stdev;
    }

    @Override
    public double normalize(double originalValue) {
        return (originalValue - mean) / stdev;
    }

    @Override
    public double denormalize(double normalizedValue) {
        return normalizedValue * stdev + mean;
    }

    public double[] normalize(double[] originalValues) {
        double[] normValues = new double[originalValues.length];
        for (int i = 0; i < normValues.length; i++) {
            normValues[i] = this.normalize(originalValues[i]);
        }
        return normValues;
    }

    public double[] denormalize(double[] normalizedValues) {
        double[] denormValues = new double[normalizedValues.length];
        for (int i = 0; i < denormValues.length; i++) {
            denormValues[i] = this.denormalize(normalizedValues[i]);
        }
        return denormValues;
    }

    public static String output(ZNormalizer norm) {
        return norm.mean + "\t" + norm.stdev;
    }

    public static ZNormalizer input(String str) {
        String[] sstr = str.split("\t");
        double mean = Double.parseDouble(sstr[0]);
        double stdv = Double.parseDouble(sstr[1]);
        return new ZNormalizer(mean, stdv);
    }

    public static void main(String[] args) {
        double[] data = {2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85,
            3.85, 4.85, 3.85, 2.22, 1.45, 1.34};
        ZNormalizer n = new ZNormalizer(data);
        System.out.println("mean = " + n.mean);
        System.out.println("stdev = " + n.stdev);
        double[] normData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normData[i] = n.normalize(data[i]);
            System.out.println(normData[i]);
        }
        System.out.println(MiscUtils.arrayToString(normData));
        System.out.println(StatUtils.mean(normData));
        System.out.println(StatUtils.standardDeviation(normData));
    }
}
