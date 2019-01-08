package util.normalizer;

import util.StatUtils;

/**
 *
 * @author vietan
 */
public class MinMaxNormalizer extends AbstractNormalizer {

    private final double oldMin;
    private final double oldMax;
    private final double newMin;
    private final double newMax;
    private final double oldDiff;
    private final double newDiff;

    public MinMaxNormalizer(double[] data, double newMin, double newMax) {
        this.oldMin = StatUtils.min(data);
        this.oldMax = StatUtils.max(data);
        this.newMin = newMin;
        this.newMax = newMax;

        this.oldDiff = this.oldMax - this.oldMin;
        this.newDiff = this.newMax - this.newMin;
    }

    public MinMaxNormalizer(double min, double max, double newMin, double newMax) {
        this.oldMin = min;
        this.oldMax = max;
        this.newMin = newMin;
        this.newMax = newMax;

        this.oldDiff = this.oldMax - this.oldMin;
        this.newDiff = this.newMax - this.newMin;
    }

    @Override
    public double normalize(double oriValue) {
        return (oriValue - oldMin) * newDiff / oldDiff + newMin;
    }

    @Override
    public double denormalize(double normValue) {
        return (normValue - newMin) * oldDiff / newDiff + oldMin;
    }

    public double[] normalize(double[] oriValues) {
        double[] normValues = new double[oriValues.length];
        for (int i = 0; i < normValues.length; i++) {
            normValues[i] = this.normalize(oriValues[i]);
        }
        return normValues;
    }

    public static String output(MinMaxNormalizer norm) {
        if (norm == null) {
            return "null";
        }
        return norm.oldMin + "\t" + norm.oldMax + "\t" + norm.newMin + "\t" + norm.newMax;
    }

    public static MinMaxNormalizer input(String str) {
        if (str.equals("null")) {
            return null;
        }
        String[] sstr = str.split("\t");
        double oldMin = Double.parseDouble(sstr[0]);
        double oldMax = Double.parseDouble(sstr[1]);
        double newMin = Double.parseDouble(sstr[2]);
        double newMax = Double.parseDouble(sstr[3]);
        return new MinMaxNormalizer(oldMin, oldMax, newMin, newMax);
    }
}
