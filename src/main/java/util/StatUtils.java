package util;

import java.util.ArrayList;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import sampling.util.SparseCount;
import util.normalizer.AbstractNormalizer;
import util.normalizer.MinMaxNormalizer;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class StatUtils {
    // for computing digamma

    public static final double c1 = -0.5;
    public static final double c2 = -1.0 / 12;
    public static final double c4 = 1.0 / 120;
    public static final double c6 = -1.0 / 252;
    private static final PearsonsCorrelation corr = new PearsonsCorrelation();

    /**
     * Scale the values of each feature to the range [0,1] and return the
     * maximum value of each feature in the training data. Note that this only
     * applies to positive feature values.
     *
     * Note the values of the training data should be positive.
     *
     * @param data Training data in a sparse format
     * @param numFeatures Number of features
     * @return An array storing the maximum value of each feature
     */
    public static double[] scaleTrainingData(SparseVector[] data, int numFeatures) {
        int D = data.length;
        double[] maxVals = new double[numFeatures];
        for (int ii = 1; ii <= numFeatures; ii++) {
            double maxVal = 0.0;
            for (int d = 0; d < D; d++) {
                Double val = data[d].get(ii);
                if (val != null && val > maxVal) {
                    maxVal = val;
                }
            }

            // scale
            for (int d = 0; d < D; d++) {
                Double val = data[d].get(ii);
                if (val != null) {
                    data[d].set(ii, val / maxVal);
                }
            }

            // store max value for each feature
            maxVals[ii - 1] = maxVal;
        }
        return maxVals;
    }

    /**
     * Obtain min-max normalizers from training data.
     *
     * @param data Training data from which the normalizers are obtained
     * @param numFeatures Number of features
     * @return
     */
    public static MinMaxNormalizer[] minmaxNormalizeTrainingData(
            SparseVector[] data, int numFeatures) {
        int D = data.length;
        MinMaxNormalizer[] norms = new MinMaxNormalizer[numFeatures];
        for (int ii = 0; ii < numFeatures; ii++) {
            double[] featVals = new double[D];
            for (int dd = 0; dd < D; dd++) {
                featVals[dd] = data[dd].get(ii);
            }

            // skip uninformed feature
            if (min(featVals) == max(featVals)) {
                norms[ii] = null;
                continue;
            }

            norms[ii] = new MinMaxNormalizer(featVals, 0.0, 1.0);

            // scale
            double[] normVal = norms[ii].normalize(featVals);
            for (int dd = 0; dd < D; dd++) {
                data[dd].set(ii, normVal[dd]);
            }
        }
        return norms;
    }

    /**
     * Obtain z-normalizers from training data.
     *
     * @param data
     * @param numFeatures
     * @return
     */
    public static ZNormalizer[] zNormalizeTrainingData(SparseVector[] data,
            int numFeatures) {
        int D = data.length;
        ZNormalizer[] norms = new ZNormalizer[numFeatures];
        for (int ii = 0; ii < numFeatures; ii++) {
            double[] featVals = new double[D];
            for (int dd = 0; dd < D; dd++) {
                featVals[dd] = data[dd].get(ii);
            }

            // uninformed feature
            if (min(featVals) == max(featVals)) {
                norms[ii] = null;
                continue;
            }

            norms[ii] = new ZNormalizer(featVals);

            // scale
            double[] normVals = norms[ii].normalize(featVals);
            for (int dd = 0; dd < D; dd++) {
                data[dd].set(ii, normVals[dd]);
            }
        }
        return norms;
    }

    /**
     * Scale the value of the test data using the list of maximum values
     * obtained from the training data. Note that this only applies to positive
     * feature values.
     *
     * @param data Test data in a sparse format
     * @param maxVals Array storing the maximum value of each feature in the
     * training data
     */
    public static void scaleTestData(SparseVector[] data, double[] maxVals) {
        for (SparseVector vec : data) {
            for (int ii : vec.getIndices()) {
                double maxVal = maxVals[ii - 1];
                if (maxVal == 0) {
                    vec.set(ii, 0.0);
                } else {
                    vec.set(ii, vec.get(ii) / maxVal);
                }
            }
        }
    }

    /**
     * Normalize data given the normalizers. The normalizers are usually
     * obtained from the training data, and this method is to normalize test
     * data.
     *
     * @param data Data to be normalized
     * @param norms Pre-learned normalizers
     */
    public static void normalizeTestData(SparseVector[] data,
            AbstractNormalizer[] norms) {
        for (int ii = 0; ii < norms.length; ii++) {
            if (norms[ii] == null) {
                continue;
            }
            for (SparseVector vec : data) {
                vec.set(ii, norms[ii].normalize(vec.get(ii)));
            }
        }
    }

    public static double getL2Norm(double[] vector) {
        double sumSquare = 0.0;
        for (double val : vector) {
            sumSquare += val * val;
        }
        return Math.sqrt(sumSquare);
    }

    public static ArrayList<Integer> discretize(double[] values, int numClasses) {
        ArrayList<Integer> disVals = new ArrayList<Integer>();
        double min = StatUtils.min(values);
        double max = StatUtils.max(values) + 0.00001;
        double step = (max - min) / numClasses;
        for (int i = 0; i < values.length; i++) {
            int cls = (int) ((values[i] - min) / step);
            disVals.add(cls);
        }
        return disVals;
    }

    public static ArrayList<Integer> discretize(ArrayList<Double> values, int numClasses) {
        ArrayList<Integer> disVals = new ArrayList<Integer>();
        double min = StatUtils.min(values);
        double max = StatUtils.max(values) + 0.00001;
        double step = (max - min) / numClasses;
        for (int i = 0; i < values.size(); i++) {
            int cls = (int) ((values.get(i) - min) / step);
            disVals.add(cls);
        }
        return disVals;
    }

    public static double factorial(int n) {
        double factorial = 1;
        for (int i = 1; i <= n; i++) {
            factorial *= i;
        }
        return factorial;
    }

    public static double computePredictedRSquared(double[] trueValues, double[] predValues) {
        double mean = mean(trueValues);
        double num = 0;
        for (int i = 0; i < trueValues.length; i++) {
            double diff = trueValues[i] - predValues[i];
            num += diff * diff;
        }

        double den = 0;
        for (int i = 0; i < trueValues.length; i++) {
            double diff = trueValues[i] - mean;
            den += diff * diff;
        }
        return 1 - num / den;
    }

    public static double computeCorrelationCoefficient(double[] trueValues, double[] predValues) {
        return corr.correlation(trueValues, predValues);
    }

    public static double computeMeanAbsoluteError(double[] trueValues, double[] predValues) {
        double sae = 0.0;
        for (int ii = 0; ii < trueValues.length; ii++) {
            double absDiff = Math.abs(trueValues[ii] - predValues[ii]);
            sae += absDiff;
        }
        return sae / trueValues.length;
    }

    public static double computeMeanSquaredError(double[] trueValues, double[] predValues) {
        double sse = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            double diff = trueValues[i] - predValues[i];
            sse += diff * diff;
        }
        return sse / trueValues.length;
    }

    public static double computeRSquared(double[] trueValues, double[] predValues) {
        double mean = StatUtils.mean(trueValues);
        double totalSS = 0.0;
        double errSS = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            totalSS += (trueValues[i] - mean) * (trueValues[i] - mean);
            errSS += (trueValues[i] - predValues[i]) * (trueValues[i] - predValues[i]);
        }
        return 1.0 - (errSS / totalSS);
    }

    public static double logNormalProbability(double observation,
            double mean, double stdv) {
        double llh = 0;
        llh -= 0.5 * Math.log(2 * Math.PI);
        llh -= Math.log(stdv);
        llh += -(observation - mean) * (observation - mean) / (2 * stdv * stdv);
        return llh;
    }

    public static double normalProbability(double observation, double mean,
            double stdv) {
        double var = stdv * stdv;
        double diff = observation - mean;
        double prob = (1 / stdv) * Math.exp(-diff * diff / (2 * var));
        return prob;
    }

    /**
     * Compute the value of digamma function
     *
     * @param x
     * @return
     */
    public static double digamma(double x) {
        double y, y2, sum = 0;
        for (y = x; y < 10; y++) {
            sum -= 1.0 / y;
        }
        y2 = 1.0 / (y * y);
        sum += Math.log(y) + c1 / y + y2 * (c2 + y2 * (c4 + y2 * c6));
        return sum;
    }

    /**
     * Compute digamma difference
     *
     * @param x
     * @param d
     * @return
     */
    public static double digammaDiff(double x, int d) {
        double sum = 0;
        int dcutoff = 16;
        if (d > dcutoff) {
            return (digamma(x + d) - digamma(x));
        }
        for (int i = 0; i < d; ++i) {
            sum += 1 / (x + i);
        }
        return (sum);
    }
    // for computing trigamma
    public static final double L1 = 0.0001;
    public static final double L2 = 5.0;
    public static final double B2 = 1 / 6.0;
    public static final double B4 = -1 / 30.0;
    public static final double B6 = 1 / 42.0;
    public static final double B8 = -1 / 30.0;

    /**
     * Compute the value of trigamma function See:
     * http://www.jstor.org/stable/2346249
     *
     * @param x
     * @return
     */
    public static double trigamma(double x) {
        if (x < L1) {
            return 1 / (x * x);
        }
        double y, z;
        for (y = 0; x < L2; x++) {
            y += 1 / (x * x);
        }
        z = 1 / (x * x);
        return (y + z / 2 + (1 + z * (B2 + z * (B4 + z * (B6 + z * B8)))) / x);
    }

    /**
     * Compute trigamma difference
     *
     * @param x
     * @param d
     * @return
     */
    public static double trigammaDiff(double x, int d) {
        int tcutoff = 10;
        double sum = 0;
        if (d > tcutoff) {
            return (trigamma(x + d) - trigamma(x));
        }
        for (int i = 0; i < d; ++i) {
            sum -= 1 / ((x + i) * (x + i));
        }
        return sum;
    }

    public static int[] bin(double[] data, int numBins, double min, double max) {
        int[] bins = new int[numBins];

        double stepSize = (max - min) / numBins;

        for (double value : data) {
            int binIndex = (int) ((value - min) / stepSize);
            if (binIndex == numBins) { // correct for the max value
                binIndex = numBins - 1;
            }
            bins[binIndex]++;
        }

        return bins;
    }

    public static int[] bin(double[] data, int numBins) {
        int[] bins = new int[numBins];

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double value : data) {
            if (value > max) {
                max = value;
            }
            if (value < min) {
                min = value;
            }
        }
        double stepSize = (max - min) / numBins;

        for (double value : data) {
            int binIndex = (int) ((value - min) / stepSize);
            if (binIndex == numBins) { // correct for the max value
                binIndex = numBins - 1;
            }
            bins[binIndex]++;
        }

        return bins;
    }

    public static int[] bin(ArrayList<Double> data, int numBins) {
        int[] bins = new int[numBins];

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double value : data) {
            if (value > max) {
                max = value;
            }
            if (value < min) {
                min = value;
            }
        }

        double stepSize = (max - min) / numBins;
        for (double value : data) {
            int binIndex = (int) ((value - min) / stepSize);
            if (binIndex == numBins) // correct for the max value
            {
                binIndex = numBins - 1;
            }
            bins[binIndex]++;
        }

        return bins;
    }

    public static double standardDeviation(ArrayList<Double> values) {
        if (values.size() <= 1) {
            return 0.0;
        }

        double mean = mean(values);
        double ssd = 0.0;
        for (int i = 0; i < values.size(); i++) {
            ssd += (values.get(i) - mean) * (values.get(i) - mean);
        }
        return Math.sqrt(ssd / (values.size() - 1));
    }

    public static double standardDeviation(double[] values) {
        if (values.length <= 1) {
            return 0.0;
        }

        double mean = mean(values);
        double ssd = 0.0;
        for (int i = 0; i < values.length; i++) {
            ssd += (values[i] - mean) * (values[i] - mean);
        }
        return Math.sqrt(ssd / (values.length - 1));
    }

    public static double standardDeviation(SparseCount counts) {
        int countSum = counts.getCountSum();
        if (countSum <= 1) {
            return 0.0;
        }
        double mean = mean(counts);
        double ssd = 0.0;
        for (int i : counts.getIndices()) {
            double diff = i - mean;
            ssd += diff * diff * counts.getCount(i);
        }
        return Math.sqrt(ssd / (countSum - 1));
    }

    public static double mean(SparseCount counts) {
        double total = 0.0;
        for (int i : counts.getIndices()) {
            total += i * counts.getCount(i);
        }
        return total / counts.getCountSum();
    }

    public static double mean(int[] values) {
        return (double) sum(values) / values.length;
    }

    public static double mean(double[] values) {
        return sum(values) / values.length;
    }

    public static double mean(ArrayList<Double> values) {
        return sum(values) / values.size();
    }

    public static double sum(ArrayList<Double> values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum;
    }

    public static double sum(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum;
    }

    public static int sum(int[] values) {
        int sum = 0;
        for (int value : values) {
            sum += value;
        }
        return sum;
    }

    public static double median(ArrayList<Double> values) {
        double[] arr = new double[values.size()];
        for (int ii = 0; ii < arr.length; ii++) {
            arr[ii] = values.get(ii);
        }
        return median(arr);
    }

    public static double median(double[] values) {
        double[] copy = new double[values.length];
        System.arraycopy(values, 0, copy, 0, values.length);
        java.util.Arrays.sort(copy);
        int length = values.length;
        if (length % 2 != 0) {
            return copy[length / 2];
        } else {
            return ((double) (copy[length / 2 - 1] + copy[length / 2])) / 2;
        }
    }

    public static int max(int[] array) {
        int max = -Integer.MAX_VALUE;
        for (int a : array) {
            if (a > max) {
                max = a;
            }
        }
        return max;
    }

    public static double max(double[] array) {
        double max = -Double.MAX_VALUE;
        for (double a : array) {
            if (a > max) {
                max = a;
            }
        }
        return max;
    }

    public static double min(int[] array) {
        int min = Integer.MAX_VALUE;
        for (int a : array) {
            if (a < min) {
                min = a;
            }
        }
        return min;
    }

    public static double min(ArrayList<Double> array) {
        double min = Double.MAX_VALUE;
        for (double a : array) {
            if (a < min) {
                min = a;
            }
        }
        return min;
    }

    public static double max(ArrayList<Double> array) {
        double max = -Double.MAX_VALUE;
        for (double a : array) {
            if (a > max) {
                max = a;
            }
        }
        return max;
    }

    public static double min(double[] array) {
        double min = Double.MAX_VALUE;
        for (double a : array) {
            if (a < min) {
                min = a;
            }
        }
        return min;
    }

    /**
     * Compute the cosine similarity between two vectors
     *
     * @param p1 The first vector
     * @param p2 The second vector
     * @return The cosine similarity between the two vectors
     */
    public static double cosineSimilarity(double[] p1, double[] p2) {
        double dotProduct = dotProduct(p1, p2);
        double norm1 = norm(p1);
        double norm2 = norm(p2);
        if (norm1 == 0 || norm2 == 0) {
            return 0;
        }
        return dotProduct / (norm1 * norm2);
    }

    /**
     * JS divergence
     *
     * @param p1
     * @param p2
     * @return
     */
    public static double JSDivergence(double[] p1, double[] p2) {
        double[] p = new double[p1.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = 0.5 * (p1[i] + p2[i]);
        }
        double kl1 = KLDivergenceAsymmetric(p1, p);
        double kl2 = KLDivergenceAsymmetric(p2, p);
        return 0.5 * (kl1 + kl2);
    }

    public static double[] getSelectionalAssociation(double[] p, double[] q) {
        if (p.length != q.length) {
            throw new RuntimeException("Dimensions mismatch. " + p.length + " vs. " + q.length);
        }
        double[] selectAssocs = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            selectAssocs[i] = p[i] * Math.log(p[i] - q[i]);
        }
        return selectAssocs;
    }

    /**
     * Asymmetric KL divergence
     *
     * @param p1
     * @param p2
     * @return
     */
    public static double KLDivergenceAsymmetric(double[] p1, double[] p2) {
        assert (p1.length == p2.length);
        double klDiv = 0.0;
        for (int i = 0; i < p1.length; ++i) {
            if (p1[i] == 0) {
                continue;
            }
            if (p2[i] == 0) {
                return Double.POSITIVE_INFINITY;
            }
            klDiv += p1[i] * Math.log(p1[i] / p2[i]);
        }
        return klDiv / Math.log(2);
    }

    /**
     * Symmetric KL divergence
     *
     * @param dist_1
     * @param dist_2
     * @return
     */
    public static double KLDivergenceSymmetric(double[] dist_1, double[] dist_2) {
        double a_kl_12 = KLDivergenceAsymmetric(dist_1, dist_2);
        double a_kl_21 = KLDivergenceAsymmetric(dist_2, dist_1);
        return 0.5 * (a_kl_12 + a_kl_21);
    }

    /**
     * Normalize to a distribution in which the sum of all the elements is 1
     *
     * @param values
     * @return
     */
    public static double[] normalize(double[] values) {
        double sum = sum(values);
        double[] norm = new double[values.length];
        for (int i = 0; i < norm.length; i++) {
            norm[i] = values[i] / sum;
        }
        return norm;
    }

    /**
     * Compute the entropy of a discrete distribution
     *
     * @param distribution
     * @return
     */
    public static double entropy(double[] distribution) {
        double entropy = 0;
        for (double p : distribution) {
            if (p != 0) {
                entropy += p * (Math.log(p) / Math.log(2.0));
            }
        }
        return -entropy;
    }

    public static double dotProduct(double[] v1, ArrayList<Double> v2) {
        if (v1.length != v2.size()) {
            throw new RuntimeException("Vectors have different lengths. "
                    + v1.length + " vs. " + v2.size());
        }
        double dotprod = 0;
        for (int i = 0; i < v1.length; i++) {
            dotprod += v1[i] * v2.get(i);
        }
        return dotprod;
    }

    public static double dotProduct(double[] v1, double[] v2) {
        double p = 0;
        for (int i = 0; i < v1.length; i++) {
            p += v1[i] * v2[i];
        }
        return p;
    }

    public static double dotProduct(ArrayList<Double> w, ArrayList<Double> f) {
        if (w.size() != f.size()) {
            throw new RuntimeException("Vectors have different lengths. "
                    + w.size() + " vs. " + f.size());
        }

        double dotprod = 0;
        for (int i = 0; i < w.size(); i++) {
            dotprod += w.get(i) * f.get(i);
        }
        return dotprod;
    }

    public static double sumSquare(ArrayList<Double> vs) {
        double sum_sqrt = 0;
        for (double v : vs) {
            sum_sqrt += v * v;
        }
        return sum_sqrt;
    }

    public static double norm(ArrayList<Double> v) {
        double sum_sqrt = sum(v);
        return Math.sqrt(sum_sqrt);
    }

    public static double norm(double[] v) {
        double sum_sqrt = 0;
        for (int i = 0; i < v.length; i++) {
            sum_sqrt += v[i] * v[i];
        }
        return Math.sqrt(sum_sqrt);
    }

    public static ArrayList<Double> sumVector(ArrayList<Double> v1, ArrayList<Double> v2) {
        ArrayList<Double> sumV = new ArrayList<Double>();
        for (int i = 0; i < v1.size(); i++) {
            sumV.add(v1.get(i) + v2.get(i));
        }
        return sumV;
    }
}
