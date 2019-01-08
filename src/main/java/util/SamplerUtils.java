package util;

import cc.mallet.util.Randoms;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import sampling.util.SparseCount;

/**
 *
 * @author vietan
 */
public class SamplerUtils {

    public static final long RAND_SEED = 1123581321;
    public static final double MAX_LOG = Math.log(Double.MAX_VALUE);
    public static final double HALF_LOG_TWO_PI = Math.log(2 * Math.PI) / 2;
    public static final double EULER_MASCHERONI = -0.5772156649015328606065121;
    public static Random rand = new Random(RAND_SEED);
    public static Randoms randoms = new Randoms((int)RAND_SEED);

    public static void resetRand() {
        rand = new Random(RAND_SEED);
    }
    
    public static double[] sampleMultinomial(double[] dirVector) {
        double[] ts = new double[dirVector.length];
        double sum = 0.0;
        for (int v = 0; v < dirVector.length; v++) {
            ts[v] = randoms.nextGamma(dirVector[v], 1);
            sum += ts[v];
        }

        // normalize
        for (int v = 0; v < dirVector.length; v++) {
            ts[v] /= sum;
            if (ts[v] == 0) {
                ts[v] = 10E-4;
            }
        }
        return ts;
    }

    /**
     * Sample number of components m that a DP(alpha, G0) has after n samples.
     * This was first published by Antoniak (1974).
     *
     * @param alpha
     * @param n
     * @return
     */
    public static int randAntoniak(double alpha, int n) {
        int totalCount = 0;
        int numSamples = 20;

        for (int ii = 0; ii < numSamples; ii++) {
            int count = 0;
            for (int r = 0; r < n; r++) {
                double prob = alpha / (alpha + r);
                if (rand.nextDouble() < prob) {
                    count++;
                }
            }
            totalCount += count;
        }
        return totalCount / numSamples;
    }

    public static int[] getSortedTopic(double[] distribution) {
        int[] sortedTopic = new int[distribution.length];
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < sortedTopic.length; v++) {
            rankItems.add(new RankingItem<Integer>(v, distribution[v]));
        }
        Collections.sort(rankItems);
        for (int i = 0; i < rankItems.size(); i++) {
            sortedTopic[i] = rankItems.get(i).getObject();
        }
        return sortedTopic;
    }
    private static double[] cc = {76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155,
        0.1208650973866179e-2, -0.5395239384953e-5};

    public static int maxIndex(double[] arrays) {
        int mIdx = -1;
        double maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < arrays.length; i++) {
            if (arrays[i] > maxValue) {
                maxValue = arrays[i];
                mIdx = i;
            }
        }
        return mIdx;
    }

    public static int maxIndex(int[] counts) {
        int mIdx = -1;
        double maxValue = -Integer.MAX_VALUE;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > maxValue) {
                maxValue = counts[i];
                mIdx = i;
            }
        }
        return mIdx;
    }

    public static int maxIndex(ArrayList<Double> list) {
        int mIdx = -1;
        double maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > maxValue) {
                maxValue = list.get(i);
                mIdx = i;
            }
        }
        return mIdx;
    }

    /**
     * Compute the log joint probability of table assignments
     *
     * @param customerCounts Number of customers assigned to each table
     * @param pseudoCustomerCount The pseudo number of customers (DP
     * hyperparameter)
     */
    public static double getAssignmentJointLogProbability(
            ArrayList<Integer> customerCounts,
            double pseudoCustomerCount) {
        if (customerCounts.isEmpty()) {
            return 0.0;
        }

        int numTables = customerCounts.size();
        int numTotalCustomers = 0;
        int maxNumTotalCustomers = Integer.MIN_VALUE;

        for (int customerCount : customerCounts) {
            numTotalCustomers += customerCount;
            if (customerCount > maxNumTotalCustomers) {
                maxNumTotalCustomers = customerCount;
            }
        }

        double[] cachedLogs = new double[maxNumTotalCustomers];
        for (int i = 0; i < maxNumTotalCustomers; i++) {
            cachedLogs[i] = Math.log(i + 1);
        }

        double logprob = numTables * Math.log(pseudoCustomerCount);
        for (int customerCount : customerCounts) {
            for (int i = 0; i < customerCount; i++) {
                logprob += cachedLogs[i];
            }
        }
        for (int x = 1; x <= numTotalCustomers; x++) {
            logprob -= Math.log(x - 1 + pseudoCustomerCount);
        }
        return logprob;
    }

    public static double getGaussian(double aMean, double aVariance) {
        return aMean + rand.nextGaussian() * Math.sqrt(aVariance);
    }

    public static double logGammaStirling(double x) {
        if (x == 0) {
            return logGammaStirling(1E-300);
        }

        if (x < 1E-300) // if x is too small, use the exact formula instead
        {
            return logGammaDefinition(x);
        }

        int i;
        double y, t, r;

        y = x;
        t = x + 5.5;
        t -= (x + 0.5) * Math.log(t);
        r = 1.000000000190015;
        for (i = 0; i < 6; ++i) {
            r += cc[i] / (y += 1.0);
        }
        return -t + Math.log(2.5066282746310005 * r / x);
    }

    /**
     * This calculates a log gamma function exactly. It's extremely inefficient
     * -- use this for comparison only.
     */
    public static double logGammaDefinition(double z) {
        double result = EULER_MASCHERONI * z - Math.log(z);
        for (int k = 1; k < 10000000; k++) {
            result += (z / k) - Math.log(1 + (z / k));
        }
        return result;
    }

    /**
     * Compute log likelihood for a single symmetric multinomial
     *
     * @param obs Count vector of observations
     * @param sum Total number of observations
     * @param prior_val A single element in the symmetric prior vector
     */
    public static double computeLogLhood(int[] obs, int sum, double prior_val) {
        double size = obs.length;
        double prior_sum = size * prior_val;
        double val = 0.0;

        val += SamplerUtils.logGammaStirling(prior_sum);
        val -= size * SamplerUtils.logGammaStirling(prior_val);
        for (int ii = 0; ii < obs.length; ++ii) {
            val += SamplerUtils.logGammaStirling(prior_val + (double) obs[ii]);
        }
        val -= SamplerUtils.logGammaStirling(sum + prior_sum);
        return val;
    }

    public static double computeLogLhood(SparseCount obs, double[] prior_mean, double concentration) {
        double val = 0.0;
        val += logGammaStirling(concentration);
        val -= logGammaStirling(obs.getCountSum() + concentration);
        for (int i = 0; i < prior_mean.length; i++) {
            double pseudoCount = concentration * prior_mean[i];
            val -= logGammaStirling(pseudoCount);
            val += logGammaStirling(pseudoCount + obs.getCount(i));
        }
        return val;
    }

    public static double computeLogLhood(SparseCount obs, double[] priorVals) {
        double val = 0.0;
        double priorValSum = StatUtils.sum(priorVals);
        val += logGammaStirling(priorValSum);
        val -= logGammaStirling(obs.getCountSum() + priorValSum);
        for (int i = 0; i < priorVals.length; i++) {
            val -= logGammaStirling(priorVals[i]);
            val += logGammaStirling(priorVals[i] + obs.getCount(i));
        }
        return val;
    }

    /**
     * Compute log likelihood for an asymmetric multinomial when the prior is
     * expressed using a mean vector and a concentration parameter
     *
     * @param obs Count vector of observations
     * @param sum Total number of observations
     * @param prior_mean Mean vector
     * @param concentration Concentration parameter
     */
    public static double computeLogLhood(int[] obs, int sum,
            double[] prior_mean, double concentration) {
        double val = 0.0;
        val += logGammaStirling(concentration);
        val -= logGammaStirling(sum + concentration);
        for (int i = 0; i < obs.length; i++) {
            double pseudoCount = concentration * prior_mean[i];
            val -= logGammaStirling(pseudoCount);
            val += logGammaStirling(pseudoCount + obs[i]);
        }
        return val;
    }

    /**
     * Compute log likelihood for an asymmetric multinomial when the prior is
     * expressed using a mean vector and a concentration parameter
     *
     * @param obs Count vector of observations
     * @param sum Total number of observations
     * @param prior_mean Mean vector
     * @param concentration Concentration parameter
     */
    public static double computeLogLhoodForDebug(int[] obs, int sum,
            double[] prior_mean, double concentration) {
        double val = 0.0;
        val += logGammaStirling(concentration);
        val -= logGammaStirling(sum + concentration);
        for (int i = 0; i < obs.length; i++) {
            double pseudoCount = concentration * prior_mean[i];
            val -= logGammaStirling(pseudoCount);
            val += logGammaStirling(pseudoCount + obs[i]);

            System.out.println("i=" + i + "\t"
                    + pseudoCount + "\t"
                    + logGammaStirling(pseudoCount) + "\t"
                    + obs[i] + "\t"
                    + (pseudoCount + obs[i]) + "\t"
                    + logGammaStirling(pseudoCount + obs[i]));
        }
        return val;
    }

    public static double computeLogLhood(double[] mult, double[] prior_mean,
            double concentration) {
        double val = 0.0;
        val += logGammaStirling(concentration);
        for (int ii = 0; ii < mult.length; ii++) {
            double pseudoCount = concentration * prior_mean[ii];
            val -= logGammaStirling(pseudoCount);
            val += (pseudoCount - 1) * Math.log(mult[ii]);
        }
        return val;
    }

    /**
     * Compute log likelihood for an asymmetric multinomial when the prior is
     * expressed using a vector
     *
     * @param obs Count vector of observations
     * @param sum Total number prior_valsof observations
     * @param prior_vals Prior vector
     */
    public static double computeLogLhood(int[] obs, int sum, double[] prior_vals) {
        double prior_sum = 0;
        for (double p : prior_vals) {
            prior_sum += p;
        }

        double val = 0.0;
        val += SamplerUtils.logGammaStirling(prior_sum);

        for (double p : prior_vals) {
            val -= SamplerUtils.logGammaStirling(p);
        }

        for (int ii = 0; ii < obs.length; ++ii) {
            val += SamplerUtils.logGammaStirling(prior_vals[ii] + (double) obs[ii]);
        }

        val -= SamplerUtils.logGammaStirling(sum + prior_sum);

        return val;
    }

    /**
     * Compute the log probability that a given multinomial distribution were
     * drawn from a given Dirichlet distribution.
     *
     * @param mult The multinomial distribution
     * @param prior_vals The parameter vector of the Dirichlet distribution
     */
    public static double computeLogLhood(double[] mult, double[] prior_vals) {
        double val = 0.0;

        double prior_sum = StatUtils.sum(prior_vals);
        val += logGammaStirling(prior_sum);
        for (double p : prior_vals) {
            val -= logGammaStirling(p);
        }
        for (int ii = 0; ii < mult.length; ii++) {
            val += (prior_vals[ii] - 1) * Math.log(mult[ii]);
        }
        return val;
    }

    /**
     * Compute log likelihood for multiple symmetric multinomials
     */
    public static double computeRepeatedLogLhood(ArrayList<int[]> obs, ArrayList<Integer> sum, double prior_val) {
        int K = obs.size();
        double val = 0.0;
        for (int i = 0; i < K; i++) {
            val += computeLogLhood(obs.get(i), sum.get(i), prior_val);
        }
        return val;
    }

    /**
     * Compute log likelihood for multiple symmetric multinomials
     */
    public static double computeRepeatedLogLhood(int[][] obs, int[] sum, double prior_val) {
        int K = obs.length;
        double val = 0.0;
        for (int ii = 0; ii < K; ++ii) {
            val += computeLogLhood(obs[ii], sum[ii], prior_val);
        }
        return val;
    }

    /**
     * Scale sample from a pdf
     */
    public static int scaleSample(double[] distribution) {
        double[] cumm_probs = new double[distribution.length];
        System.arraycopy(distribution, 0, cumm_probs, 0, cumm_probs.length);
        for (int i = 1; i < cumm_probs.length; i++) {
            cumm_probs[i] += cumm_probs[i - 1];
        }

        double randValue = rand.nextDouble() * cumm_probs[cumm_probs.length - 1];

        int index;
        for (index = 0; index < cumm_probs.length; index++) {
            if (randValue < cumm_probs[index]) {
                break;
            }
        }
        return index;
    }

    /**
     * Scale sample from a pdf
     */
    public static int scaleSample(ArrayList<Double> distribution) {
        double[] cumm_probs = new double[distribution.size()];
        for (int i = 0; i < cumm_probs.length; i++) {
            cumm_probs[i] = distribution.get(i);
        }
        for (int i = 1; i < cumm_probs.length; i++) {
            cumm_probs[i] += cumm_probs[i - 1];
        }

        double randValue = rand.nextDouble() * cumm_probs[cumm_probs.length - 1];

        int index;
        for (index = 0; index < cumm_probs.length; index++) {
            if (randValue < cumm_probs[index]) {
                break;
            }
        }
        return index;
    }

    public static int scaleSample(double[] weights, double sum) {
        double b = 0, r = rand.nextDouble() * sum;
        int i;
        for (i = 0; i < weights.length; i++) {
            b += weights[i];
            if (b > r) {
                break;
            }
        }
//        System.out.println("weights: " + MiscUtils.arrayToString(weights));
//        System.out.println("r: " + r 
//                + ". sum = " + sum
//                + ". i = " + i
//                + "\n");
        return i;
    }

    public static int logMaxRescaleSample(ArrayList<Double> logDistList) {
        double[] logDist = new double[logDistList.size()];
        for (int i = 0; i < logDist.length; i++) {
            logDist[i] = logDistList.get(i);
        }
        return logMaxRescaleSample(logDist);
    }

    public static int logMaxRescaleSample(double[] logDist) {
        double sum = 0.0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logDist.length; i++) {
            if (logDist[i] > max) {
                max = logDist[i];
            }
        }
        double[] weights = new double[logDist.length];
        for (int i = 0; i < logDist.length; i++) {
            weights[i] = Math.exp(logDist[i] - max);
            sum += weights[i];
        }
        return scaleSample(weights, sum);
    }

//    public static int logScaleSampleNew(double[] logPdf){
//        
//    }
    /**
     * Scale sample from a pdf in the log space
     */
    public static int logScaleSample(double[] logPdf) {
        double[] logCdf = new double[logPdf.length];
        logCdf[0] = logPdf[0];
        for (int i = 1; i < logPdf.length; i++) {
            logCdf[i] = logAdd(logCdf[i - 1], logPdf[i]);
        }
        double logmax = logCdf[logCdf.length - 1];
        //System.out.println("logmax = " + logmax);

        int n = (int) (logmax / MAX_LOG);
        if (n > 0) {
            for (int i = 0; i < logCdf.length; i++) {
                if (logCdf[i] < MAX_LOG * n) {
                    logCdf[i] = 0;
                } else {
                    logCdf[i] -= MAX_LOG * n;
                }
            }
        }

        double max = Math.exp(logCdf[logCdf.length - 1]);
        double randVal = rand.nextDouble();
        double sampledLogVal = Math.log(randVal * max);
        int index;
        for (index = 0; index < logCdf.length; index++) {
            if (sampledLogVal < logCdf[index]) {
                break;
            }
        }

        //debug
//        System.out.println("logpdf: " + MiscUtils.arrayToString(logPdf));
//        System.out.println("logcdf: " + MiscUtils.arrayToString(logCdf));
//        System.out.println("max = " + max 
//                + ". randval = " + MiscUtils.formatDouble(randVal) 
//                + ". sampleLogVal = " + MiscUtils.formatDouble(sampledLogVal)
//                + ". index = " + index);
//        System.out.println();

        return index;
    }

    public static int logMinRescaleSample(double[] logPdf) {
        double[] scaledLogPdf = new double[logPdf.length];
        double min = StatUtils.min(logPdf);
        for (int i = 0; i < scaledLogPdf.length; i++) {
            scaledLogPdf[i] = logPdf[i] - min;
        }
        return logScaleSample(scaledLogPdf);
    }

    public static int logMinRescaleSample(ArrayList<Double> logPdf) {
        double[] logPdfArr = new double[logPdf.size()];
        for (int i = 0; i < logPdfArr.length; i++) {
            logPdfArr[i] = logPdf.get(i);
        }
        return logMinRescaleSample(logPdfArr);
    }

    /**
     * Compute log(X) using log(X + Y) and log(Y)
     *
     * @param logXAndY log(X + Y)
     * @param logY log(Y)
     * @return log(X)
     */
    public static double logMinus(double logXAndY, double logY) {
        double diffLog = logXAndY - logY;
        if (diffLog > 20) {
            return logXAndY;
        }
        return logY + Math.log(Math.expm1(diffLog));
    }

    /**
     * Compute log(X + Y) from logX and logY
     *
     * @param logX log of X
     * @param logY log of Y
     * @return log(X + Y)
     */
    public static double logAdd(double logX, double logY) {
        // 1. make X the max
        if (logY > logX) {
            double temp = logX;
            logX = logY;
            logY = temp;
        }
        // 2. now X is bigger
        if (logX == Double.NEGATIVE_INFINITY) {
            return logX;
        }
        // 3. how far "down" (think decibels) is logY from logX?
        //    if it's really small (20 orders of magnitude smaller), then ignore
        double negDiff = logY - logX;
        if (negDiff < -20) {
            return logX;
        }
        // 4. otherwise use some nice algebra to stay in the log domain
        //    (except for negDiff)
        //return logX + java.lang.Math.log(1.0 + java.lang.Math.exp(negDiff));
        return logX + Math.log1p(Math.exp(negDiff));
    }

    // From libbow, dirichlet.c
    // Written by Tom Minka <minka@stat.cmu.edu>
    public static double logGamma(double x) {
        double result, y, xnum, xden;
        int i;
        final double d1 = -5.772156649015328605195174e-1;
        final double p1[] = {
            4.945235359296727046734888e0, 2.018112620856775083915565e2,
            2.290838373831346393026739e3, 1.131967205903380828685045e4,
            2.855724635671635335736389e4, 3.848496228443793359990269e4,
            2.637748787624195437963534e4, 7.225813979700288197698961e3
        };
        final double q1[] = {
            6.748212550303777196073036e1, 1.113332393857199323513008e3,
            7.738757056935398733233834e3, 2.763987074403340708898585e4,
            5.499310206226157329794414e4, 6.161122180066002127833352e4,
            3.635127591501940507276287e4, 8.785536302431013170870835e3
        };
        final double d2 = 4.227843350984671393993777e-1;
        final double p2[] = {
            4.974607845568932035012064e0, 5.424138599891070494101986e2,
            1.550693864978364947665077e4, 1.847932904445632425417223e5,
            1.088204769468828767498470e6, 3.338152967987029735917223e6,
            5.106661678927352456275255e6, 3.074109054850539556250927e6
        };
        final double q2[] = {
            1.830328399370592604055942e2, 7.765049321445005871323047e3,
            1.331903827966074194402448e5, 1.136705821321969608938755e6,
            5.267964117437946917577538e6, 1.346701454311101692290052e7,
            1.782736530353274213975932e7, 9.533095591844353613395747e6
        };
        final double d4 = 1.791759469228055000094023e0;
        final double p4[] = {
            1.474502166059939948905062e4, 2.426813369486704502836312e6,
            1.214755574045093227939592e8, 2.663432449630976949898078e9,
            2.940378956634553899906876e10, 1.702665737765398868392998e11,
            4.926125793377430887588120e11, 5.606251856223951465078242e11
        };
        final double q4[] = {
            2.690530175870899333379843e3, 6.393885654300092398984238e5,
            4.135599930241388052042842e7, 1.120872109616147941376570e9,
            1.488613728678813811542398e10, 1.016803586272438228077304e11,
            3.417476345507377132798597e11, 4.463158187419713286462081e11
        };
        final double c[] = {
            -1.910444077728e-03, 8.4171387781295e-04,
            -5.952379913043012e-04, 7.93650793500350248e-04,
            -2.777777777777681622553e-03, 8.333333333333333331554247e-02,
            5.7083835261e-03
        };
        final double a = 0.6796875;

        if ((x <= 0.5) || ((x > a) && (x <= 1.5))) {
            if (x <= 0.5) {
                result = -Math.log(x);
                /*
                 * Test whether X < machine epsilon.
                 */
                if (x + 1 == 1) {
                    return result;
                }
            } else {
                result = 0;
                x = (x - 0.5) - 0.5;
            }
            xnum = 0;
            xden = 1;
            for (i = 0; i < 8; i++) {
                xnum = xnum * x + p1[i];
                xden = xden * x + q1[i];
            }
            result += x * (d1 + x * (xnum / xden));
        } else if ((x <= a) || ((x > 1.5) && (x <= 4))) {
            if (x <= a) {
                result = -Math.log(x);
                x = (x - 0.5) - 0.5;
            } else {
                result = 0;
                x -= 2;
            }
            xnum = 0;
            xden = 1;
            for (i = 0; i < 8; i++) {
                xnum = xnum * x + p2[i];
                xden = xden * x + q2[i];
            }
            result += x * (d2 + x * (xnum / xden));
        } else if (x <= 12) {
            x -= 4;
            xnum = 0;
            xden = -1;
            for (i = 0; i < 8; i++) {
                xnum = xnum * x + p4[i];
                xden = xden * x + q4[i];
            }
            result = d4 + x * (xnum / xden);
        } /*
         * X > 12
         */ else {
            y = Math.log(x);
            result = x * (y - 1) - y * 0.5 + .9189385332046727417803297;
            x = 1 / x;
            y = x * x;
            xnum = c[6];
            for (i = 0; i < 6; i++) {
                xnum = xnum * y + c[i];
            }
            xnum *= x;
            result += xnum;
        }
        return result;
    }
}
