package sampling.likelihood;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 * Implementation of a truncated stick breaking process.
 *
 * @author vietan
 */
public class TruncatedStickBreaking {

    private int dimension;
    private double mean;
    private double scale;
    private int[] counts; // store N_k
    private int[] backwardCounts; // store N_{\geq k}

    public TruncatedStickBreaking(int l, double m, double s) {
        this.dimension = l;
        this.mean = m;
        this.scale = s;
        this.counts = new int[this.dimension];
        this.backwardCounts = new int[this.dimension];
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public void setScale(double scale) {
        this.scale = scale;
    }

    public double getLogLikelihood(double newMean, double newScale) {
        double v = newMean * newScale;
        double w = (1 - newMean) * newScale;
        double[] pi = new double[dimension]; // expected value of posterior
        for (int i = 0; i < dimension; i++) {
            pi[i] = Math.exp(getLogProbability(i));
        }

        double[] v_ks = new double[dimension - 1];
        double[] w_ks = new double[dimension - 1];
        for (int k = 0; k < v_ks.length; k++) {
            v_ks[k] = v + counts[k];
            w_ks[k] = w + backwardCounts[k] - counts[k];
        }

        double llh = 0.0;
        for (int k = 0; k < dimension - 1; k++) {
            llh += SamplerUtils.logGamma(v_ks[k] + w_ks[k])
                    - SamplerUtils.logGamma(v_ks[k])
                    - SamplerUtils.logGamma(w_ks[k])
                    + (v_ks[k] - 1) * Math.log(pi[k]);
        }
        double w_d = w + backwardCounts[dimension - 2] - counts[dimension - 2];
        llh += (w_d - 1) * Math.log(pi[dimension - 1]);

        if (dimension > 2) {
            double[] sumPi = new double[dimension - 2];
            sumPi[0] = pi[0];
            for (int i = 1; i < sumPi.length; i++) {
                sumPi[i] = sumPi[i - 1] + pi[i];
            }
            for (int i = 0; i < sumPi.length; i++) {
                llh += (w_ks[i] - (v_ks[i + 1] + w_ks[i + 1])) * Math.log(1 - sumPi[i]);
            }
        }
        return llh;
    }

    public double getLogLikelihood() {
        double v = mean * scale;
        double w = (1 - mean) * scale;
        double[] pi = new double[dimension]; // expected value of posterior
        for (int i = 0; i < dimension; i++) {
            pi[i] = Math.exp(getLogProbability(i));
        }

        double[] v_ks = new double[dimension - 1];
        double[] w_ks = new double[dimension - 1];
        for (int k = 0; k < v_ks.length; k++) {
            v_ks[k] = v + counts[k];
            w_ks[k] = w + backwardCounts[k] - counts[k];
        }

        double llh = 0.0;
        for (int k = 0; k < dimension - 1; k++) {
            llh += SamplerUtils.logGamma(v_ks[k] + w_ks[k])
                    - SamplerUtils.logGamma(v_ks[k])
                    - SamplerUtils.logGamma(w_ks[k])
                    + (v_ks[k] - 1) * Math.log(pi[k]);
        }
        double w_d = w + backwardCounts[dimension - 2] - counts[dimension - 2];
        llh += (w_d - 1) * Math.log(pi[dimension - 1]);

        if (dimension > 2) {
            double[] sumPi = new double[dimension - 2];
            sumPi[0] = pi[0];
            for (int i = 1; i < sumPi.length; i++) {
                sumPi[i] = sumPi[i - 1] + pi[i];
            }
            for (int i = 0; i < sumPi.length; i++) {
                llh += (w_ks[i] - (v_ks[i + 1] + w_ks[i + 1])) * Math.log(1 - sumPi[i]);
            }
        }
        return llh;
    }

    public int[] getCounts() {
        return this.counts;
    }

    public int getCount(int index) {
        return this.counts[index];
    }

    public int getCountSum() {
        return this.backwardCounts[0];
    }

    public void setCounts(int[] counts) {
        if (dimension != counts.length) {
            throw new RuntimeException("Dimension mismatched. " + dimension + " vs. " + counts.length);
        }
        this.counts = counts;
        this.backwardCounts = new int[this.counts.length];
        this.backwardCounts[dimension - 1] = this.counts[dimension - 1];
        for (int j = dimension - 2; j >= 0; j--) {
            this.backwardCounts[j] = this.counts[j] + this.backwardCounts[j + 1];
        }
    }

    public void changeCount(int k, int delta) {
        this.counts[k] += delta;
        for (int i = 0; i <= k; i++) {
            this.backwardCounts[i] += delta;
        }

        if (this.counts[k] < 0) {
            throw new RuntimeException("Negative count " + this.counts[k]);
        } else if (this.backwardCounts[k] < 0) {
            throw new RuntimeException("Negative backward count " + this.backwardCounts[k]);
        }
    }

    public void increment(int k) {
        this.counts[k]++;
        for (int i = 0; i <= k; i++) {
            this.backwardCounts[i]++;
        }
    }

    public void decrement(int k) {
        this.counts[k]--;
        for (int i = 0; i <= k; i++) {
            this.backwardCounts[i]--;
        }

        if (this.counts[k] < 0) {
            throw new RuntimeException("Negative count " + this.counts[k]);
        } else if (this.backwardCounts[k] < 0) {
            throw new RuntimeException("Negative backward count " + this.backwardCounts[k]);
        }
    }

    public double getProbability(int k) {
        double score = (this.mean * this.scale + this.counts[k])
                / (this.scale + this.backwardCounts[k]);
        for (int j = 0; j < k; j++) {
            score *= ((1 - this.mean) * this.scale + (this.backwardCounts[j] - this.counts[j]))
                    / (this.scale + this.backwardCounts[j]);
        }
        return score;
    }

    public double getLogProbability(int k) {
        double score = (this.mean * this.scale + this.counts[k])
                / (this.scale + this.backwardCounts[k]);
        for (int j = 0; j < k; j++) {
            score *= ((1 - this.mean) * this.scale + (this.backwardCounts[j] - this.counts[j]))
                    / (this.scale + this.backwardCounts[j]);
        }
        return Math.log(score);
    }

    private double getSingleNodeLogProbability(
            double priorHead, double priorTail,
            int numHeads, int numTails) {
        double priorSum = priorHead + priorTail;
        double val = 0.0;
        int c = 0;
        for (int ii = 0; ii < numHeads; ii++) {
            val += Math.log(priorHead + ii) - Math.log(priorSum + c);
            c++;
        }
        for (int jj = 0; jj < numTails; jj++) {
            val += Math.log(priorTail + jj) - Math.log(priorSum + c);
            c++;
        }
        return val;
    }

    public double getLogProbability(int[] obs) {
        int[] backObs = new int[obs.length];
        backObs[obs.length-1] = obs[obs.length-1];
        for(int l=obs.length-2; l>=0; l--){
            backObs[l] = backObs[l+1] + obs[l];
        }
                
        double val = 0.0;
        for(int l=0; l<obs.length; l++) {
            double priorHead = mean * scale + counts[l];
            double priorTail = (1-mean) * scale + (backwardCounts[l] - counts[l]);
            int numHeads = obs[l];
            int numTails = backObs[l] - obs[l];
            val += getSingleNodeLogProbability(priorHead, priorTail, numHeads, numTails);
        }
        return val;
    }

    public void validate(String msg) {
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] < 0) {
                throw new RuntimeException("Negative count i = " + i + ". count = " + counts[i]);
            }
        }
    }

    public double[] getDistribution() {
        double[] distribution = new double[dimension];
        double sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            distribution[i] = getProbability(i);
            sum += distribution[i];
        }
        for (int i = 0; i < dimension; i++) {
            distribution[i] /= sum;
        }
        return distribution;
    }

    public double[] getEmpiricalDistribution() {
        double[] empDist = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            empDist[i] = (double) this.getCount(i) / this.getCountSum();
        }
        return empDist;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(MiscUtils.arrayToString(this.counts));
        return str.toString();
    }

    public static String output(TruncatedStickBreaking tsb) {
        StringBuilder str = new StringBuilder();
        str.append(tsb.dimension)
                .append("\t").append(tsb.mean)
                .append("\t").append(tsb.scale);
        for (int j = 0; j < tsb.dimension; j++) {
            str.append("\t").append(tsb.counts[j]);
        }
        return str.toString();
    }

    public static TruncatedStickBreaking input(String str) {
        String[] sstr = str.split("\t");
        int dim = Integer.parseInt(sstr[0]);
        double mean = Double.parseDouble(sstr[1]);
        double scale = Double.parseDouble(sstr[2]);
        TruncatedStickBreaking tsb = new TruncatedStickBreaking(dim, mean, scale);
        int[] counts = new int[dim];
        for (int i = 0; i < counts.length; i++) {
            counts[i] = Integer.parseInt(sstr[i + 3]);
        }
        tsb.setCounts(counts);
        return tsb;
    }

    public static void outputTruncatedStickBreakings(TruncatedStickBreaking[] tsbs, String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < tsbs.length; i++) {
            writer.write("dimension\t" + tsbs[i].dimension + "\n");
            writer.write("mean\t" + tsbs[i].mean + "\n");
            writer.write("scale\t" + tsbs[i].scale + "\n");
            for (int j = 0; j < tsbs[i].counts.length; j++) {
                writer.write(tsbs[i].counts[j] + "\t");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public static TruncatedStickBreaking[] inputTruncatedStickBreakings(String filepath) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        ArrayList<TruncatedStickBreaking> tsbList = new ArrayList<TruncatedStickBreaking>();
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            int dim = Integer.parseInt(line.split("\t")[1]);
            double mean = Double.parseDouble(reader.readLine().split("\t")[1]);
            double scale = Double.parseDouble(reader.readLine().split("\t")[1]);
            sline = reader.readLine().split("\t");
            int[] counts = new int[sline.length];
            for (int i = 0; i < counts.length; i++) {
                counts[i] = Integer.parseInt(sline[i]);
            }

            TruncatedStickBreaking tsb = new TruncatedStickBreaking(dim, mean, scale);
            tsb.setCounts(counts);
            tsbList.add(tsb);
        }
        reader.close();
        return tsbList.toArray(new TruncatedStickBreaking[tsbList.size()]);
    }

    public static void main(String[] args) {
        int dim = 3;
        double mean = 0.5;
        double scale = 100;
        TruncatedStickBreaking stick = new TruncatedStickBreaking(dim, mean, scale);
        for (int i = 0; i < 3; i++) {
            stick.increment(0);
        }
        for (int i = 0; i < 3; i++) {
            stick.increment(1);
        }
        for (int i = 0; i < 3; i++) {
            stick.increment(2);
        }
        for (int i = 0; i < dim; i++) {
            System.out.println(i + " " + stick.getLogProbability(i) + ". " + Math.exp(stick.getLogProbability(i)));
        }
        System.out.println(MiscUtils.arrayToString(stick.getCounts()));
        System.out.println(MiscUtils.arrayToString(stick.backwardCounts));
        System.out.println(MiscUtils.arrayToString(stick.getDistribution()));
        System.out.println(stick.getLogLikelihood());
        
        int[] obs = {1, 2, 3};
        System.out.println(Math.exp(stick.getLogProbability(obs)));
        
        stick = new TruncatedStickBreaking(dim, mean, scale);
        for (int i = 0; i < 2; i++) {
            stick.increment(0);
        }
        for (int i = 0; i < 1; i++) {
            stick.increment(1);
        }
        for (int i = 0; i < 0; i++) {
            stick.increment(2);
        }
        
        System.out.println(MiscUtils.arrayToString(stick.getCounts()));
        System.out.println(MiscUtils.arrayToString(stick.backwardCounts));
        int[] counts = {2, 1, 0};
        for(int i=0; i<counts.length; i++) {
            stick.changeCount(i, -counts[i]);
        }
        System.out.println(MiscUtils.arrayToString(stick.getCounts()));
        System.out.println(MiscUtils.arrayToString(stick.backwardCounts));
    }
}
