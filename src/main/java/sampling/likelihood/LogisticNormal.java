package sampling.likelihood;

import java.util.ArrayList;
import java.util.HashMap;
import sampling.AbstractDiscreteFiniteLikelihoodModel;

/**
 *
 * @author vietan
 */
public class LogisticNormal extends AbstractDiscreteFiniteLikelihoodModel {

    public static final int Q = 10;
    private double[] priorMean;
    private double[] priorVariance;
    private double[] mean;
    private double[] variance;
    private double[] distribution;
    private ArrayList<double[]> auxDistributions;

    public LogisticNormal(int dim, double[] pm, double[] pv) {
        super(dim);
        this.priorMean = pm;
        this.priorVariance = pv;

        this.mean = new double[this.dimension];
        this.variance = new double[this.dimension];
        this.distribution = null;
    }

    @Override
    public void sampleFromPrior() {
        for (int i = 0; i < dimension; i++) {
            this.mean[i] = rand.nextGaussian() * priorVariance[i] + priorMean[i];
        }

        this.distribution = new double[this.dimension];
        this.updateDistribution();
    }

    @Override
    public double[] getDistribution() {
        return distribution;
    }

    public void updateDistribution() {
        this.distribution = new double[dimension];
        double sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            this.distribution[i] = Math.exp(mean[i]);
            sum += distribution[i];
        }
        for (int i = 0; i < dimension; i++) {
            distribution[i] /= sum;
        }

        // clear the auxiliary distributions
        this.auxDistributions = null;
    }

    /**
     * Get the log probability of a given observation. If this is initialized,
     * this method simply returns the value.
     *
     * On the other hand, if this is a new component, Q distributions are drawn
     * from the prior and the log likelihood is averaged over the Q values.
     *
     * @param obs The observation
     */
    @Override
    public double getLogLikelihood(int obs) {
        double llh = 0.0;
        if (this.distribution != null) {
            llh = Math.log(this.distribution[obs]);
        } else {
            if (this.auxDistributions == null) {
                this.auxDistributions = new ArrayList<double[]>();
                for (int q = 0; q < Q; q++) {
                    double[] auxDist = new double[dimension];
                    double sumExp = 0.0;
                    for (int v = 0; v < dimension; v++) {
                        double r = rand.nextGaussian();

                        auxDist[v] = Math.exp(r * priorVariance[v] + priorMean[v]);
                        sumExp += auxDist[v];
                    }

                    for (int v = 0; v < dimension; v++) {
                        auxDist[v] /= sumExp;
                    }
                    this.auxDistributions.add(auxDist);
                }
            }

            for (int q = 0; q < Q; q++) {
                llh += Math.log(this.auxDistributions.get(q)[obs]);
            }
            llh /= Q;
        }
        return llh;
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;
        for (int obs : this.getUniqueObservations()) {
            llh += this.getCount(obs) * getLogLikelihood(obs);
        }
        return llh;
    }

    public double getLogLikelihood(HashMap<Integer, Integer> obsCounts) {
        double llh = 0.0;
        for (int obs : obsCounts.keySet()) {
            llh += obsCounts.get(obs) * getLogLikelihood(obs);
        }
        return llh;
    }

    @Override
    public String getModelName() {
        return "Logistic-Normal";
    }

    public double[] getVariance() {
        return this.variance;
    }

    public double[] getMean() {
        return this.mean;
    }

    public void setMean(int index, double value) {
        this.mean[index] = value;
    }

    public void setVariance(int index, double value) {
        this.variance[index] = value;
    }

    public double getMean(int index) {
        return this.mean[index];
    }

    public double getVariance(int index) {
        return this.variance[index];
    }

    public double getPriorMean(int index) {
        return this.priorMean[index];
    }

    public double getPriorVariance(int index) {
        return this.priorVariance[index];
    }
}
