package sampler.dynamic;

import sampling.util.SparseCount;

/**
 * Implement the state for linear state-space model. Each state is associated
 * with a mean vector, a diagonal variance matrix and a set of observations.
 *
 * The mean vector and the variance matrix can represent different things,
 * depending on the stage of the forward-backward algorithm.
 *
 * Initially, the mean and variance of state t will be set to those of state
 * (t-1)
 *
 * After forward filtering, the mean and variance of state t is the parameters
 * for the Gaussian capturing the probability of state t given observation 1...t
 * P(s_t | x_1, x_2, ..., x_t)
 *
 * After backward smoothing, the mean and variance of state t is the parameters
 * for the Gaussian capturing the probability of state t given all observations
 * P(s_t | x_1, x_2, ..., x_T)
 *
 * @author vietan
 */
public class State {

    private double[] mean;
    private double[] variance;  // diagonal matrix
    private SparseCount observations;

    public State(int dim) {
        this.mean = new double[dim];
        this.variance = new double[dim];

        this.observations = new SparseCount();
    }

    public State(double[] m, double[] v) {
        this.mean = m;
        this.variance = v;

        this.observations = new SparseCount();
    }

    public double[] getLogisticNormalDistribution() {
        double[] dist = new double[getDimension()];
        double sum = 0.0;
        for (int i = 0; i < getDimension(); i++) {
            dist[i] = Math.exp(this.mean[i]);
            sum += dist[i];
        }
        for (int i = 0; i < getDimension(); i++) {
            dist[i] /= sum;
        }
        return dist;
    }

    public int getDimension() {
        return this.mean.length;
    }

    public double getMean(int index) {
        return this.mean[index];
    }

    public double getVariance(int index) {
        return this.variance[index];
    }

    public void setMean(int index, double m) {
        this.mean[index] = m;
    }

    public void setVariance(int index, double v) {
        this.variance[index] = v;
    }

    public SparseCount getObservations() {
        return this.observations;
    }

    public int getCountSum() {
        return this.observations.getCountSum();
    }

    public int getCount(int obs) {
        return this.observations.getCount(obs);
    }

    public void decrement(int obs) {
        this.observations.decrement(obs);
    }

    public void increment(int obs) {
        this.observations.increment(obs);
    }

    public double[] getMean() {
        return mean;
    }

    public void setMean(double[] mean) {
        this.mean = mean;
    }

    public double[] getVariance() {
        return variance;
    }

    public void setVariance(double[] variance) {
        this.variance = variance;
    }
}
