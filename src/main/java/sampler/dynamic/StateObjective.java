package sampler.dynamic;

import cc.mallet.optimize.Optimizable;
import sampling.util.SparseCount;

/**
 *
 * @author vietan
 */
public class StateObjective implements Optimizable.ByGradientValue {

    private double[] preMean; // mean of previous state
    private double[] preVariance; // updated diagonal covariance matrix of previous state (\Sigma_{t-1} + \sigma^2 I)
    private SparseCount counts; // observations
    private double[] parameters; // [V x 1]

    public StateObjective(double[] pm, double[] pv, SparseCount obs) {
        this.preMean = pm;
        this.preVariance = pv;
        this.counts = obs;

        // initialize with the mean of the previous state
        this.parameters = new double[preMean.length];
        System.arraycopy(preMean, 0, parameters, 0, preMean.length);
    }

    @Override
    public double getValue() {
        double value = 0.0;

        for (int obs : counts.getIndices()) {
            value += counts.getCount(obs) * parameters[obs];
        }

        double sumExp = 0.0;
        for (int i = 0; i < getNumParameters(); i++) {
            sumExp += Math.exp(parameters[i]);
        }
        value -= counts.getCountSum() * Math.log(sumExp);

        double logprior = 0.0;
        for (int i = 0; i < getNumParameters(); i++) {
            logprior += parameters[i] * parameters[i] / preVariance[i];
        }
        value -= 0.5 * logprior;

        return value;
    }

    @Override
    public void getValueGradient(double[] gradient) {
        double sumExp = 0;
        for (int i = 0; i < getNumParameters(); i++) {
            sumExp += Math.exp(parameters[i]);
        }

        for (int i = 0; i < getNumParameters(); i++) {
            double grad = counts.getCount(i);
            grad -= counts.getCountSum() * Math.exp(parameters[i]) / sumExp;
            grad -= (parameters[i] - preMean[i]) / preVariance[i];
            gradient[i] = grad;
        }
    }

    @Override
    public int getNumParameters() {
        return this.parameters.length;
    }

    @Override
    public double getParameter(int i) {
        return parameters[i];
    }

    @Override
    public void getParameters(double[] buffer) {
        assert (buffer.length == parameters.length);
        System.arraycopy(parameters, 0, buffer, 0, buffer.length);
    }

    @Override
    public void setParameter(int i, double r) {
        this.parameters[i] = r;
    }

    @Override
    public void setParameters(double[] newParameters) {
        assert (newParameters.length == parameters.length);
        System.arraycopy(newParameters, 0, parameters, 0, parameters.length);
    }
}
