package optimization;

import cc.mallet.optimize.Optimizable;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class RidgeLogisticRegressionOptimizable implements Optimizable.ByGradientValue {

    private final int[] labels;                 // [N]-dim vector
    private final double[] params;              // [K]-dim vector
    private final SparseVector[] designMatrix;  // [N]x[K] sparse matrix
    private final int N; // number of instances
    private final int K; // number of features
    private final double paramMean;
    private final double paramVar;
    private final double[] paramVars;

    public RidgeLogisticRegressionOptimizable(int[] labels,
            double[] params,
            SparseVector[] designMatrix,
            double mean,
            double var) {
        this.labels = labels;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        this.paramMean = mean;
        this.paramVar = var;
        this.paramVars = null;
    }
    
    public RidgeLogisticRegressionOptimizable(int[] labels,
            double[] params,
            SparseVector[] designMatrix,
            double mean,
            double[] vars) {
        this.labels = labels;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        this.paramMean = mean;
        this.paramVar = 0.0;
        this.paramVars = vars;
    }

    public double getMean(int k) {
        return this.paramMean;
    }

    public double getVariance(int k) {
        if (paramVars == null) {
            return this.paramVar;
        }
        return this.paramVars[k];
    }

    @Override
    public double getValue() {
        double llh = 0.0;
        for (int nn = 0; nn < N; nn++) {
            double dotProb = designMatrix[nn].dotProduct(params);
            llh += labels[nn] * dotProb - Math.log(Math.exp(dotProb) + 1);
        }

        double lprior = 0.0;
        for (int kk = 0; kk < K; kk++) {
            double diff = params[kk] - getMean(kk);
            lprior -= 0.5 * diff * diff / getVariance(kk);
        }
        return (llh + lprior);
    }

    @Override
    public void getValueGradient(double[] gradient) {
        double[] llhGrad = new double[K];
        for (int nn = 0; nn < N; nn++) {
            double dotprod = designMatrix[nn].dotProduct(params);
            double expDotprod = Math.exp(dotprod);
            double pred = expDotprod / (expDotprod + 1);
            for (int kk = 0; kk < K; kk++) {
                llhGrad[kk] += (labels[nn] - pred) * designMatrix[nn].get(kk);
            }
        }

        double[] lpGrad = new double[K];
        for (int kk = 0; kk < K; kk++) {
            lpGrad[kk] -= (params[kk] - getMean(kk)) / (getVariance(kk));
        }

        for (int k = 0; k < K; k++) {
            gradient[k] = (llhGrad[k] + lpGrad[k]);
        }
    }
    
    @Override
    public int getNumParameters() {
        return this.K;
    }

    @Override
    public double getParameter(int i) {
        return params[i];
    }

    @Override
    public void getParameters(double[] buffer) {
        assert (buffer.length == params.length);
        System.arraycopy(params, 0, buffer, 0, buffer.length);
    }

    @Override
    public void setParameter(int i, double r) {
        this.params[i] = r;
    }

    @Override
    public void setParameters(double[] newParameters) {
        assert (newParameters.length == params.length);
        System.arraycopy(newParameters, 0, params, 0, params.length);
    }
}
