package optimization;

import cc.mallet.optimize.Optimizable;
import java.util.ArrayList;
import util.SparseVector;

/**
 * Solving L2-norm multiple linear regression using L-BFGS.
 *
 * @author vietan
 */
public class RidgeLinearRegressionOptimizable implements Optimizable.ByGradientValue {

    // inputs
    private final double[] values;              // [N]-dim vector
    private final double[] params;              // [K]-dim vector
    private final SparseVector[] designMatrix;  // [N]x[K] sparse matrix
    private final int N; // number of instances
    private final int K; // number of features
    private final double[] rhoSquares;
    private final double mu;
    private final double sigmaSquare;
    private final double[] sigmaSquares;

    public RidgeLinearRegressionOptimizable(double[] values,
            double[] params,
            SparseVector[] designMatrix,
            double rho,
            double mu,
            double sigma) {
        this.values = values;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        double rhoSquare = rho * rho;
        this.rhoSquares = new double[N];
        for (int nn = 0; nn < N; nn++) {
            this.rhoSquares[nn] = rhoSquare;
        }
        this.mu = mu;
        this.sigmaSquare = sigma * sigma;
        this.sigmaSquares = null;
    }
    
    public RidgeLinearRegressionOptimizable(double[] values,
            double[] params,
            SparseVector[] designMatrix,
            double[] rhos,
            double mu,
            double sigma) {
        this.values = values;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        this.rhoSquares = new double[N];
        for(int nn=0; nn<N; nn++) {
            this.rhoSquares[nn] = rhos[nn] * rhos[nn];
        }
        this.mu = mu;
        this.sigmaSquare = sigma * sigma;
        this.sigmaSquares = null;
    }

    public RidgeLinearRegressionOptimizable(double[] values,
            double[] params,
            SparseVector[] designMatrix,
            double[] rhos,
            double mu,
            double[] sigmas) {
        this.values = values;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        this.rhoSquares = new double[N];
        for (int nn = 0; nn < N; nn++) {
            this.rhoSquares[nn] = rhos[nn] * rhos[nn];
        }
        this.mu = mu;
        this.sigmaSquare = -1; // dummy value
        this.sigmaSquares = new double[sigmas.length];
        for (int ii = 0; ii < this.sigmaSquares.length; ii++) {
            this.sigmaSquares[ii] = sigmas[ii] * sigmas[ii];
        }
    }
    
    public RidgeLinearRegressionOptimizable(double[] values,
            double[] params,
            SparseVector[] designMatrix,
            double rho,
            double mu,
            double[] sigmas) {
        this.values = values;
        this.params = params;
        this.designMatrix = designMatrix;
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        this.rhoSquares = new double[N];
        for (int nn = 0; nn < N; nn++) {
            this.rhoSquares[nn] = rho * rho;
        }
        this.mu = mu;
        this.sigmaSquare = -1; // dummy value
        this.sigmaSquares = new double[sigmas.length];
        for (int ii = 0; ii < this.sigmaSquares.length; ii++) {
            this.sigmaSquares[ii] = sigmas[ii] * sigmas[ii];
        }
    }

    public RidgeLinearRegressionOptimizable(ArrayList<Double> values,
            double[] params,
            ArrayList<SparseVector> designMatrix,
            double rho,
            double mu,
            double[] sigmas) {
        this.params = params;
        this.values = new double[values.size()];
        this.designMatrix = new SparseVector[designMatrix.size()];
        this.N = this.designMatrix.length;
        this.K = this.params.length;

        for (int n = 0; n < N; n++) {
            this.values[n] = values.get(n);
            this.designMatrix[n] = designMatrix.get(n);
        }

        this.rhoSquares = new double[N];
        for(int nn=0; nn<N; nn++) {
            this.rhoSquares[nn] = rho * rho;
        }
        this.mu = mu;
        this.sigmaSquare = -1; // dummy value
        this.sigmaSquares = new double[sigmas.length];
        for (int ii = 0; ii < this.sigmaSquares.length; ii++) {
            this.sigmaSquares[ii] = sigmas[ii] * sigmas[ii];
        }
    }

    public double getMu(int k) {
        return this.mu;
    }

    public double getSigmaSquare(int k) {
        if (this.sigmaSquares == null) {
            return this.sigmaSquare;
        }
        return this.sigmaSquares[k];
    }

    @Override
    public double getValue() {
        double llh = 0.0;
        for (int n = 0; n < N; n++) {
            double dotprod = dotprod(n);
            double diff = values[n] - dotprod;
            llh += diff * diff / rhoSquares[n];
        }
        llh /= (-2 * N);

        double lprior = 0.0;
        for (int k = 0; k < K; k++) {
            double diff = params[k] - getMu(k);
            lprior += diff * diff / (-2 * getSigmaSquare(k));
        }
        lprior /= N;
        return llh + lprior;
    }

    @Override
    public void getValueGradient(double[] gradient) {
        double[] llhGrad = new double[K];
        for (int n = 0; n < N; n++) {
            double dotprod = dotprod(n);
            for (int k : this.designMatrix[n].getIndices()) {
                double grad = (values[n] - dotprod) * designMatrix[n].get(k)
                        / (rhoSquares[n] * N);
                llhGrad[k] += grad;
            }
        }

        double[] lpGrad = new double[K];
        for (int k = 0; k < K; k++) {
            lpGrad[k] -= (params[k] - getMu(k)) / (N * getSigmaSquare(k));
        }

        for (int k = 0; k < K; k++) {
            gradient[k] = llhGrad[k] + lpGrad[k];
        }
    }

    private double dotprod(int n) {
        double dotprod = 0.0;
        for (int k : designMatrix[n].getIndices()) {
            dotprod += params[k] * designMatrix[n].get(k);
        }
        return dotprod;
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
