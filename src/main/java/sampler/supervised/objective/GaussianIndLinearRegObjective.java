package sampler.supervised.objective;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import java.util.Random;
import util.MiscUtils;
import util.SamplerUtils;
import util.SparseVector;

/**
 * Bayesian version of IndependentLinearRegressionObject where the parameters
 * are drawn from a Gaussian prior
 *
 * @author vietan
 */
public class GaussianIndLinearRegObjective implements Optimizable.ByGradientValue {

    private double[][] designMatrix; // [D x N] - matrix Z
    private double[] responses; // [D x 1] - vector y
    // variance of the Gaussian generating the observation
    private double rhoSquare;
    // hyperparameters for the Guassian prior
    private double[] mus; // mean
    private double[] sigmaSquares; // variance
    private double[] parameters; // [N x 1]
    // precompute unchanged values
    private double[] ZTy;
    private double[][] ZTZ;
    private int N;
    private int D;

    public GaussianIndLinearRegObjective(
            double[] curParams,
            double[][] designMatrix, // the design matrix [D x N]
            double[] responses, // response vector [D]
            double rho,
            double mu,
            double sigma) {
        this.designMatrix = designMatrix;
        this.responses = responses;

        this.rhoSquare = rho * rho;
        this.mus = new double[curParams.length];
        this.sigmaSquares = new double[curParams.length];
        for (int i = 0; i < curParams.length; i++) {
            this.mus[i] = mu;
            this.sigmaSquares[i] = sigma * sigma;
        }

        this.N = curParams.length;
        this.D = this.responses.length;

        this.parameters = new double[N];
        System.arraycopy(curParams, 0, this.parameters, 0, N);

        if (this.N != this.designMatrix[0].length) {
            throw new RuntimeException("Dimension mismatched"
                    + ". # parameters = " + N
                    + ". Size of design matrix = " + designMatrix[0].length);
        }

        // precompute Z^T y
        this.ZTy = new double[N];
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                ZTy[n] += designMatrix[d][n] * responses[d];
            }
        }

        // precompute Z^T Z
        this.ZTZ = new double[N][N];
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < N; m++) {
                for (int d = 0; d < D; d++) {
                    this.ZTZ[n][m] += designMatrix[d][n] * designMatrix[d][m];
                }
            }
        }
    }

    public GaussianIndLinearRegObjective(
            double[] curParams,
            double[][] designMatrix, // the design matrix [D x N]
            double[] responses, // response vector [D]
            double rho,
            double[] mus, double[] sigmas) {
        this.designMatrix = designMatrix;
        this.responses = responses;

        this.mus = mus;
        this.rhoSquare = rho * rho;
        this.sigmaSquares = new double[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            this.sigmaSquares[i] = sigmas[i] * sigmas[i];
        }

        this.N = curParams.length;
        this.D = this.responses.length;

        this.parameters = new double[N];
        System.arraycopy(curParams, 0, this.parameters, 0, N);

        if (this.N != this.designMatrix[0].length) {
            throw new RuntimeException("Dimension mismatched"
                    + ". # parameters = " + N
                    + ". Size of design matrix = " + designMatrix[0].length);
        }

        // precompute Z^T y
        this.ZTy = new double[N];
        for (int n = 0; n < N; n++) {
            for (int d = 0; d < D; d++) {
                ZTy[n] += designMatrix[d][n] * responses[d];
            }
        }

        // precompute Z^T Z
        this.ZTZ = new double[N][N];
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < N; m++) {
                for (int d = 0; d < D; d++) {
                    this.ZTZ[n][m] += designMatrix[d][n] * designMatrix[d][m];
                }
            }
        }
    }

    @Override
    public double getValue() {
        // observation log likelihood
        double llh = 0.0;
        for (int d = 0; d < responses.length; d++) {
            double obsMean = 0.0; // dot product
            for (int n = 0; n < designMatrix[d].length; n++) {
                obsMean += designMatrix[d][n] * parameters[n];
            }
            double diff = responses[d] - obsMean;
            llh += diff * diff;
        }
        llh /= (- 2 * rhoSquare);

        // log prior
        double lprior = 0.0;
        for (int n = 0; n < this.getNumParameters(); n++) {
            double diff = parameters[n] - mus[n];
            lprior += diff * diff / (- 2 * sigmaSquares[n]);
        }

        return llh + lprior;
    }

    @Override
    public void getValueGradient(double[] gradient) {
        // gradiend of the log likelihood term
        double[] llhGrad = new double[N];
        for (int n = 0; n < N; n++) {
            llhGrad[n] = ZTy[n];
            for (int m = 0; m < N; m++) {
                llhGrad[n] -= ZTZ[n][m] * parameters[m];
            }
            llhGrad[n] /= (rhoSquare);
        }

        // gradient of the log prior
        double[] lprior = new double[N];
        for (int n = 0; n < N; n++) {
            lprior[n] = -(parameters[n] - mus[n]) / sigmaSquares[n];
        }

        for (int n = 0; n < N; n++) {
            gradient[n] = llhGrad[n] + lprior[n];
        }
    }

    @Override
    public int getNumParameters() {
        return this.N;
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

    public static void main(String[] args) {
        Random rand = new Random(11235813);
        double sigma = 1.0;
        double rho = 100;
        double mean = 0.0;

        int D = 10000;
        int V = 5;

        double[] trueParams = new double[V];
        for (int v = 0; v < V; v++) {
            trueParams[v] = SamplerUtils.getGaussian(mean, sigma);
        }

        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                double w = rand.nextFloat();
                designMatrix[d][v] = w;
            }
        }

        // generate response
        double[] responseVector = new double[D];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                responseVector[d] += designMatrix[d][v] * trueParams[v];
            }
        }
        for (int d = 0; d < D; d++) {
            responseVector[d] = SamplerUtils.getGaussian(responseVector[d], rho);
        }

        double[] initParams = new double[V];
        for (int v = 0; v < V; v++) {
            initParams[v] = SamplerUtils.getGaussian(mean, sigma);
        }
        System.out.println("I params: " + MiscUtils.arrayToString(initParams));

        GaussianIndLinearRegObjective obj = 
                new GaussianIndLinearRegObjective(initParams, 
                        designMatrix, responseVector, rho, mean, sigma);

        LimitedMemoryBFGS opt = new LimitedMemoryBFGS(obj);
        boolean converged = false;
        try {
            converged = opt.optimize();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("converged: " + converged);

        System.out.println("T params: " + MiscUtils.arrayToString(trueParams));

        double[] ps = new double[V];
        for (int v = 0; v < V; v++) {
            ps[v] = obj.getParameter(v);
        }
        System.out.println("L params: " + MiscUtils.arrayToString(ps));
    }
}
