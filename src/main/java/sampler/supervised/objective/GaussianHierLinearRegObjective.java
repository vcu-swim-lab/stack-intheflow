/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.supervised.objective;

import cc.mallet.optimize.Optimizable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author vietan
 */
public class GaussianHierLinearRegObjective implements Optimizable.ByGradientValue {

    private double[][] designMatrix; // [D x N] - matrix Z
    private double[] responses; // [D x 1] - vector y
    // variance of the Gaussian generating the observation
    private double rhoSquare;
    // hyperparameters for the Guassian prior
    private double mu; // mean
    private double[] sigmaSquares; // variance
    private double[] parameters; // [N x 1]
    // precompute unchanged values
    private double[] ZTy;
    private double[][] ZTZ;
    private int N;
    private int D;
    private HashMap<Integer, Integer> upwardLink;
    private HashMap<Integer, ArrayList<Integer>> downwardLinks;

    public GaussianHierLinearRegObjective(
            double[] curParams,
            double[][] designMatrix, // the design matrix [D x N]
            double[] responses, // response vector [D]
            double rho,
            double mu, double sigma,
            HashMap<Integer, Integer> upwardLink,
            HashMap<Integer, ArrayList<Integer>> downwardLinks) {
        this.designMatrix = designMatrix;
        this.responses = responses;
        this.upwardLink = upwardLink;
        this.downwardLinks = downwardLinks;

        this.rhoSquare = rho * rho;
        this.mu = mu;
        this.sigmaSquares = new double[curParams.length];
        for (int i = 0; i < curParams.length; i++) {
            this.sigmaSquares[i] = sigma * sigma;
        }

        this.N = curParams.length;
        this.D = this.responses.length;

        this.parameters = new double[N];
        for (int i = 0; i < N; i++) {
            this.parameters[i] = curParams[i];
        }

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

    public GaussianHierLinearRegObjective(
            double[] curParams,
            double[][] designMatrix, // the design matrix [D x N]
            double[] responses, // response vector [D]
            double rho,
            double mu, double[] sigmas,
            HashMap<Integer, Integer> upwardLink,
            HashMap<Integer, ArrayList<Integer>> downwardLinks) {
        this.designMatrix = designMatrix;
        this.responses = responses;
        this.upwardLink = upwardLink;
        this.downwardLinks = downwardLinks;

        this.mu = mu;
        this.rhoSquare = rho * rho;
        this.sigmaSquares = new double[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            this.sigmaSquares[i] = sigmas[i] * sigmas[i];
        }

        this.N = curParams.length;
        this.D = this.responses.length;

        this.parameters = new double[N];
        for (int i = 0; i < N; i++) {
            this.parameters[i] = curParams[i];
        }

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
            double diff;
            int parentIndex = upwardLink.get(n);
            if (parentIndex == -1) {
                diff = parameters[n] - mu;
            } else {
                diff = parameters[n] - parameters[parentIndex];
            }
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
            // uplink
            double uplinkPrior = 0.0;
            int parentIndex = upwardLink.get(n);
            if (parentIndex == -1) {
                uplinkPrior += parameters[n] - mu;
            } else {
                uplinkPrior += parameters[n] - parameters[parentIndex];
            }
            uplinkPrior /= (-sigmaSquares[n]);

            // downlinks
            double downlinkPrior = 0.0;
            for (int child : downwardLinks.get(n)) {
                downlinkPrior += (parameters[n] - parameters[child]) / (-sigmaSquares[child]);
            }

            lprior[n] = uplinkPrior + downlinkPrior;
        }
//        for(int n=0; n<N; n++)
//            lprior[n] = - (parameters[n] - mus[n]) / sigmaSquares[n];

        // combine
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
}
