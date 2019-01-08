package sampler.dynamic;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import java.util.Arrays;
import java.util.Random;
import util.MiscUtils;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class StateSpaceModel {

    static double[][] trueDist;
    private int dimension;
    private double sigma;
    private int[][] observations;
    private State[] states;
    // internal variables
    private int T;
    private double sigmaSquare;

    public StateSpaceModel(int dim, double sigma, int[][] obs) {
        this.dimension = dim;
        this.sigma = sigma;
        this.observations = obs;

        this.T = this.observations.length;
        this.sigmaSquare = this.sigma * this.sigma;

        this.states = new State[this.dimension];
        for (int t = 0; t < T; t++) {
            this.states[t] = new State(dimension);
            for (int o : observations[t]) {
                this.states[t].increment(o);
            }
        }
    }

    public void forward() {
        // initialize first state
        double[] mean0 = new double[dimension];
        Arrays.fill(mean0, 0.0);
        double[] var0 = new double[dimension];
        Arrays.fill(var0, sigma);

        double[] preMean = mean0;
        double[] preVar = var0;

        int numConverged = 0;
        for (int t = 0; t < this.T; t++) {
            StateObjective objective = new StateObjective(preMean, preVar, states[t].getObservations());
            Optimizer optimizer = new LimitedMemoryBFGS(objective);
            boolean converged = false;
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                // This exception may be thrown if L-BFGS
                //  cannot step in the current direction.
                // This condition does not necessarily mean that
                //  the optimizer has failed, but it doesn't want
                //  to claim to have succeeded... 
                // do nothing
            }

            if (converged) {
                numConverged++;
            }

            for (int i = 0; i < dimension; i++) {
                states[t].setMean(i, objective.getParameter(i));
            }

            // compute diagonal approximation of the Hessian
            double[] exps = new double[dimension];
            double sumExp = 0.0;
            for (int i = 0; i < dimension; i++) {
                exps[i] = Math.exp(states[t].getMean(i));
                sumExp += exps[i];
            }

            for (int i = 0; i < dimension; i++) {
                double prob = exps[i] / sumExp;
                double negHess =
                        1.0 / preVar[i]
                        + states[t].getCountSum() * prob * (1 - prob);
                states[t].setVariance(i, 1.0 / negHess);
            }

            // update 
            for (int i = 0; i < dimension; i++) {
                preMean[i] = states[t].getMean(i);
                preVar[t] = states[t].getVariance(i) + this.sigmaSquare;
            }

            System.out.println("State " + t + ". " + converged);
            System.out.println("Mean:\t" + MiscUtils.arrayToString(states[t].getMean()));
            System.out.println("Var:\t" + MiscUtils.arrayToString(states[t].getVariance()));
            System.out.println("Dist:\t" + MiscUtils.arrayToString(states[t].getLogisticNormalDistribution()));
            System.out.println("True:\t" + MiscUtils.arrayToString(trueDist[t]));

            System.out.println("Obs:\t" + MiscUtils.arrayToString(observations[t]));
            System.out.println();
        }
        System.out.println("# converged = " + numConverged);
    }

    /**
     * Perform backward smoothing
     */
    public void backward() {
        double[] posMean = states[T - 1].getMean();
        double[] posVar = states[T - 1].getVariance();

        for (int t = T - 2; t >= 0; t--) {
            double[] curMean = states[t].getMean();
            double[] curVar = states[t].getVariance();
            double[] tempK = new double[dimension];
            for (int v = 0; v < dimension; v++) {
                tempK[v] = curVar[v] / (curVar[v] + this.sigmaSquare);
            }

            // debug
            System.out.println("Backward t = " + t);
//            System.out.println("cur mean:\t" + MiscUtils.arrayToString(curMean));
//            System.out.println("cur var: \t" + MiscUtils.arrayToString(curVar));
            System.out.println("cur dist:\t" + MiscUtils.arrayToString(states[t].getLogisticNormalDistribution()));

            for (int v = 0; v < dimension; v++) {
                double newMean = curMean[v] + tempK[v] * (posMean[v] - curMean[v]);
                states[t].setMean(v, newMean);
                double newVar = curVar[v] + tempK[v] * posVar[v] * tempK[v] - tempK[v] * curVar[v];
                states[t].setVariance(v, newVar);
            }

            // debug
//            System.out.println("new mean:\t" + MiscUtils.arrayToString(states[t].getMean()));
//            System.out.println("new var: \t" + MiscUtils.arrayToString(states[t].getVariance()));
            System.out.println("new dist:\t" + MiscUtils.arrayToString(states[t].getLogisticNormalDistribution()));
            System.out.println("true dist:\t" + MiscUtils.arrayToString(trueDist[t]));
            System.out.println();
        }
    }

    public static void main(String[] args) {
        try {
            System.err.close();

            testForward();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testForward() throws Exception {
        int T = 10;
        int dim = 10;
        int N = 1000;
        double sigma = 1.0;

        Random rand = new Random(1234);

        double[] preMean = new double[dim];
        Arrays.fill(preMean, 0.0);

        // debug
        StateSpaceModel.trueDist = new double[T][];

        int[][] observations = new int[T][N];
        for (int t = 0; t < T; t++) {
            System.out.println("t = " + t);

            double[] mean = new double[dim];
            for (int i = 0; i < dim; i++) {
                mean[i] = preMean[i] + rand.nextGaussian() * sigma;
            }

            System.out.println("Mean\t" + MiscUtils.arrayToString(mean));

            double[] logisticNorm = new double[mean.length];
            double sum = 0.0;
            for (int i = 0; i < dim; i++) {
                logisticNorm[i] = Math.exp(mean[i]);
                sum += logisticNorm[i];
            }
            for (int i = 0; i < dim; i++) {
                logisticNorm[i] /= sum;
            }

            System.out.println("LogNorm\t" + MiscUtils.arrayToString(logisticNorm));
            StateSpaceModel.trueDist[t] = logisticNorm;

            for (int n = 0; n < N; n++) {
                observations[t][n] = SamplerUtils.scaleSample(logisticNorm);
            }

            System.out.println("Obs\t" + MiscUtils.arrayToString(observations[t]));
            System.out.println();

            preMean = mean;
        }

        StateSpaceModel model = new StateSpaceModel(dim, sigma, observations);
        model.forward();
        model.backward();
    }
}
