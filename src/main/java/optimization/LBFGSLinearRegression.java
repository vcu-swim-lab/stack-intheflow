package optimization;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractLinearModel;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;
import util.MiscUtils;
import util.SparseVector;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class LBFGSLinearRegression extends AbstractLinearModel {

    private final double mu;
    private final double sigma;
    private final double rho;

    public LBFGSLinearRegression(String basename,
            double mu, double sigma, double rho) {
        super(basename);
        this.mu = mu;
        this.sigma = sigma;
        this.rho = rho;
    }

    @Override
    public String getName() {
        return this.name
                + "_r-" + MiscUtils.formatDouble(rho)
                + "_m-" + MiscUtils.formatDouble(mu)
                + "_s-" + MiscUtils.formatDouble(sigma);
    }

    public double getMu() {
        return this.mu;
    }

    public double getSigma() {
        return this.sigma;
    }

    public void trainAdaptive(int[][] docWords, ArrayList<Integer> docIndices,
            double[] docResponses, int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        double[] responses = new double[D];
        double[] rhos = new double[D];
        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            responses[ii] = docResponses[dd];
            designMatrix[ii] = new SparseVector(V);
            double val = 1.0 / docWords[dd].length;
            for (int nn = 0; nn < docWords[dd].length; nn++) {
                designMatrix[ii].change(docWords[dd][nn], val);
            }
            rhos[ii] = this.rho / docWords[dd].length;
        }

        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension() + ". " + V);
        }
        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, new double[V], designMatrix, rhos, mu, sigma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

        try {
            optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        this.weights = new double[V];
        for (int kk = 0; kk < V; kk++) {
            this.weights[kk] = optimizable.getParameter(kk);
        }
    }

    public void train(int[][] docWords, ArrayList<Integer> docIndices,
            double[] docResponses, int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        double[] responses = new double[D];
        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            responses[ii] = docResponses[dd];
            designMatrix[ii] = new SparseVector(V);
            double val = 1.0 / docWords[dd].length;
            for (int nn = 0; nn < docWords[dd].length; nn++) {
                designMatrix[ii].change(docWords[dd][nn], val);
            }
        }
        train(designMatrix, responses, V);
    }

    public void train(SparseVector[] designMatrix, double[] responses, int K) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension() + ". " + K);
        }
        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, new double[K], designMatrix, rho, mu, sigma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

        try {
            optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        this.weights = new double[K];
        for (int kk = 0; kk < K; kk++) {
            this.weights[kk] = optimizable.getParameter(kk);
        }
    }

    public void train(SparseVector[] designMatrix, double[] responses, double[] initParams) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, initParams, designMatrix, rho, mu, sigma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

        try {
            optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        this.weights = new double[initParams.length];
        for (int kk = 0; kk < initParams.length; kk++) {
            this.weights[kk] = optimizable.getParameter(kk);
        }
    }

    public double[] test(int[][] docWords, ArrayList<Integer> docIndices, int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            designMatrix[ii] = new SparseVector(V);
            double val = 1.0 / docWords[dd].length;
            for (int nn = 0; nn < docWords[dd].length; nn++) {
                designMatrix[ii].change(docWords[dd][nn], val);
            }
        }

        return test(designMatrix);
    }

    public double[] test(SparseVector[] designMatrix) {
        if (verbose) {
            System.out.println("Testing ...");
            System.out.println("--- # instances: " + designMatrix.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        double[] predictions = new double[designMatrix.length];
        for (int d = 0; d < predictions.length; d++) {
            predictions[d] = designMatrix[d].dotProduct(weights);
        }
        return predictions;
    }

    public void tune(File outputFile,
            int[][] trDocWords,
            ArrayList<Integer> trDocIndices,
            double[] trDocResponses,
            int[][] deDocWords,
            ArrayList<Integer> deDocIndices,
            double[] deDocResponses,
            int K) {
        if (trDocIndices == null) {
            trDocIndices = new ArrayList<>();
            for (int dd = 0; dd < trDocWords.length; dd++) {
                trDocIndices.add(dd);
            }
        }
        int Dtr = trDocIndices.size();
        double[] trResponses = new double[Dtr];
        SparseVector[] trDesignMatrix = new SparseVector[Dtr];
        for (int ii = 0; ii < Dtr; ii++) {
            int dd = trDocIndices.get(ii);
            trResponses[ii] = trDocResponses[dd];
            trDesignMatrix[ii] = new SparseVector(K);
            double val = 1.0 / trDocWords[dd].length;
            for (int nn = 0; nn < trDocWords[dd].length; nn++) {
                trDesignMatrix[ii].change(trDocWords[dd][nn], val);
            }
        }

        if (deDocIndices == null) {
            deDocIndices = new ArrayList<>();
            for (int dd = 0; dd < deDocWords.length; dd++) {
                deDocIndices.add(dd);
            }
        }
        int Dde = deDocIndices.size();
        double[] deResponses = new double[Dde];
        SparseVector[] deDesignMatrix = new SparseVector[Dde];
        for (int ii = 0; ii < Dde; ii++) {
            int dd = deDocIndices.get(ii);
            deResponses[ii] = deDocResponses[dd];
            deDesignMatrix[ii] = new SparseVector(K);
            double val = 1.0 / deDocWords[dd].length;
            for (int nn = 0; nn < deDocWords[dd].length; nn++) {
                deDesignMatrix[ii].change(deDocWords[dd][nn], val);
            }
        }

        tune(outputFile, trDesignMatrix, trResponses, deDesignMatrix, deResponses, K, null, null);
    }

    public void tune(File outputFile,
            SparseVector[] trDesignMatrix, double[] trResponses,
            SparseVector[] deDesignMatrix, double[] deResponses, int K,
            ArrayList<Double> rhoList, ArrayList<Double> sigmaList) {
        if (verbose) {
            System.out.println("Tuning ...");
            System.out.println("--- Output file: " + outputFile);
            System.out.println("--- Train: # instances: " + trDesignMatrix.length);
            System.out.println("--- Dev: # instances: " + deDesignMatrix.length);
            System.out.println("--- # features: " + trDesignMatrix[0].getDimension() + ". " + K);
        }

        if (rhoList == null) {
            rhoList = new ArrayList<>();
            rhoList.add(0.1);
            rhoList.add(0.25);
            rhoList.add(0.5);
            rhoList.add(1.0);
        }
        if (sigmaList == null) {
            sigmaList = new ArrayList<>();
            sigmaList.add(0.1);
            sigmaList.add(0.5);
            sigmaList.add(1.0);
            sigmaList.add(2.5);
            sigmaList.add(5.0);
            sigmaList.add(10.0);
            sigmaList.add(15.0);
            sigmaList.add(50.0);
        }

        double m = 0.0;
        try {
            boolean header = true;
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (double r : rhoList) {
                for (double s : sigmaList) {
                    if (verbose) {
                        System.out.println("*** rho = " + r + ". mu = " + m
                                + ". sigma = " + s);
                    }

                    RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                            trResponses, new double[K], trDesignMatrix, r, m, s);
                    LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
                    try {
                        optimizer.optimize();
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    double[] ws = new double[K];
                    for (int kk = 0; kk < K; kk++) {
                        ws[kk] = optimizable.getParameter(kk);
                    }

                    double[] preds = new double[deDesignMatrix.length];
                    for (int dd = 0; dd < preds.length; dd++) {
                        preds[dd] = deDesignMatrix[dd].dotProduct(ws);
                    }

                    RegressionEvaluation eval = new RegressionEvaluation(deResponses, preds);
                    eval.computeCorrelationCoefficient();
                    eval.computeMeanSquareError();
                    eval.computeMeanAbsoluteError();
                    eval.computeRSquared();
                    eval.computePredictiveRSquared();
                    ArrayList<Measurement> measurements = eval.getMeasurements();
                    measurements.add(new Measurement("N", preds.length));

                    if (header) {
                        writer.write("Rho\tSigma");
                        for (Measurement measurement : measurements) {
                            writer.write("\t" + measurement.getName());
                        }
                        writer.write("\n");
                        header = false;
                    }
                    writer.write(r + "\t" + s);
                    for (Measurement measurement : measurements) {
                        writer.write("\t" + measurement.getValue());
                    }
                    writer.write("\n");
                }
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }
}
