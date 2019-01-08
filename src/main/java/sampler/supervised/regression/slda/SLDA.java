package sampler.supervised.regression.slda;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL2Norm;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import regression.Regressor;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 * This is obsolete. Use sampler.supervised.regression.SLDA instead.
 * 
 * @author vietan
 */
public class SLDA extends AbstractSampler implements Regressor<ResponseTextDataset> {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int RHO = 2;
    public static final int MU = 3;
    public static final int SIGMA = 4;
    protected int K;
    protected int V;
    protected int D;
    // inputs
    protected int[][] words;
    protected double[] responses;
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    // optimization
    protected double[] docRegressMeans;
    protected double[][] designMatrix;
    // topical regression
    protected double[] regParams;
    // internal
    protected int numTokensChanged = 0;
    protected int numTokens = 0;
    private String optType = "lbfgs";

    public void setOptimizerType(String ot) {
        this.optType = ot;
    }

    public void configure(SLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(RHO),
                sampler.hyperparams.get(MU),
                sampler.hyperparams.get(SIGMA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho, // standard deviation of Gaussian for document observations
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.hyperparams.add(rho);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(hyperparams.get(RHO)));
            logln("--- topic mu:\t" + MiscUtils.formatDouble(hyperparams.get(MU)));
            logln("--- topic sigma:\t" + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    @Override
    public String getName() {
        return this.name;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_ot-").append(optType);
        this.name = str.toString();
    }

    public void train(int[][] ws, double[] rs) {
        this.words = ws;
        this.responses = rs;
        this.D = this.words.length;

        // statistics
        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
            int[] histogram = StatUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
    }

    protected void evaluateRegressPrediction(double[] trueVals, double[] predVals) {
        RegressionEvaluation eval = new RegressionEvaluation(trueVals, predVals);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeMeanAbsoluteError();
        eval.computeRSquared();
        eval.computePredictiveRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }
    }

    public double[] getPredictedResponses() {
        return this.docRegressMeans;
    }

    @Override
    public void train(ResponseTextDataset trainData) {
        train(trainData.getWords(), trainData.getResponses());
    }

    @Override
    public void test(ResponseTextDataset testData) {
        test(testData.getWords(), new File(getSamplerFolderPath(), IterPredictionFolder));
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        // compute the design matrix for lexical regression
        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        // optimize regression parameters
        updateTopicRegressionParameters();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluateRegressPrediction(responses, docRegressMeans);
        }
    }

    protected void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        regParams = new double[K];
        for (int k = 0; k < K; k++) {
            regParams[k] = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        docRegressMeans = new double[D];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                docTopics[d].increment(z[d][n]);
                topicWords[z[d][n]].increment(words[d][n]);
            }
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        double lda_alpha = hyperparams.get(ALPHA);
        double lda_beta = hyperparams.get(BETA);

        lda.configure(folder, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaZFile = new File(lda.getSamplerFolderPath(), "model.zip");
            if (ldaZFile.exists()) {
                lda.inputState(ldaZFile);
            } else {
                lda.initialize();
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaZFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
            ldaZ = lda.getZ();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(true);

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = ldaZ[d][n];
                docTopics[d].increment(z[d][n]);
                topicWords[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report) {
                IOUtils.createFolder(reportFolderPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating report folder."
                    + " " + reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = getClass().toString() + ". " + getSamplerName()
                        + "\nIter " + iter
                        + "\tllh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter <= BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            // update the regression parameters
            int step = (int) Math.log(iter + 1) + 1;
            if (iter % step == 0) {
                updateTopicRegressionParameters();
            }

            // parameter optimization
            if (iter % LAG == 0 && iter > BURN_IN) {
                if (paramOptimized) { // slice sampling
                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        for (double p : sparams) {
                            System.out.println(p);
                        }
                    }
                }
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                evaluateRegressPrediction(responses, docRegressMeans);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n");
                System.out.println();
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Sample topic assignments for all tokens. This is a bit faster than using
     * sampleZ.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe Whether the response variable of this document is observed
     */
    protected void sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        double totalBeta = V * hyperparams.get(BETA);
        double sqrtRho = Math.sqrt(hyperparams.get(RHO));
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    docRegressMeans[d] -= regParams[z[d][n]] / words[d].length;
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k] = Math.log(docTopics[d].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log((topicWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                            / (topicWords[k].getCountSum() + totalBeta));
                    if (observe) {
                        double mean = docRegressMeans[d] + regParams[k] / words[d].length;
                        logprobs[k] += StatUtils.logNormalProbability(responses[d], mean, sqrtRho);
                    }
                }

                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

                if (z[d][n] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[d][n] = sampledZ;

                if (addToModel) {
                    topicWords[z[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docTopics[d].increment(z[d][n]);
                    docRegressMeans[d] += regParams[z[d][n]] / words[d].length;
                }
            }
        }
    }

    /**
     * Sample topic assignment for a token
     *
     * @param d Document index
     * @param n Token index
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe Whether the response variable of this document is observed
     */
    protected void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (removeFromModel) {
            topicWords[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            docTopics[d].decrement(z[d][n]);
            docRegressMeans[d] -= regParams[z[d][n]] / words[d].length;
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] = docTopics[d].getLogLikelihood(k)
                    + topicWords[k].getLogLikelihood(words[d][n]);
            if (observe) {
                double mean = docRegressMeans[d] + regParams[k] / (words[d].length);
                logprobs[k] += StatUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));
            }
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (z[d][n] != sampledZ) {
            numTokensChanged++; // for debugging
        }
        // update
        z[d][n] = sampledZ;

        if (addToModel) {
            topicWords[z[d][n]].increment(words[d][n]);
        }
        if (addToData) {
            docTopics[d].increment(z[d][n]);
            docRegressMeans[d] += regParams[z[d][n]] / words[d].length;
        }
    }

    protected void updateTopicRegressionParameters() {
        if (optType.equals("gurobi")) {
            optimizeTopicRegressionParametersGurobi();
        } else if (optType.equals("lbfgs")) {
            optimizeTopicRegressionParametersLBFGS();
        } else {
            throw new RuntimeException("Optimization type " + optType + " is not supported");
        }
    }

    private void optimizeTopicRegressionParametersLBFGS() {
        designMatrix = new double[D][K];
        for (int d = 0; d < D; d++) {
            designMatrix[d] = docTopics[d].getEmpiricalDistribution();
        }

        GaussianIndLinearRegObjective optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses,
                hyperparams.get(RHO),
                hyperparams.get(MU),
                hyperparams.get(SIGMA));

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
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
            ex.printStackTrace();
        }

        if (verbose) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int kk = 0; kk < K; kk++) {
            regParams[kk] = optimizable.getParameter(kk);
        }

        // update current predictions
        updatePredictionValues();
    }

    private void optimizeTopicRegressionParametersGurobi() {
        designMatrix = new double[D][K];
        for (int d = 0; d < D; d++) {
            designMatrix[d] = docTopics[d].getEmpiricalDistribution();
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));
        mlr.setMean(hyperparams.get(MU));
        mlr.setSigma(hyperparams.get(SIGMA));
        regParams = mlr.solveExact();

        // update current predictions
        updatePredictionValues();
    }

    /**
     * Update the document predicted values given the new set of assignments.
     */
    protected void updatePredictionValues() {
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            this.docRegressMeans[d] = StatUtils.dotProduct(regParams, empDist);
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood();
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood();
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            double mean = StatUtils.dotProduct(regParams, empDist);
            responseLlh += StatUtils.logNormalProbability(
                    responses[d],
                    mean,
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k],
                    hyperparams.get(MU),
                    Math.sqrt(hyperparams.get(SIGMA)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". response: " + MiscUtils.formatDouble(responseLlh)
                    + ". regParam: " + MiscUtils.formatDouble(regParamLlh));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            double mean = StatUtils.dotProduct(regParams, empDist);
            responseLlh += StatUtils.logNormalProbability(
                    responses[d],
                    mean,
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < K; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k],
                    hyperparams.get(MU),
                    Math.sqrt(hyperparams.get(SIGMA)));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.docTopics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topicWords[k].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            docTopics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topicWords[k].validate(msg);
        }
    }

    @Override
    public void output(File samplerFile) {
        this.outputState(samplerFile.getAbsolutePath());
    }

    @Override
    public void input(File samplerFile) {
        this.inputModel(samplerFile.getAbsolutePath());
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(regParams[k]).append("\n");
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");

                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        validate("Done reading state from " + filepath);
    }

    protected void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                regParams[k] = Double.parseDouble(reader.readLine());
                topicWords[k] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    protected void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                docTopics[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k = 0; k < K; k++) {
            sortedTopics.add(new RankingItem<Integer>(k, regParams[k]));
        }
        Collections.sort(sortedTopics);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int ii = 0; ii < K; ii++) {
                int k = sortedTopics.get(ii).getObject();
                double[] distrs = topicWords[k].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + k
                        + ", " + topicWords[k].getCountSum()
                        + ", " + MiscUtils.formatDouble(regParams[k])
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    /**
     * Output topic coherence
     *
     * @param file Output file
     * @param topicCoherence Topic coherence
     */
    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            double[] distribution = this.topicWords[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + topicWords[k].getCountSum()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void test(int[][] newWords, File iterPredFolder) {
        if (verbose) {
            logln("Test sampling ...");
        }
        this.setTestConfigurations(BURN_IN, MAX_ITER, LAG);
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        try {
            IOUtils.createFolder(iterPredFolder);
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                sampleNewDocuments(
                        new File(reportFolder, filename).getAbsolutePath(),
                        newWords,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    /**
     * Perform sampling on test documents using a single model learned during
     * training time.
     *
     * @param stateFile The state file of the trained model
     * @param newWords Test documents
     * @param outputResultFile Prediction file
     */
    protected void sampleNewDocuments(
            String stateFile,
            int[][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Perform regression using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

        // initialize structure
        initializeDataStructure();

        if (verbose) {
            logln("test data");
            logln("--- V = " + V);
            logln("--- D = " + D);
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // initialize assignments
        sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);

        this.updatePredictionValues();

        if (verbose) {
            logln("After initialization");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < this.testMaxIter; iter++) {
            sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);

            this.updatePredictionValues();

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }

                double[] predResponses = new double[D];
                System.arraycopy(docRegressMeans, 0, predResponses, 0, D);
                predResponsesList.add(predResponses);
            }
        }

        if (verbose) {
            logln("After iterating");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k = 0; k < topicWords.length; k++) {
                topicWordCount += topicWords[k].getCountSum();
            }

            logln("\t--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("\t--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        PredictionUtils.outputSingleModelRegressions(
                new File(outputResultFile),
                predResponsesList);
    }
    // End prediction ----------------------------------------------------------

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        if (K < 10) { // only print out when number of topics is small
            str.append(MiscUtils.arrayToString(regParams)).append("\n");
            for (int k = 0; k < K; k++) {
                str.append(topicWords[k].getCountSum()).append(", ");
            }
            str.append("\n");
        }
        return str.toString();
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     */
    public static void parallelTest(int[][] newWords, File iterPredFolder, SLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                SLDATestRunner runner = new SLDATestRunner(sampler,
                        newWords, stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }

    public static void main(String[] args) {
        run(args);
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + SLDA.class.getName() + " -help";
    }

    private static void addOptions() {
        // directories
        addOption("output", "Output folder");
        addOption("dataset", "Dataset");
        addOption("data-folder", "Processed data folder");
        addOption("format-folder", "Folder holding formatted data");
        addOption("format-file", "Format file");

        addSamplingOptions();

        addOption("alpha", "alpha");
        addOption("beta", "beta");
        addOption("rho", "rho");
        addOption("mu", "mu");
        addOption("sigma", "sigma");

        addOption("init", "Initialization");
        addOption("K", "Number of topics");

        addOption("prediction-folder", "Prediction folder");
        addOption("evaluation-folder", "Evaluation folder");

        options.addOption("paramOpt", false, "Whether hyperparameter "
                + "optimization using slice sampling is performed");
        options.addOption("z", false, "whether standardize (z-score normalization)");

        options.addOption("z", false, "z-normalization");
        options.addOption("train", false, "train");
        options.addOption("test", false, "test");
    }

    public static void run(String[] args) {
        try {
            parser = new BasicParser(); // create the command line parser
            options = new Options(); // create the Options
            addOptions();
            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }
            runModel();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Use option -help for all options.");
        }
    }

    protected static void runModel() {
        // sampling configurations
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 500);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 50);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 2.5);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 10);
        String optType = CLIUtils.getStringArgument(cmd, "opt-type", "gurobi");

        // directories
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo");
        String resultFolder = CLIUtils.getStringArgument(cmd, "output",
                "demo/amazon-data/format-response/models");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        ResponseTextDataset data = new ResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        if (cmd.hasOption("topic-coherence")) {
            data.prepareTopicCoherence(numTopWords);
        }
        if (cmd.hasOption("z")) {
            data.zNormalize();
        } else {
            System.out.println("[WARNING] Running with unnormalized responses.");
        }

        String init = CLIUtils.getStringArgument(cmd, "init", "random");
        InitialState initState;
        if (init.equals("random")) {
            initState = InitialState.RANDOM;
        } else if (init.equals("preset")) {
            initState = InitialState.PRESET;
        } else {
            throw new RuntimeException("Initialization " + init + " not supported");
        }

        SLDA sampler = new SLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());
        sampler.setOptimizerType(optType);

        sampler.configure(resultFolder,
                data.getWordVocab().size(), K,
                alpha, beta, rho, mu, sigma, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(data.getWords(), data.getResponses());
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
        }

        if (cmd.hasOption("test")) {
            File predictionFolder = new File(sampler.getSamplerFolderPath(),
                    CLIUtils.getStringArgument(cmd, "prediction-folder", "predictions"));
            IOUtils.createFolder(predictionFolder);

            File evaluationFolder = new File(sampler.getSamplerFolderPath(),
                    CLIUtils.getStringArgument(cmd, "evaluation-folder", "evaluations"));
            IOUtils.createFolder(evaluationFolder);

            // test in parallel
            SLDA.parallelTest(data.getWords(), predictionFolder, sampler);

            double[] predictions = PredictionUtils.evaluateRegression(
                    predictionFolder, evaluationFolder, data.getDocIds(),
                    data.getResponses());

            PredictionUtils.outputRegressionPredictions(
                    new File(predictionFolder,
                    AbstractExperiment.PREDICTION_FILE),
                    data.getDocIds(), data.getResponses(), predictions);
            PredictionUtils.outputRegressionResults(
                    new File(evaluationFolder,
                    AbstractExperiment.RESULT_FILE), data.getResponses(),
                    predictions);
        }
    }
}

class SLDATestRunner implements Runnable {

    SLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public SLDATestRunner(SLDA sampler,
            int[][] newWords,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        SLDA testSampler = new SLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
