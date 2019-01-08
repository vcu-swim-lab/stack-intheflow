package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import optimization.RidgeLinearRegressionOptimizable;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;

/**
 * Implementation of Supervised Latent Dirichlet Allocation (SLDA).
 *
 * @author vietan
 */
public class SLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected double rho;
    protected double mu;
    protected double sigma;
    // inputs
    protected int[][] words; // original documents
    protected double[] responses; // [D]: responses of selected documents
    protected ArrayList<Integer> docIndices; // [D]: indices of selected documents
    protected int K;
    protected int V;
    // derive
    protected int D;
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    protected double[] regParams;
    // optimization
    protected double[] docMeans;
    protected SparseVector[] designMatrix;
    // internal
    protected double sqrtRho;
    protected boolean hasBias;

    public SLDA() {
        this.basename = "SLDA";
    }

    public SLDA(String bname) {
        this.basename = bname;
    }

    public void configure(SLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.rho,
                sampler.mu,
                sampler.sigma,
                sampler.initState,
                sampler.paramOptimized,
                sampler.hasBias,
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
            double rho, // variance of Gaussian for document observations
            double mu, // mean of Gaussian for regression parameters
            double sigma, // variance of Gaussian for regression parameters
            InitialState initState,
            boolean paramOpt, boolean hasBias,
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

        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.sqrtRho = Math.sqrt(this.rho);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.hasBias = hasBias;
        this.prefix += initState.toString();
        this.setName();

        this.report = true;

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta: \t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- topic mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- topic sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- has bias:\t" + hasBias);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_" + this.basename
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(hyperparams.get(ALPHA))
                + "_b-" + formatter.format(hyperparams.get(BETA))
                + "_r-" + formatter.format(rho)
                + "_m-" + formatter.format(mu)
                + "_s-" + formatter.format(sigma)
                + "_opt-" + this.paramOptimized
                + "_bias-" + this.hasBias;
    }

    public DirMult[] getTopicWords() {
        return this.topicWords;
    }

    public int[][] getZs() {
        return this.z;
    }

    public double[][] getThetas() {
        double[][] thetas = new double[D][];
        for (int dd = 0; dd < D; dd++) {
            thetas[dd] = this.docTopics[dd].getDistribution();
        }
        return thetas;
    }

    public double[][] getPhis() {
        double[][] phis = new double[K][];
        for (int kk = 0; kk < K; kk++) {
            phis[kk] = this.topicWords[kk].getDistribution();
        }
        return phis;
    }

    public double[] getRegressionParameters() {
        return this.regParams;
    }

    public double[] getPredictedValues() {
        return this.docMeans;
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

    /**
     * Set up data.
     *
     * @param docWords All documents
     * @param docIndices Indices of documents under consideration
     * @param docResponses Responses of all documents, null if test data
     */
    public void setupData(int[][] docWords, ArrayList<Integer> docIndices,
            double[] docResponses) {
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.responses = null;
        if (docResponses != null) { // null if test data
            this.responses = new double[D]; // responses of considered documents
        }
        this.numTokens = 0;
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            if (docResponses != null) {
                this.responses[ii] = docResponses[dd];
            }
            this.words[ii] = docWords[dd];
            this.numTokens += this.words[ii].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            if (docResponses != null) {
                logln("--- responses:");
                logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
                logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
                int[] histogram = StatUtils.bin(responses, 10);
                for (int ii = 0; ii < histogram.length; ii++) {
                    logln("--- --- " + ii + "\t" + histogram[ii]);
                }
            }
        }
    }

    /**
     * Set up training data.
     *
     * @param docWords All documents
     * @param docIndices Indices of documents under consideration
     * @param docResponses Responses of all documents
     */
    public void train(int[][] docWords, ArrayList<Integer> docIndices,
            double[] docResponses) {
        setupData(docWords, docIndices, docResponses);
    }

    /**
     * Set up test data.
     *
     * @param docWords Test documents
     * @param docIndices Indices of test documents
     * @param stateFile File storing trained model
     * @param predictionFile File storing predictions at different test
     * iterations using the given trained model
     * @return Prediction on all documents using the given model
     */
    public double[] test(int[][] docWords, ArrayList<Integer> docIndices,
            File stateFile, File predictionFile) {
        setTestConfigurations(BURN_IN / 2, MAX_ITER / 2, LAG / 2);
        inputModel(stateFile.toString());
        setupData(docWords, docIndices, null);
        initializeDataStructure();
        if (hasBias) {
            for (int dd = 0; dd < D; dd++) {
                docMeans[dd] = regParams[K];
            }
        }

        // store predictions at different test iterations
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();

        // sample topic assignments for test document
        for (iter = 0; iter < this.testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + ". current thread: " + Thread.currentThread().getId();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
            }

            // store prediction (on all documents) at a test iteration
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double[] predResponses = new double[D];
                System.arraycopy(docMeans, 0, predResponses, 0, D);
                predResponsesList.add(predResponses);
            }
        }

        // store predictions if necessary
        if (predictionFile != null) {
            PredictionUtils.outputSingleModelRegressions(predictionFile, predResponsesList);
        }

        // average over all stored predictions
        double[] predictions = new double[D];
        for (int dd = 0; dd < D; dd++) {
            for (double[] predResponses : predResponsesList) {
                predictions[dd] += predResponses[dd] / predResponsesList.size();
            }
        }
        return predictions;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        initialize(null);
    }

    /**
     * Initialized with seeded distributions.
     *
     * @param topicWordPrior Word distribution for each topic
     */
    public void initialize(double[][] topicWordPrior) {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;
        initializeModelStructure(topicWordPrior);
        initializeDataStructure();
        initializeAssignments();
        updateTopicRegressionParameters(); // optimize regression parameters

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluateRegressPrediction(responses, docMeans);
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (topics != null && topics.length != K) {
            throw new MismatchRuntimeException(topics.length, K);
        }

        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (topics != null) { // seeded prior
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, topics[k]);
            } else { // uninformed prior
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }

        int numParams = K;
        if (hasBias) {
            numParams++;
        }
        regParams = new double[numParams];
        for (int kk = 0; kk < numParams; kk++) {
            regParams[kk] = SamplerUtils.getGaussian(mu, sigma);
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int dd = 0; dd < D; dd++) {
            z[dd] = new int[words[dd].length];
        }

        docTopics = new DirMult[D];
        for (int ii = 0; ii < D; ii++) {
            docTopics[ii] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        docMeans = new double[D];
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
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = rand.nextInt(K);
                docTopics[dd].increment(z[dd][nn]);
                topicWords[z[dd][nn]].increment(words[dd][nn]);
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
        double lda_alpha = hyperparams.get(ALPHA);
        double lda_beta = hyperparams.get(BETA);
        LDA lda = runLDA(words, K, V, null, null, lda_alpha, lda_beta,
                lda_burnin, lda_maxiter, lda_samplelag);
        int[][] ldaZ = lda.getZs();

        // initialize assignments
        for (int dd = 0; dd < D; dd++) {
            for (int n = 0; n < words[dd].length; n++) {
                z[dd][n] = ldaZ[dd][n];
                docTopics[dd].increment(z[dd][n]);
                topicWords[z[dd][n]].increment(words[dd][n]);
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
            isReporting = isReporting();
            if (isReporting) {
                // store llh after every iteration
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            // update the regression parameters
            updateTopicRegressionParameters();

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

            if (isReporting && debug) {
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
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (isReporting) {
            logln("+++ Sampling Zs ...");
        }
        numTokensChanged = 0;
        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (removeFromModel) {
                    topicWords[z[dd][nn]].decrement(words[dd][nn]);
                }
                if (removeFromData) {
                    docTopics[dd].decrement(z[dd][nn]);
                    docMeans[dd] -= regParams[z[dd][nn]] / words[dd].length;
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k] = Math.log(docTopics[dd].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[k].getProbability(words[dd][nn]));
                    if (observe) {
                        double mean = docMeans[dd] + regParams[k] / words[dd].length;
                        logprobs[k] += StatUtils.logNormalProbability(responses[dd], mean, sqrtRho);
                    }
                }

                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

                if (z[dd][nn] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[dd][nn] = sampledZ;

                if (addToModel) {
                    topicWords[z[dd][nn]].increment(words[dd][nn]);
                }
                if (addToData) {
                    docTopics[dd].increment(z[dd][nn]);
                    docMeans[dd] += regParams[z[dd][nn]] / words[dd].length;
                }
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # token changed: " + numTokensChanged
                    + ". change ratio: " + (double) numTokensChanged / numTokens
                    + "\n");
        }
        return eTime;
    }

    /**
     * Update regression parameters by optimizing using L-BFGS.
     */
    private long updateTopicRegressionParameters() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();
        designMatrix = new SparseVector[D];
        for (int dd = 0; dd < D; dd++) {
            if (hasBias) {
                designMatrix[dd] = new SparseVector(K + 1);
                designMatrix[dd].set(K, 1.0);
            } else {
                designMatrix[dd] = new SparseVector(K);
            }
            for (int k : docTopics[dd].getSparseCounts().getIndices()) {
                double val = (double) docTopics[dd].getCount(k) / z[dd].length;
                designMatrix[dd].change(k, val);
            }
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, regParams, designMatrix, rho, mu, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        for (int kk = 0; kk < optimizable.getNumParameters(); kk++) {
            regParams[kk] = optimizable.getParameter(kk);
        }

        // update current predictions
        this.docMeans = new double[D];
        for (int dd = 0; dd < D; dd++) {
            for (int kk : designMatrix[dd].getIndices()) {
                this.docMeans[dd] += designMatrix[dd].get(kk) * regParams[kk];
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            evaluateRegressPrediction(responses, docMeans);
            logln("--- " + designMatrix.length + " x " + optimizable.getNumParameters());
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
        }
        return eTime;
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
        for (int ii = 0; ii < D; ii++) {
            double[] empDist = docTopics[ii].getEmpiricalDistribution();
            double mean = 0.0;
            if (hasBias) {
                mean += regParams[K];
            }
            for (int kk = 0; kk < K; kk++) {
                mean += regParams[kk] * empDist[kk];
            }
            responseLlh += StatUtils.logNormalProbability(responses[ii],
                    mean, sqrtRho);
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < regParams.length; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k], mu, Math.sqrt(sigma));
        }

        if (isReporting()) {
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
            wordLlh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V,
                    topicWords[k].getCenterVector());
        }

        double topicLlh = 0.0;
        for (int dd = 0; dd < D; dd++) {
            topicLlh += docTopics[dd].getLogLikelihood(newParams.get(ALPHA) * K,
                    docTopics[dd].getCenterVector());
        }

        double responseLlh = 0.0;
        for (int ii = 0; ii < D; ii++) {
            double[] empDist = docTopics[ii].getEmpiricalDistribution();
            double mean = 0.0;
            if (hasBias) {
                mean += regParams[K];
            }
            for (int kk = 0; kk < K; kk++) {
                mean += regParams[kk] * empDist[kk];
            }
            responseLlh += StatUtils.logNormalProbability(responses[ii],
                    mean, sqrtRho);
        }

        double regParamLlh = 0.0;
        for (int k = 0; k < regParams.length; k++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[k], mu, Math.sqrt(sigma));
        }

        double llh = wordLlh + topicLlh + responseLlh + regParamLlh;
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
            if (hasBias) {
                modelStr.append(regParams[K]).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int dd = 0; dd < D; dd++) {
                assignStr.append(dd).append("\n");
                assignStr.append(DirMult.output(docTopics[dd])).append("\n");

                for (int n = 0; n < z[dd].length; n++) {
                    assignStr.append(z[dd][n]).append("\t");
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
            throw new RuntimeException("Exception while loading from " + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    protected void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeModelStructure(null);

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
            if (hasBias) {
                regParams[K] = Double.parseDouble(reader.readLine());
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
                for (int n = 0; n < z[d].length; n++) {
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

    @Override
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

            if (hasBias) {
                writer.write("Bias: " + regParams[K] + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    public void outputPosterior(File my_file) {
        double[][] postTops = new double[K][];
        for (int i = 0; i < K; i++) {
            postTops[i] = topicWords[i].getDistribution();
        }
        IOUtils.output2DArray(my_file, postTops);
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newDocIndices Indices of test documents
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     */
    public static double[] parallelTest(int[][] newWords,
            ArrayList<Integer> newDocIndices,
            File iterPredFolder,
            SLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        double[] avgPredictions = null;
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            ArrayList<File> partPredFiles = new ArrayList<>();
            for (String filename : filenames) { // all learned models
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                SLDATestRunner runner = new SLDATestRunner(sampler,
                        newWords, newDocIndices, stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
                partPredFiles.add(partialResultFile);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // average predictions
            avgPredictions = PredictionUtils.computeMultipleAverage(partPredFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
        return avgPredictions;
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + SLDA.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        return "java -cp \"dist/segan.jar:lib/*\" sampler.supervised.regression.SLDA "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-supervised/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-supervised/amazon-data.dat "
                + "--info-file demo/amazon-data/format-supervised/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-supervised "
                + "--burnIn 100 "
                + "--maxIter 250 "
                + "--sampleLag 30 "
                + "--report 5 "
                + "--K 50 "
                + "--alpha 0.1 "
                + "--beta 0.1 "
                + "--rho 1.0 "
                + "--sigma 1.0 "
                + "--mu 0.0 "
                + "--init random "
                + "-v -d -z";
    }

    private static int findIndex(String[] set, String q) {
        for (int i = 0; i < set.length; i++) {
            if (set[i].equals(q)) {
                return i;
            }
        }
        System.out.println(q);
        return -1;
    }

    private static void addOpitions() throws Exception {
        parser = new BasicParser();
        options = new Options();

        // data input
        addOption("dataset", "Dataset");
        addOption("word-voc-file", "Word vocabulary file");
        addOption("word-file", "Document word file");
        addOption("info-file", "Document info file");
        addOption("selected-docs-file", "(Optional) Indices of selected documents");
        addOption("prior-topic-file", "File containing prior topics");

        // predictions
        addOption("prediction-folder", "Folder containing predictions");
        addOption("evaluation-folder", "Folder containing evaluations");

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("alpha", "Alpha");
        addOption("beta", "Beta");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigma", "Sigma");
        addOption("K", "Number of topics");
        addOption("num-top-words", "Number of top words per topic");

        // running
        options.addOption("train", false, "Train");
        options.addOption("test", false, "Test");
        options.addOption("parallel", false, "Parallel");
        options.addOption("bias", false, "Bias");

        // configurations
        addOption("init", "Initialization");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
        options.addOption("example", false, "Example command");
    }

    private static void runModel() throws Exception {
        // sampling configurations
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 20);
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 500);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 50);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 25);
        boolean paramOpt = cmd.hasOption("paramOpt");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
        InitialState initState;
        switch (init) {
            case "random":
                initState = InitialState.RANDOM;
                break;
            case "preset":
                initState = InitialState.PRESET;
                break;
            default:
                throw new RuntimeException("Initialization " + init + " not supported");
        }

        // model parameters
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 1.0);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);

        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");
        String docInfoFile = cmd.getOptionValue("info-file");

        // data output
        String outputFolder = cmd.getOptionValue("output-folder");

        ResponseTextDataset data = new ResponseTextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile),
                new File(docInfoFile),
                null);
        int V = data.getWordVocab().size();
        boolean hasBias = cmd.hasOption("bias");

        SLDA sampler = new SLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(outputFolder, V, K,
                alpha, beta, rho, mu, sigma,
                initState, paramOpt, hasBias,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        double[] docResponses = data.getResponses();
        if (cmd.hasOption("z")) { // z-normalization
            ZNormalizer zNorm = new ZNormalizer(docResponses);
            docResponses = zNorm.normalize(docResponses);
        }

        ArrayList<Integer> selectedDocIndices = null;
        if (cmd.hasOption("selected-docs-file")) {
            String selectedDocFile = cmd.getOptionValue("selected-docs-file");
            selectedDocIndices = new ArrayList<>();
            BufferedReader reader = IOUtils.getBufferedReader(selectedDocFile);
            String line;
            while ((line = reader.readLine()) != null) {
                //int docIdx = findIndex(data.getDocIds(), line.trim());
                int docIdx = Integer.parseInt(line);
                if (docIdx >= data.getDocIds().length) {
                    throw new RuntimeException("Out of bound. Doc index " + docIdx);
                }
                selectedDocIndices.add(docIdx);
            }
            reader.close();
        }

        double[][] priorTopics = null;
        if (cmd.hasOption("prior-topic-file")) {
            String priorTopicFile = cmd.getOptionValue("prior-topic-file");
            priorTopics = IOUtils.input2DArray(new File(priorTopicFile));
        }

        if (cmd.hasOption("train")) {
            sampler.train(data.getWords(), selectedDocIndices, docResponses);
            sampler.initialize(priorTopics);
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            sampler.outputPosterior(new File(samplerFolder, "posterior.csv"));
            IOUtils.output2DArray(new File(samplerFolder, "phis.txt"), sampler.getPhis());
            IOUtils.outputArray(new File(samplerFolder, "etas.txt"), sampler.regParams);
        }

        if (cmd.hasOption("test")) {
            File predictionFolder = new File(sampler.getSamplerFolderPath(),
                    CLIUtils.getStringArgument(cmd, "prediction-folder", "predictions"));
            IOUtils.createFolder(predictionFolder);

            File evaluationFolder = new File(sampler.getSamplerFolderPath(),
                    CLIUtils.getStringArgument(cmd, "evaluation-folder", "evaluations"));
            IOUtils.createFolder(evaluationFolder);

            double[] predictions;
            if (cmd.hasOption("parallel")) { // predict using all models
                File iterPredFolder = new File(sampler.getSamplerFolderPath(), "iter-preds");
                IOUtils.createFolder(iterPredFolder);
                SLDA.parallelTest(data.getWords(), selectedDocIndices, iterPredFolder, sampler);
                predictions = PredictionUtils.evaluateRegression(
                        iterPredFolder, evaluationFolder, data.getDocIds(),
                        docResponses);
            } else { // predict using the final model
                predictions = sampler.test(data.getWords(), selectedDocIndices,
                        sampler.getFinalStateFile(), null);
            }

            // output predictions and results
            String[] selectedIds;
            double[] responses;
            if (selectedDocIndices == null) {
                selectedIds = data.getDocIds();
                responses = docResponses;
            } else {
                int numDocs = selectedDocIndices.size();
                selectedIds = new String[numDocs];
                responses = new double[numDocs];
                for (int q = 0; q < numDocs; q++) {
                    int index = selectedDocIndices.get(q);
                    selectedIds[q] = data.getDocIds()[index];
                    responses[q] = docResponses[index];
                }
            }

            PredictionUtils.outputRegressionPredictions(
                    new File(predictionFolder, AbstractExperiment.PREDICTION_FILE),
                    selectedIds, responses, predictions);
            PredictionUtils.outputRegressionResults(
                    new File(evaluationFolder, AbstractExperiment.RESULT_FILE), responses,
                    predictions);
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOpitions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            } else if (cmd.hasOption("example")) {
                System.out.println(getExampleCmd());
                return;
            }

            runModel();

            // date and time
            DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
            Date dateobj = new Date();
            long eTime = (System.currentTimeMillis() - sTime) / 1000;
            System.out.println("Elapsed time: " + eTime + "s");
            System.out.println("End time: " + df.format(dateobj));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}

class SLDATestRunner implements Runnable {

    SLDA sampler;
    int[][] newWords;
    ArrayList<Integer> newDocIndices;
    String stateFile;
    String outputFile;

    public SLDATestRunner(SLDA sampler,
            int[][] newWords,
            ArrayList<Integer> newDocIndices,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newDocIndices = newDocIndices;
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
            testSampler.test(newWords, newDocIndices, new File(stateFile), new File(outputFile));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
