package sampler.backup;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL2Norm;
import sampler.LDA;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class DoubleSLDA extends AbstractSampler {

    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int RHO = 2;
    public static final int ETA_MEAN = 3;
    public static final int ETA_VAR = 4;
    public static final int LAMBDA_MEAN = 5;
    public static final int LAMBDA_VAR = 6;
    protected int K;
    protected int V;
    protected int D;
    protected int numPostives;
    // inputs
    protected int[][] words; // [D] x [N_d]
    protected double[] responses; // [D] continuous response variables
    protected int[] labels; // [D] binary labels
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    // topical regression
    protected double[] etas;    // response regression parameters
    protected double[] lambdas; // label regression parameters
    private double[] docResponseDotProds;
    private double[] docLabelDotProds;
    // prediction
    private int testBurnIn = BURN_IN;
    private int testMaxIter = MAX_ITER;
    private int testSampleLag = LAG;
    // internal
    private int numTokensChanged = 0;
    private int numTokens = 0;
    private ArrayList<double[]> etasOverTime;
    private ArrayList<double[]> lambdasOverTime;

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double etaMean,
            double etaSigma,
            double lambdaMean,
            double lambdaSigma,
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
        this.hyperparams.add(etaMean);
        this.hyperparams.add(etaSigma);
        this.hyperparams.add(lambdaMean);
        this.hyperparams.add(lambdaSigma);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.etasOverTime = new ArrayList<double[]>();
        this.lambdasOverTime = new ArrayList<double[]>();
        this.setName();

        if (!debug) {
            System.err.close();
        }

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(hyperparams.get(RHO)));
            logln("--- response mean:\t" + MiscUtils.formatDouble(hyperparams.get(ETA_MEAN)));
            logln("--- response variance:\t" + MiscUtils.formatDouble(hyperparams.get(ETA_VAR)));
            logln("--- label mean:\t" + MiscUtils.formatDouble(hyperparams.get(LAMBDA_MEAN)));
            logln("--- label variance:\t" + MiscUtils.formatDouble(hyperparams.get(LAMBDA_VAR)));

            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- # tokens:\t" + numTokens);

            logln("--- responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
            int[] histogram = StatUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }

            logln("--- labels:");
            int numPositives = 0;
            for (int ii = 0; ii < labels.length; ii++) {
                if (labels[ii] == POSITVE) {
                    numPositives++;
                }
            }
            logln("--- --- # positives\t" + numPositives);
            logln("--- --- # negatives\t" + (D - numPositives));
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_double-sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_es-").append(formatter.format(hyperparams.get(ETA_VAR)))
                .append("_ls-").append(formatter.format(hyperparams.get(LAMBDA_VAR)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void train(int[][] ws, double[] rs, int[] ls) {
        this.words = ws;
        this.responses = rs;
        this.labels = ls;
        this.D = this.words.length;

        this.numPostives = 0;
        for (int d = 0; d < D; d++) {
            if (labels[d] == POSITVE) {
                numPostives++;
            }
        }

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        updateEtas();

        updateLambdas();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluateRegressPrediction();
            evaluateLabelPrediction();
        }
    }

    private void evaluateLabelPrediction() {
        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<RankingItem<Integer>>();
        for (int d = 0; d < D; d++) {
            double expDotProd = Math.exp(docLabelDotProds[d]);
            double docPred = expDotProd / (expDotProd + 1);
            rankDocs.add(new RankingItem<Integer>(d, docPred));
        }
        Collections.sort(rankDocs);
        int[] preds = new int[D];
        for (int ii = 0; ii < numPostives; ii++) {
            int d = rankDocs.get(ii).getObject();
            preds[d] = POSITVE;
        }

        ClassificationEvaluation eval = new ClassificationEvaluation(labels, preds);
        eval.computePRF1();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }
    }

    private void evaluateRegressPrediction() {
        RegressionEvaluation eval = new RegressionEvaluation(responses, docResponseDotProds);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }
    }

    @Override
    public void sample() {
        this.initialize();
        this.iterate();
    }

    private void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        etas = new double[K];
        for (int k = 0; k < K; k++) {
            etas[k] = SamplerUtils.getGaussian(hyperparams.get(ETA_MEAN), hyperparams.get(ETA_VAR));
        }

        lambdas = new double[K];
        for (int k = 0; k < K; k++) {
            lambdas[k] = SamplerUtils.getGaussian(hyperparams.get(LAMBDA_MEAN), hyperparams.get(LAMBDA_VAR));
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
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

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
                lda.outputTopicTopWords(
                        new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
            ldaZ = lda.getZ();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
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

        RegressionEvaluation eval;
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            // store regression parameters after every iteration
            double[] snapEtas = new double[K];
            System.arraycopy(etas, 0, snapEtas, 0, K);
            this.etasOverTime.add(snapEtas);

            double[] snapLambdas = new double[K];
            System.arraycopy(lambdas, 0, snapLambdas, 0, K);
            this.lambdasOverTime.add(snapLambdas);

            if (verbose && iter % REP_INTERVAL == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\n" + getCurrentState());
                }
            }

            // sample topic assignments
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            // update the regression parameters
            updateEtas();

            // update the label parameters
            updateLambdas();

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
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
                logln("--- response regression");
                evaluateRegressPrediction();
                logln("--- label prediction");
                evaluateLabelPrediction();

                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n");
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
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

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        "hyperparameters.txt").getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
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
    private void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (removeFromModel) {
            topicWords[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            docTopics[d].decrement(z[d][n]);
            docResponseDotProds[d] -= etas[z[d][n]] / words[d].length;
            docLabelDotProds[d] -= lambdas[z[d][n]] / words[d].length;
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k]
                    = docTopics[d].getLogLikelihood(k)
                    + topicWords[k].getLogLikelihood(words[d][n]);
            if (observe) {
                double responseMean = docResponseDotProds[d] + etas[k] / (words[d].length);
                logprobs[k] += StatUtils.logNormalProbability(responses[d],
                        responseMean, Math.sqrt(hyperparams.get(RHO)));

                double dotProd = docLabelDotProds[d] + lambdas[z[d][n]] / words[d].length;
                logprobs[k] += getLabelLogLikelihood(labels[d], dotProd);
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
            docResponseDotProds[d] += etas[z[d][n]] / words[d].length;
            docLabelDotProds[d] += lambdas[z[d][n]] / words[d].length;
        }
    }

    private void updateLambdas() {
        if (lambdas == null) {
            this.lambdas = new double[K];
            for (int k = 0; k < K; k++) {
                this.lambdas[k] = SamplerUtils.getGaussian(hyperparams.get(LAMBDA_MEAN),
                        Math.sqrt(hyperparams.get(LAMBDA_VAR)));
            }
        }

        double[][] designMatrix = new double[D][K];
        for (int d = 0; d < D; d++) {
            designMatrix[d] = docTopics[d].getEmpiricalDistribution();
        }

        L2NormLogLinearObjective optimizable = new L2NormLogLinearObjective(
                lambdas, designMatrix, labels,
                hyperparams.get(LAMBDA_MEAN),
                hyperparams.get(LAMBDA_VAR));

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (verbose) {
            logln("--- converged: " + converged);
        }

        // update regression parameters
        for (int k = 0; k < K; k++) {
            lambdas[k] = optimizable.getParameter(k);
        }

        // update current predictions
        this.docLabelDotProds = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            for (int k = 0; k < K; k++) {
                this.docLabelDotProds[d] += lambdas[k] * empDist[k];
            }
        }
    }

    private void updateEtas() {
        double[][] designMatrix = new double[D][K];
        for (int d = 0; d < D; d++) {
            designMatrix[d] = docTopics[d].getEmpiricalDistribution();
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));
        mlr.setMean(hyperparams.get(ETA_MEAN));
        mlr.setSigma(hyperparams.get(ETA_VAR));
        etas = mlr.solve();

        // update current predictions
        this.docResponseDotProds = new double[D];
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            for (int k = 0; k < K; k++) {
                this.docResponseDotProds[d] += etas[k] * empDist[k];
            }
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
        double labelLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();

            // response
            double mean = StatUtils.dotProduct(etas, empDist);
            responseLlh += StatUtils.logNormalProbability(
                    responses[d],
                    mean,
                    Math.sqrt(hyperparams.get(RHO)));

            // label
            double dotProd = StatUtils.dotProduct(lambdas, empDist);
            labelLlh += getLabelLogLikelihood(labels[d], dotProd);
        }

        double etaLlh = 0.0;
        double lambdaLlh = 0.0;
        for (int k = 0; k < K; k++) {
            etaLlh += StatUtils.logNormalProbability(
                    etas[k],
                    hyperparams.get(ETA_MEAN),
                    Math.sqrt(hyperparams.get(ETA_VAR)));

            lambdaLlh += StatUtils.logNormalProbability(
                    lambdas[k],
                    hyperparams.get(LAMBDA_MEAN),
                    Math.sqrt(hyperparams.get(LAMBDA_VAR)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". response: " + MiscUtils.formatDouble(responseLlh)
                    + ". label: " + MiscUtils.formatDouble(labelLlh)
                    + ". etaLlh: " + MiscUtils.formatDouble(etaLlh)
                    + ". lambdaLlh: " + MiscUtils.formatDouble(labelLlh));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + labelLlh
                + etaLlh
                + lambdaLlh;
        return llh;
    }

    private double getLabelLogLikelihood(int label, double dotProb) {
        double logNorm = Math.log(Math.exp(dotProb) + 1);
        if (label == POSITVE) {
            return dotProb - logNorm;
        } else {
            return -logNorm;
        }
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
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
                modelStr.append(etas[k]).append("\n");
                modelStr.append(lambdas[k]).append("\n");
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
            System.exit(1);
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

    private void inputModel(String zipFilepath) {
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
                etas[k] = Double.parseDouble(reader.readLine());
                lambdas[k] = Double.parseDouble(reader.readLine());
                topicWords[k] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    private void inputAssignments(String zipFilepath) throws Exception {
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
            sortedTopics.add(new RankingItem<Integer>(k, etas[k]));
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
                        + ", " + MiscUtils.formatDouble(etas[k])
                        + ", " + MiscUtils.formatDouble(lambdas[k])
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }
}

class L2NormLogLinearObjective implements Optimizable.ByGradientValue {

    private double[] parameters;
    private double[][] designMatrix;
    private int[] labels; // 1 or 0
    private double paramMean;
    private double paramVar;
    private int D;

    public L2NormLogLinearObjective(
            double[] curParams,
            double[][] designMatrix,
            int[] labels,
            double paramMean, double paramVar) {
        this.parameters = new double[curParams.length];
        System.arraycopy(curParams, 0, this.parameters, 0, curParams.length);

        this.designMatrix = designMatrix;
        this.labels = labels;
        this.paramMean = paramMean;
        this.paramVar = paramVar;

        this.D = designMatrix.length;
    }

    @Override
    public double getValue() {
        double value = 0;
        // log likelihood
        for (int d = 0; d < D; d++) {
            double dotProb = 0.0;
            for (int k = 0; k < getNumParameters(); k++) {
                dotProb += this.parameters[k] * this.designMatrix[d][k];
            }

            value += labels[d] * dotProb;
            value -= Math.log(Math.exp(dotProb) + 1);
        }

        // regularizer
        for (int k = 0; k < getNumParameters(); k++) {
            double diff = this.parameters[k] - this.paramMean;
            value -= (diff * diff) / (2 * this.paramVar);
        }
        return value;
    }

    @Override
    public void getValueGradient(double[] gradient) {
        double[] tempGrad = new double[getNumParameters()];

        // likelihood
        for (int d = 0; d < D; d++) {
            double dotprod = 0.0;
            for (int k = 0; k < getNumParameters(); k++) {
                dotprod += parameters[k] * designMatrix[d][k];
            }
            double expDotprod = Math.exp(dotprod);
            double pred = expDotprod / (expDotprod + 1);

            for (int k = 0; k < getNumParameters(); k++) {
                tempGrad[k] += (labels[d] - pred) * designMatrix[d][k];
            }
        }

        // regularizer
        for (int k = 0; k < getNumParameters(); k++) {
            tempGrad[k] -= (parameters[k] - paramMean) / paramVar;
        }
        System.arraycopy(tempGrad, 0, gradient, 0, getNumParameters());
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
