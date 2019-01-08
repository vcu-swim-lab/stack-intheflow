package sampler.supervised.classification;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractSampler;
import data.LabelTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import optimization.RidgeLogisticRegressionOptimizable;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.LDA;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RankingEvaluation;

/**
 *
 * @author vietan
 */
public class BinarySLDA extends AbstractSampler {

    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected double mean;
    protected double sigma;
    // data statistics
    protected int K;
    protected int V;
    protected int D;
    // inputs
    protected int[][] words; // [D] x [N_d]
    protected int[] labels; // [D] binary labels
    protected ArrayList<Integer> docIndices; // [D]: indices of selected documents
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    protected double[] lambdas; // label regression parameters
    private double[] docLabelDotProds;
    private Set<Integer> positives;

    public BinarySLDA() {
        this.basename = "binary-SLDA";
    }

    public BinarySLDA(String bname) {
        this.basename = bname;
    }

    public void configure(BinarySLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.mean,
                sampler.sigma,
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
        this.mean = lambdaMean;
        this.sigma = lambdaSigma;

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

        if (!debug) {
            System.err.close();
        }

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- label mean:\t" + MiscUtils.formatDouble(mean));
            logln("--- label variance:\t" + MiscUtils.formatDouble(sigma));

            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_m-").append(formatter.format(mean))
                .append("_s-").append(formatter.format(sigma));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    /**
     * Set training data.
     *
     * @param docWords All documents
     * @param docIndices Indices of documents under consideration
     * @param docLabels Responses of all documents
     */
    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            int[] docLabels) {
        this.words = docWords;
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.numTokens = 0;
        this.D = this.docIndices.size();
        this.labels = new int[this.D];
        this.positives = new HashSet<Integer>();
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.numTokens += this.words[dd].length;
            this.labels[ii] = docLabels[dd];
            if (this.labels[ii] == POSITVE) {
                this.positives.add(ii);
            }
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- responses:");
            int posCount = this.positives.size();
            logln("--- --- # postive: " + posCount + " (" + ((double) posCount / D) + ")");
            logln("--- --- # negative: " + (D - posCount));
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

        updateLambdas();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluatePerformances();
        }
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
        updateLambdas();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
            evaluatePerformances();
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (topics != null && topics.length != K) {
            throw new IllegalArgumentException("Mismatch: " + topics.length + ", " + K);
        }

        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (topics != null) { // seeded prior
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, topics[k]);
            } else { // uninformed prior
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }

        lambdas = new double[K];
        for (int k = 0; k < K; k++) {
            lambdas[k] = SamplerUtils.getGaussian(mean, sigma);
        }
    }

    private void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        lambdas = new double[K];
        for (int k = 0; k < K; k++) {
            lambdas[k] = SamplerUtils.getGaussian(mean, sigma);
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            z[ii] = new int[words[docIndices.get(ii)].length];
        }

        docTopics = new DirMult[D];
        for (int ii = 0; ii < D; ii++) {
            docTopics[ii] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        docLabelDotProds = new double[D];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int nn = 0; nn < words[ii].length; nn++) {
                z[ii][nn] = rand.nextInt(K);
                docTopics[ii].increment(z[ii][nn]);
                topicWords[z[ii][nn]].increment(words[dd][nn]);
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
            File ldaZFile = new File(lda.getSamplerFolderPath(), basename + ".zip");
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
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);

        // initialize assignments
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int n = 0; n < words[ii].length; n++) {
                z[ii][n] = ldaZ[ii][n];
                docTopics[ii].increment(z[ii][n]);
                topicWords[z[ii][n]].increment(words[dd][n]);
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

            if (isReporting()) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "Iter " + iter + " / " + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            long topicTime = sampleZ(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            long lambdaTime = updateLambdas();

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

            if (isReporting()) {
                logln("--- training label prediction");
                evaluatePerformances();
                logln("--- --- Time. topic: " + topicTime + ". lambda: " + lambdaTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n");
            }

            if (debug) {
                validate("iter " + iter);
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
                        HyperparameterFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Sample topic assignment for a token
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe Whether the response variable of this document is observed
     * @return Elapsed time
     */
    private long sampleZ(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        long sTime = System.currentTimeMillis();
        double totalBeta = V * hyperparams.get(BETA);
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (removeFromModel) {
                    topicWords[z[ii][nn]].decrement(words[dd][nn]);
                }
                if (removeFromData) {
                    docTopics[ii].decrement(z[ii][nn]);
                    docLabelDotProds[ii] -= lambdas[z[ii][nn]] / words[dd].length;
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k] = Math.log(docTopics[ii].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log((topicWords[k].getCount(words[dd][nn]) + hyperparams.get(BETA))
                                    / (topicWords[k].getCountSum() + totalBeta));
                    if (observe) {
                        double dotProd = docLabelDotProds[ii]
                                + lambdas[z[ii][nn]] / words[dd].length;
                        logprobs[k] += getLabelLogLikelihood(labels[ii], dotProd);
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

                if (z[ii][nn] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[ii][nn] = sampledZ;

                if (addToModel) {
                    topicWords[z[ii][nn]].increment(words[dd][nn]);
                }
                if (addToData) {
                    docTopics[ii].increment(z[ii][nn]);
                    docLabelDotProds[ii] += lambdas[z[ii][nn]] / words[dd].length;
                }
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Update parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    private long updateLambdas() {
        long sTime = System.currentTimeMillis();
        if (lambdas == null) {
            this.lambdas = new double[K];
            for (int k = 0; k < K; k++) {
                this.lambdas[k] = SamplerUtils.getGaussian(mean, sigma);
            }
        }

        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            designMatrix[ii] = new SparseVector(K);
            for (int k : docTopics[ii].getSparseCounts().getIndices()) {
                double val = (double) docTopics[ii].getCount(k) / z[ii].length;
                designMatrix[ii].change(k, val);
            }
        }

        RidgeLogisticRegressionOptimizable optimizable = new RidgeLogisticRegressionOptimizable(
                labels, lambdas, designMatrix, mean, sigma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (isReporting()) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int k = 0; k < K; k++) {
            lambdas[k] = optimizable.getParameter(k);
        }

        // update current predictions
        this.docLabelDotProds = new double[D];
        for (int ii = 0; ii < D; ii++) {
            docLabelDotProds[ii] = designMatrix[ii].dotProduct(lambdas);
        }

        return System.currentTimeMillis() - sTime;
    }

    /**
     * Compute the current prediction values.
     */
    private double[] computePredictionValues() {
        double[] predResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double expDotProd = Math.exp(docLabelDotProds[d]);
            double docPred = expDotProd / (expDotProd + 1);
            predResponses[d] = docPred;
        }
        return predResponses;
    }

    private void evaluatePerformances() {
        double[] predVals = computePredictionValues();
        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<RankingItem<Integer>>();
        for (int d = 0; d < D; d++) {
            rankDocs.add(new RankingItem<Integer>(d, predVals[d]));
        }
        Collections.sort(rankDocs);
        int[] preds = new int[D];
        for (int ii = 0; ii < this.positives.size(); ii++) {
            int d = rankDocs.get(ii).getObject();
            preds[d] = POSITVE;
        }

        ClassificationEvaluation eval = new ClassificationEvaluation(labels, preds);
        eval.computePRF1();
        for (Measurement measurement : eval.getMeasurements()) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }

        RankingEvaluation rankEval = new RankingEvaluation(predVals, positives);
        rankEval.computeAUCs();
        for (Measurement measurement : rankEval.getMeasurements()) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
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

        double labelLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();

            // label
            double dotProd = StatUtils.dotProduct(lambdas, empDist);
            labelLlh += getLabelLogLikelihood(labels[d], dotProd);
        }

        double lambdaLlh = 0.0;
        for (int k = 0; k < K; k++) {
            lambdaLlh += StatUtils.logNormalProbability(
                    lambdas[k], mean, Math.sqrt(sigma));
        }

        double llh = wordLlh + topicLlh + labelLlh + lambdaLlh;
        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". label: " + MiscUtils.formatDouble(labelLlh)
                    + ". lambdaLlh: " + MiscUtils.formatDouble(labelLlh)
                    + ". llh = " + MiscUtils.formatDouble(llh));
        }

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
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
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
            sortedTopics.add(new RankingItem<Integer>(k, lambdas[k]));
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
                        + ", " + MiscUtils.formatDouble(lambdas[k])
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
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
            for (String filename : filenames) {
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
     * @throws java.lang.Exception
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
        labels = null; // for evaluation
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
            for (DirMult topicWord : topicWords) {
                topicWordCount += topicWord.getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // initialize assignments
        sampleZ(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);

        if (verbose) {
            logln("After initialization");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (DirMult topicWord : topicWords) {
                topicWordCount += topicWord.getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.length + ". " + topicWordCount);
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < this.testMaxIter; iter++) {
            sampleZ(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }

                // update current dot products
                this.docLabelDotProds = new double[D];
                for (int d = 0; d < D; d++) {
                    double[] empDist = docTopics[d].getEmpiricalDistribution();
                    for (int k = 0; k < K; k++) {
                        this.docLabelDotProds[d] += lambdas[k] * empDist[k];
                    }
                }

                // compute prediction values
                double[] predResponses = computePredictionValues();
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
            for (DirMult topicWord : topicWords) {
                topicWordCount += topicWord.getCountSum();
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

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + BinarySLDA.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        return "java -cp \"dist/segan.jar:lib/*\" sampler.supervised.classification.BinarySLDA "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-binary/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-binary/amazon-data.dat "
                + "--info-file demo/amazon-data/format-binary/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-supervised "
                + "--burnIn 100 "
                + "--maxIter 250 "
                + "--sampleLag 30 "
                + "--report 5 "
                + "--K 50 "
                + "--alpha 0.1 "
                + "--beta 0.1 "
                + "--sigma 1.0 "
                + "--mu 0.0 "
                + "--init random "
                + "-v -d";
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
        addOption("mu", "Mu");
        addOption("sigma", "Sigma");
        addOption("K", "Number of topics");
        addOption("num-top-words", "Number of top words per topic");

        options.addOption("train", false, "Train");
        options.addOption("test", false, "Test");

        // configurations
        addOption("init", "Initialization");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
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

        LabelTextDataset data = new LabelTextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile),
                new File(docInfoFile),
                null);
        int V = data.getWordVocab().size();

        BinarySLDA sampler = new BinarySLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(outputFolder, V, K,
                alpha, beta, mu, sigma,
                initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        ArrayList<Integer> selectedDocIndices = null;
        if (cmd.hasOption("selected-docs-file")) {
            String selectedDocFile = cmd.getOptionValue("selected-docs-file");
            selectedDocIndices = new ArrayList<>();
            BufferedReader reader = IOUtils.getBufferedReader(selectedDocFile);
            String line;
            while ((line = reader.readLine()) != null) {
                int docIdx = Integer.parseInt(line);
                if (docIdx >= data.getDocIds().length) {
                    throw new RuntimeException("Out of bound. Doc index " + docIdx);
                }
                selectedDocIndices.add(Integer.parseInt(line));
            }
            reader.close();
        }

        double[][] priorTopics = null;
        if (cmd.hasOption("prior-topic-file")) {
            String priorTopicFile = cmd.getOptionValue("prior-topic-file");
            priorTopics = IOUtils.input2DArray(new File(priorTopicFile));
        }

        if (cmd.hasOption("train")) {
            System.out.println("here");
            sampler.train(data.getWords(), selectedDocIndices, data.getSingleLabels());
            sampler.initialize(priorTopics);
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

            double[] predictions;
            /*predictions = sampler.test(data.getWords(), selectedDocIndices,
             sampler.getFinalStateFile(), null);

             // output predictions and results
             int numDocs = selectedDocIndices.size();
             String[] selectedIds = new String[numDocs];
             double[] responses = new double[numDocs];
             for (int q = 0; q < numDocs; q++) {
             int index = selectedDocIndices.get(q);
             selectedIds[q] = data.getDocIds()[index];
             responses[q] = docResponses[index];
             }
	       

             PredictionUtils.outputRegressionPredictions(
             new File(predictionFolder,
             AbstractExperiment.PREDICTION_FILE),
             selectedIds, responses, predictions);
             PredictionUtils.outputRegressionResults(
             new File(evaluationFolder,
             AbstractExperiment.RESULT_FILE), responses,
             predictions);*/
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

    BinarySLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public SLDATestRunner(BinarySLDA sampler,
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
        BinarySLDA testSampler = new BinarySLDA();
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

//class L2NormLogLinearObjective implements Optimizable.ByGradientValue {
//
//    private double[] parameters;
//    private double[][] designMatrix;
//    private int[] labels; // 1 or 0
//    private double paramMean;
//    private double paramVar;
//    private int D;
//
//    public L2NormLogLinearObjective(
//            double[] curParams,
//            double[][] designMatrix,
//            int[] labels,
//            double paramMean, double paramVar) {
//        this.parameters = new double[curParams.length];
//        System.arraycopy(curParams, 0, this.parameters, 0, curParams.length);
//
//        this.designMatrix = designMatrix;
//        this.labels = labels;
//        this.paramMean = paramMean;
//        this.paramVar = paramVar;
//
//        this.D = designMatrix.length;
//    }
//
//    @Override
//    public double getValue() {
//        double value = 0;
//        // log likelihood
//        for (int d = 0; d < D; d++) {
//            double dotProb = 0.0;
//            for (int k = 0; k < getNumParameters(); k++) {
//                dotProb += this.parameters[k] * this.designMatrix[d][k];
//            }
//
//            value += labels[d] * dotProb;
//            value -= Math.log(Math.exp(dotProb) + 1);
//        }
//
//        // regularizer
//        for (int k = 0; k < getNumParameters(); k++) {
//            double diff = this.parameters[k] - this.paramMean;
//            value -= (diff * diff) / (2 * this.paramVar);
//        }
//        return value;
//    }
//
//    @Override
//    public void getValueGradient(double[] gradient) {
//        double[] tempGrad = new double[getNumParameters()];
//
//        // likelihood
//        for (int d = 0; d < D; d++) {
//            double dotprod = 0.0;
//            for (int k = 0; k < getNumParameters(); k++) {
//                dotprod += parameters[k] * designMatrix[d][k];
//            }
//            double expDotprod = Math.exp(dotprod);
//            double pred = expDotprod / (expDotprod + 1);
//
//            for (int k = 0; k < getNumParameters(); k++) {
//                tempGrad[k] += (labels[d] - pred) * designMatrix[d][k];
//            }
//        }
//
//        // regularizer
//        for (int k = 0; k < getNumParameters(); k++) {
//            tempGrad[k] -= (parameters[k] - paramMean) / paramVar;
//        }
//        System.arraycopy(tempGrad, 0, gradient, 0, getNumParameters());
//    }
//
//    @Override
//    public int getNumParameters() {
//        return this.parameters.length;
//    }
//
//    @Override
//    public double getParameter(int i) {
//        return parameters[i];
//    }
//
//    @Override
//    public void getParameters(double[] buffer) {
//        assert (buffer.length == parameters.length);
//        System.arraycopy(parameters, 0, buffer, 0, buffer.length);
//    }
//
//    @Override
//    public void setParameter(int i, double r) {
//        this.parameters[i] = r;
//    }
//
//    @Override
//    public void setParameters(double[] newParameters) {
//        assert (newParameters.length == parameters.length);
//        System.arraycopy(newParameters, 0, parameters, 0, parameters.length);
//    }
//}
