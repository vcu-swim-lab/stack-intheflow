package sampler.supervised.regression.shdp;

import cc.mallet.types.Dirichlet;
import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import optimization.GurobiMLRL2Norm;
import regression.Regressor;
import sampler.LDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 * Implementation of supervised hierarchical Dirichlet process using direct
 * assignment Gibbs sampler.
 *
 * @author vietan
 */
public class SHDP extends AbstractSampler implements Regressor<ResponseTextDataset> {

    public static final int NEW_COMPONENT_INDEX = -1;
    public static final int ALPHA_GLOBAL = 0;
    public static final int ALPHA_LOCAL = 1;
    public static final int BETA = 2;
    public static final int MU = 3;
    public static final int SIGMA = 4;
    public static final int RHO = 5;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K; // initial number of tables
    protected int[][] words;
    protected double[] responses;
    private SparseCount[] docTopics;
    private SBP<Topic> topicWords;
    protected double[] docRegressMeans;
    private int[][] z;
    private int numTokensChange;
    private SparseVector sbpWeights;

    public void configure(SHDP sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.hyperparams.get(ALPHA_GLOBAL),
                sampler.hyperparams.get(ALPHA_LOCAL),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(MU),
                sampler.hyperparams.get(SIGMA),
                sampler.hyperparams.get(RHO),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V,
            double alpha_global, double alpha_local, double beta,
            double mu, double sigma, double rho,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_global);
        this.hyperparams.add(alpha_local);
        this.hyperparams.add(beta);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma);
        this.hyperparams.add(rho);

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

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- alpha-global:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_GLOBAL)));
            logln("--- alpha-local:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_LOCAL)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(hyperparams.get(MU)));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(hyperparams.get(RHO)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    public void setK(int K) {
        this.K = K;
    }

    @Override
    public String getName() {
        return this.name;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_SHDP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_ag-").append(formatter.format(hyperparams.get(ALPHA_GLOBAL)))
                .append("_al-").append(formatter.format(hyperparams.get(ALPHA_LOCAL)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public double[] getPredictedResponses() {
        return this.docRegressMeans;
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

    @Override
    public void train(ResponseTextDataset trainData) {
        train(trainData.getWords(), trainData.getResponses());
    }

    @Override
    public void test(ResponseTextDataset testData) {
        test(testData.getWords(), new File(getSamplerFolderPath(), IterPredictionFolder));
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

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        optimizeRegressionParameters();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- --- Done initializing. \n" + getCurrentState());
        }
    }

    protected void initializeModelStructure() {
        this.topicWords = new SBP();

        this.sbpWeights = new SparseVector();
        this.sbpWeights.set(NEW_COMPONENT_INDEX, 1.0);
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int dd = 0; dd < D; dd++) {
            z[dd] = new int[words[dd].length];
        }

        docTopics = new SparseCount[D];
        for (int dd = 0; dd < D; dd++) {
            docTopics[dd] = new SparseCount();
        }

        docRegressMeans = new double[D];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializePresetAssignments();
                break;
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, !REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
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
        if (K == 0) {// this is not set
            K = 50;
        }
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(folder, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaFile = new File(lda.getSamplerFolderPath(), "model.zip");
            if (ldaFile.exists()) {
                logln("--- Loading LDA from " + ldaFile);
                lda.inputState(ldaFile);
            } else {
                logln("--- LDA file not found " + ldaFile + ". Sampling LDA ...");
                lda.initialize();
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
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
        if (verbose) {
            logln("--- LDA loaded. Start initializing assingments ...");
        }

        for (int kk = 0; kk < K; kk++) {
            DirMult topicWord = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            double regParam = SamplerUtils.getGaussian(hyperparams.get(MU),
                    hyperparams.get(SIGMA));
            Topic topic = new Topic(topicWord, regParam);
            topicWords.createNewComponent(kk, topic);
        }

        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords.getComponent(z[dd][nn]).topic.increment(words[dd][nn]);
            }
        }

        // initialize tau
        double mean = 1.0 / (K + hyperparams.get(ALPHA_GLOBAL));
        for (int kk = 0; kk < K; kk++) {
            this.sbpWeights.set(kk, mean);
        }
        this.sbpWeights.set(NEW_COMPONENT_INDEX,
                hyperparams.get(ALPHA_GLOBAL) / (K + hyperparams.get(ALPHA_GLOBAL)));

        updateSBPWeights();
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
            numTokensChange = 0;

            this.iterate(iter);

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
                String str = "Iter " + iter + "\t llh = " + loglikelihood
                        + ". # token changed: " + numTokensChange
                        + ". change ratio: " + (double) numTokensChange / numTokens
                        + "\n" + getCurrentState();
                if (iter <= BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
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

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }

            if (debug) {
                validate("iter " + iter);
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
            throw new RuntimeException("Exception iter = " + iter);
        }
    }

    public void iterate(int iteration) {
        // sample topic assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
            }
        }

        updateSBPWeights();

        // update the regression parameters
        int step = (int) Math.log(iter + 1) + 1;
        if (iter % step == 0) {
            optimizeRegressionParameters();
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            evaluateRegressPrediction(responses, docRegressMeans);
        }
    }

    private void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observed) {
        int curZ = z[d][n];
        if (removeFromData) {
            this.docTopics[d].decrement(curZ);
            this.docRegressMeans[d] -= topicWords.getComponent(curZ).regParam / words[d].length;
        }

        if (removeFromModel) {
            this.topicWords.getComponent(curZ).topic.decrement(words[d][n]);
            if (this.topicWords.getComponent(curZ).topic.isEmpty()) {
                topicWords.removeComponent(curZ);
                sbpWeights.remove(curZ);
                updateSBPWeights();
            }
        }

        ArrayList<Integer> indices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        for (int k : topicWords.getIndices()) {
            indices.add(k);
            double logprior = Math.log(docTopics[d].getCount(k) + hyperparams.get(ALPHA_LOCAL) * sbpWeights.get(k));
            double loglh = topicWords.getComponent(k).topic.getLogLikelihood(words[d][n]);
            double lp = logprior + loglh;
            if (observed) {
                double mean = docRegressMeans[d]
                        + topicWords.getComponent(k).regParam / (words[d].length);
                lp += StatUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));
            }
            logprobs.add(lp);
        }

        if (addToModel) { // for test time
            indices.add(NEW_COMPONENT_INDEX);
            double logprior = Math.log(hyperparams.get(ALPHA_LOCAL) * sbpWeights.get(NEW_COMPONENT_INDEX));
            double loglh = Math.log(1.0 / V);
            double lp = logprior + loglh;
            if (observed) {
                double mean = docRegressMeans[d] + hyperparams.get(MU) / (words[d].length);
                double var = hyperparams.get(SIGMA)
                        / (words[d].length * words[d].length) + hyperparams.get(RHO);
                double resLlh = StatUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
                lp += resLlh;
            }
            logprobs.add(lp);
        }

        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        int newZ = indices.get(sampledIdx);

        if (curZ != newZ) {
            numTokensChange++;
        }

        boolean newTopic = false;
        if (newZ == NEW_COMPONENT_INDEX) {
            newZ = topicWords.getNextIndex();
            DirMult topicWord = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            double regParam = SamplerUtils.getGaussian(hyperparams.get(MU),
                    hyperparams.get(SIGMA));
            Topic topic = new Topic(topicWord, regParam);
            topicWords.createNewComponent(newZ, topic);
            sbpWeights.set(newZ, 0.0);
            newTopic = true;
        }
        z[d][n] = newZ;

        if (addToData) {
            this.docTopics[d].increment(newZ);
            this.docRegressMeans[d] += topicWords.getComponent(newZ).regParam / words[d].length;
        }

        if (addToModel) {
            this.topicWords.getComponent(newZ).topic.increment(words[d][n]);
        }

        if (newTopic) { // if new topic is created, update SBP weights
            updateSBPWeights();
        }
    }

    private void updateSBPWeights() {
        if (sbpWeights.size() != topicWords.getNumComponents() + 1) {
            throw new RuntimeException("Mismatch: " + sbpWeights.size()
                    + " vs. " + topicWords.getNumComponents());
        }

        SparseCount counts = new SparseCount();
        for (int k : topicWords.getIndices()) {
            for (int dd = 0; dd < D; dd++) {
                int count = docTopics[dd].getCount(k);
                if (count > 1) {
                    int c = SamplerUtils.randAntoniak(
                            hyperparams.get(ALPHA_LOCAL) * sbpWeights.get(k),
                            count);
                    counts.changeCount(k, c);
                } else {
                    counts.changeCount(k, count);
                }
            }
        }
        double[] dirPrior = new double[topicWords.getNumComponents() + 1];
        ArrayList<Integer> indices = new ArrayList<Integer>();

        int idx = 0;
        for (int kk : topicWords.getIndices()) {
            indices.add(kk);
            dirPrior[idx++] = counts.getCount(kk);
        }

        indices.add(NEW_COMPONENT_INDEX);
        dirPrior[idx] = hyperparams.get(ALPHA_GLOBAL);

        Dirichlet dir = new Dirichlet(dirPrior);
        double[] wts = dir.nextDistribution();
        this.sbpWeights = new SparseVector();
        for (int ii = 0; ii < wts.length; ii++) {
            this.sbpWeights.set(indices.get(ii), wts[ii]);
        }
    }

    private void optimizeRegressionParameters() {
        ArrayList<Integer> sortedIndices = topicWords.getSortedIndices();

        double[][] designMatrix = new double[D][topicWords.getNumComponents()];
        for (int d = 0; d < D; d++) {
            for (int k : docTopics[d].getIndices()) {
                int idx = sortedIndices.indexOf(k);
                designMatrix[d][idx] = (double) docTopics[d].getCount(k) / words[d].length;
            }
        }
        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));
        mlr.setMean(hyperparams.get(MU));
        mlr.setSigma(hyperparams.get(SIGMA));
        double[] weights = mlr.solve();

        for (int ii = 0; ii < weights.length; ii++) {
            int idx = sortedIndices.get(ii);
            topicWords.getComponent(idx).regParam = weights[ii];
        }
        updatePredictionValues();
    }

    protected void updatePredictionValues() {
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            double score = 0.0;
            for (int k : docTopics[d].getIndices()) {
                score += topicWords.getComponent(k).regParam
                        * (double) docTopics[d].getCount(k) / words[d].length;
                this.docRegressMeans[d] = score;
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double obsLlh = 0.0;
        for (int idx : topicWords.getIndices()) {
            obsLlh += topicWords.getComponent(idx).topic.getLogLikelihood();
        }

        double assignLp = 0.0;

//        for (SHDPDish dish : globalRestaurant.getTables()) {
//            obsLlh += dish.getContent().getLogLikelihood();
//        }
        double dishRegLlh = 0.0;
        for (int idx : topicWords.getIndices()) {
            dishRegLlh += StatUtils.logNormalProbability(
                    topicWords.getComponent(idx).regParam,
                    hyperparams.get(MU), Math.sqrt(hyperparams.get(SIGMA)));
        }

        double resLlh = 0.0;
        double[] regValues = getPredictedResponses();
        for (int d = 0; d < D; d++) {
            resLlh += StatUtils.logNormalProbability(responses[d],
                    regValues[d], Math.sqrt(hyperparams.get(RHO)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** obs llh: " + MiscUtils.formatDouble(obsLlh)
                    + ". res llh: " + MiscUtils.formatDouble(resLlh)
                    + ". assignments: " + MiscUtils.formatDouble(assignLp)
                    + ". global reg: " + MiscUtils.formatDouble(dishRegLlh));
        }

        double llh = obsLlh + dishRegLlh;
        return llh;
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
            if (docTopics[d].getCountSum() != words[d].length) {
                throw new RuntimeException(msg + ". Num tokens mismatch");
            }
        }
        for (int k : topicWords.getIndices()) {
            topicWords.getComponent(k).topic.validate(msg);
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append("# components: ").append(topicWords.getNumComponents())
                .append("\n");
        return str.toString();
    }

    protected String printDebug() {
        StringBuilder str = new StringBuilder();
        double totalWeight = 0.0;
        for (int k : topicWords.getSortedIndices()) {
            double[] distrs = topicWords.getComponent(k).topic.getDistribution();
            String[] topWords = getTopWords(distrs, 10);
            str.append(">>> ").append(k)
                    .append(": ").append(MiscUtils.formatDouble(sbpWeights.get(k)))
                    .append(", ").append(topicWords.getComponent(k).topic.getCountSum())
                    .append(", ").append(MiscUtils.formatDouble(topicWords.getComponent(k).regParam))
                    .append("\t");
            for (String w : topWords) {
                str.append(" ").append(w);
            }
            str.append("\n");
            totalWeight += sbpWeights.get(k);
        }
        str.append(">>>").append(NEW_COMPONENT_INDEX)
                .append(": ").append(MiscUtils.formatDouble(sbpWeights.get(NEW_COMPONENT_INDEX)))
                .append("\n");
        totalWeight += sbpWeights.get(NEW_COMPONENT_INDEX);
        str.append("# components: ").append(topicWords.getNumComponents())
                .append("\ttotal weights: ").append(totalWeight)
                .append("\n");
        return str.toString();
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
            for (int k : topicWords.getSortedIndices()) {
                modelStr.append(k).append("\n");
                Topic component = topicWords.getComponent(k);
                modelStr.append(sbpWeights.get(k)).append("\n");
                modelStr.append(component.regParam).append("\n");
                modelStr.append(DirMult.output(component.topic)).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
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
            this.topicWords = new SBP();
            this.sbpWeights = new SparseVector();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            String line;
            while ((line = reader.readLine()) != null) {
                int k = Integer.parseInt(line);
                double weight = Double.parseDouble(reader.readLine());
                this.sbpWeights.set(k, weight);

                double param = Double.parseDouble(reader.readLine());
                DirMult topicWord = DirMult.input(reader.readLine());
                Topic topic = new Topic(topicWord, param);
                this.topicWords.createNewComponent(k, topic);
            }
            this.topicWords.fillInactives();
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

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                    docTopics[d].increment(z[d][n]);
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
        for (int k : topicWords.getIndices()) {
            sortedTopics.add(new RankingItem<Integer>(k, topicWords.getComponent(k).regParam));
        }
        Collections.sort(sortedTopics);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int ii = 0; ii < sortedTopics.size(); ii++) {
                int k = sortedTopics.get(ii).getObject();
                Topic component = topicWords.getComponent(k);
                double[] distrs = component.topic.getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + k
                        + ", " + component.topic.getCountSum()
                        + ", " + MiscUtils.formatDouble(component.regParam)
                        + ", " + sbpWeights.get(k)
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
            for (int k : topicWords.getIndices()) {
                topicWordCount += topicWords.getComponent(k).topic.getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.getNumComponents() + ". " + topicWordCount);
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleZ(d, n, !REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            }
        }

        this.updatePredictionValues();

        if (verbose) {
            logln("After initialization");
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docTopics[d].getCountSum();
            }

            int topicWordCount = 0;
            for (int k : topicWords.getIndices()) {
                topicWordCount += topicWords.getComponent(k).topic.getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.getNumComponents() + ". " + topicWordCount);
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < this.testMaxIter; iter++) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, !REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
                }
            }

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
            for (int k : topicWords.getIndices()) {
                topicWordCount += topicWords.getComponent(k).topic.getCountSum();
            }

            logln("--- docTopics: " + docTopics.length + ". " + docTopicCount);
            logln("--- topicWords: " + topicWords.getNumComponents() + ". " + topicWordCount);
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        PredictionUtils.outputSingleModelRegressions(
                new File(outputResultFile),
                predResponsesList);
    }

    class Topic {

        DirMult topic;
        double regParam;

        public Topic(DirMult tw, double rp) {
            this.topic = tw;
            this.regParam = rp;
        }

        public void validate(String msg) {
            topic.validate(msg);
        }
    }

    class SBP<C> {

        HashMap<Integer, C> actives;
        private SortedSet<Integer> inactives;
        int totalCount;

        public SBP() {
            this.actives = new HashMap<Integer, C>();
            this.inactives = new TreeSet<Integer>();
        }

        public ArrayList<Integer> getSortedIndices() {
            ArrayList<Integer> sortedIndices = new ArrayList<Integer>();
            for (int ii : getIndices()) {
                sortedIndices.add(ii);
            }
            Collections.sort(sortedIndices);
            return sortedIndices;
        }

        public int getNumComponents() {
            return this.actives.size();
        }

        public Set<Integer> getIndices() {
            return this.actives.keySet();
        }

        public boolean isEmpty() {
            return actives.isEmpty();
        }

        public boolean isActive(int idx) {
            return this.actives.containsKey(idx);
        }

        public C getComponent(int idx) {
            return this.actives.get(idx);
        }

        public void removeComponent(int idx) {
            this.inactives.add(idx);
            this.actives.remove(idx);
        }

        public void createNewComponent(int idx, C c) {
            if (isActive(idx)) {
                throw new RuntimeException("Component " + idx + " exists");
            }
            if (inactives.contains(idx)) {
                inactives.remove(idx);
            }
            this.actives.put(idx, c);
        }

        public void fillInactives() {
            int maxTableIndex = -1;
            for (int idx : actives.keySet()) {
                if (idx > maxTableIndex) {
                    maxTableIndex = idx;
                }
            }
            this.inactives = new TreeSet<Integer>();
            for (int ii = 0; ii < maxTableIndex; ii++) {
                if (!isActive(ii)) {
                    inactives.add(ii);
                }
            }
        }

        public int getNextIndex() {
            int newIdx;
            if (this.inactives.isEmpty()) {
                newIdx = this.actives.size();
            } else {
                newIdx = this.inactives.first();
            }
            return newIdx;
        }
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     */
    public static void parallelTest(int[][] newWords, File iterPredFolder, SHDP sampler) {
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
                SHDPTestRunner runner = new SHDPTestRunner(sampler,
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
}

class SHDPTestRunner implements Runnable {

    SHDP sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public SHDPTestRunner(SHDP sampler,
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
        SHDP testSampler = new SHDP();
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
