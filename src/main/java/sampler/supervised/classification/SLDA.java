package sampler.supervised.classification;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import sampler.LDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class SLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected double sigma;
    protected double gamma;
    // inputs
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // D x L_d
    protected SparseCount[] docLabels; // D x L (sparse)
    protected int K; // number of topics
    protected int L; // number of labels
    protected int V; // vocab size
    protected int D; // number of documents
    // latent
    protected int[][] z;
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected double[][] v;         // L x K
    protected double[][] scores;    // D x L
    // internal
    private int numLabels;
    // info
    protected ArrayList<String> labelVocab;

    public void setLabelVocab(ArrayList<String> lVoc) {
        this.labelVocab = lVoc;
    }

    public void configure(SLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.L,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.sigma,
                sampler.gamma,
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V, int K, int L,
            double alpha, double beta,
            double sigma,
            double gamma,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.K = K;
        this.L = L;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.sigma = sigma;
        this.gamma = gamma;

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
            logln("--- num labels:\t" + L);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
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
                .append("_SLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_s-").append(sigma)
                .append("_g-").append(gamma)
                .append("-");
        for (double hp : this.hyperparams) {
            str.append(hp).append("-");
        }
        str.append("opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void train(int[][] ws, int[][] labels) {
        this.words = ws;
        this.D = this.words.length;
        this.labels = labels;
        this.docLabels = new SparseCount[D];
        for (int d = 0; d < D; d++) {
            this.docLabels[d] = new SparseCount();
            for (int ii = 0; ii < labels[d].length; ii++) {
                this.docLabels[d].increment(labels[d][ii]);
            }
        }

        // statistics
        this.numTokens = 0;
        this.numLabels = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
            this.numLabels += labels[d].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- # labels:\t" + numLabels);
        }
    }

    public void test(int[][] ws,
            File iterPredFolder) {
        if (verbose) {
            logln("Test sampling ...");
        }

        // start testing
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
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
                        ws,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    public void updateScores() {
        for (int d = 0; d < D; d++) {
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < K; k++) {
                    this.scores[d][l] += v[l][k] * docTopics[d].getCount(k) / words[d].length;
                }
            }
        }
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

//        updateVs();
        updateManualVs();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. " + getCurrentState());
            getLogLikelihood();
        }
    }

    protected void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }

        v = new double[L][K];
        for (int ll = 0; ll < L; ll++) {
            for (int kk = 0; kk < K; kk++) {
                v[ll][kk] = SamplerUtils.getGaussian(0.0, sigma);
            }
        }
    }

    protected void initializeDataStructure() {
        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        scores = new double[D][L];
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
        updateScores();
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

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
            long sampleTime = sampleZs(REMOVE, ADD, REMOVE, ADD);
//            long updateTime = updateVs();
            long updateTime = updateManualVs();

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter
                        + "\tllh = " + loglikelihood
                        + "\tsample-time = " + sampleTime
                        + "\tupdate-time = " + updateTime
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

            if (verbose && iter % REP_INTERVAL == 0) {
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

    private long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        long sTime = System.currentTimeMillis();
        double totalBeta = V * hyperparams.get(BETA);
        numTokensChanged = 0;

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    for (int l = 0; l < L; l++) {
                        scores[d][l] -= v[l][z[d][n]] / words[d].length;
                    }
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k] = Math.log(docTopics[d].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log((topicWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (topicWords[k].getCountSum() + totalBeta));

                    if (labels != null) {
                        double totalScore = 0.0;
                        double[] updatedScores = new double[L];
                        for (int l = 0; l < L; l++) {
                            updatedScores[l] = scores[d][l] + v[l][k] / words[d].length;
                            totalScore = SamplerUtils.logAdd(totalScore, updatedScores[l]);
                        }
                        for (int ll : labels[d]) {
                            logprobs[k] += updatedScores[ll] - totalScore;
                        }
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

                // debug
                if (sampledZ == K) {
                    for (int k = 0; k < K; k++) {
                        double labelLlh = 1.0;
                        double totalScore = 0.0;
                        double[] updatedScores = new double[L];
                        for (int l = 0; l < L; l++) {
                            updatedScores[l] = Math.exp(scores[d][l] + v[l][k] / words[d].length);
                            totalScore += updatedScores[l];

                            System.out.println("l = " + l
                                    + "\t" + scores[d][l]
                                    + "\t" + v[l][k]
                                    + "\t" + updatedScores[l]
                                    + "\t" + totalScore);
                        }

                        for (int ll : labels[d]) {
                            labelLlh *= updatedScores[ll] / totalScore;
                        }
                        logln("iter = " + iter
                                + ". d = " + d
                                + ". n = " + n
                                + ". " + MiscUtils.formatDouble((docTopics[d].getCount(k) + hyperparams.get(ALPHA)))
                                + ". " + MiscUtils.formatDouble((topicWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                        / (topicWords[k].getCountSum() + totalBeta))
                                + ". " + (labelLlh)
                                + ". " + logprobs[k]);
                    }
                }

                if (sampledZ != z[d][n]) {
                    numTokensChanged++;
                }

                z[d][n] = sampledZ;
                if (addToModel) {
                    topicWords[z[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docTopics[d].increment(z[d][n]);
                    for (int l = 0; l < L; l++) {
                        scores[d][l] += v[l][z[d][n]] / words[d].length;
                    }
                }
            }
        }

        return System.currentTimeMillis() - sTime;
    }

    private long updateVs() {
        long sTime = System.currentTimeMillis();

        // debug
        printLabelLogLikelihood();

        Objective obj = new Objective(v);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(obj);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (verbose) {
            logln("--- converged: " + converged);
        }

        int count = 0;
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                v[l][k] = obj.getParameter(count++);
            }
        }

        updateScores();

        // debug
        printLabelLogLikelihood();

        return System.currentTimeMillis() - sTime;
    }

    private long updateManualVs() {
        long sTime = System.currentTimeMillis();

        // debug
//        printLabelLogLikelihood();
        // precompute
        SparseVector[] expDocDotProds = new SparseVector[D];
        double[] docNorms = new double[D];
        for (int d = 0; d < D; d++) {
            double[] expDocDPs = new double[L];
            for (int l = 0; l < L; l++) {
                for (int k : docTopics[d].getSparseCounts().getIndices()) {
                    expDocDPs[l] += v[l][k] * docTopics[d].getCount(k) / words[d].length;
                }
                expDocDPs[l] = Math.exp(expDocDPs[l]);
            }
            docNorms[d] = StatUtils.sum(expDocDPs);

            // sparse vector
            expDocDotProds[d] = new SparseVector();
            for (int l : docLabels[d].getIndices()) {
                expDocDotProds[d].set(l, expDocDPs[l]);
            }
        }

        // likelihood
        double[][] grads = new double[L][K];
        for (int d = 0; d < D; d++) {
            for (int l : docLabels[d].getIndices()) {
                for (int k = 0; k < K; k++) {
                    grads[l][k] += docLabels[d].getCount(l)
                            * (docTopics[d].getCount(k) / words[d].length)
                            * (1 - expDocDotProds[d].get(l) / docNorms[d]);
                }
            }
        }

        // prior
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                grads[l][k] -= v[l][k] / sigma;
            }
        }

        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                v[l][k] += gamma * grads[l][k];
            }
        }

        updateScores();

        // debug
//        printLabelLogLikelihood();
        return System.currentTimeMillis() - sTime;
    }

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
        sampleZs(!REMOVE, !ADD, !REMOVE, ADD);

        updateScores();

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

        // sample an store predictions
        double[][] predictedScores = new double[D][L];
        int count = 0;
        for (iter = 0; iter < testMaxIter; iter++) {
            sampleZs(!REMOVE, !ADD, REMOVE, ADD);
            updateScores();

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }
                for (int dd = 0; dd < D; dd++) {
                    for (int ll = 0; ll < L; ll++) {
                        predictedScores[dd][ll] += scores[dd][ll];
                    }
                }
                count++;
            }
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        for (int dd = 0; dd < D; dd++) {
            for (int ll = 0; ll < L; ll++) {
                predictedScores[dd][ll] /= count;
            }
        }
        PredictionUtils.outputSingleModelClassifications(
                new File(outputResultFile), predictedScores);
    }

    private void printLabelLogLikelihood() {
        double labelLlh = 0.0;
        for (int d = 0; d < D; d++) {
            double[] docDotProds = new double[L];
            double docTotal = 0.0;
            for (int l = 0; l < L; l++) {
                for (int k : docTopics[d].getSparseCounts().getIndices()) {
                    docDotProds[l] += v[l][k] * docTopics[d].getCount(k) / words[d].length;
                }
                docTotal = SamplerUtils.logAdd(docTotal, docDotProds[l]);
            }

            for (int l : docLabels[d].getIndices()) {
                labelLlh += docLabels[d].getCount(l) * (docDotProds[l] - docTotal);
            }
        }

        double paramPrior = 0.0;
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                paramPrior -= v[l][k] * v[l][k];
            }
        }
        paramPrior /= 2 * sigma;

        logln("label = " + MiscUtils.formatDouble(labelLlh)
                + ". param = " + MiscUtils.formatDouble(paramPrior));
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
            double[] docDotProds = new double[L];
            double docTotal = 0.0;
            for (int l = 0; l < L; l++) {
                for (int k : docTopics[d].getSparseCounts().getIndices()) {
                    docDotProds[l] += v[l][k] * docTopics[d].getCount(k) / words[d].length;
                }
                docTotal = SamplerUtils.logAdd(docTotal, docDotProds[l]);
            }

            for (int l : docLabels[d].getIndices()) {
                labelLlh += docLabels[d].getCount(l) * (docDotProds[l] - docTotal);
            }
        }

        double paramPrior = 0.0;
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                paramPrior -= v[l][k] * v[l][k];
            }
        }
        paramPrior /= 2 * sigma;

        logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                + ". topic: " + MiscUtils.formatDouble(topicLlh)
                + ". label: " + MiscUtils.formatDouble(labelLlh)
                + ". param: " + MiscUtils.formatDouble(paramPrior));
        double llh = wordLlh
                + topicLlh
                + labelLlh
                + paramPrior;
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
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }

            for (int l = 0; l < L; l++) {
                modelStr.append(l).append("\n");
                for (int k = 0; k < K; k++) {
                    modelStr.append(v[l][k]).append("\t");
                }
                modelStr.append("\n");
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
                topicWords[k] = DirMult.input(reader.readLine());
            }

            for (int l = 0; l < L; l++) {
                int labelIdx = Integer.parseInt(reader.readLine());
                if (labelIdx != l) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                String[] sline = reader.readLine().split("\t");
                for (int k = 0; k < K; k++) {
                    v[l][k] = Double.parseDouble(sline[k]);
                }
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

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int k = 0; k < K; k++) {
                double[] distrs = topicWords[k].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + k
                        + ", " + topicWords[k].getCountSum()
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

    public static void parallelTest(int[][] newWords, File iterPredFolder, SLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                ClassSLDATestRunner runner = new ClassSLDATestRunner(sampler,
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

    class Objective implements Optimizable.ByGradientValue {

        private final double[] parameters;

        public Objective(double[][] curParams) {
            parameters = new double[L * K];
            int count = 0;
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < K; k++) {
                    parameters[count++] = curParams[l][k];
                }
            }
        }

        private double getParameter(int l, int k) {
            return parameters[l * K + k];
        }

        @Override
        public double getValue() {
            double llh = 0.0;
            for (int d = 0; d < D; d++) {
                double[] docDotProds = new double[L];
                double docTotal = 0.0;
                for (int l = 0; l < L; l++) {
                    for (int k : docTopics[d].getSparseCounts().getIndices()) {
                        docDotProds[l] += getParameter(l, k) * docTopics[d].getCount(k) / words[d].length;
                    }
                    docTotal = SamplerUtils.logAdd(docTotal, docDotProds[l]);
                }

                for (int l : docLabels[d].getIndices()) {
                    llh += docLabels[d].getCount(l) * (docDotProds[l] - docTotal);
                }
            }

            double prior = 0.0;
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < K; k++) {
                    double vv = getParameter(l, k);
                    prior -= vv * vv;
                }
            }
            prior /= 2 * sigma;

            // debug
            System.out.println("llh = " + llh + ". prior = " + prior);

            return prior + llh;
        }

        @Override
        public void getValueGradient(double[] gradient) {
            // precompute
            SparseVector[] expDocDotProds = new SparseVector[D];
            double[] docNorms = new double[D];
            for (int d = 0; d < D; d++) {
                double[] expDocDPs = new double[L];
                for (int l = 0; l < L; l++) {
                    for (int k : docTopics[d].getSparseCounts().getIndices()) {
                        expDocDPs[l] += getParameter(l, k) * docTopics[d].getCount(k) / words[d].length;
                    }
                    expDocDPs[l] = Math.exp(expDocDPs[l]);
                }
                docNorms[d] = StatUtils.sum(expDocDPs);

                // sparse vector
                expDocDotProds[d] = new SparseVector();
                for (int l : docLabels[d].getIndices()) {
                    expDocDotProds[d].set(l, expDocDPs[l]);
                }
            }

            // likelihood
            double[][] grads = new double[L][K];
            for (int d = 0; d < D; d++) {
                for (int l : docLabels[d].getIndices()) {
                    for (int k = 0; k < K; k++) {
                        grads[l][k] += docLabels[d].getCount(l)
                                * (docTopics[d].getCount(k) / words[d].length)
                                * (1 - expDocDotProds[d].get(l) / docNorms[d]);
                    }
                }
            }

            // prior
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < K; k++) {
                    grads[l][k] -= getParameter(l, k) / sigma;
                }
            }

            // update
            int count = 0;
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < K; k++) {
                    gradient[count++] = grads[l][k];
                }
            }
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
}

class ClassSLDATestRunner implements Runnable {

    SLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public ClassSLDATestRunner(SLDA sampler,
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
