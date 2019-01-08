package sampler.supervised.regression;

import cc.mallet.util.Randoms;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import optimization.GurobiMLRL2Norm;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import regression.Regressor;
import sampler.RLDA;
import sampler.RecursiveLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.TopicTreeNode;
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
 *
 * @author vietan
 */
public class SHLDA extends AbstractSampler
        implements Regressor<ResponseTextDataset> {

    public static Randoms randoms = new Randoms(1123581321);
    public static final String LEXICAL_REG_OVERTIME = "lexical-weights-overtime.txt";
    private static final STable NULL_TABLE = null;
    public static final int STAY = 0;
    public static final int PASS = 1;
    public static final Double WEIGHT_THRESHOLD = 10e-2;
    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    // hyperparameter indices
    public static final int ALPHA = 0;
    public static final int RHO = 1; // response variable variance
    public static final int TAU_MEAN = 2; // lexical regression parameter mean
    public static final int TAU_SIGMA = 3; // lexical regression parameter variance
    // hyperparameters
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;    // regression parameter means
    protected double[] sigmas; // regression parameter variances
    protected double[] pis;    // level distribution parameter
    // input data
    protected String[][] rawSentences;
    protected int[][][] words;  // [D] x [S_d] x [N_ds]: words
    protected int[][] docWords;
    protected double[] responses; // [D]: response variables of each author
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    // input statistics
    protected int sentCount;
    protected int tokenCount;
    // pre-computed hyperparameters
    protected double logAlpha;
    protected double sqrtRho;
    protected double[] sqrtSigmas;
    protected double[] logGammas;
    protected PathAssumption pathAssumption;
    // latent variables
    private DirMult[] docLevelDist;
    private STable[][] c; // path assigned to sentences
    private int[][][] z; // level assigned to tokens
    private double[] lexParams;
    private double[][] lexDesignMatrix;
    // state structure
    private SNode globalTreeRoot; // tree
    private Restaurant<STable, Integer, SNode>[] localRestaurants; // franchise
    // for regression
    protected double[] docValues;
    protected int[][][] sentLevelCounts;
    // auxiliary
    protected double[] uniform;
    protected int numSentAsntsChange;
    protected int numTableAsgnsChange;
    protected ArrayList<String> authorVocab;
    protected int[] initBranchFactor = new int[]{16, 3};
    private int numAccepts;
    private int numProposes;
    private String seededAssignmentFile;

    public void setInitialBranchingFactor(int[] bf) {
        this.initBranchFactor = bf;
    }

    public void setResponses(double[] responses) {
        this.responses = responses;
    }

    public void setRawSentences(String[][] rawSents) {
        this.rawSentences = rawSents;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public File getIterationPredictionFolder() {
        return new File(getSamplerFolderPath(), IterPredictionFolder);
    }

    public void configure(SHLDA sampler) {
        this.configure(sampler.folder, sampler.V, sampler.L,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(RHO),
                sampler.hyperparams.get(TAU_MEAN),
                sampler.hyperparams.get(TAU_SIGMA),
                sampler.betas,
                sampler.gammas,
                sampler.mus,
                sampler.sigmas,
                sampler.pis,
                sampler.initBranchFactor,
                sampler.initState,
                sampler.pathAssumption,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V, int L,
            double alpha,
            double rho,
            double tau_mean,
            double tau_scale,
            double[] betas,
            double[] gammas,
            double[] mus,
            double[] sigmas,
            double[] pis,
            int[] initBranchFactor,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.V = V;
        this.L = L;

        this.betas = betas;
        this.gammas = gammas;
        this.mus = mus;
        this.sigmas = sigmas;
        this.pis = pis;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(rho);
        this.hyperparams.add(tau_mean);
        this.hyperparams.add(tau_scale);
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }
        for (double gamma : gammas) {
            this.hyperparams.add(gamma);
        }
        for (double mu : mus) {
            this.hyperparams.add(mu);
        }
        for (double sigma : sigmas) {
            this.hyperparams.add(sigma);
        }
        for (double pi : pis) {
            this.hyperparams.add(pi);
        }

        if (initBranchFactor != null) {
            this.initBranchFactor = initBranchFactor;
        }

        this.updatePrecomputedHyperparameters();

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.pathAssumption = pathAssumption;
        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.setName();

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gammas must have length " + (this.L - 1)
                    + ". Current length = " + this.gammas.length);
        }
        if (this.mus.length != this.L) {
            throw new RuntimeException("Vector mus must have length " + this.L
                    + ". Current length = " + this.mus.length);
        }
        if (this.sigmas.length != this.L) {
            throw new RuntimeException("Vector sigmas must have length " + this.L
                    + ". Current length = " + this.sigmas.length);
        }
        if (this.pis.length != this.L) {
            throw new RuntimeException("Vector pis must have length " + this.L
                    + ". Current length = " + this.pis.length);
        }

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- V = " + V);
            logln("--- L = " + L);

            logln("--- folder\t" + folder);
            logln("--- max level:\t" + L);
            logln("--- alpha:\t" + hyperparams.get(ALPHA));
            logln("--- rho:\t" + hyperparams.get(RHO));
            logln("--- tau mean:\t" + hyperparams.get(TAU_MEAN));
            logln("--- tau scale:\t" + hyperparams.get(TAU_SIGMA));

            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- reg mus:\t" + MiscUtils.arrayToString(mus));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- pis:\t" + MiscUtils.arrayToString(pis));

            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.pathAssumption);
        }
    }

    private void updatePrecomputedHyperparameters() {
        logAlpha = Math.log(hyperparams.get(ALPHA));
        sqrtRho = Math.sqrt(hyperparams.get(RHO));
        sqrtSigmas = new double[sigmas.length];
        for (int i = 0; i < sqrtSigmas.length; i++) {
            sqrtSigmas[i] = Math.sqrt(sigmas[i]);
        }
        logGammas = new double[gammas.length];
        for (int i = 0; i < logGammas.length; i++) {
            logGammas[i] = Math.log(gammas[i]);
        }
    }

    @Override
    public String getName() {
        return this.name;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_SHLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG);
        for (double h : hyperparams) {
            str.append("-").append(MiscUtils.formatDouble(h));
        }
        str.append("_opt-").append(this.paramOptimized);
        str.append("_").append(this.paramOptimized);
        for (int ii = 0; ii < initBranchFactor.length; ii++) {
            str.append("-").append(initBranchFactor[ii]);
        }
        this.name = str.toString();
    }

    private void prepareDataStatistics() {
        // statistics
        sentCount = 0;
        tokenCount = 0;
        for (int d = 0; d < D; d++) {
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
            }
        }

        // document words
        docWords = new int[D][];
        for (int d = 0; d < D; d++) {
            int docLength = 0;
            for (int s = 0; s < words[d].length; s++) {
                docLength += words[d][s].length;
            }
            docWords[d] = new int[docLength];
            int count = 0;
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docWords[d][count++] = words[d][s][n];
                }
            }
        }
    }

    @Override
    public void train(ResponseTextDataset trainData) {
        train(trainData.getSentenceWords(),
                trainData.getResponses());
    }

    public void train(int[][][] ws, double[] rs) {
        this.words = ws;
        this.responses = rs;
        this.D = this.words.length;

        this.prepareDataStatistics();

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + tokenCount);
            logln("--- # sentences:\t" + sentCount);
            logln("--- response distributions:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
            int[] histogram = StatUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
    }

    @Override
    public void test(ResponseTextDataset testData) {
        test(testData.getSentenceWords(),
                new File(getSamplerFolderPath(), IterPredictionFolder));
    }

    public void test(int[][][] newWords, File iterPredFolder) {
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
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                sampleNewDocuments(
                        new File(reportFolder, filename), newWords,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    /**
     * Compute the regression sum from the topic tree for a set of tokens with
     * known level assignments given the path
     *
     * @param pathNode The leaf node of the given path
     * @param levelAssignments Array containing level assignments
     *
     * @return The regression sum
     */
    private double computeTopicWeightFullPath(SNode pathNode, int[] levelAssignments) {
        int[] levelCounts = new int[L];
        for (int n = 0; n < levelAssignments.length; n++) {
            int level = levelAssignments[n];
            levelCounts[level]++;
        }

        double regSum = 0.0;
        SNode[] path = getPathFromNode(pathNode);
        for (int lvl = 0; lvl < path.length; lvl++) {
            regSum += levelCounts[lvl] * path[lvl].getRegressionParameter();
        }
        return regSum;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    SNode[] getPathFromNode(SNode node) {
        SNode[] path = new SNode[node.getLevel() + 1];
        SNode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    private void updateAuthorValues() {
        docValues = new double[D];
        for (int d = 0; d < D; d++) {
            // topic
            double docTopicVal = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }

                docTopicVal += computeTopicWeightFullPath(c[d][s].getContent(), z[d][s]);
            }
            double tVal = docTopicVal / docWords[d].length;
            docValues[d] += tVal;

            // lexical
            double docLexicalVal = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docLexicalVal += lexParams[words[d][s][n]];
                }
            }
            double lVal = docLexicalVal / docWords[d].length;
            docValues[d] += lVal;
        }
    }

    private boolean isValidSentence(int d, int s) {
        return this.words[d][s].length > 0;
    }

    private void evaluateRegressPrediction(double[] trueVals, double[] predVals) {
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

    public double[] getRegressionValues() {
        double[] predVals = new double[D];
        for (int d = 0; d < D; d++) {
            predVals[d] = getPredictedAuthorResponse(d);
        }
        return predVals;
    }

    public double getPredictedAuthorResponse(int d) {
        return docValues[d];
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
//        sampleTopics();
        updateParameters();

        if (verbose) {
            logln("--- --- Done initializing.\n" + getCurrentState());
            logln(printGlobalTree());
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
            getLogLikelihood();
        }

        if (debug) {
            validate("Initialized");
            evaluateRegressPrediction(responses, getRegressionValues());
        }
    }

    /**
     * Initialize model structure.
     */
    protected void initializeModelStructure() {
        DirMult dmModel = new DirMult(V, betas[0] * V, uniform);
        double regParam = 0.0;
        this.globalTreeRoot = new SNode(iter, 0, 0, dmModel, regParam, null);
        this.lexParams = new double[V];
    }

    /**
     * Initialize data-specific structures.
     */
    protected void initializeDataStructure() {
        this.docLevelDist = new DirMult[D];
        for (int d = 0; d < D; d++) {
            this.docLevelDist[d] = new DirMult(pis);
        }

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<STable, Integer, SNode>();
        }

        this.sentLevelCounts = new int[D][][];
        for (int d = 0; d < D; d++) {
            this.sentLevelCounts[d] = new int[words[d].length][L];
        }

        this.c = new STable[D][];
        this.z = new int[D][][];
        for (int d = 0; d < D; d++) {
            c[d] = new STable[words[d].length];
            z[d] = new int[words[d].length][];
            for (int s = 0; s < words[d].length; s++) {
                z[d][s] = new int[words[d][s].length];
            }
        }

        // partial design matrix
        this.lexDesignMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    int w = words[d][s][n];
                    lexDesignMatrix[d][w]++;
                }
            }
            for (int v = 0; v < V; v++) {
                lexDesignMatrix[d][v] /= docWords[d].length;
            }
        }
        this.docValues = new double[D];
    }

    /**
     * Initialize assignments.
     */
    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                initializeRecursiveLDAAssignments();
                break;
            case SEEDED:
                if (this.seededAssignmentFile == null) {
                    throw new RuntimeException("Seeded assignment file is not "
                            + "initialized.");
                }
                initializeRecursiveLDAAssignmentsSeeded(seededAssignmentFile);
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    public void setSeededAssignmentFile(String f) {
        this.seededAssignmentFile = f;
    }

    protected void initializeRecursiveLDAAssignments() {
        int[][] seededAssignments = null;
        initializeRecursiveLDAAssignmentsSeeded(seededAssignments);
    }

    protected void initializeRecursiveLDAAssignmentsSeeded(String seededFile) {
        int[][] seededAssignments = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(seededFile);
            int numDocs = Integer.parseInt(reader.readLine());
            if (numDocs != D) {
                throw new RuntimeException("Number of documents is incorrect. "
                        + numDocs + " vs. " + D);
            }
            seededAssignments = new int[D][];
            for (int d = 0; d < D; d++) {
                String[] sline = reader.readLine().split(" ");
                seededAssignments[d] = new int[sline.length];
                for (int n = 0; n < sline.length; n++) {
                    seededAssignments[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            logln(">>> Seeded assignments cannot be loaded from " + seededFile);
        }
        initializeRecursiveLDAAssignmentsSeeded(seededAssignments);
    }

    private void initializeRecursiveLDAAssignmentsSeeded(int[][] seededAssignments) {
        if (verbose) {
            logln("--- Initializing assignments using recursive LDA ...");
        }
        RecursiveLDA rLDA = new RecursiveLDA();
        rLDA.setVerbose(verbose);
        rLDA.setDebug(debug);
        rLDA.setLog(false);
        rLDA.setReport(false);

        double[] empBackgroundTopic = new double[V];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    empBackgroundTopic[words[d][s][n]]++;
                }
            }
        }

        for (int v = 0; v < V; v++) {
            empBackgroundTopic[v] /= tokenCount;
        }

        int init_burnin = 25;
        int init_maxiter = 50;
        int init_samplelag = 5;

        double[] init_alphas = {0.1, 0.1};
        double[] init_betas = {0.1, 0.1};
        double ratio = 1000;

        rLDA.configure(folder, docWords,
                V, initBranchFactor, ratio, init_alphas, init_betas, initState,
                paramOptimized, init_burnin, init_maxiter, init_samplelag, 1);

        try {
            File lldaZFile = new File(rLDA.getSamplerFolderPath(), "model.zip");
            if (lldaZFile.exists()) {
                rLDA.inputState(lldaZFile);
            } else {
                rLDA.initialize();
                rLDA.iterate(seededAssignments);
                IOUtils.createFolder(rLDA.getSamplerFolderPath());
                rLDA.outputState(lldaZFile);
            }
            rLDA.setWordVocab(wordVocab);
            rLDA.outputTopicTopWords(new File(rLDA.getSamplerFolderPath(), TopWordFile), 20);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing");
        }

        setLog(true);
        this.globalTreeRoot.setTopic(empBackgroundTopic);

        HashMap<RLDA, SNode> nodeMap = new HashMap<RLDA, SNode>();
        nodeMap.put(rLDA.getRoot(), globalTreeRoot);

        Queue<RLDA> queue = new LinkedList<RLDA>();
        queue.add(rLDA.getRoot());
        while (!queue.isEmpty()) {
            RLDA rldaNode = queue.poll();

            for (RLDA rldaChild : rldaNode.getChildren()) {
                queue.add(rldaChild);
            }

            if (rldaNode.getParent() == null) {
                continue;
            }

            int rLDAIndex = rldaNode.getIndex();
            int level = rldaNode.getLevel();

            if (rLDA.hasBackground() && level == 1
                    && rLDAIndex == RecursiveLDA.BACKGROUND) {
                continue; // skip background node
            }

            DirMult topic = new DirMult(V, betas[level] * V, 1.0 / V);
            double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
            SNode parent = nodeMap.get(rldaNode.getParent());
            int sNodeIndex = parent.getNextChildIndex();
            SNode node = new SNode(iter, sNodeIndex, level, topic, regParam, parent);
            node.setTopic(rldaNode.getParent().getTopics()[rLDAIndex].getDistribution());
            parent.addChild(sNodeIndex, node);

            nodeMap.put(rldaNode, node);

            level++;
            if (level == rLDA.getNumLevels()) {
                for (int ii = 0; ii < rldaNode.getTopics().length; ii++) {
                    DirMult subtopic = new DirMult(V, betas[level] * V, 1.0 / V);
                    double subregParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
                    SNode leaf = new SNode(iter, ii, level, subtopic, subregParam, node);
                    leaf.setTopic(rldaNode.getTopics()[ii].getDistribution());
                    node.addChild(ii, leaf);
                }
            }
        }

        if (verbose) {
            logln(printGlobalTree());
            outputTopicTopWords(new File(getSamplerFolderPath(), "init-" + TopWordFile), 15);
        }

        // initialize assignments
        for (int d = 0; d < D; d++) {
            HashMap<SNode, STable> nodeTableMap = new HashMap<SNode, STable>();

            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }
                SparseCount obs = new SparseCount();
                for (int n = 0; n < words[d][s].length; n++) {
                    obs.increment(words[d][s][n]);
                }

                // recursively sample a node for this sentence
                SNode leafNode = recurseNode(globalTreeRoot, obs);
                STable table = nodeTableMap.get(leafNode);
                if (table == null) {
                    int tabIdx = localRestaurants[d].getNextTableIndex();
                    table = new STable(iter, tabIdx, leafNode, d);
                    localRestaurants[d].addTable(table);
                    addTableToPath(leafNode);
                    nodeTableMap.put(leafNode, table);
                }
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // sample level for each token
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int n = 0; n < words[d][s].length; n++) {
                    double[] logprobs = new double[L];
                    for (int ll = 0; ll < L; ll++) {
                        logprobs[ll] = docLevelDist[d].getLogLikelihood(ll)
                                + path[ll].getLogProbability(words[d][s][n]);

//                        if (d == 1) {
//                            System.out.println("s = " + s
//                                    + ". n = " + n
//                                    + ". ll = " + ll
//                                    + ". " + MiscUtils.formatDouble(docLevelDist[d].getLogLikelihood(ll))
//                                    + ". " + MiscUtils.formatDouble(path[ll].getLogProbability(words[d][s][n]))
//                                    + ". " + MiscUtils.formatDouble(logprobs[ll]));
//                        }
                    }
                    int lvl = SamplerUtils.logMaxRescaleSample(logprobs);

                    // debug
//                    if (d == 1) {
//                        System.out.println(">>> " + lvl);
//                    }

                    z[d][s][n] = lvl;
                    sentLevelCounts[d][s][z[d][s][n]]++;
                    docLevelDist[d].increment(z[d][s][n]);
                    path[z[d][s][n]].getContent().increment(words[d][s][n]);
                }
            }
        }

        if (verbose && debug) {
            validate("After initializing assignments");
            outputTopicTopWords(new File(getSamplerFolderPath(), "init-assigned-" + TopWordFile), 15);
        }
    }

    private SNode recurseNode(SNode node, SparseCount obs) {
        if (node.getLevel() == L - 1) {
            return node;
        }
        ArrayList<SNode> children = new ArrayList<SNode>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        for (SNode child : node.getChildren()) {
            children.add(child);
            logprobs.add(node.getLogProbability(obs));
        }
        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        SNode sampledNode = children.get(sampledIdx);
        return recurseNode(sampledNode, obs);
    }

    protected void inspectRestaurant(int d) {
        int docTokenCount = 0;
        int docNonEmptySentCount = 0;
        for (int s = 0; s < words[d].length; s++) {
            if (words[d][s].length != 0) {
                docNonEmptySentCount++;
                docTokenCount += words[d][s].length;
            }
        }

        System.out.println("\n\n>>>d: " + d
                + ". # sentences:" + words[d].length
                + ". # non-empty sentences: " + docNonEmptySentCount
                + ". # tokens: " + docTokenCount
                + "\t # tables: " + localRestaurants[d].getNumTables());
        for (int ll = 0; ll < L; ll++) {
            System.out.println(">>> >>> level: " + ll + "\t" + docLevelDist[d].getCount(ll));
        }
        for (STable table : localRestaurants[d].getTables()) {
            SNode node = table.getContent();
            System.out.print("Table " + table.getTableId()
                    + "\t#c: " + table.getNumCustomers()
                    + "\t" + node.toString()
                    + ":::");
            double[] nodeTopic = node.getTopic();
            String[] topWords = getTopWords(nodeTopic, 10);
            System.out.print("\t");
            for (String word : topWords) {
                System.out.print(word + "\t");
            }
            System.out.println();
            for (int s : table.getCustomers()) {
                System.out.print(">>>s=" + s);
                for (int n = 0; n < words[d][s].length; n++) {
                    System.out.print("; " + wordVocab.get(words[d][s][n])
                            + " (" + z[d][s][n] + ")");
                }
                System.out.println();
                System.out.println("\t" + rawSentences[d][s]);
                System.out.println();
            }
            System.out.println();
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();
        File repFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report && !repFolderPath.exists()) {
                IOUtils.createFolder(repFolderPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            double[] storeWeights = new double[V];
            System.arraycopy(lexParams, 0, storeWeights, 0, V);

            if (verbose) {
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\n*** *** # sents change: " + numSentAsntsChange
                        + " / " + sentCount
                        + " (" + (double) numSentAsntsChange / sentCount + ")"
                        + "\n*** *** # tables change: " + numTableAsgnsChange
                        + " / " + globalTreeRoot.getNumTables()
                        + " (" + (double) numTableAsgnsChange / globalTreeRoot.getNumTables() + ")"
                        + "\n*** *** # accept: " + numAccepts
                        + " / " + numProposes
                        + " (" + (double) numAccepts / numProposes + ")"
                        + "\n" + getCurrentState()
                        + "\n";
                if (iter <= BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            numTableAsgnsChange = 0;
            numSentAsntsChange = 0;
            numProposes = 0;
            numAccepts = 0;

            long tabSent = 0;
            long pathTab = 0;

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (!isValidSentence(d, s)) {
                        continue;
                    }
                    tabSent += sampleSentenceAssignmentsApprox(d, s, REMOVE, ADD,
                            REMOVE, ADD, OBSERVED, EXTEND);
                }

                for (STable table : this.localRestaurants[d].getTables()) {
                    pathTab += samplePathForTable(d, table,
                            REMOVE, ADD, REMOVE, ADD,
                            OBSERVED, EXTEND);
                }
            }

            long updateParam = updateParameters();

            long sampleTopics = sampleTopics();

            logln("Time spent. Iter = " + iter
                    + ". tab->sen: " + tabSent
                    + ". pat->tab: " + pathTab
                    + ". upd->par: " + updateParam
                    + ". sam->top: " + sampleTopics);

            if (verbose) {
                evaluateRegressPrediction(responses, getRegressionValues());
            }

            if (iter > BURN_IN && iter % LAG == 0) {
                if (paramOptimized) {
                    if (verbose) {
                        logln("--- --- Slice sampling ...");
                    }

                    sliceSample();
                    this.sampledParams.add(this.cloneHyperparameters());

                    if (verbose) {
                        logln("--- ---- " + MiscUtils.listToString(hyperparams));
                    }
                }
            }

            if (debug) {
                this.validate("Iteration " + iter);
            }

            float elapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
            logln("Elapsed time iterating: " + elapsedSeconds + " seconds");
            System.out.println();

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(repFolderPath, "iter-" + iter + ".zip"));
                outputTopicTopWords(new File(repFolderPath,
                        "iter-" + iter + "-top-words.txt"), 15);
            }
        }

        // output final model
        if (report) {
            outputState(new File(repFolderPath, "iter-" + iter + ".zip"));
            outputTopicTopWords(new File(repFolderPath,
                    "iter-" + iter + "-top-words.txt"), 15);
        }

        if (verbose) {
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        float elapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + elapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    private double getPathResponseLogLikelihood(int d, SNode[] path, int[] levelCounts) {
        double denom = docWords[d].length;
        double addReg = 0.0;
        int level;
        for (level = 0; level < path.length; level++) {
            addReg += path[level].getRegressionParameter() * levelCounts[level] / denom;
        }
        double mean = docValues[d] + addReg;
        double resLlh = StatUtils.logNormalProbability(
                responses[d], mean, sqrtRho);
        return resLlh;
    }

    /**
     * Sample a path on the global tree for a table.
     *
     * @param d The restaurant index
     * @param table The table
     * @param removeFromModel Whether the current assignment should be removed
     * @param addToModel Whether the new assignment should be added
     * @param observed Whether the response variable is observed
     * @param extend Whether the global tree is extendable
     */
    private long samplePathForTable(int d, STable table,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observed, boolean extend) {
        long sTime = System.currentTimeMillis();
        SNode curNode = table.getContent();
        double denom = docWords[d].length;

        // observation counts of this table per level
        SparseCount[] tabObsCountPerLevel = getTableObsCountPerLevel(d, table);

        if (observed) {
            for (int s : table.getCustomers()) {
                SNode[] curPath = getPathFromNode(table.getContent());
                for (int n = 0; n < words[d][s].length; n++) {
                    docValues[d] -= curPath[z[d][s][n]].getRegressionParameter() / denom;
                }
            }
        }

        if (removeFromModel) {
            removeObservationsFromPath(table.getContent(), tabObsCountPerLevel);
        }

        if (removeFromData) {
            for (int s : table.getCustomers()) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docLevelDist[d].decrement(z[d][s][n]);
                }
            }
            removeTableFromPath(table.getContent());
        }

        // debug
//        boolean condition = d < 5;

        // current assignment
        int[] curLevelCounts = new int[L];
        for (int s : table.getCustomers()) {
            for (int n = 0; n < words[d][s].length; n++) {
                curLevelCounts[z[d][s][n]]++;
            }
        }
        while (curNode.isEmpty()) {
            curNode = curNode.getParent();
        }
        SNode[] path = getPathFromNode(curNode);
        double curResLlh = getPathResponseLogLikelihood(d, path, curLevelCounts);

        // debug
//        if (condition) {
//            logln("table = " + table.getTableId());
//            for (SNode p : path) {
//                logln("path " + p.toString());
//            }
//            for (int s : table.getCustomers()) {
//                System.out.println("s = " + s + ". " + MiscUtils.arrayToString(z[d][s]));
//            }
//            System.out.println(">>> cur res: " + curResLlh);
//        }

        // propose a path and level assignments
        SNode proposedNode = samplePathFromPrior(globalTreeRoot, extend);
        SNode[] proposedPath = getPathFromNode(proposedNode);

        HashMap<Integer, int[]> proposedZs = new HashMap<Integer, int[]>();
        for (int s : table.getCustomers()) {
            int[] ppZs = new int[words[d][s].length];
            for (int n = 0; n < words[d][s].length; n++) {
                double[] lps = new double[L];
                for (int ll = 0; ll < L; ll++) {
                    // word log likelihood
                    double wordLlh;
                    if (ll < proposedPath.length) {
                        wordLlh = proposedPath[ll].getLogProbability(words[d][s][n]);
                    } else { // approx using ancestor
                        wordLlh = proposedPath[proposedPath.length - 1].getLogProbability(words[d][s][n]);
                    }
                    lps[ll] = docLevelDist[d].getLogLikelihood(ll) + wordLlh;
                }
                int idx = SamplerUtils.logMaxRescaleSample(lps);
                ppZs[n] = idx;
            }
            proposedZs.put(s, ppZs);
        }

        int[] newLevelCounts = new int[L];
        for (int s : proposedZs.keySet()) {
            int[] ppZs = proposedZs.get(s);
            for (int n = 0; n < ppZs.length; n++) {
                newLevelCounts[ppZs[n]]++;
            }
        }
        double newResLlh = getPathResponseLogLikelihood(d, proposedPath, newLevelCounts);

        // debug
//        if (condition) {
//            logln("\nd = " + d);
//            for (SNode p : proposedPath) {
//                logln("pp-path " + p.toString());
//            }
//            for (int s : table.getCustomers()) {
//                System.out.println("s = " + s + ". " + MiscUtils.arrayToString(proposedZs.get(s)));
//            }
//            System.out.println(">>> new res: " + newResLlh);
//            System.out.println();
//        }

        // >>> TODO: need to recompute this
        double ratio = Math.min(1, Math.exp(newResLlh - curResLlh));
        SNode newLeaf;
        double randNum = rand.nextDouble();

        // debug
//        if (d < 10) {
//            logln("d = " + d
//                    + ". tab = " + table.getTableId()
//                    + "\t" + MiscUtils.formatDouble(newResLlh)
//                    + "\t" + MiscUtils.formatDouble(curResLlh)
//                    + "\t" + MiscUtils.formatDouble(ratio)
//                    + "\t" + randNum);
//        }

        if (randNum < ratio) { // accept
            newLeaf = proposedNode;
            for (int s : proposedZs.keySet()) {
                int[] ppZs = proposedZs.get(s);
                System.arraycopy(ppZs, 0, z[d][s], 0, words[d][s].length);
            }
            tabObsCountPerLevel = getTableObsCountPerLevel(d, table);
            numAccepts++;
        } else {
            newLeaf = curNode;
        }
        numProposes++;

        // debug
        if (curNode == null || curNode.equals(newLeaf)) {
            numTableAsgnsChange++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (newLeaf.getLevel() < L - 1) {
            newLeaf = this.createNewPath(newLeaf);
        }

        // update
        table.setContent(newLeaf);

        if (addToModel) {
            addObservationsToPath(table.getContent(), tabObsCountPerLevel);
        }

        if (addToData) {
            for (int s : table.getCustomers()) {
                for (int n = 0; n < words[d][s].length; n++) {
                    docLevelDist[d].increment(z[d][s][n]);
                }
            }
            addTableToPath(table.getContent());
        }

        if (observed) {
            for (int s : table.getCustomers()) {
                SNode[] curPath = getPathFromNode(table.getContent());
                for (int n = 0; n < words[d][s].length; n++) {
                    docValues[d] += curPath[z[d][s][n]].getRegressionParameter() / denom;
                }
            }
        }

        return System.currentTimeMillis() - sTime;
    }

    /**
     * Recursively sample a path from prior distribution.
     *
     * @param curNode The current node
     */
    private SNode samplePathFromPrior(SNode curNode, boolean extend) {
        if (isLeafNode(curNode)) {
            return curNode;
        }

        ArrayList<Integer> children = new ArrayList<Integer>();
        ArrayList<Double> probs = new ArrayList<Double>();
        for (SNode child : curNode.getChildren()) {
            children.add(child.getIndex());
            probs.add((double) child.getNumTables());
        }
        if (extend) {
            children.add(PSEUDO_NODE_INDEX);
            probs.add(gammas[curNode.getLevel()]);
        }

        int idx = SamplerUtils.scaleSample(probs);
        int nodeIdx = children.get(idx);
        if (nodeIdx == PSEUDO_NODE_INDEX) {
            return curNode;
        } else {
            return samplePathFromPrior(curNode.getChild(nodeIdx), extend);
        }
    }

    /**
     * Sample the assignments for a sentence. This is done as follows
     *
     * * Remove the table assignment of the current sentence and all level
     * assignments of tokens in this sentence.
     *
     * * Block sample both a new table assignment AND all level assignments for
     * tokens in the sentence. For existing table, sample level assignments
     * using the current path. For a new table, sample a path from prior and
     * sample level assignments on this sampled path.
     *
     * * Update with the new table assignment and level assignments
     *
     * @param d Document index
     * @param s Sentence index
     * @param removeFromModel Whether the current observations should be removed
     * from the tree
     * @param addToModel Whether the observations should be added to the tree
     * with the newly sampled assignments
     * @param removeFromData Whether the current table assignment should be
     * removed
     * @param addToData Whether the new table assignment should be added
     * @param observed Whether the response variable is observed
     * @param extend Whether the tree is changeable
     */
    private long sampleSentenceAssignmentsApprox(int d, int s,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observed, boolean extend) {
        long sTime = System.currentTimeMillis();

        STable curTable = this.c[d][s];
        int[] curZs = new int[words[d][s].length];
        System.arraycopy(z[d][s], 0, curZs, 0, curZs.length);
        SparseCount[] sentObsCountPerLevel = getSentObsCountPerLevel(d, s);
        double denom = docWords[d].length;

        // remove current assignments
        if (observed) {
            SNode[] curPath = getPathFromNode(curTable.getContent());
            for (int n = 0; n < words[d][s].length; n++) {
                docValues[d] -= curPath[curZs[n]].getRegressionParameter() / denom;
            }
        }
        if (removeFromData) {
            for (int n = 0; n < words[d][s].length; n++) {
                docLevelDist[d].decrement(z[d][s][n]);
                sentLevelCounts[d][s][z[d][s][n]]--;
            }
            localRestaurants[d].removeCustomerFromTable(s, curTable.getIndex());
            if (curTable.isEmpty()) {
                removeTableFromPath(curTable.getContent());
                localRestaurants[d].removeTable(curTable.getIndex());
            }
        }
        if (removeFromModel) {
            removeObservationsFromPath(curTable.getContent(), sentObsCountPerLevel);
        }

        // sample 
        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> tableLps = new ArrayList<Double>();

        HashMap<SNode, int[]> proposedZs = new HashMap<SNode, int[]>();
        HashMap<SNode, Double> proposedLps = new HashMap<SNode, Double>();

        // path log prior
        HashMap<SNode, Double> pathLogpriors = new HashMap<SNode, Double>();
        computePathLogPrior(pathLogpriors, globalTreeRoot, 0.0, extend);

        // --- new table (randomly choose a path from prior)
        ArrayList<SNode> allPathNodes = new ArrayList<SNode>();
        ArrayList<Double> allPathLogpriors = new ArrayList<Double>();
        for (SNode node : pathLogpriors.keySet()) {
            allPathNodes.add(node);
            allPathLogpriors.add(pathLogpriors.get(node));
        }
        int newTabSampledIdx = SamplerUtils.logMaxRescaleSample(allPathLogpriors);
        SNode newTabSampledNode = allPathNodes.get(newTabSampledIdx);
        double newTabLp = 0.0;
        int[] newTableZs = new int[words[d][s].length];
        SNode[] path = getPathFromNode(newTabSampledNode);

        // sample level for each token
        for (int n = 0; n < words[d][s].length; n++) {
            double[] lps = new double[L];

            for (int ll = 0; ll < L; ll++) {
                // word log likelihood
                double wordLlh;
                if (ll < path.length) {
                    wordLlh = path[ll].getLogProbability(words[d][s][n]);
                } else { // approx using ancestor
                    wordLlh = path[path.length - 1].getLogProbability(words[d][s][n]);
                }
                lps[ll] = docLevelDist[d].getLogLikelihood(ll) + wordLlh;
            }

            int idx = SamplerUtils.logMaxRescaleSample(lps);
            newTabLp += lps[idx];
            newTableZs[n] = idx;
        }
        proposedZs.put(newTabSampledNode, newTableZs);
        proposedLps.put(newTabSampledNode, newTabLp);


        // propose assignments
        proposeTokenAssignments(d, s, proposedZs, proposedLps);

        // path response llhs
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (observed) {
            computePathResponseLogLikelihood(pathResLlhs, docValues[d], denom,
                    responses[d], proposedZs);
        }

        // debug
//        boolean condition = (d == 10);
//        if (condition) {
//            logln("ppZs: " + proposedZs.size()
//                    + "\tppLps: " + proposedLps.size()
//                    + "\tprior" + pathLogpriors.size()
//                    + "\tres: " + pathResLlhs.size());
//        }

        // sample table for this sentence

        // --- existing tables
        for (STable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());

            SNode pathNode = table.getContent();
            double resLlh = 0.0;
            if (observed) {
                resLlh = pathResLlhs.get(pathNode);
            }
            double lp = Math.log(table.getNumCustomers())
                    + proposedLps.get(pathNode)
                    + resLlh;
            tableLps.add(lp);

            // debug
//            if (condition) {
//                logln("iter = " + iter
//                        + ". d = " + d
//                        + ". s = " + s
//                        + ". table: " + table.getIndex()
//                        + ". # custs: " + table.getNumCustomers()
//                        + ". leaf: " + table.getContent().getPathString()
//                        + ". " + MiscUtils.formatDouble(Math.log(table.getNumCustomers()))
//                        + ". " + MiscUtils.formatDouble(proposedLps.get(pathNode))
//                        + ". " + MiscUtils.formatDouble(resLlh)
//                        + ". " + MiscUtils.formatDouble(lp));
//            }
        }

        tableIndices.add(PSEUDO_TABLE_INDEX);
        double newTabLogPrior = logAlpha;
        double newTabWordLlh = proposedLps.get(newTabSampledNode);
        double newTabResLlh = pathResLlhs.get(newTabSampledNode);
        double newTabLogprob = newTabLogPrior + newTabWordLlh + newTabResLlh;
        tableLps.add(newTabLogprob);

        // debug
//        if (condition) {
//            logln("iter = " + iter
//                    + ". d = " + d
//                    + ". s = " + s
//                    + ". new table: "
//                    + ". " + MiscUtils.formatDouble(newTabLogPrior)
//                    + ". " + MiscUtils.formatDouble(newTabWordLlh)
//                    + ". " + MiscUtils.formatDouble(newTabResLlh)
//                    + ". " + MiscUtils.formatDouble(newTabLogprob));
//        }

        // update new assignments
        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(tableLps);
        int tableIdx = tableIndices.get(sampledIndex);

        // debug
//        if (condition) {
//            logln(">>> sIdx: " + sampledIndex
//                    + "\t" + tableIdx);
//        }

        if (curTable != null && curTable.getIndex() != tableIdx) {
            numSentAsntsChange++;
        }

        STable table;
        int[] newZs;
        if (tableIdx == PSEUDO_NODE_INDEX) {
            // sample path for a new table
            int newTableIdx = localRestaurants[d].getNextTableIndex();
            table = new STable(iter, newTableIdx, null, d);
            localRestaurants[d].addTable(table);

            newZs = proposedZs.get(newTabSampledNode);
            if (!isLeafNode(newTabSampledNode)) {
                newTabSampledNode = createNewPath(newTabSampledNode);
            }
            table.setContent(newTabSampledNode);
            addTableToPath(table.getContent());
        } else {
            table = localRestaurants[d].getTable(tableIdx);
            newZs = proposedZs.get(table.getContent());
        }

        c[d][s] = table;
        System.arraycopy(newZs, 0, z[d][s], 0, words[d][s].length);
        sentObsCountPerLevel = getSentObsCountPerLevel(d, s);

        if (addToData) {
            for (int n = 0; n < words[d][s].length; n++) {
                docLevelDist[d].increment(z[d][s][n]);
                sentLevelCounts[d][s][z[d][s][n]]++;
            }
            localRestaurants[d].addCustomerToTable(s, c[d][s].getIndex());
        }

        if (addToModel) {
            addObservationsToPath(c[d][s].getContent(), sentObsCountPerLevel);
        }

        if (observed) {
            SNode[] curPath = getPathFromNode(curTable.getContent());
            for (int n = 0; n < words[d][s].length; n++) {
                docValues[d] += curPath[curZs[n]].getRegressionParameter() / denom;
            }
        }

        return System.currentTimeMillis() - sTime;
    }

    private long sampleSentenceAssignmentsExact(int d, int s,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observed, boolean extend) {
        long sTime = System.currentTimeMillis();

        STable curTable = this.c[d][s];
        int[] curZs = new int[words[d][s].length];
        System.arraycopy(z[d][s], 0, curZs, 0, curZs.length);
        SparseCount[] sentObsCountPerLevel = getSentObsCountPerLevel(d, s);

        double denom = docWords[d].length;
        if (observed) {
            SNode[] curPath = getPathFromNode(curTable.getContent());
            for (int n = 0; n < words[d][s].length; n++) {
                docValues[d] -= curPath[curZs[n]].getRegressionParameter() / denom;
            }
        }
        if (removeFromData) {
            for (int n = 0; n < words[d][s].length; n++) {
                docLevelDist[d].decrement(z[d][s][n]);
                sentLevelCounts[d][s][z[d][s][n]]--;
            }
            localRestaurants[d].removeCustomerFromTable(s, curTable.getIndex());
            if (curTable.isEmpty()) {
                removeTableFromPath(curTable.getContent());
                localRestaurants[d].removeTable(curTable.getIndex());
            }
        }
        if (removeFromModel) {
            removeObservationsFromPath(curTable.getContent(), sentObsCountPerLevel);
        }

        // for each possible path, sample a new set of level assignments for
        // tokens in this sentence.
        HashMap<SNode, int[]> proposedZs = new HashMap<SNode, int[]>();
        HashMap<SNode, Double> proposedLps = new HashMap<SNode, Double>();
        proposeTokenAssignments(d, s, proposedZs, proposedLps, extend);

        // path log prior
        HashMap<SNode, Double> pathLogpriors = new HashMap<SNode, Double>();
        computePathLogPrior(pathLogpriors, globalTreeRoot, 0.0, extend);

        // path response llhs
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (observed) {
            computePathResponseLogLikelihood(pathResLlhs, docValues[d],
                    denom, responses[d], proposedZs);
        }

        // sample table for this sentence
        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> tableLps = new ArrayList<Double>();
        // --- existing tables
        for (STable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());

            SNode pathNode = table.getContent();
            double resLlh = 0.0;
            if (observed) {
                resLlh = pathResLlhs.get(pathNode);
            }
            double lp = Math.log(table.getNumCustomers())
                    + proposedLps.get(pathNode)
                    + resLlh;
            tableLps.add(lp);

            // debug
//            if (d == 0) {
//                logln("iter = " + iter
//                        + ". d = " + d
//                        + ". s = " + s
//                        + ". table: " + table.getIndex()
//                        + ". leaf: " + table.getContent().getPathString()
//                        + ". " + MiscUtils.formatDouble(Math.log(table.getNumCustomers()))
//                        + ". " + MiscUtils.formatDouble(proposedLps.get(pathNode))
//                        + ". " + MiscUtils.formatDouble(resLlh)
//                        + ". " + MiscUtils.formatDouble(lp));
//            }
        }

        // --- new table
        tableIndices.add(PSEUDO_TABLE_INDEX);
        double newTabLp = logAlpha
                + computeMarginals(pathLogpriors, proposedLps, pathResLlhs, observed);
        tableLps.add(newTabLp);

        // debug
//        if (d == 0) {
//            logln("iter = " + iter
//                    + ". d = " + d
//                    + ". s = " + s
//                    + ". new table: "
//                    + ". " + MiscUtils.formatDouble(newTabLp));
//        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(tableLps);
        int tableIdx = tableIndices.get(sampledIndex);

        // debug
//        if (d == 0) {
//            logln(">>> sIdx: " + sampledIndex
//                    + "\t" + tableIdx);
//        }

        if (curTable != null && curTable.getIndex() != tableIdx) {
            numSentAsntsChange++;
        }

        STable table;
        int[] newZs;
        if (tableIdx == PSEUDO_NODE_INDEX) {
            // sample path for a new table
            int newTableIdx = localRestaurants[d].getNextTableIndex();
            table = new STable(iter, newTableIdx, null, d);
            localRestaurants[d].addTable(table);

            SNode newNode = samplePath(pathLogpriors, proposedLps, pathResLlhs, observed);
            newZs = proposedZs.get(newNode);
            if (!isLeafNode(newNode)) {
                newNode = createNewPath(newNode);
            }
            table.setContent(newNode);
            addTableToPath(table.getContent());
        } else {
            table = localRestaurants[d].getTable(tableIdx);
            newZs = proposedZs.get(table.getContent());

        }

        c[d][s] = table;
        System.arraycopy(newZs, 0, z[d][s], 0, words[d][s].length);
        sentObsCountPerLevel = getSentObsCountPerLevel(d, s);

        if (addToData) {
            for (int n = 0; n < words[d][s].length; n++) {
                docLevelDist[d].increment(z[d][s][n]);
                sentLevelCounts[d][s][z[d][s][n]]++;
            }
            localRestaurants[d].addCustomerToTable(s, c[d][s].getIndex());
        }

        if (addToModel) {
            addObservationsToPath(c[d][s].getContent(), sentObsCountPerLevel);
        }

        if (observed) {
            SNode[] curPath = getPathFromNode(curTable.getContent());
            for (int n = 0; n < words[d][s].length; n++) {
                docValues[d] += curPath[curZs[n]].getRegressionParameter() / denom;
            }
        }

        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample a path
     *
     * @param logPriors Path log priors
     * @param wordLlhs Path word log likelihoods
     * @param resLlhs Path response variable log likelihoods
     */
    SNode samplePath(
            HashMap<SNode, Double> logPriors,
            HashMap<SNode, Double> wordLlhs,
            HashMap<SNode, Double> resLlhs,
            boolean observed) {
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode node : logPriors.keySet()) {
            double lp = logPriors.get(node) + wordLlhs.get(node);
            if (observed) {
                lp += resLlhs.get(node);
            }
            pathList.add(node);
            logProbs.add(lp);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode path = pathList.get(sampledIndex);
        return path;
    }

    /**
     * Propose the assignments for all tokens in a given sentence considering
     * only paths that are current assigned to table in the document.
     *
     * @param d The document index
     * @param s The sentence index
     * @param ppAssignments Map to store the proposed assignments
     * @param ppLogprobs Map to store the corresponding log probabilities of the
     * proposed assignments
     */
    void proposeTokenAssignments(int d, int s,
            HashMap<SNode, int[]> ppAssignments,
            HashMap<SNode, Double> ppLogprobs) {
        // log prior of each level: shared across document
        double[] logpriors = new double[L];
        for (int ll = 0; ll < L; ll++) {
            logpriors[ll] = docLevelDist[d].getLogLikelihood(ll);
        }

        for (STable table : localRestaurants[d].getTables()) {
            SNode node = table.getContent();
            if (ppAssignments.containsKey(node)) {
                continue;
            }

            int[] asgns = new int[words[d][s].length];
            double lp = 0.0;
            SNode[] path = getPathFromNode(node);

            // sample level for each token
            for (int n = 0; n < words[d][s].length; n++) {
                double[] lps = new double[L];

                for (int ll = 0; ll < L; ll++) {
                    // word log likelihood
                    double wordLlh;
                    if (ll < path.length) {
                        wordLlh = path[ll].getLogProbability(words[d][s][n]);
                    } else { // approx using ancestor
                        wordLlh = path[path.length - 1].getLogProbability(words[d][s][n]);
                    }
                    lps[ll] = logpriors[ll] + wordLlh;
                }

                int idx = SamplerUtils.logMaxRescaleSample(lps);
                asgns[n] = idx;
                lp += lps[idx];
            }

            ppAssignments.put(node, asgns);
            ppLogprobs.put(node, lp);
        }
    }

    /**
     * Propose the assignments for all tokens in a given sentence considering
     * all possible paths in the global tree.
     *
     * @param d The document index
     * @param s The sentence index
     * @param ppAssignments Map to store the proposed assignments
     * @param ppLogprobs Map to store the corresponding log probabilities of the
     * proposed assignments
     * @param extend Whether extending the tree
     */
    void proposeTokenAssignments(int d, int s,
            HashMap<SNode, int[]> ppAssignments,
            HashMap<SNode, Double> ppLogprobs,
            boolean extend) {
        // log prior of each level: shared across document
        double[] logpriors = new double[L];
        for (int ll = 0; ll < L; ll++) {
            logpriors[ll] = docLevelDist[d].getLogLikelihood(ll);
        }

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            if (!extend && !isLeafNode(node)) {
                continue;
            }

            int[] asgns = new int[words[d][s].length];
            double lp = 0.0;
            SNode[] path = getPathFromNode(node);

            // sample level for each token
            for (int n = 0; n < words[d][s].length; n++) {
                double[] lps = new double[L];

                for (int ll = 0; ll < L; ll++) {
                    // word log likelihood
                    double wordLlh;
                    if (ll < path.length) {
                        wordLlh = path[ll].getLogProbability(words[d][s][n]);
                    } else { // approx using ancestor
                        wordLlh = path[path.length - 1].getLogProbability(words[d][s][n]);
                    }
                    lps[ll] = logpriors[ll] + wordLlh;
                }

                int idx = SamplerUtils.logMaxRescaleSample(lps);
                asgns[n] = idx;
                lp += lps[idx];
            }

            ppAssignments.put(node, asgns);
            ppLogprobs.put(node, lp);
        }
    }

    /**
     * Compute the log probability of a new table, marginalized over all
     * possible paths
     *
     * @param pathLogPriors The log priors of each path
     * @param pathWordLogLikelihoods The word likelihoods
     * @param pathResLogLikelihoods The response variable likelihoods
     */
    double computeMarginals(
            HashMap<SNode, Double> pathLogPriors,
            HashMap<SNode, Double> pathWordLogLikelihoods,
            HashMap<SNode, Double> pathResLogLikelihoods,
            boolean resObserved) {
        double marginal = 0.0;
        for (SNode node : pathLogPriors.keySet()) {
            double logprior = pathLogPriors.get(node);
            double loglikelihood = pathWordLogLikelihoods.get(node);

            double lp = logprior + loglikelihood;
            if (resObserved) {
                lp += pathResLogLikelihoods.get(node);
            }

            if (marginal == 0.0) {
                marginal = lp;
            } else {
                marginal = SamplerUtils.logAdd(marginal, lp);
            }
        }
        return marginal;
    }

    /**
     * Compute the log likelihoods of an author response when assigning a set of
     * tokens (represented by their level assignments) to each path in the tree.
     *
     * @param nodeResLogProbs Map to store the results
     * @param curAuthorVal Current author value
     * @param denom
     */
    void computePathResponseLogLikelihood(
            HashMap<SNode, Double> nodeResLogProbs,
            double curAuthorVal, double denom,
            double authorResponse,
            HashMap<SNode, int[]> proposedZs) {
        for (SNode pathNode : proposedZs.keySet()) {
            SNode[] path = getPathFromNode(pathNode);

            int[] ppZs = proposedZs.get(pathNode);
            int[] levelCounts = new int[L];
            for (int ii = 0; ii < ppZs.length; ii++) {
                levelCounts[ppZs[ii]]++;
            }

            double addReg = 0.0;
            int level;
            for (level = 0; level < path.length; level++) {
                addReg += path[level].getRegressionParameter() * levelCounts[level] / denom;
            }
            double authorMean = curAuthorVal + addReg;
            double resLlh = StatUtils.logNormalProbability(
                    authorResponse, authorMean, sqrtRho);
            nodeResLogProbs.put(pathNode, resLlh);
        }
    }

    /**
     * Compute the log probability of the response variable when the given table
     * is assigned to each path
     *
     * @param d The document index
     * @param table The table
     */
    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            STable table,
            boolean extend) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();
//        int author = authors[d];
//        double denom = docWords[d].length * authorDocIndices[author].length;
//
//        Stack<SNode> stack = new Stack<SNode>();
//        stack.add(globalTreeRoot);
//        while (!stack.isEmpty()) {
//            SNode node = stack.pop();
//
//            for (SNode child : node.getChildren()) {
//                stack.add(child);
//            }
//            if (!extend && !isLeafNode(node)) {
//                continue;
//            }
//
//            SNode[] path = getPathFromNode(node);
//            double addReg = 0.0;
//            for (int level = 0; level < path.length; level++) {
//                for (int s : table.getCustomers()) {
//                    addReg += path[level].getRegressionParameter()
//                            * sentLevelCounts[d][s][level] / denom;
//                }
//            }
//
//            double authorMean = docValues[author] + addReg;
//            double resLlh = StatisticsUtils.logNormalProbability(
//                    responses[author], authorMean, sqrtRho);
//            resLlhs.put(node, resLlh);
//
//        }
        return resLlhs;
    }

    /**
     * Recursively compute the log probability of each path in the global tree
     *
     * @param nodeLogProbs HashMap to store the results
     * @param curNode Current node in the recursive call
     * @param parentLogProb The log probability passed from the parent node
     */
    void computePathLogPrior(
            HashMap<SNode, Double> nodeLogProbs,
            SNode curNode,
            double parentLogProb,
            boolean extend) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumTables() + gammas[curNode.getLevel()]);
            newWeight += logGammas[curNode.getLevel()] - logNorm;

            for (SNode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumTables()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight, extend);
            }
        }
        if (!extend && !isLeafNode(curNode)) {
            return;
        }
        nodeLogProbs.put(curNode, newWeight);
    }

    /**
     * Compute the log probability of assigning a set of words to each path in
     * the tree
     *
     * @param nodeDataLlhs HashMap to store the result
     * @param curNode The current node in recursive calls
     * @param tokenCountPerLevel Token counts per level
     * @param parentDataLlh The value passed from the parent node
     */
    void computePathWordLogLikelihood(
            HashMap<SNode, Double> nodeDataLlhs,
            SNode curNode,
            SparseCount[] tokenCountPerLevel,
            double parentDataLlh,
            boolean extend) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getLogProbability(tokenCountPerLevel[level]);

        // populate to child nodes
        for (SNode child : curNode.getChildren()) {
            computePathWordLogLikelihood(nodeDataLlhs, child, tokenCountPerLevel,
                    parentDataLlh + nodeDataLlh, extend);
        }

        if (!extend && !isLeafNode(curNode)) {
            return;
        }
        // store the data llh from the root to this current node
        double storeDataLlh = parentDataLlh + nodeDataLlh;
        level++;
        while (level < L) { // if this is an internal node, add llh of new child node
            DirMult dirMult;
            if (curNode.getTopic() == null) {
                dirMult = new DirMult(V, betas[level] * V, 1.0 / V);
            } else {
                dirMult = new DirMult(V, betas[level] * V, curNode.getTopic());
            }
            storeDataLlh += dirMult.getLogLikelihood(tokenCountPerLevel[level].getObservations());
            level++;
        }
        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    /**
     * Get the observation counts per level of a given table
     *
     * @param d The document index
     * @param table The table
     */
    SparseCount[] getTableObsCountPerLevel(int d, STable table) {
        // observations of sentences currently being assign to this table
        SparseCount[] obsCountPerLevel = new SparseCount[L];
        for (int l = 0; l < L; l++) {
            obsCountPerLevel[l] = new SparseCount();
        }

        for (int s : table.getCustomers()) {
            for (int n = 0; n < words[d][s].length; n++) {
                int level = z[d][s][n];
                int obs = words[d][s][n];
                obsCountPerLevel[level].increment(obs);
            }
        }
        return obsCountPerLevel;
    }

    /**
     * Sample all topics. This is done by performing bottom-up smoothing and
     * top-down sampling.
     */
    protected long sampleTopics() {
        if (verbose) {
            logln("Sampling topics ...");
        }
        long sTime = System.currentTimeMillis();
        // get all leaves of the tree
        ArrayList<SNode> leaves = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<SNode> queue = new LinkedList<SNode>();
        for (SNode leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            SNode node = queue.poll();
            if (node.isLeaf()) {
                continue;
            }

            SNode parent = node.getParent();
            if (!node.isRoot() && !queue.contains(parent)) {
                queue.add(parent);
            }

            if (this.pathAssumption == PathAssumption.MINIMAL) {
                node.getPseudoCountsFromChildrenMin();
            } else if (this.pathAssumption == PathAssumption.MAXIMAL) {
                node.getPseudoCountsFromChildrenMax();
            } else {
                throw new RuntimeException("Path assumption " + this.pathAssumption
                        + " is not supported.");
            }
        }

        // top-down sampling to get topics
        queue = new LinkedList<SNode>();
        queue.add(globalTreeRoot);
        while (!queue.isEmpty()) {
            SNode node = queue.poll();
            for (SNode child : node.getChildren()) {
                queue.add(child);
            }
            node.sampleTopic(betas[node.getLevel()] * V, betas[node.getLevel()]);
        }
        return System.currentTimeMillis() - sTime;
    }

    private long updateParameters() {
        if (verbose) {
            logln("--- Optimizing regression parameters ...");
        }
        long sTime = System.currentTimeMillis();

        ArrayList<SNode> flattenTree = flattenTreeWithoutRoot();
        int numNodes = flattenTree.size();

        double[] paramSigmas = new double[V + numNodes];
        double[] paramMeans = new double[V + numNodes];
        for (int v = 0; v < V; v++) {
            paramMeans[v] = hyperparams.get(TAU_MEAN);
            paramSigmas[v] = hyperparams.get(TAU_SIGMA);
        }

        HashMap<SNode, Integer> nodeIndices = new HashMap<SNode, Integer>();
        for (int i = 0; i < flattenTree.size(); i++) {
            SNode node = flattenTree.get(i);
            nodeIndices.put(node, i);
            paramSigmas[V + i] = sigmas[node.getLevel()];
            paramMeans[V + i] = mus[node.getLevel()];
        }

        double[][] designMatrix = new double[D][V + numNodes];
        for (int d = 0; d < D; d++) {
            System.arraycopy(lexDesignMatrix[d], 0, designMatrix[d], 0, V);
            double[] docTopicDist = new double[numNodes];
            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 1; l < L; l++) {
                    int nodeIdx = nodeIndices.get(path[l]);
                    docTopicDist[nodeIdx] += (double) sentLevelCounts[d][s][l]
                            / docWords[d].length;
                }
            }
            for (int ii = 0; ii < numNodes; ii++) {
                designMatrix[d][ii + V] += docTopicDist[ii] / docWords[d].length;
            }
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setSigmas(paramSigmas);
        mlr.setMeans(paramMeans);
        mlr.setRho(hyperparams.get(RHO));
        double[] weights = mlr.solve();
        System.arraycopy(weights, 0, lexParams, 0, V);
        for (int i = 0; i < numNodes; i++) {
            flattenTree.get(i).setRegressionParameter(weights[i + V]);
        }

        this.updateAuthorValues();
        long etime = System.currentTimeMillis() - sTime;
        return etime;
    }

    /**
     * Get the observation counts per level of a sentence given the current
     * token assignments.
     *
     * @param d The document index
     * @param s The sentence index
     */
    protected SparseCount[] getSentObsCountPerLevel(int d, int s) {
        SparseCount[] counts = new SparseCount[L];
        for (int ll = 0; ll < L; ll++) {
            counts[ll] = new SparseCount();
        }
        for (int n = 0; n < words[d][s].length; n++) {
            int type = words[d][s][n];
            int level = z[d][s][n];
            counts[level].increment(type);
        }
        return counts;
    }

    /**
     * Check whether a given node is a leaf node.
     *
     * @param node The node
     */
    boolean isLeafNode(SNode node) {
        return node.getLevel() == L - 1;
    }

    /**
     * Parse the node path string.
     *
     * @param nodePath The node path string
     */
    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    /**
     * Get a node in the tree given a parsed path
     *
     * @param parsedPath The parsed path
     */
    private SNode getNode(int[] parsedPath) {
        SNode node = globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    /**
     * Get a list of node in the current tree without the root.
     */
    ArrayList<SNode> flattenTreeWithoutRoot() {
        ArrayList<SNode> flattenTree = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (!node.isRoot()) {
                flattenTree.add(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return flattenTree;
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    void addTableToPath(SNode leafNode) {
        SNode node = leafNode;
        while (node != null) {
            node.incrementNumTables();
            node = node.getParent();
        }
    }

    /**
     * Remove a customer from a path. A path is specified by the pointer to its
     * leaf node. The number of customers at each node on the path will be
     * decremented. If the number of customers at a node is 0, the node will be
     * removed.
     *
     * @param leafNode The leaf node of the path
     * @return Return the node that specifies the path that the leaf node is
     * removed from. If a lower-level node has no customer, it will be removed
     * and the lowest parent node on the path that has non-zero number of
     * customers will be returned.
     */
    SNode removeTableFromPath(SNode leafNode) {
        SNode retNode = leafNode;
        SNode node = leafNode;
        while (node != null) {
            node.decrementNumTables();
            if (node.isEmpty()) {
                retNode = node.getParent();
                node.getParent().removeChild(node.getIndex());
            }
            node = node.getParent();
        }
        return retNode;
    }

    /**
     * Add a set of observations (given their level assignments) to a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    SNode[] addObservationsToPath(SNode leafNode, SparseCount[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Add a set of observations to a node. This will (1) add the set of
     * observations to the node's topic, and (2) add the token counts for
     * switches from the root to the node.
     *
     * @param node The node
     * @param observations The set of observations
     */
    void addObservationsToNode(SNode node, SparseCount observations) {
        for (int obs : observations.getIndices()) {
            int count = observations.getCount(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    /**
     * Remove a set of observations (given their level assignments) from a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    SNode[] removeObservationsFromPath(SNode leafNode, SparseCount[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations from a node. This will (1) remove the set of
     * observations from the node's topic, and (2) remove the token counts from
     * the switches from the root to the node.
     *
     * @param node The node
     * @param observations The set of observations
     */
    void removeObservationsFromNode(SNode node, SparseCount observations) {
        for (int obs : observations.getIndices()) {
            int count = observations.getCount(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    /**
     * Create a new path from an internal node.
     *
     * @param internalNode The internal node
     */
    SNode createNewPath(SNode internalNode) {
        SNode node = internalNode;
        for (int l = internalNode.getLevel(); l < L - 1; l++) {
            node = this.createNode(node);
        }
        return node;
    }

    /**
     * Create a node given a parent node
     *
     * @param parent The parent node
     */
    SNode createNode(SNode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmm = new DirMult(V, betas[level] * V, uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SNode child = new SNode(iter, nextChildIndex, level, dmm, regParam, parent);
        return parent.addChild(nextChildIndex, child);
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            wordLlh += node.getLogProbability(node.getContent().getSparseCounts());
            regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                    mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));
            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double levelLp = 0.0;
        double restLgprob = 0.0;
        for (int d = 0; d < D; d++) {
            levelLp += docLevelDist[d].getLogLikelihood();
            restLgprob += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));
        }

        double resLlh = 0.0;
        if (responses != null) {
            double[] regValues = getRegressionValues();
            for (int d = 0; d < D; d++) {
                resLlh += StatUtils.logNormalProbability(responses[d],
                        regValues[d], sqrtRho);
            }
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                + ". rest = " + MiscUtils.formatDouble(restLgprob)
                + ". level = " + MiscUtils.formatDouble(levelLp)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLogProb + levelLp + regParamLgprob + resLlh + restLgprob;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        throw new RuntimeException("Hyperparameter optimization is not supported");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
        throw new RuntimeException("Hyperparameter optimization is not supported");
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(printGlobalTreeSummary()).append("\n");
        str.append(printLocalRestaurantSummary()).append("\n");
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

    public void inputFinalModel() {
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        this.inputModel(new File(reportFolder, "iter-" + MAX_ITER + ".zip").getAbsolutePath());
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath + "\n");
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int v = 0; v < V; v++) {
                modelStr.append(lexParams[v]).append("\t");
            }
            modelStr.append("\n");

            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumTables()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");
                modelStr.append(DirMult.outputDistribution(node.getContent().
                        getSamplingDistribution())).append("\n");

                for (SNode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                assignStr.append(DirMult.output(docLevelDist[d])).append("\n");
                for (STable table : localRestaurants[d].getTables()) {
                    assignStr.append(table.getIndex()).append("\n");
                    assignStr.append(table.getIterationCreated()).append("\n");
                    assignStr.append(table.getContent().getPathString()).append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (isValidSentence(d, s)) {
                        assignStr.append(d)
                                .append(":").append(s)
                                .append("\t").append(c[d][s].getIndex())
                                .append("\n");
                    }
                }
            }

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (isValidSentence(d, s)) {
                        for (int n = 0; n < words[d][s].length; n++) {
                            assignStr.append(d)
                                    .append(":").append(s)
                                    .append(":").append(n)
                                    .append("\t").append(z[d][s][n])
                                    .append("\n");
                        }
                    }
                }
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
            logln("--- Reading state from " + filepath + "\n");
        }
        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }

        if (debug) {
            validate("--- Loaded.");
        }
    }

    void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath + "\n");
        }
        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);

            // lexical weights
            String line = reader.readLine();
            String[] sline = line.split("\t");
            this.lexParams = new double[V];
            for (int v = 0; v < V; v++) {
                this.lexParams[v] = Double.parseDouble(sline[v]);
            }

            // topic tree
            HashMap<String, SNode> nodeMap = new HashMap<String, SNode>();
            while ((line = reader.readLine()) != null) {
                String pathStr = line;
                int iterCreated = Integer.parseInt(reader.readLine());
                int numTables = Integer.parseInt(reader.readLine());
                double regParam = Double.parseDouble(reader.readLine());
                DirMult dmm = DirMult.input(reader.readLine());
                double[] dist = DirMult.inputDistribution(reader.readLine());
                dmm.setSamplingDistribution(dist);

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                SNode parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }

                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;
                SNode node = new SNode(iterCreated, nodeIndex,
                        nodeLevel, dmm, regParam, parent);
                node.setTopic(dist);

                node.changeNumTables(numTables);

                if (node.getLevel() == 0) {
                    globalTreeRoot = node;
                }

                if (parent != null) {
                    parent.addChild(node.getIndex(), node);
                }

                nodeMap.put(pathStr, node);
            }
            reader.close();

            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                if (!isLeafNode(node)) {
                    node.fillInactiveChildIndices();
                    for (SNode child : node.getChildren()) {
                        stack.add(child);
                    }
                }
            }

            validateModel("Loading model " + filename);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath + "\n");
        }
        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
        BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);

        String[] sline;

        for (int d = 0; d < D; d++) {
            sline = reader.readLine().split("\t");
            if (d != Integer.parseInt(sline[0])) {
                throw new RuntimeException("Mismatch. "
                        + d + " vs. " + sline[0]);
            }
            int numTables = Integer.parseInt(sline[1]);
            docLevelDist[d] = DirMult.input(reader.readLine());

            for (int i = 0; i < numTables; i++) {
                int tabIndex = Integer.parseInt(reader.readLine());
                int iterCreated = Integer.parseInt(reader.readLine());
                SNode leafNode = getNode(parseNodePath(reader.readLine()));
                STable table = new STable(iterCreated, tabIndex, leafNode, d);
                localRestaurants[d].addTable(table);
            }
        }

        for (int d = 0; d < D; d++) {
            localRestaurants[d].fillInactiveTableIndices();
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }
                sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + s)) {
                    throw new RuntimeException("Mismatch");
                }
                int tableIndex = Integer.parseInt(sline[1]);
                STable table = localRestaurants[d].getTable(tableIndex);
                c[d][s] = table;
                localRestaurants[d].addCustomerToTable(s, tableIndex);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }
                STable table = c[d][s];
                SNode[] path = getPathFromNode(table.getContent());
                for (int n = 0; n < words[d][s].length; n++) {
                    sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + s + ":" + n)) {
                        throw new RuntimeException("Mismatch. "
                                + sline[0] + " vs. "
                                + (d + ":" + s + ":" + n));
                    }
                    z[d][s][n] = Integer.parseInt(sline[1]);
                    path[z[d][s][n]].getContent().increment(words[d][s][n]);
                    sentLevelCounts[d][s][z[d][s][n]]++;
                }
            }
        }

        reader.close();
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);

        validateModel(msg);

        validateAssignments(msg);
    }

    protected void validateModel(String msg) {
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            if (!isLeafNode(node)) {
                int childNumCusts = 0;

                for (SNode child : node.getChildren()) {
                    childNumCusts += child.getNumTables();
                    stack.add(child);
                }

                if (childNumCusts != node.getNumTables()) {
                    throw new RuntimeException(msg + ". Numbers of customers mismatch. "
                            + node.toString());
                }
            }

            if (this.isLeafNode(node) && node.isEmpty()) {
                throw new RuntimeException(msg + ". Leaf node " + node.toString()
                        + " is empty");
            }
        }
    }

    protected void validateAssignments(String msg) {
        // validate sentence assignments
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                if (words[d][s].length == 0) {
                    if (this.c[d][s] != NULL_TABLE) {
                        throw new RuntimeException("Assigning empty sentence to "
                                + "a table. Sentence " + d + "-" + s + " is "
                                + "assigned to " + this.c[d][s].toString());
                    }
                }
            }
        }

        for (int d = 0; d < D; d++) {
            int totalCusts = 0;
            for (STable table : localRestaurants[d].getTables()) {
                totalCusts += table.getNumCustomers();
            }

            int numValidSentences = 0;
            for (int s = 0; s < words[d].length; s++) {
                if (isValidSentence(d, s)) {
                    numValidSentences++;
                }
            }
            if (totalCusts != numValidSentences) {
                for (STable table : localRestaurants[d].getTables()) {
                    System.out.println(table.toString() + ". customers: " + table.getCustomers().toString());
                }
                throw new RuntimeException(msg + ". Numbers of customers in restaurant " + d
                        + " mismatch. " + totalCusts + " vs. " + words[d].length);
            }

            HashMap<STable, Integer> tableCustCounts = new HashMap<STable, Integer>();
            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }

                Integer count = tableCustCounts.get(c[d][s]);

                if (count == null) {
                    tableCustCounts.put(c[d][s], 1);
                } else {
                    tableCustCounts.put(c[d][s], count + 1);
                }
            }

            if (tableCustCounts.size() != localRestaurants[d].getNumTables()) {
                throw new RuntimeException(msg + ". Numbers of tables mismatch in"
                        + " restaurant " + d
                        + ". " + tableCustCounts.size()
                        + " vs. " + localRestaurants[d].getNumTables());
            }

            for (STable table : localRestaurants[d].getTables()) {
                if (table.getNumCustomers() != tableCustCounts.get(table)) {
                    System.out.println("Table: " + table.toString());

                    for (int s : table.getCustomers()) {
                        System.out.println("--- s = " + s + ". " + c[d][s].toString());
                    }
                    System.out.println(tableCustCounts.get(table));


                    throw new RuntimeException(msg + ". Number of customers "
                            + "mismatch. Table " + table.toString()
                            + ". " + table.getNumCustomers() + " vs. " + tableCustCounts.get(table));
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                if (words[d][s].length != StatUtils.sum(sentLevelCounts[d][s])) {
                    throw new RuntimeException("Counts mismatch. d = " + d
                            + ". s = " + s
                            + ". " + words[d][s].length
                            + " vs. " + StatUtils.sum(sentLevelCounts[d][s]));
                }
            }
        }

        for (int d = 0; d < D; d++) {
            int[] levelCounts = new int[L];
            for (STable table : localRestaurants[d].getTables()) {
                for (int s : table.getCustomers()) {
                    for (int n = 0; n < words[d][s].length; n++) {
                        levelCounts[z[d][s][n]]++;
                    }
                }
            }

            for (int ll = 0; ll < L; ll++) {
                if (levelCounts[ll] != docLevelDist[d].getCount(ll)) {
                    throw new RuntimeException(msg + ". Counts mismatch."
                            + " d = " + d
                            + " level = " + ll
                            + " " + levelCounts[ll]
                            + " vs. " + docLevelDist[d].getCount(ll));
                }
            }
        }

        for (int d = 0; d < D; d++) {
            int numNonEmptySentences = 0;
            for (int s = 0; s < words[d].length; s++) {
                if (words[d][s].length > 0) {
                    numNonEmptySentences++;
                }
            }

            int numCustomers = localRestaurants[d].getTotalNumCustomers();
            if (numCustomers != numNonEmptySentences) {
                throw new RuntimeException(msg + ". Number of non-empty sentences"
                        + " mismatch. " + numNonEmptySentences + " vs. "
                        + numCustomers);
            }
        }
    }

    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        int[] nodeCountPerLevel = new int[L];
        int[] obsCountPerLevel = new int[L];

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;
        int numEmptyLeaves = 0;
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            nodeCountPerLevel[node.getLevel()]++;
            obsCountPerLevel[node.getLevel()] += node.getContent().getCountSum();

            totalObs += node.getContent().getCountSum();

            if (isLeafNode(node)
                    && node.getContent().getSparseCounts().getCountSum() == 0) {
                numEmptyLeaves++;
            }

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("global tree:\n\t>>> node count per level: ");
        for (int l = 0; l < L; l++) {
            str.append(l).append("(")
                    .append(nodeCountPerLevel[l])
                    .append(", ").append(obsCountPerLevel[l])
                    .append(", ").append(MiscUtils.formatDouble(
                    (double) obsCountPerLevel[l] / nodeCountPerLevel[l]))
                    .append(");\t");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs)
                .append("\n\t>>> # customers = ").append(globalTreeRoot.getNumTables())
                .append("\n\t>>> # empty leaves = ").append(numEmptyLeaves);
        return str.toString();
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        str.append("global tree\n");

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;

        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    .append("\n");

            totalObs += node.getContent().getCountSum();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # customers = ").append(globalTreeRoot.getNumTables())
                .append("\n");
        return str.toString();
    }

    public String printLocalRestaurantSummary() {
        StringBuilder str = new StringBuilder();
        str.append("local restaurants:\n");
        int[] numTables = new int[D];
        int totalTableCusts = 0;
        for (int d = 0; d < D; d++) {
            numTables[d] = localRestaurants[d].getNumTables();
            for (STable table : localRestaurants[d].getTables()) {
                totalTableCusts += table.getNumCustomers();
            }
        }
        str.append("\t>>> # tables:")
                .append(". min: ").append(MiscUtils.formatDouble(StatUtils.min(numTables)))
                .append(". max: ").append(MiscUtils.formatDouble(StatUtils.max(numTables)))
                .append(". avg: ").append(MiscUtils.formatDouble(StatUtils.mean(numTables)))
                .append(". total: ").append(MiscUtils.formatDouble(StatUtils.sum(numTables)))
                .append("\n");
        str.append("\t>>> # customers: ").append(totalTableCusts);
        return str.toString();
    }

    public String printLocalRestaurants() {
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            logln("restaurant d = " + d
                    + ". # tables: " + localRestaurants[d].getNumTables()
                    + ". # total customers: " + localRestaurants[d].getTotalNumCustomers());
            for (STable table : localRestaurants[d].getTables()) {
                logln("--- table: " + table.toString());
            }
            System.out.println();
        }
        return str.toString();
    }

    public String printLocalRestaurant(int d) {
        StringBuilder str = new StringBuilder();
        str.append("restaurant d = ").append(d)
                .append(". # tables: ").append(localRestaurants[d].getNumTables())
                .append(". # total customers: ").append(localRestaurants[d]
                .getTotalNumCustomers()).append("\n");
        for (STable table : localRestaurants[d].getTables()) {
            str.append("--- table: ").append(table.toString()).append("\n");
        }
        return str.toString();
    }

    public void outputLexicalParameters(File filepath) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing lexical weights to " + filepath);
        }

        ArrayList<RankingItem<Integer>> sortedWeights = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            sortedWeights.add(new RankingItem<Integer>(v, lexParams[v]));
        }
        Collections.sort(sortedWeights);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (int v = 0; v < V; v++) {
                RankingItem<Integer> rankItem = sortedWeights.get(v);
                int lexIdx = rankItem.getObject();
                writer.write(lexIdx
                        + "\t" + this.wordVocab.get(lexIdx)
                        + "\t" + this.lexParams[lexIdx]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing lexical weights to "
                    + filepath);
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
            MimnoTopicCoherence topicCoherence) {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                for (SNode child : node.getChildren()) {
                    stack.add(child);
                }

                double[] distribution = node.getTopic();
                int[] topic = SamplerUtils.getSortedTopic(distribution);
                double score = topicCoherence.getCoherenceScore(topic);
                writer.write(node.getPathString()
                        + "\t" + node.getLevel()
                        + "\t" + node.getIterationCreated()
                        + "\t" + node.getContent().getCountSum()
                        + "\t" + score);
                for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                    writer.write("\t" + this.wordVocab.get(topic[i]));
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing "
                    + "topic coherence to " + file);
        }
    }

    /**
     * Output top words for each topic in the tree to text file.
     *
     * @param outputFile The output file
     * @param numWords Number of top words
     */
    public void outputTopicTopWords(File outputFile, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            ArrayList<RankingItem<SNode>> rankChildren = new ArrayList<RankingItem<SNode>>();
            for (SNode child : node.getChildren()) {
                rankChildren.add(new RankingItem<SNode>(child, child.getRegressionParameter()));
            }
            Collections.sort(rankChildren);
            for (RankingItem<SNode> item : rankChildren) {
                stack.add(item.getObject());
            }

            if (node.getIterationCreated() >= MAX_ITER - LAG) {
                continue;
            }

            // debug
            HashMap<Integer, Integer> cs = node.getContent().getObservations();
            ArrayList<RankingItem<Integer>> ranks = new ArrayList<RankingItem<Integer>>();
            for (int ii : cs.keySet()) {
                ranks.add(new RankingItem<Integer>(ii, cs.get(ii)));
            }
            Collections.sort(ranks);

            double[] nodeTopic = node.getTopic();
            String[] topWords = getTopWords(nodeTopic, numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getIterationCreated())
                    .append("; t:").append(node.getNumTables())
                    .append("; o:").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.getRegressionParameter()))
                    .append(")");
            for (String topWord : topWords) {
                str.append(" ").append(topWord);
            }
            str.append("\n\n");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(str.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topics "
                    + outputFile);
        }
    }

    public void outputHTML(
            File htmlFile,
            String[] docIds,
            String[][] rawSentences,
            int numSents,
            int numWords,
            String url) {
        if (verbose) {
            logln("--- Outputing result to HTML file " + htmlFile);
        }

        // rank sentences for each path
        HashMap<SNode, ArrayList<RankingItem<String>>> pathRankSentMap = getRankingSentences();

        StringBuilder str = new StringBuilder();
        str.append("<!DOCTYPE html>\n<html>\n");

        // header containing styles and javascript functions
        str.append("<head>\n");
        str.append("<link type=\"text/css\" rel=\"stylesheet\" "
                + "href=\"http://argviz.umiacs.umd.edu/teaparty/framing.css\">\n"); // style
        str.append("<script type=\"text/javascript\" "
                + "src=\"http://argviz.umiacs.umd.edu/teaparty/framing.js\"></script>\n"); // script
        str.append("</head>\n"); // end head

        // start body
        str.append("<body>\n");
        str.append("<table>\n");
        str.append("<tbody>\n");

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            ArrayList<RankingItem<SNode>> rankChildren = new ArrayList<RankingItem<SNode>>();
            for (SNode child : node.getChildren()) {
                rankChildren.add(new RankingItem<SNode>(child, child.getRegressionParameter()));
            }
            Collections.sort(rankChildren);
            for (RankingItem<SNode> rankChild : rankChildren) {
                stack.add(rankChild.getObject());
            }

            double[] nodeTopic = node.getTopic();
            String[] topWords = getTopWords(nodeTopic, numWords);

            if (node.getLevel() == 1) {
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td>\n")
                        .append("[Topic ").append(node.getPathString())
                        .append("] ")
                        .append(" (")
                        .append(node.getNumChildren())
                        .append("; ").append(node.getNumTables())
                        .append("; ").append(node.getContent().getCountSum())
                        .append("; ").append(formatter.format(node.getRegressionParameter()))
                        .append(")");
                for (String topWord : topWords) {
                    str.append(" ").append(topWord);
                }
                str.append("</td>\n");
                str.append("</tr>\n");
            } else if (node.getLevel() == 2) {
                ArrayList<RankingItem<String>> rankSents = pathRankSentMap.get(node);
                if (rankSents == null || rankSents.size() < 10) {
                    continue;
                }
                Collections.sort(rankSents);

                // node info
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td>\n")
                        .append("<a style=\"text-decoration:underline;color:blue;\" onclick=\"showme('")
                        .append(node.getPathString())
                        .append("');\" id=\"toggleDisplay\">")
                        .append("[")
                        .append(node.getPathString())
                        .append("]</a>")
                        .append(" (").append(node.getNumTables())
                        .append("; ").append(node.getContent().getCountSum())
                        .append("; ").append(formatter.format(node.getRegressionParameter()))
                        .append(")");
                for (String topWord : topWords) {
                    str.append(" ").append(topWord);
                }
                str.append("</td>\n");
                str.append("</tr>\n");

                // sentences
                str.append("<tr class=\"level").append(L).append("\"")
                        .append(" id=\"").append(node.getPathString()).append("\"")
                        .append(" style=\"display:none;\"")
                        .append(">\n");
                str.append("<td>\n");

                for (int ii = 0; ii < Math.min(numSents, rankSents.size()); ii++) {
                    RankingItem<String> sent = rankSents.get(ii);
                    int d = Integer.parseInt(sent.getObject().split("-")[0]);
                    int s = Integer.parseInt(sent.getObject().split("-")[1]);

                    String debateId = docIds[d].substring(0, docIds[d].indexOf("_"));
                    if (debateId.startsWith("pre-")) {
                        debateId = debateId.substring(4);
                    }
                    str.append("<a href=\"")
                            .append(url)
                            .append(debateId).append(".xml")
                            .append("\" ")
                            .append("target=\"_blank\">")
                            .append(docIds[d]).append("_").append(s)
                            .append("</a> ")
                            .append(rawSentences[d][s])
                            .append("<br/>\n");
                    str.append("</br>");
                }
                str.append("</td>\n</tr>\n");
            }
        }

        str.append("</tbody>\n");
        str.append("</table>\n");
        str.append("</body>\n");
        str.append("</html>");

        // output to file
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(htmlFile);
            writer.write(str.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing HTML to "
                    + htmlFile);
        }
    }

    private HashMap<SNode, ArrayList<RankingItem<String>>> getRankingSentences() {
        HashMap<SNode, ArrayList<RankingItem<String>>> pathRankSentMap =
                new HashMap<SNode, ArrayList<RankingItem<String>>>();
        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                SNode pathNode = table.getContent();
                ArrayList<RankingItem<String>> rankSents = pathRankSentMap.get(pathNode);
                if (rankSents == null) {
                    rankSents = new ArrayList<RankingItem<String>>();
                }
                for (int s : table.getCustomers()) {
                    if (words[d][s].length < 10) { // filter out too short sentences
                        continue;
                    }

                    double logprob = 0.0;
                    for (int n = 0; n < words[d][s].length; n++) {
                        if (z[d][s][n] == L - 1) {
                            logprob += pathNode.getLogProbability(words[d][s][n]);
                        }
                    }
                    if (logprob != 0) {
                        rankSents.add(new RankingItem<String>(d + "-" + s, logprob / words[d][s].length));
                    }
                }
                pathRankSentMap.put(pathNode, rankSents);
            }
        }
        return pathRankSentMap;
    }

    protected void sampleNewDocuments(
            File stateFile,
            int[][][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Perform regression using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        this.words = newWords;
        this.responses = null;
        this.D = this.words.length;

        this.prepareDataStatistics();

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + tokenCount);
            logln("--- # sentences:\t" + sentCount);
        }

        try {
            inputModel(stateFile.getAbsolutePath());
//            updateAverageTopicsPerLevel();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + stateFile);
        }

        // initialize structure for test data
        initializeDataStructure();

        if (verbose) {
            logln("Initialized data structure");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        // initialize random assignments
        initializeTestAssignments();

        updateAuthorValues();

        if (verbose) {
            logln("Initialized random assignments");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        if (debug) {
            validateAssignments("Initialized");
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < testMaxIter; iter++) {
            double loglikelihood = getLogLikelihood();

            if (verbose) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\n*** *** # sents change: " + numSentAsntsChange
                        + " / " + sentCount
                        + " (" + (double) numSentAsntsChange / sentCount + ")"
                        + "\n*** *** # tables change: " + numTableAsgnsChange
                        + " / " + globalTreeRoot.getNumTables()
                        + " (" + (double) numTableAsgnsChange / globalTreeRoot.getNumTables() + ")"
                        + "\n" + getCurrentState()
                        + "\n";
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            numSentAsntsChange = 0;

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (!isValidSentence(d, s)) {
                        continue;
                    }

                    // if this document has only 1 sentence, no sampling is needed
                    if (words[d].length > 1) {
                        sampleSentenceAssignmentsExact(d, s, !REMOVE, !ADD,
                                REMOVE, ADD, !OBSERVED, !EXTEND);
                    }
                }

//                for (STable table : this.localRestaurants[d].getTables()) {
//                    samplePathForTable(d, table,
//                            !REMOVE, !ADD, REMOVE, ADD,
//                            !OBSERVED, !EXTEND);
//                }
            }

            updateAuthorValues();

            if (verbose && iter % testSampleLag == 0) {
                logln("--- iter = " + iter + " / " + testMaxIter);
            }

            if (iter >= testBurnIn && iter % testSampleLag == 0) {
                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);
            }

            if (responses != null) {
                logln("--- iter = " + iter + " / " + testMaxIter);
                double[] predResponses = getRegressionValues();
                evaluateRegressPrediction(responses, predResponses);
            }
        }

        // output result during test time 
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        PredictionUtils.outputSingleModelRegressions(
                new File(outputResultFile),
                predResponsesList);
    }

    private void initializeTestAssignments() {
        for (int d = 0; d < D; d++) {
            HashMap<SNode, STable> nodeTableMap = new HashMap<SNode, STable>();

            for (int s = 0; s < words[d].length; s++) {
                if (!isValidSentence(d, s)) {
                    continue;
                }
                SparseCount obs = new SparseCount();
                for (int n = 0; n < words[d][s].length; n++) {
                    obs.increment(words[d][s][n]);
                }

                // recursively sample a node for this sentence
                SNode leafNode = recurseNode(globalTreeRoot, obs);
                STable table = nodeTableMap.get(leafNode);
                if (table == null) {
                    int tabIdx = localRestaurants[d].getNextTableIndex();
                    table = new STable(iter, tabIdx, leafNode, d);
                    localRestaurants[d].addTable(table);
                    addTableToPath(leafNode);
                    nodeTableMap.put(leafNode, table);
                }
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // sample level for each token
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int n = 0; n < words[d][s].length; n++) {
                    double[] logprobs = new double[L];
                    for (int ll = 0; ll < L; ll++) {
                        logprobs[ll] = docLevelDist[d].getLogLikelihood(ll)
                                + path[ll].getLogProbability(words[d][s][n]);
                    }
                    int lvl = SamplerUtils.logMaxRescaleSample(logprobs);
                    z[d][s][n] = lvl;
                    sentLevelCounts[d][s][z[d][s][n]]++;
                    docLevelDist[d].increment(z[d][s][n]);
                }
            }
        }
    }

    class SNode extends TopicTreeNode<SNode, DirMult> {

        private final int born;
        private int numTables;
        private double regression;

        SNode(int iter, int index, int level,
                DirMult content,
                double regParam,
                SNode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numTables = 0;
            this.regression = regParam;
        }

        public int getIterationCreated() {
            return this.born;
        }

        /**
         * Get the log probability of a set of observations given the topic at
         * this node.
         *
         * @param obs The set of observations
         */
        double getLogProbability(SparseCount obs) {
            if (this.getTopic() == null) {
                return this.content.getLogLikelihood(obs.getObservations());
            } else {
                double val = 0.0;
                for (int o : obs.getIndices()) {
                    val += obs.getCount(o) * this.getLogProbability(o);
                }
                return val;
            }
        }

        @Override
        public double getLogProbability(int obs) {
            return this.content.getLogLikelihood(obs);
        }

        double getLogJointProbability(double gamma) {
            ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
            for (SNode child : this.getChildren()) {
                numChildrenCusts.add(child.getNumTables());
            }
            return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts,
                    gamma);
        }

        double getRegressionParameter() {
            return this.regression;
        }

        void setRegressionParameter(double reg) {
            this.regression = reg;
        }

        int getNumTables() {
            return this.numTables;
        }

        void decrementNumTables() {
            this.numTables--;
        }

        void incrementNumTables() {
            this.numTables++;
        }

        void changeNumTables(int delta) {
            this.numTables += delta;
        }

        boolean isEmpty() {
            return this.numTables == 0;
        }

        @Override
        public void sampleTopic(double beta, double gamma) {
            int V = content.getDimension();
            double[] meanVector = new double[V];
            Arrays.fill(meanVector, beta / V);
            if (!this.isRoot()) { // root
                double[] parentTopic = (parent.getContent()).getSamplingDistribution();
                for (int v = 0; v < V; v++) {
                    meanVector[v] += parentTopic[v] * gamma;
                }
            }
            SparseCount observations = this.content.getSparseCounts();
            for (int obs : observations.getIndices()) {
                meanVector[obs] += observations.getCount(obs);
            }
            for (int obs : this.pseudoCounts.getIndices()) {
                meanVector[obs] += this.pseudoCounts.getCount(obs);
            }

            double[] ts = new double[V];
            double sum = 0.0;
            for (int v = 0; v < V; v++) {
                ts[v] = randoms.nextGamma(meanVector[v], 1);
                if (ts[v] <= 0) {
                    ts[v] = 0.001;
                }
                sum += ts[v];
            }
            // normalize
            for (int v = 0; v < V; v++) {
                ts[v] /= sum;
            }

//            Dirichlet dir = new Dirichlet(meanVector);
//            double[] topic = dir.nextDistribution();
//
            for (int v = 0; v < V; v++) {
                if (ts[v] == 0) {
                    throw new RuntimeException("v: " + v
                            + "\t" + ts[v]
                            + "\t" + meanVector[v]
                            + "\t" + sum);
                }
            }

            this.setTopic(ts);
        }

        @Override
        public void setTopic(double[] topic) {
            this.content.setSamplingDistribution(topic);
            this.logTopics = new double[topic.length];
            for (int v = 0; v < topic.length; v++) {
                this.logTopics[v] = Math.log(topic[v]);
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getPathString())
                    .append(" (").append(born).append(")")
                    .append(" #ch = ").append(getNumChildren())
                    .append(", #c = ").append(getNumTables())
                    .append(", #o = ").append(getContent().getCountSum())
                    .append(", reg = ").append(MiscUtils.formatDouble(regression))
                    .append("]");
            return str.toString();
        }

        void validate(String msg) {
            int maxChildIndex = PSEUDO_NODE_INDEX;
            for (SNode child : this.getChildren()) {
                if (maxChildIndex < child.getIndex()) {
                    maxChildIndex = child.getIndex();
                }
            }

            for (int i = 0; i < maxChildIndex; i++) {
                if (!inactiveChildren.contains(i) && !hasChild(i)) {
                    throw new RuntimeException(msg + ". Child inactive indices"
                            + " have not been updated. Node: " + this.toString()
                            + ". Index " + i + " is neither active nor inactive");
                }
            }
        }
    }

    class STable extends FullTable<Integer, SNode> {

        private final int born;
        private final int restIndex;

        public STable(int iter, int index,
                SNode content, int restId) {
            super(index, content);
            this.born = iter;
            this.restIndex = restId;
        }

        public int getRestaurantIndex() {
            return this.restIndex;
        }

        public boolean containsCustomer(int c) {
            return this.customers.contains(c);
        }

        public int getIterationCreated() {
            return this.born;
        }

        public String getTableId() {
            return restIndex + ":" + index;
        }

        @Override
        public int hashCode() {
            String hashCodeStr = getTableId();
            return hashCodeStr.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if ((obj == null) || (this.getClass() != obj.getClass())) {
                return false;
            }
            STable r = (STable) (obj);

            return r.index == this.index
                    && r.restIndex == this.restIndex;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getTableId())
                    .append(", ").append(born)
                    .append(", ").append(getNumCustomers())
                    .append("]")
                    .append(" >> ").append(getContent() == null ? "null" : getContent().toString());
            return str.toString();
        }
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newAuthors Authors of new documents
     * @param numAuthors Number of authors in the test data
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     */
//    public static void parallelTest(int[][][] newWords, int[] newAuthors, int numAuthors,
//            File iterPredFolder, AuthorSHLDA_MH sampler) {
//        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
//        if (!reportFolder.exists()) {
//            throw new RuntimeException("Report folder not found. " + reportFolder);
//        }
//
//        // debug
//        System.out.println("# new docs: " + newWords.length);
//        System.out.println("# new authors: " + newAuthors.length);
//        System.out.println("# new authors: " + numAuthors);
//
//        String[] filenames = reportFolder.list();
//        try {
//            IOUtils.createFolder(iterPredFolder);
//            ArrayList<Thread> threads = new ArrayList<Thread>();
//            for (int i = 0; i < filenames.length; i++) {
//                String filename = filenames[i];
//                if (!filename.contains("zip")) {
//                    continue;
//                }
//
//                File stateFile = new File(reportFolder, filename);
//                File partialResultFile = new File(iterPredFolder,
//                        IOUtils.removeExtension(filename) + ".txt");
//                AuthorSHLDAMHTestRunner runner = new AuthorSHLDAMHTestRunner(sampler,
//                        newWords, newAuthors,
//                        numAuthors, stateFile.getAbsolutePath(),
//                        partialResultFile.getAbsolutePath());
//                Thread thread = new Thread(runner);
//                threads.add(thread);
//            }
//
//            // run MAX_NUM_PARALLEL_THREADS threads at a time
//            runThreads(threads);
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while sampling during parallel test.");
//        }
//    }
    public static void main(String[] args) {
        run(args);
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + SHLDA.class.getName() + " -help";
    }

    private static void addOptions() {
        // directories
        addOption("output", "Output folder");
        addOption("dataset", "Dataset");
        addOption("data-folder", "Processed data folder");
        addOption("format-folder", "Folder holding formatted data");
        addOption("format-file", "Format file");

        // sampler configurations
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval.");
        addOption("seeded-asgn-file", "Directory of file containing the "
                + "seeded assignments.");

        // hyperparameters
        addOption("T", "Lexical regression regularizer");
        addOption("init-branch-factor", "Initial branching factors at each level. "
                + "The length of this array should be equal to L-1 (where L "
                + "is the number of levels in the tree).");
        addOption("num-topics", "Number of topics for initialization");
        addOption("num-frames", "Number of frames per-topic for initialization");
        addOption("gem-mean", "GEM mean. [0.5]");
        addOption("gem-scale", "GEM scale. [50]");
        addOption("betas", "Dirichlet hyperparameter for topic distributions."
                + " [1, 0.5, 0.25] for a 3-level tree.");
        addOption("gammas", "DP hyperparameters. [1.0, 1.0] for a 3-level tree");
        addOption("mus", "Prior means for topic regression parameters."
                + " [0.0, 0.0, 0.0] for a 3-level tree and standardized"
                + " response variable.");
        addOption("sigmas", "Prior variances for topic regression parameters."
                + " [0.0001, 0.5, 1.0] for a 3-level tree and stadardized"
                + " response variable.");
        addOption("rho", "Prior variance for response variable. [1.0]");
        addOption("tau-mean", "Prior mean of lexical regression parameters. [0.0]");
        addOption("tau-scale", "Prior scale of lexical regression parameters. [1.0]");
        addOption("num-lex-items", "Number of non-zero lexical regression parameters."
                + " Defaule: vocabulary size.");

        // cross validation
        addOption("cv-folder", "Cross validation folder");
        addOption("num-folds", "Number of folds");
        addOption("fold", "The cross-validation fold to run");
        addOption("run-mode", "Running mode");

        options.addOption("paramOpt", false, "Whether hyperparameter "
                + "optimization using slice sampling is performed");
        options.addOption("z", false, "whether standardize (z-score normalization)");
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "Help");

        options.addOption("train", false, "train");
        options.addOption("dev", false, "development");
        options.addOption("test", false, "test");
        options.addOption("z", false, "z-normalization");

        addOption("prediction-folder", "Prediction folder");
        addOption("evaluation-folder", "Evaluation folder");
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
            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModel();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Use option -help for all options.");
        }
    }

    private static void runModel() throws Exception {
        // sampling configurations
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 5);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 10);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        // directories
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder",
                "demo");
        String resultFolder = CLIUtils.getStringArgument(cmd, "output",
                "demo/amazon-data/format-response/model");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-response");
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

        // parameters
        int L = CLIUtils.getIntegerArgument(cmd, "tree-height", 3);

        double[] defaultPis = new double[L];
        for (int ii = 0; ii < L; ii++) {
            defaultPis[ii] = 1000 / 3;
        }
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis", defaultPis, ",");

        double[] defaultBetas = new double[L];
        defaultBetas[0] = 1;
        for (int i = 1; i < L; i++) {
            defaultBetas[i] = 1.0 / (i + 1);
        }
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", defaultBetas, ",");

        double[] defaultGammas = new double[L - 1];
        for (int i = 0; i < defaultGammas.length; i++) {
            defaultGammas[i] = 1.0 / (i + 1);
        }
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", defaultGammas, ",");

        InitialState initState = InitialState.PRESET;

        double[] defaultMus = new double[L];
        for (int i = 0; i < L; i++) {
            defaultMus[i] = 0.0;
        }
        double[] mus = CLIUtils.getDoubleArrayArgument(cmd, "mus", defaultMus, ",");

        double[] defaultSigmas = new double[L];
        defaultSigmas[0] = 0.0001; // root node
        defaultSigmas[1] = 50;
        defaultSigmas[2] = 250;

        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", defaultSigmas, ",");
        double tau_mean = CLIUtils.getDoubleArgument(cmd, "tau-mean", 0.0);
        double tau_scale = CLIUtils.getDoubleArgument(cmd, "tau-sigma", 1000);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 1.0);
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 5);

        // initialization
        String branchFactorStr = CLIUtils.getStringArgument(cmd,
                "init-branch-factor", "16-3");
        String[] sstr = branchFactorStr.split("-");
        int[] branch = new int[sstr.length];
        for (int ii = 0; ii < branch.length; ii++) {
            branch[ii] = Integer.parseInt(sstr[ii]);
        }

        SHLDA sampler = new SHLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        String seededAsgnFile = cmd.getOptionValue("seeded-asgn-file");
        sampler.setSeededAssignmentFile(seededAsgnFile);
        initState = InitialState.SEEDED;

        sampler.configure(resultFolder,
                data.getWordVocab().size(), L,
                alpha,
                rho,
                tau_mean, tau_scale,
                betas, gammas,
                mus, sigmas,
                pis,
                branch,
                initState,
                PathAssumption.MAXIMAL,
                paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File samplerFolder = new File(resultFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);

        // train
        if (cmd.hasOption("train")) {
            sampler.train(data.getSentenceWords(), data.getResponses());
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
            sampler.test(data);

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

    private static void runCrossValidation() throws Exception {
    }
}