/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.backup;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.LDA;
import sampler.supervised.objective.GaussianHierLinearRegObjective;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.FullTable;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class SentCrpNCrpSampler extends AbstractSampler {

    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    protected boolean supervised = true;
    public static final int ALPHA = 0;
    public static final int RHO = 1; // response variable variances
    public static final int GEM_MEAN = 2;
    public static final int GEM_SCALE = 3;
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;    // regression parameter means
    protected double[] sigmas; // regression parameter variances
    private double sqrtRho;
    private double[] sqrtSigmas;
    private double[] logGammas;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K; // initial number of tables
    protected int[][][] words;  // [D] x [S_d] x [N_ds]: words
    protected double[] responses; // [D]
    private STable[][] c; // table assigned to sentences
    private int[][][] z; // level assigned to tokens
    private SNode globalTreeRoot;
    private Restaurant<STable, Integer, SNode>[] localRestaurants;
    private TruncatedStickBreaking[] docLevelDists;
    private SparseCount[][] sentLevelCounts;
    private double[] lexicalWeights;
    private double[] docLexicalWeightSum;
    private ArrayList<double[]> lexicalWeightsOverTime;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private int sentCount;
    private int tokenCount;
    private int[] docTokenCounts;
    private double[] uniform;
    private DirMult[] emptyModels;
    private int numTokenAssignmentsChange;
    private int numSentAssignmentsChange;
    private int numTableAssignmentsChange;
    private int numConverged;

    public void configure(String folder,
            int[][][] words,
            double[] responses,
            int V, int L,
            double alpha,
            double rho,
            double gem_mean,
            double gem_scale,
            double[] betas,
            double[] gammas,
            double[] mus,
            double[] sigmas,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;
        this.responses = responses;

        this.V = V;
        this.L = L;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;
        this.mus = mus;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(rho);
        this.hyperparams.add(gem_mean);
        this.hyperparams.add(gem_scale);
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

        this.updatePrecomputedHyperparameters();

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

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

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        sentCount = 0;
        tokenCount = 0;
        docTokenCounts = new int[D];
        for (int d = 0; d < D; d++) {
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
                docTokenCounts[d] += words[d][s].length;
            }
        }

        logln("--- V = " + V);
        logln("--- # documents = " + D); // number of groups
        logln("--- # sentences = " + sentCount);
        logln("--- # tokens = " + tokenCount);

        if (!debug) {
            System.err.close();
        }
    }

    public void setSupervised(boolean s) {
        this.supervised = s;
    }

    public boolean isSupervised() {
        return this.supervised;
    }

    private void updatePrecomputedHyperparameters() {
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

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_lex-MSHLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_gm-").append(formatter.format(hyperparams.get(GEM_MEAN)))
                .append("_gs-").append(formatter.format(hyperparams.get(GEM_SCALE)));
        int count = GEM_SCALE + 1;
        str.append("_b");
        for (int i = 0; i < betas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_g");
        for (int i = 0; i < gammas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_m");
        for (int i = 0; i < mus.length; i++) {
            str.append("-").append(formatter.format(mus[i]));
        }
        str.append("_s");
        for (int i = 0; i < sigmas.length; i++) {
            str.append("-").append(formatter.format(sigmas[i]));
        }
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void setK(int K) {
        this.K = K;
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

        if (verbose) {
            logln("--- --- Done initializing.\n" + getCurrentState());
            logln(printGlobalTree());
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        if (debug) {
            validate("Initialized");
        }
    }

    private void initializeModelStructure() {
        int rootLevel = 0;
        int rootIndex = 0;
        DirMult dmModel = new DirMult(V, betas[rootLevel], uniform);
        double regParam = SamplerUtils.getGaussian(mus[rootLevel], sigmas[rootLevel]);
        this.globalTreeRoot = new SNode(iter, rootIndex, rootLevel, dmModel, regParam, null);

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }
    }

    private void initializeDataStructure() {
        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<STable, Integer, SNode>();
        }

        this.docLevelDists = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            this.docLevelDists[d] = new TruncatedStickBreaking(L, hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
        }

        this.sentLevelCounts = new SparseCount[D][];
//        this.sentRegressionSums = new double[D][];
        for (int d = 0; d < D; d++) {
            this.sentLevelCounts[d] = new SparseCount[words[d].length];
//            this.sentRegressionSums[d] = new double[words[d].length];
            for (int s = 0; s < words[d].length; s++) {
                this.sentLevelCounts[d][s] = new SparseCount();
            }
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
    }

    private void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            case PRESET:
                this.initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments ...");
        }

        if (this.K == 0) {
            this.K = 50;
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

        int[][] flattenWords = new int[D][];
        for (int d = 0; d < D; d++) {
            flattenWords[d] = new int[docTokenCounts[d]];
            int idx = 0;
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    flattenWords[d][idx++] = words[d][s][n];
                }
            }
        }

        lda.configure(null, flattenWords, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            String ldaFile = this.folder + "lda-init-" + K + ".txt";
            File ldaZFile = new File(ldaFile);
            if (ldaZFile.exists()) {
                ldaZ = inputLDAInitialization(ldaFile);
            } else {
                lda.sample();
                ldaZ = lda.getZ();
                outputLDAInitialization(ldaFile, ldaZ);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(this.folder, "lda-topwords.txt"), 15);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        setLog(true);

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int k = 0; k < K; k++) {
                STable table = new STable(iter, k, null, d);
                this.localRestaurants[d].addTable(table);
            }

            // assign table to sentence
            int idx = 0;
            for (int s = 0; s < words[d].length; s++) {
                int[] sentTopicCounts = new int[K];
                for (int n = 0; n < words[d][s].length; n++) {
                    int topicIdx = ldaZ[d][idx++];
                    sentTopicCounts[topicIdx]++;
                }

                int singleTopic = SamplerUtils.maxIndex(sentTopicCounts);
                c[d][s] = this.localRestaurants[d].getTable(singleTopic);
                localRestaurants[d].addCustomerToTable(s, c[d][s].getIndex());
            }

            // initialize all tokens to the leaf level
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    z[d][s][n] = L - 1;
                    docLevelDists[d].increment(z[d][s][n]);
                    sentLevelCounts[d][s].increment(z[d][s][n]);
                }
            }

            // assign tree path to table
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (STable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }

                samplePathForTable(d, table, !REMOVE, ADD, !OBSERVED, EXTEND);
            }

            // remove empty table
            for (int tIndex : emptyTables) {
                this.localRestaurants[d].removeTable(tIndex);
            }
        }
    }

    private int[][] inputLDAInitialization(String filepath) {
        if (verbose) {
            logln("--- --- LDA init file found. Loading from " + filepath);
        }

        int[][] ldaZ = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            int numDocs = Integer.parseInt(reader.readLine());
            ldaZ = new int[numDocs][];
            for (int d = 0; d < numDocs; d++) {
                String[] sline = reader.readLine().split("\t")[1].split(" ");
                ldaZ[d] = new int[sline.length];
                for (int n = 0; n < ldaZ[d].length; n++) {
                    ldaZ[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return ldaZ;
    }

    private void outputLDAInitialization(String filepath, int[][] z) {
        if (verbose) {
            logln("--- --- Outputing LDA init state to file " + filepath);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            writer.write(z.length + "\n");
            for (int d = 0; d < z.length; d++) {
                writer.write(z[d].length + "\t");
                for (int n = 0; n < z[d].length; n++) {
                    writer.write(z[d][n] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // create a new path for each table
                SNode node = globalTreeRoot;
                for (int l = 0; l < L - 1; l++) {
                    node = createNode(node);
                }
                addTableToPath(node);
                table.setContent(node);

                // sample level
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelForToken(d, s, n, !REMOVE, ADD, !OBSERVED);
                }

                if (d > 0 || s > 0) {
                    sampleTableForSentence(d, s, REMOVE, ADD, !OBSERVED, EXTEND);
                }

                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelForToken(d, s, n, REMOVE, ADD, !OBSERVED);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, EXTEND);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();

        try {
            if (report) {
                IOUtils.createFolder(this.folder + this.getSamplerFolder() + ReportFolder);
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

            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens: " + numTokenAssignmentsChange
                            + "\t # sents: " + numSentAssignmentsChange
                            + "\t # tables: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState()
                            + "\n");
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens: " + numTokenAssignmentsChange
                            + "\t # sents: " + numSentAssignmentsChange
                            + "\t # tables: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState()
                            + "\n");
                }
            }

            numTableAssignmentsChange = 0;
            numSentAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;
            numConverged = 0;

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    sampleTableForSentence(d, s, REMOVE, ADD, OBSERVED, EXTEND);

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelForToken(d, s, n, REMOVE, ADD, OBSERVED);
                    }
                }

                for (STable table : this.localRestaurants[d].getTables()) {
                    samplePathForTable(d, table, REMOVE, ADD, OBSERVED, EXTEND);
                }
            }

            if (isSupervised()) {
                updateRegressionParameters();
            }

            if (verbose && isSupervised()) {
                double[] trPredResponses = getRegressionValues();
                RegressionEvaluation eval = new RegressionEvaluation(
                        (responses),
                        (trPredResponses));
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
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
            System.out.println();

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
                try {
                    outputTopicTopWords(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + "-top-words.txt", 15);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            }
        }

        if (verbose) {
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        if (report) {
            outputState(this.folder + this.getSamplerFolder() + "final.zip");
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    private void addTableToPath(SNode leafNode) {
        SNode node = leafNode;
        while (node != null) {
            node.incrementNumCustomers();
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
    private SNode removeTableFromPath(SNode leafNode) {
        SNode retNode = leafNode;
        SNode node = leafNode;
        while (node != null) {
            node.decrementNumCustomers();
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
    private SNode[] addObservationsToPath(SNode leafNode, HashMap<Integer, Integer>[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations (given their level assignments) from a path
     *
     * @param leafNode The leaf node identifying the path
     * @param observations The observations per level
     */
    private SNode[] removeObservationsFromPath(SNode leafNode, HashMap<Integer, Integer>[] observations) {
        SNode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
        return path;
    }

    /**
     * Remove a set of observations from a node
     *
     * @param node The node
     * @param observations The set of observations
     */
    private void removeObservationsFromNode(SNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    /**
     * Add a set of observations to a node
     *
     * @param node The node
     * @param observations The set of observations
     */
    private void addObservationsToNode(SNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    private SNode createNewPath(SNode internalNode) {
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
    private SNode createNode(SNode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmm = new DirMult(V, betas[level], uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SNode child = new SNode(iter, nextChildIndex, level, dmm, regParam, parent);
        return parent.addChild(nextChildIndex, child);
    }

    /**
     * Sample a table assignment for a sentence
     *
     * @param d The document index
     * @param s The sentence index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response is observed
     * @param extend Whether the structure is extendable
     */
    private void sampleTableForSentence(int d, int s, boolean remove, boolean add,
            boolean observed, boolean extend) {
        STable curTable = c[d][s];

        HashMap<Integer, Integer>[] sentObsCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            sentObsCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[d][s].length; n++) {
            int type = words[d][s][n];
            int level = z[d][s][n];
            Integer count = sentObsCountPerLevel[level].get(type);
            if (count == null) {
                sentObsCountPerLevel[level].put(type, 1);
            } else {
                sentObsCountPerLevel[level].put(type, count + 1);
            }
        }

        if (remove) {
            removeObservationsFromPath(c[d][s].getContent(), sentObsCountPerLevel);
            localRestaurants[d].removeCustomerFromTable(s, c[d][s].getIndex());
            if (c[d][s].isEmpty()) {
                removeTableFromPath(c[d][s].getContent());
                localRestaurants[d].removeTable(c[d][s].getIndex());
            }
        }

        double preSum = 0.0;
        if (observed) {
            // this should be pre-computed and updated within each document d
            for (int t = 0; t < words[d].length; t++) {
                if (t == s) // skip the current sentence
                {
                    continue;
                }
                preSum += computeRegressionSum(d, t);
            }
        }

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logProbs = new ArrayList<Double>();

        // existing tables
        for (STable table : localRestaurants[d].getTables()) {
            double logprior = Math.log(table.getNumCustomers());
            SNode[] path = getPathFromNode(table.getContent());
            double wordLlh = 0.0;
            for (int l = 0; l < L; l++) {
                wordLlh += path[l].getContent().getLogLikelihood(sentObsCountPerLevel[l]);
            }

            double resLlh = 0.0;
            if (observed) {
                double addSum = 0.0;
                for (int l = 0; l < L; l++) {
                    addSum += path[l].getRegressionParameter() * sentLevelCounts[d][s].getCount(l);
                }

                double mean = (preSum + addSum) / docTokenCounts[d];
                resLlh = StatUtils.logNormalProbability(responses[d], mean, sqrtRho);
            }

            double lp = logprior + wordLlh + resLlh;
            logProbs.add(lp);
            tableIndices.add(table.getIndex());

            // debug
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". table: " + table.toString()
//                    + ". log prior = " + MiscUtils.formatDouble(logprior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (extend) {
            // log priors
            computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

            // word log likelihoods
            double[] dataLlhNewTopic = new double[L];
            for (int l = 1; l < L; l++) // skip the root
            {
                dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(sentObsCountPerLevel[l]);
            }
            computePathWordLogLikelihood(pathWordLlhs, globalTreeRoot, sentObsCountPerLevel, dataLlhNewTopic, 0.0);

            // debug
            if (pathLogPriors.size() != pathWordLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }

            // response log likelihoods
            if (supervised && observed) {
                pathResLlhs = computePathResponseLogLikelihood(d, s, preSum);

                if (pathLogPriors.size() != pathResLlhs.size()) {
                    throw new RuntimeException("Numbers of paths mismatch");
                }
            }

            double logPrior = Math.log(hyperparams.get(ALPHA));
            double marginals = computeMarginals(pathLogPriors, pathWordLlhs, pathResLlhs, observed);

            double lp = logPrior + marginals;
            logProbs.add(lp);
            tableIndices.add(PSEUDO_TABLE_INDEX);

            // debug
//            logln("iter = " + iter + ". d = " + d + ". s = " + s
//                    + ". new table"
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". marginal = " + MiscUtils.formatDouble(marginals)
//                    + ". lp = " + MiscUtils.formatDouble(lp));
        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        int tableIdx = tableIndices.get(sampledIndex);

        // debug
//        logln(">>> idx = " + sampledIndex + ". tabIdx = " + tableIdx + "\n");

        if (curTable != null && curTable.getIndex() != tableIdx) {
            numSentAssignmentsChange++;
        }

        STable table;
        if (tableIdx == PSEUDO_NODE_INDEX) {
            int newTableIdx = localRestaurants[d].getNextTableIndex();
            table = new STable(iter, newTableIdx, null, d);
            localRestaurants[d].addTable(table);

            SNode newNode = samplePath(pathLogPriors, pathWordLlhs, pathResLlhs, observed);
            if (!isLeafNode(newNode)) {
                newNode = createNewPath(newNode);
            }
            table.setContent(newNode);
            addTableToPath(table.getContent());
        } else {
            table = localRestaurants[d].getTable(tableIdx);
        }

        c[d][s] = table;
//        sentRegressionSums[d][s] = computeRegressionSum(d, s);

        if (add) {
            addObservationsToPath(table.getContent(), sentObsCountPerLevel);
            localRestaurants[d].addCustomerToTable(s, table.getIndex());
        }
    }

    /**
     * Sample a level for a token
     *
     * @param d The document index
     * @param s The sentence index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response variable is observed
     */
    private void sampleLevelForToken(int d, int s, int n, boolean remove, boolean add, boolean observed) {
        STable curTable = c[d][s];
        SNode[] curPath = getPathFromNode(curTable.getContent());

        if (remove) {
            docLevelDists[d].decrement(z[d][s][n]);
            sentLevelCounts[d][s].decrement(z[d][s][n]);
            curPath[z[d][s][n]].getContent().decrement(words[d][s][n]);
        }

        double preSum = 0.0;
        if (observed) {
            for (int t = 0; t < words[d].length; t++) {
                preSum += computeRegressionSum(d, t);
            }
            preSum -= curPath[z[d][s][n]].getRegressionParameter();
        }

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            double logPrior = docLevelDists[d].getLogProbability(l);
//            double logPrior = docLevelDists[d].getLogLikelihood(l);
            double wordLlh = curPath[l].getContent().getLogLikelihood(words[d][s][n]);
            double resLlh = 0.0;
            if (observed) {
                double sum = preSum + curPath[l].getRegressionParameter();
                double mean = sum / docTokenCounts[d];
                resLlh = StatUtils.logNormalProbability(responses[d], mean, sqrtRho);
            }
            logprobs[l] = logPrior + wordLlh + resLlh;

            // debug
//            logln("iter = " + iter + ". " + d + ":" + s + ":" + n
//                    + ". l = " + l + ". count = " + docLevelDists[d].getCount(l)
//                    + ". log prior = " + MiscUtils.formatDouble(logPrior)
//                    + ". word llh = " + MiscUtils.formatDouble(wordLlh)
//                    + ". res llh = " + MiscUtils.formatDouble(resLlh)
//                    + ". lp = " + MiscUtils.formatDouble(logprobs[l]));
        }

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);

        // debug
//        logln("--->>> sampled level = " + sampledL + "\n");

        if (z[d][s][n] != sampledL) {
            numTokenAssignmentsChange++;
        }

        // update and increment
        z[d][s][n] = sampledL;

        if (add) {
            docLevelDists[d].increment(z[d][s][n]);
            sentLevelCounts[d][s].increment(z[d][s][n]);
            curPath[z[d][s][n]].getContent().increment(words[d][s][n]);
//            sentRegressionSums[d][s] += curPath[z[d][s][n]].getRegressionParameter();
        }
    }

    /**
     * Sample a path on the global tree for a table
     *
     * @param d The restaurant index
     * @param table The table
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param observed Whether the response variable is observed
     * @param extend Whether the global tree is extendable
     */
    private void samplePathForTable(int d, STable table, boolean remove, boolean add, boolean observed, boolean extend) {
        SNode curLeaf = table.getContent();

        // observations of sentences currently being assign to this table
        HashMap<Integer, Integer>[] obsCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            obsCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int s : table.getCustomers()) {
            for (int n = 0; n < words[d][s].length; n++) {
                int level = z[d][s][n];
                int obs = words[d][s][n];

                Integer count = obsCountPerLevel[level].get(obs);
                if (count == null) {
                    obsCountPerLevel[level].put(obs, 1);
                } else {
                    obsCountPerLevel[level].put(obs, count + 1);
                }
            }
        }

        // data likelihood for new nodes at each level
        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) // skip the root
        {
            dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(obsCountPerLevel[l]);
        }

        boolean condition = false;
        if (condition) {
            logln("iter = " + iter + ". d = " + d + ". tabIdx = " + table.getTableId());
            logln(printGlobalTree());
            logln(printLocalRestaurant(d));
        }

        if (remove) {
            removeObservationsFromPath(table.getContent(), obsCountPerLevel);
            removeTableFromPath(table.getContent());
        }

        if (condition) {
            logln("After remove. iter = " + iter + ". d = " + d + ". tabIdx = " + table.getTableId());
            logln(printGlobalTree());
            logln(printLocalRestaurant(d));
        }

        // log priors
        HashMap<SNode, Double> pathLogPriors = new HashMap<SNode, Double>();
        computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

        // word log likelihoods
        HashMap<SNode, Double> pathWordLlhs = new HashMap<SNode, Double>();
        computePathWordLogLikelihood(pathWordLlhs, globalTreeRoot, obsCountPerLevel, dataLlhNewTopic, 0.0);

        // debug
        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        // response log likelihoods
        HashMap<SNode, Double> pathResLlhs = new HashMap<SNode, Double>();
        if (supervised && observed) {
            double preSum = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                if (table.containsCustomer(s)) // skip sentences current assigned to this table
                {
                    continue;
                }
                preSum += computeRegressionSum(d, s);
            }
            pathResLlhs = computePathResponseLogLikelihood(d, table, preSum);

            if (pathLogPriors.size() != pathResLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }
        }

        // sample
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode path : pathLogPriors.keySet()) {
            if (!extend && !isLeafNode(path)) {
                continue;
            }

            double lp = pathLogPriors.get(path) + pathWordLlhs.get(path);
            if (supervised && observed) {
                lp += pathResLlhs.get(path);
            }

            logProbs.add(lp);
            pathList.add(path);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode newLeaf = pathList.get(sampledIndex);

        // debug
        if (curLeaf == null || curLeaf.equals(newLeaf)) {
            numTableAssignmentsChange++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (newLeaf.getLevel() < L - 1) {
            newLeaf = this.createNewPath(newLeaf);
        }

        // update
        table.setContent(newLeaf);

        if (add) {
            addTableToPath(newLeaf);
            addObservationsToPath(newLeaf, obsCountPerLevel);
        }
    }

    private void optimizeHierarchical() {
        // debug
        double resLlhBefore = 0.0;
        if (debug) {
            double[] regValues = getRegressionValues();
            for (int d = 0; d < D; d++) {
                resLlhBefore += StatUtils.logNormalProbability(responses[d],
                        regValues[d], sqrtRho);
            }
        }

        ArrayList<SNode> flattenTree = flattenTree();
        int numNodes = flattenTree.size();

        // current regression parameters and priors
        double[] regParams = new double[numNodes];
        double[] priorStdvs = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            SNode node = flattenTree.get(i);
            regParams[i] = node.getRegressionParameter();
            priorStdvs[i] = sigmas[node.getLevel()];
        }

        HashMap<Integer, Integer> uplink = new HashMap<Integer, Integer>();
        HashMap<Integer, ArrayList<Integer>> downlinks = new HashMap<Integer, ArrayList<Integer>>();
        for (int n = 0; n < numNodes; n++) {
            SNode curNode = flattenTree.get(n);

            // up
            SNode parentNode = curNode.getParent();
            if (parentNode == null) {
                uplink.put(n, -1);
            } else {
                uplink.put(n, flattenTree.indexOf(parentNode));
            }

            // down
            ArrayList<Integer> children = new ArrayList<Integer>();
            for (SNode child : curNode.getChildren()) {
                children.add(flattenTree.indexOf(child));
            }
            downlinks.put(n, children);
        }

        double[][] designMatrix = new double[D][numNodes];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 0; l < L; l++) {
                    int nodeIdx = flattenTree.indexOf(path[l]);
                    int count = sentLevelCounts[d][s].getCount(l);
                    designMatrix[d][nodeIdx] += count;
                }
            }
        }

        // debug
        if (debug) {
            int sum = 0;
            for (int d = 0; d < D; d++) {
                sum += (int) StatUtils.sum(designMatrix[d]);

                if (StatUtils.sum(designMatrix[d]) != docTokenCounts[d]) {
                    throw new RuntimeException("Counts mismatch."
                            + " iter = " + iter
                            + ": " + StatUtils.sum(designMatrix[d])
                            + " vs. " + docTokenCounts[d]);
                }
            }
            if (sum != tokenCount) {
                throw new RuntimeException("Mismatch while optimizing. iter = " + iter
                        + ". " + sum + " vs. " + tokenCount);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int i = 0; i < numNodes; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }
        }

        GaussianHierLinearRegObjective opt =
                new GaussianHierLinearRegObjective(
                regParams, designMatrix, responses,
                hyperparams.get(RHO), mus[0], priorStdvs,
                uplink, downlinks);
        this.optimizer = new LimitedMemoryBFGS(opt);
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

        // if the number of observations is less than or equal to the number of parameters
        if (converged) {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < flattenTree.size(); i++) {
            flattenTree.get(i).setRegressionParameter(opt.getParameter(i));
        }

        // debug
        if (debug) {
            double resLlhAfter = 0.0;
            double[] regValues = getRegressionValues();
            for (int d = 0; d < D; d++) {
                resLlhAfter += StatUtils.logNormalProbability(responses[d],
                        regValues[d], sqrtRho);
            }
            logln("--- optimized iter = " + iter
                    + ". response llh: before = " + MiscUtils.formatDouble(resLlhBefore)
                    + ". after = " + MiscUtils.formatDouble(resLlhAfter));
        }
    }

    private void optimize() {
        // debug
        double resLlhBefore = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlhBefore += StatUtils.logNormalProbability(responses[d],
                    regValues[d], sqrtRho);
        }

        ArrayList<SNode> flattenTree = flattenTree();
        int numNodes = flattenTree.size();

        // current regression parameters and priors
        double[] regParams = new double[numNodes];
        double[] priorMeans = new double[numNodes];
        double[] priorStdvs = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            SNode node = flattenTree.get(i);
            regParams[i] = node.getRegressionParameter();
            priorMeans[i] = mus[node.getLevel()];
            priorStdvs[i] = sigmas[node.getLevel()];
        }

        // debug
//        logln("# nodes: " + numNodes);
//        for(int k=0; k<regParams.length; k++){
//            logln("\t> node: " + flattenTree.get(k).toString()
//                    + ". reg param = " + MiscUtils.formatDouble(regParams[k]));
//        }
//        System.out.println("-----------");
//        logln("Pre tree");
//        logln(printGlobalTree());
        // end debug

        double[][] designMatrix = new double[D][numNodes];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 0; l < L; l++) {
                    int nodeIdx = flattenTree.indexOf(path[l]);
                    int count = sentLevelCounts[d][s].getCount(l);
                    designMatrix[d][nodeIdx] += count;
                }
            }
        }

        // debug
        if (debug) {
            int sum = 0;
            for (int d = 0; d < D; d++) {
                sum += (int) StatUtils.sum(designMatrix[d]);

                if (StatUtils.sum(designMatrix[d]) != docTokenCounts[d]) {
                    throw new RuntimeException("Counts mismatch."
                            + " iter = " + iter
                            + ": " + StatUtils.sum(designMatrix[d])
                            + " vs. " + docTokenCounts[d]);
                }
            }
            if (sum != tokenCount) {
                throw new RuntimeException("Mismatch while optimizing. iter = " + iter
                        + ". " + sum + " vs. " + tokenCount);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int i = 0; i < numNodes; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }

            // debug
//            logln("d = " + d + ". " + MiscUtils.arrayToSVMLightString(designMatrix[d]));
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses,
                hyperparams.get(RHO),
                priorMeans, priorStdvs);
        this.optimizer = new LimitedMemoryBFGS(optimizable);
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

        // if the number of observations is less than or equal to the number of parameters
        if (converged) {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < flattenTree.size(); i++) {
            flattenTree.get(i).setRegressionParameter(optimizable.getParameter(i));
        }

        // debug
        double resLlhAfter = 0.0;
        regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlhAfter += StatUtils.logNormalProbability(responses[d],
                    regValues[d], sqrtRho);
        }
        logln("--- optimized iter = " + iter
                + ". response llh: before = " + MiscUtils.formatDouble(resLlhBefore)
                + ". after = " + MiscUtils.formatDouble(resLlhAfter));
    }

    private void updateRegressionParameters() {
        // debug
        double resLlhBefore = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlhBefore += StatUtils.logNormalProbability(responses[d],
                    regValues[d], sqrtRho);
        }

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (node.getLevel() == 1) {
                optimize(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        // debug
        double resLlhAfter = 0.0;
        regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            resLlhAfter += StatUtils.logNormalProbability(responses[d],
                    regValues[d], sqrtRho);
        }
        logln("--- optimized iter = " + iter
                + ". response llh: before = " + MiscUtils.formatDouble(resLlhBefore)
                + ". after = " + MiscUtils.formatDouble(resLlhAfter));
    }

    private void optimize(SNode root) {
        ArrayList<SNode> flattenSubtree = flattenTree(root);
        int numNodes = flattenSubtree.size();

        double[] regParams = new double[numNodes];
        double[] priorMeans = new double[numNodes];
        double[] priorStdvs = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            SNode node = flattenSubtree.get(i);
            regParams[i] = node.getRegressionParameter();
            priorMeans[i] = mus[node.getLevel()];
            priorStdvs[i] = sigmas[node.getLevel()];
        }

        double[] adjustedResponses = new double[D];
        double[][] designMatrix = new double[D][numNodes];
        for (int d = 0; d < D; d++) {
            adjustedResponses[d] = responses[d];

            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 0; l < L; l++) {
                    int count = sentLevelCounts[d][s].getCount(l);
                    int nodeIdx = flattenSubtree.indexOf(path[l]);
                    if (nodeIdx == -1) {
                        adjustedResponses[d] -= path[l].getRegressionParameter() * count / docTokenCounts[d];
                    } else {
                        designMatrix[d][nodeIdx] += count;
                    }
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int i = 0; i < numNodes; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, adjustedResponses,
                hyperparams.get(RHO),
                priorMeans, priorStdvs);
        this.optimizer = new LimitedMemoryBFGS(optimizable);
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

        // if the number of observations is less than or equal to the number of parameters
        if (converged) {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < flattenSubtree.size(); i++) {
            flattenSubtree.get(i).setRegressionParameter(optimizable.getParameter(i));
        }
    }

    private SNode samplePath(
            HashMap<SNode, Double> logPriors,
            HashMap<SNode, Double> wordLlhs,
            HashMap<SNode, Double> resLlhs,
            boolean observed) {
        ArrayList<SNode> pathList = new ArrayList<SNode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SNode node : logPriors.keySet()) {
            double lp = logPriors.get(node) + wordLlhs.get(node);
            if (supervised && observed) {
                lp += resLlhs.get(node);
            }

            pathList.add(node);
            logProbs.add(lp);
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SNode path = pathList.get(sampledIndex);
        return path;
    }

    private double computeMarginals(
            HashMap<SNode, Double> pathLogPriors,
            HashMap<SNode, Double> pathWordLogLikelihoods,
            HashMap<SNode, Double> pathResLogLikelihoods,
            boolean resObserved) {
        double marginal = 0.0;
        for (SNode node : pathLogPriors.keySet()) {
            double logprior = pathLogPriors.get(node);
            double loglikelihood = pathWordLogLikelihoods.get(node);

            double lp = logprior + loglikelihood;
            if (isSupervised() && resObserved) {
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

    private void computePathLogPrior(
            HashMap<SNode, Double> nodeLogProbs,
            SNode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
            newWeight += logGammas[curNode.getLevel()] - logNorm;

            for (SNode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }
        }
        nodeLogProbs.put(curNode, newWeight);
    }

    private void computePathWordLogLikelihood(
            HashMap<SNode, Double> nodeDataLlhs,
            SNode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(docTokenCountPerLevel[level]);

        // populate to child nodes
        for (SNode child : curNode.getChildren()) {
            computePathWordLogLikelihood(nodeDataLlhs, child, docTokenCountPerLevel,
                    dataLlhNewTopic, parentDataLlh + nodeDataLlh);
        }

        // store the data llh from the root to this current node
        double storeDataLlh = parentDataLlh + nodeDataLlh;
        level++;
        while (level < L) // if this is an internal node, add llh of new child node
        {
            storeDataLlh += dataLlhNewTopic[level++];
        }
        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            STable table,
            double preSum) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            SNode[] path = getPathFromNode(node);
            double addSum = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                for (int s : table.getCustomers()) {
                    addSum += path[level].getRegressionParameter() * sentLevelCounts[d][s].getCount(level);
                }
            }
            while (level < L) {
                int totalLevelCount = 0;
                for (int s : table.getCustomers()) {
                    int levelCount = sentLevelCounts[d][s].getCount(level);
                    addSum += levelCount * mus[level];
                    totalLevelCount += levelCount;
                }
                var += Math.pow((double) totalLevelCount / docTokenCounts[d], 2) * sigmas[level];
                level++;
            }

            double mean = (preSum + addSum) / docTokenCounts[d];
            double resLlh = StatUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return resLlhs;
    }

    private HashMap<SNode, Double> computePathResponseLogLikelihood(
            int d,
            int s,
            double preSum) {
        HashMap<SNode, Double> resLlhs = new HashMap<SNode, Double>();

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            SNode[] path = getPathFromNode(node);
            double addSum = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                addSum += path[level].getRegressionParameter() * sentLevelCounts[d][s].getCount(level);
            }
            while (level < L) {
                int levelCount = sentLevelCounts[d][s].getCount(level);
                addSum += levelCount * mus[level];
                var += Math.pow((double) levelCount / docTokenCounts[d], 2) * sigmas[level];
                level++;
            }

            double mean = (preSum + addSum) / docTokenCounts[d];
            double resLlh = StatUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        return resLlhs;
    }

    private double computeRegressionSum(int d, int s) {
        double regSum = 0.0;
        SNode[] path = getPathFromNode(c[d][s].getContent());
        for (int l = 0; l < path.length; l++) {
            regSum += path[l].getRegressionParameter() * sentLevelCounts[d][s].getCount(l);
        }
        return regSum;
    }

    /**
     * Predict the response values using the current model
     */
    public double[] getRegressionValues() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                sum += computeRegressionSum(d, s);
            }
            regValues[d] = sum / docTokenCounts[d];
        }
        return regValues;
    }

    public double[] getRegressionValuesNoRoot() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            int count = 0;
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                for (int l = 1; l < L; l++) {
                    count += sentLevelCounts[d][s].getCount(l);
                    sum += path[l].getRegressionParameter() * sentLevelCounts[d][s].getCount(l);
                }
            }
            regValues[d] = sum / count;
        }
        return regValues;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    private SNode[] getPathFromNode(SNode node) {
        SNode[] path = new SNode[node.getLevel() + 1];
        SNode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    private SNode getNode(int[] parsedPath) {
        SNode node = globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    private int[] parseTokenId(String tokenId) {
        String[] parse = tokenId.split("_");
        int[] parsedTokenId = new int[parse.length];
        for (int i = 0; i < parsedTokenId.length; i++) {
            parsedTokenId[i] = Integer.parseInt(parse[i]);
        }
        return parsedTokenId;
    }

    private String getTokenId(int turnIndex, int tokenIndex) {
        return turnIndex + "_" + tokenIndex;
    }

    private boolean isLeafNode(SNode node) {
        return node.getLevel() == L - 1;
    }

    /**
     * Get a node at a given level on a path on the tree. The path is determined
     * by its leaf node.
     *
     * @param level The level that the node is at
     * @param leafNode The leaf node of the path
     */
    private SNode getNode(int level, SNode leafNode) {
        if (!isLeafNode(leafNode)) {
            throw new RuntimeException("Exception while getting node. The given "
                    + "node is not a leaf node");
        }
        int curLevel = leafNode.getLevel();
        SNode curNode = leafNode;
        while (curLevel != level) {
            curNode = curNode.getParent();
            curLevel--;
        }
        return curNode;
    }

    private ArrayList<SNode> flattenTree(SNode root) {
        ArrayList<SNode> flattenTree = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            flattenTree.add(node);
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return flattenTree;
    }

    private ArrayList<SNode> flattenTree() {
        ArrayList<SNode> flattenTree = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            flattenTree.add(node);
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return flattenTree;
    }

    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        int[] nodeCountPerLevel = new int[L];
        int[] obsCountPerLevel = new int[L];

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            nodeCountPerLevel[node.getLevel()]++;
            obsCountPerLevel[node.getLevel()] += node.getContent().getCountSum();

            totalObs += node.getContent().getCountSum();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("global tree:\n\t>>> node count per level: ");
        for (int l = 0; l < L; l++) {
            str.append(l).append("(")
                    .append(nodeCountPerLevel[l])
                    .append(", ").append(obsCountPerLevel[l])
                    .append(");\t");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs)
                .append("\n\t>>> # customers = ").append(globalTreeRoot.getNumCustomers());
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
                .append("\n>>> # customers = ").append(globalTreeRoot.getNumCustomers())
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
                .append(". # total customers: ").append(localRestaurants[d].getTotalNumCustomers()).append("\n");
        for (STable table : localRestaurants[d].getTables()) {
            str.append("--- table: ").append(table.toString()).append("\n");
        }
        return str.toString();
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

            wordLlh += node.getContent().getLogLikelihood();

            if (supervised) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double restLgprob = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += docLevelDists[d].getLogLikelihood();

            restLgprob += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));

            if (supervised) {
                resLlh += StatUtils.logNormalProbability(responses[d],
                        regValues[d], sqrtRho);
            }
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                + ". rest = " + MiscUtils.formatDouble(restLgprob)
                + ". stick = " + MiscUtils.formatDouble(stickLgprob)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh + restLgprob;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        int count = GEM_SCALE + 1;
        double[] newBetas = new double[betas.length];
        for (int i = 0; i < newBetas.length; i++) {
            newBetas[i] = tParams.get(count++);
        }
        double[] newGammas = new double[gammas.length];
        for (int i = 0; i < newGammas.length; i++) {
            newGammas[i] = tParams.get(count++);
        }
        double[] newMus = new double[mus.length];
        for (int i = 0; i < newMus.length; i++) {
            newMus[i] = tParams.get(count++);
        }
        double[] newSigmas = new double[sigmas.length];
        for (int i = 0; i < newSigmas.length; i++) {
            newSigmas[i] = tParams.get(count++);
        }

        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);

            if (supervised) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        newMus[node.getLevel()], Math.sqrt(newSigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(newGammas[node.getLevel()]);
            }

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double restLgprob = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += docLevelDists[d].getLogLikelihood(tParams.get(GEM_MEAN), tParams.get(GEM_SCALE));

            restLgprob += localRestaurants[d].getJointProbabilityAssignments(tParams.get(ALPHA));

            if (supervised) {
                resLlh += StatUtils.logNormalProbability(responses[d],
                        regValues[d], Math.sqrt(tParams.get(RHO)));
            }
        }

        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh + restLgprob;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
        this.hyperparams = new ArrayList<Double>();
        for (double param : tParams) {
            this.hyperparams.add(param);
        }

        int count = GEM_SCALE + 1;
        betas = new double[betas.length];
        for (int i = 0; i < betas.length; i++) {
            betas[i] = tParams.get(count++);
        }
        gammas = new double[gammas.length];
        for (int i = 0; i < gammas.length; i++) {
            gammas[i] = tParams.get(count++);
        }
        mus = new double[mus.length];
        for (int i = 0; i < mus.length; i++) {
            mus[i] = tParams.get(count++);
        }
        sigmas = new double[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            sigmas[i] = tParams.get(count++);
        }

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            node.getContent().setConcentration(betas[node.getLevel()]);
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l].setConcentration(betas[l + 1]);
        }

//        for(int d=0; d<D; d++){
//            docLevelDists[d].setMean(hyperparams.get(GEM_MEAN));
//            docLevelDists[d].setScale(hyperparams.get(GEM_SCALE));
//        }
    }

    @Override
    public void validate(String msg) {
        validateModel(msg);

        validateAssignments(msg);
    }

    private void validateModel(String msg) {
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            if (!isLeafNode(node)) {
                int childNumCusts = 0;

                for (SNode child : node.getChildren()) {
                    childNumCusts += child.getNumCustomers();
                    stack.add(child);
                }

                if (childNumCusts != node.getNumCustomers()) {
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

    private void validateAssignments(String msg) {
        for (int d = 0; d < D; d++) {
            docLevelDists[d].validate(msg);
        }

        for (int d = 0; d < D; d++) {
            int totalCusts = 0;
            for (STable table : localRestaurants[d].getTables()) {
                totalCusts += table.getNumCustomers();
            }
            if (totalCusts != words[d].length) {
                for (STable table : localRestaurants[d].getTables()) {
                    System.out.println(table.toString() + ". customers: " + table.getCustomers().toString());
                }
                throw new RuntimeException(msg + ". Numbers of customers in restaurant " + d
                        + " mismatch. " + totalCusts + " vs. " + words[d].length);
            }

            HashMap<STable, Integer> tableCustCounts = new HashMap<STable, Integer>();
            for (int s = 0; s < words[d].length; s++) {
                Integer count = tableCustCounts.get(c[d][s]);

                if (count == null) {
                    tableCustCounts.put(c[d][s], 1);
                } else {
                    tableCustCounts.put(c[d][s], count + 1);
                }
            }

            if (tableCustCounts.size() != localRestaurants[d].getNumTables()) {
                throw new RuntimeException(msg + ". Numbers of tables mismatch in"
                        + " restaurant " + d);
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
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(printGlobalTreeSummary()).append("\n");
        str.append(printLocalRestaurantSummary()).append("\n");
        return str.toString();
    }

    public void outputTopicTopWords(String outputFile, int numWords)
            throws Exception {
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

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }
            if (node.getIterationCreated() >= MAX_ITER - LAG) {
                continue;
            }

            String[] topWords = getTopWords(node.getContent().getDistribution(), numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getIterationCreated())
                    .append("; ").append(node.getNumCustomers())
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.getRegressionParameter()))
                    .append(")");
            for (String topWord : topWords) {
                str.append(" ").append(topWord);
            }
            str.append("\n\n");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write(str.toString());
        writer.close();
    }

    public void outputTopicCoherence(
            String filepath,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            logln("Outputing topic coherence to file " + filepath);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);

        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();

            for (SNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getIterationCreated()
                    + "\t" + node.getNumCustomers()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }

        writer.close();
    }

    public void diagnose(String filepath) throws Exception {
        if (verbose) {
            logln("Diagnosing " + filepath);
        }

        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            double predRes = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                predRes += computeRegressionSum(d, s);
            }
            predRes /= docTokenCounts[d];

            str.append(d)
                    .append(": #tables: ").append(localRestaurants[d].getNumTables())
                    .append(". true res: ").append(responses == null ? "" : MiscUtils.formatDouble(responses[d]))
                    .append(". pred res: ").append(MiscUtils.formatDouble(predRes))
                    .append(". #tokens: ").append(docTokenCounts[d])
                    .append("\n");
            for (int s = 0; s < words[d].length; s++) {
                SNode[] path = getPathFromNode(c[d][s].getContent());
                str.append("--- s = ").append(s)
                        .append(": table id = ").append(c[d][s].getTableId());
                for (int l = L - 1; l >= 0; l--) {
                    str.append(path[l].getPathString())
                            .append("\t -> ")
                            .append(" (").append(MiscUtils.formatDouble(path[l].getRegressionParameter()))
                            .append(") ");
                }
                str.append("\n\t\t");
                for (int n = 0; n < words[d][s].length; n++) {
                    str.append(wordVocab.get(words[d][s][n]))
                            .append("-").append(z[d][s][n])
                            .append(", ");
                }
                str.append("\n");
            }
            str.append("\n");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        writer.write(str.toString());
        writer.close();
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath + "\n");
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            Stack<SNode> stack = new Stack<SNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SNode node = stack.pop();
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");

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
                for (STable table : localRestaurants[d].getTables()) {
                    assignStr.append(table.getIndex()).append("\n");
                    assignStr.append(table.getIterationCreated()).append("\n");
                    assignStr.append(table.getContent().getPathString()).append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    assignStr.append(d)
                            .append(":").append(s)
                            .append("\t").append(c[d][s].getIndex())
                            .append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        assignStr.append(d)
                                .append(":").append(t)
                                .append(":").append(n)
                                .append("\t").append(z[d][t][n])
                                .append("\n");
                    }
                }
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            data = assignStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
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
            System.exit(1);
        }
    }

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath + "\n");
        }

        // initialize
        this.initializeModelStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        HashMap<String, SNode> nodeMap = new HashMap<String, SNode>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        String line;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCustomers = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

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

            node.changeNumCustomers(numCustomers);

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
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath + "\n");
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        String[] sline;

        for (int d = 0; d < D; d++) {
            sline = reader.readLine().split("\t");
            if (d != Integer.parseInt(sline[0])) {
                throw new RuntimeException("Mismatch");
            }
            int numTables = Integer.parseInt(sline[1]);

            for (int i = 0; i < numTables; i++) {
                int tabIndex = Integer.parseInt(reader.readLine());
                int iterCreated = Integer.parseInt(reader.readLine());
                String leafPathStr = reader.readLine();

                SNode leafNode = getNode(parseNodePath(leafPathStr));
                STable table = new STable(iterCreated, tabIndex, leafNode, d);
                localRestaurants[d].addTable(table);
            }
        }

        for (int d = 0; d < D; d++) {
            localRestaurants[d].fillInactiveTableIndices();
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + s)) {
                    throw new RuntimeException("Mismatch");
                }
                int tableIndex = Integer.parseInt(sline[1]);
                c[d][s] = localRestaurants[d].getTable(tableIndex);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + s + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][s][n] = Integer.parseInt(sline[1]);
                }
            }
        }

        reader.close();
    }

    public double[] outputRegressionResults(
            double[] trueResponses,
            String predFilepath,
            String outputFile) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(predFilepath);
        String line = reader.readLine();
        String[] modelNames = line.split("\t");
        int numModels = modelNames.length;

        double[][] predResponses = new double[numModels][trueResponses.length];

        int idx = 0;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            for (int j = 0; j < numModels; j++) {
                predResponses[j][idx] = Double.parseDouble(sline[j]);
            }
            idx++;
        }
        reader.close();

        double[] finalPredResponses = new double[trueResponses.length];
        for (int d = 0; d < trueResponses.length; d++) {
            double sum = 0.0;
            for (int i = 0; i < numModels; i++) {
                sum += predResponses[i][d];
            }
            finalPredResponses[d] = sum / numModels;
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int i = 0; i < numModels; i++) {
            RegressionEvaluation eval = new RegressionEvaluation(
                    trueResponses, predResponses[i]);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();

            if (i == 0) {
                writer.write("Model");
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getName());
                }
                writer.write("\n");
            }
            writer.write(modelNames[i]);
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");
        }
        writer.close();

        return finalPredResponses;
    }
    private int[][][] x;
    private SparseCount[] docTopics;
    private ArrayList<SNode> nodeList;

    public double[] regressNewDocumentsNew(
            int[][][] newWords, double[] newResponses,
            String predFilepath) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolderpath);
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        ArrayList<String> modelList = new ArrayList<String>();

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            double[] predResponses = regressNewDocumentsNew(
                    reportFolderpath + filename,
                    newWords,
                    reportFolderpath + IOUtils.removeExtension(filename) + ".diagnose");
            predResponsesList.add(predResponses);
            modelList.add(filename);

            if (verbose) { // for debugging only
                logln("state file: " + filename
                        + ". iter = " + iter);
                RegressionEvaluation eval = new RegressionEvaluation(
                        newResponses, predResponses);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();
            }
        }

        // write prediction values
        BufferedWriter writer = IOUtils.getBufferedWriter(predFilepath);
        for (String model : modelList) // header
        {
            writer.write(model + "\t");
        }
        writer.write("\n");

        for (int r = 0; r < newWords.length; r++) {
            for (int m = 0; m < predResponsesList.size(); m++) {
                writer.write(predResponsesList.get(m)[r] + "\t");
            }
            writer.write("\n");
        }
        writer.close();

        // average predicted response over different models
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    private double[] regressNewDocumentsNew(
            String stateFile,
            int[][][] newWords,
            String diagnoseFile) throws Exception {
        if (verbose) {
            logln("\nPerform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (verbose) {
            logln("Loaded trained model");
            logln(printGlobalTreeSummary());
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

        sentCount = 0;
        tokenCount = 0;
        docTokenCounts = new int[D];
        for (int d = 0; d < D; d++) {
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
                docTokenCounts[d] += words[d][s].length;
            }
        }

        logln("--- V = " + V);
        logln("--- # documents = " + D); // number of groups
        logln("--- # sentences = " + sentCount);
        logln("--- # tokens = " + tokenCount);

        // initialize structure for test data
//        initializeDataStructure();

        // initialize structure
        nodeList = new ArrayList<SNode>();
        Stack<SNode> stack = new Stack<SNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SNode node = stack.pop();
            if (!node.isRoot() && node.getIterationCreated() < 100) {
                nodeList.add(node);
            }
            for (SNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        if (verbose) {
            logln("--- # topics: " + nodeList.size());
        }

        // initialize
        this.x = new int[D][][];
        this.docTopics = new SparseCount[D];
        for (int d = 0; d < D; d++) {
            this.x[d] = new int[words[d].length][];
            this.docTopics[d] = new SparseCount();
            for (int s = 0; s < words[d].length; s++) {
                this.x[d][s] = new int[words[d][s].length];
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleX(d, s, n, !REMOVE, ADD);
                }
            }
        }

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleX(d, s, n, REMOVE, ADD);
                    }
                }
            }

            if (verbose && iter % LAG == 0) {
                logln("--- iter = " + iter + " / " + MAX_ITER);
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = new double[D];
                for (int d = 0; d < D; d++) {
                    double predRes = 0.0;
                    for (int k = 0; k < nodeList.size(); k++) {
                        predRes += docTopics[d].getCount(k) * nodeList.get(k).getRegressionParameter()
                                / docTopics[d].getCountSum();
                    }
                    predResponses[d] = predRes;
                }

                predResponsesList.add(predResponses);
            }
        }

//        // debug
//        if(diagnoseFile != null)
//            diagnose(diagnoseFile);

        // averaging prediction responses over time
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    private void sampleX(int d, int s, int n, boolean remove, boolean add) {
        if (remove) {
            docTopics[d].decrement(x[d][s][n]);
            nodeList.get(x[d][s][n]).getContent().decrement(words[d][s][n]);
        }

        double[] logprobs = new double[nodeList.size()];
        for (int k = 0; k < logprobs.length; k++) {
            SNode node = nodeList.get(k);
            double logPrior = Math.log(docTopics[d].getCount(k) + 0.1);
            double wordLlh = node.getContent().getLogLikelihood(words[d][s][n]);
            logprobs[k] = logPrior + wordLlh;
        }
        int sampledX = SamplerUtils.logMaxRescaleSample(logprobs);

        x[d][s][n] = sampledX;

        if (add) {
            docTopics[d].increment(x[d][s][n]);
            nodeList.get(x[d][s][n]).getContent().increment(words[d][s][n]);
        }
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     */
    public double[] regressNewDocuments(
            int[][][] newWords, double[] newResponses,
            String predFilepath) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        ArrayList<String> modelList = new ArrayList<String>();

        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            double[] predResponses = regressNewDocuments(
                    reportFolderpath + filename,
                    newWords,
                    reportFolderpath + IOUtils.removeExtension(filename) + ".diagnose");
            predResponsesList.add(predResponses);
            modelList.add(filename);

            if (verbose) { // for debugging only
                logln("state file: " + filename
                        + ". iter = " + iter);
                RegressionEvaluation eval = new RegressionEvaluation(
                        newResponses, predResponses);
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();
            }

            break;
        }

        // write prediction values
        BufferedWriter writer = IOUtils.getBufferedWriter(predFilepath);
        for (String model : modelList) // header
        {
            writer.write(model + "\t");
        }
        writer.write("\n");

        for (int r = 0; r < newWords.length; r++) {
            for (int m = 0; m < predResponsesList.size(); m++) {
                writer.write(predResponsesList.get(m)[r] + "\t");
            }
            writer.write("\n");
        }
        writer.close();

        // average predicted response over different models
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    /**
     * Regressing new data using a specific model stored in a given model file
     *
     * @param stateFile File containing a stored state
     * @param newWords New words
     */
    private double[] regressNewDocuments(
            String stateFile,
            int[][][] newWords,
            String diagnoseFile) throws Exception {
        if (verbose) {
            logln("\nPerform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = null; // for evaluation
        D = words.length;

        sentCount = 0;
        tokenCount = 0;
        docTokenCounts = new int[D];
        for (int d = 0; d < D; d++) {
            sentCount += words[d].length;
            for (int s = 0; s < words[d].length; s++) {
                tokenCount += words[d][s].length;
                docTokenCounts[d] += words[d][s].length;
            }
        }

        logln("--- V = " + V);
        logln("--- # documents = " + D); // number of groups
        logln("--- # sentences = " + sentCount);
        logln("--- # tokens = " + tokenCount);

        // initialize structure for test data
        initializeDataStructure();

        if (verbose) {
            logln("Initialized data structure");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        // initialize random assignments
        initializeRandomAssignmentsNewDocuments();

        if (verbose) {
            logln("Initialized random assignments");
            logln(printGlobalTreeSummary());
            logln(printLocalRestaurantSummary());
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    if (words[d].length > 1) // if this document has only 1 sentence, no sampling is needed
                    {
                        sampleTableForSentence(d, s, REMOVE, ADD, !OBSERVED, !EXTEND);
                    }

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelForToken(d, s, n, REMOVE, ADD, !OBSERVED);
                    }
                }

                for (STable table : localRestaurants[d].getTables()) {
                    samplePathForTable(d, table, REMOVE, ADD, !OBSERVED, !EXTEND);
                }
            }

            if (verbose && iter % LAG == 0) {
                logln("--- iter = " + iter + " / " + MAX_ITER);
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);
            }
        }

        // debug
        if (diagnoseFile != null) {
            diagnose(diagnoseFile);
        }

        // averaging prediction responses over time
        double[] finalPredResponses = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int i = 0; i < predResponsesList.size(); i++) {
                sum += predResponsesList.get(i)[d];
            }
            finalPredResponses[d] = sum / predResponsesList.size();
        }
        return finalPredResponses;
    }

    private void initializeRandomAssignmentsNewDocuments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // create a new table for each sentence
                STable table = new STable(iter, s, null, d);
                localRestaurants[d].addTable(table);
                localRestaurants[d].addCustomerToTable(s, table.getIndex());
                c[d][s] = table;

                // sample level
                for (int n = 0; n < words[d][s].length; n++) {
                    // sample from prior
//                    double[] levelDist = docLevelDists[d].getDistribution();
//                    int randLevel = SamplerUtils.scaleSample(levelDist);
                    int randLevel = rand.nextInt(L);

                    // update and increment
                    z[d][s][n] = randLevel;
                    docLevelDists[d].increment(z[d][s][n]);
                    sentLevelCounts[d][s].increment(z[d][s][n]);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (STable table : localRestaurants[d].getTables()) {
                samplePathForTable(d, table, !REMOVE, ADD, !OBSERVED, !EXTEND);
            }
        }
    }

    class SNode extends TreeNode<SNode, DirMult> {

        private final int born;
        private int numCustomers;
        private double regression;

        SNode(int iter, int index, int level, DirMult content, double regParam,
                SNode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numCustomers = 0;
            this.regression = regParam;
        }

        public int getIterationCreated() {
            return this.born;
        }

        double getLogJointProbability(double gamma) {
            ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
            for (SNode child : this.getChildren()) {
                numChildrenCusts.add(child.getNumCustomers());
            }
            return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gamma);
        }

        public double getRegressionParameter() {
            return this.regression;
        }

        public void setRegressionParameter(double reg) {
            this.regression = reg;
        }

        public int getNumCustomers() {
            return this.numCustomers;
        }

        public void decrementNumCustomers() {
            this.numCustomers--;
        }

        public void incrementNumCustomers() {
            this.numCustomers++;
        }

        public void changeNumCustomers(int delta) {
            this.numCustomers += delta;
        }

        public boolean isEmpty() {
            return this.numCustomers == 0;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(getPathString())
                    .append(" (").append(born).append(")")
                    .append(" #ch = ").append(getNumChildren())
                    .append(", #c = ").append(getNumCustomers())
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

        public STable(int iter, int index, SNode content, int restId) {
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
}
