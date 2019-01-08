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
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.FullTable;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
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
public class SGHLDASampler extends AbstractSampler {

    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    public static final boolean HAS_PSEUDOCHILD = true;
    public static final int ALPHA = 0;
    public static final int RHO = 1; // response variable variances
    public static final int GEM_MEAN = 2;
    public static final int GEM_SCALE = 3;
    protected boolean supervised = true;
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
    protected int K;
    protected int[][][] words;  // [D] x [Td] x [Ndt]: words
    protected double[][] responses; // [D] x [Td]
    protected int[][][] z; // local table index
    protected int[][][] x; // level index
    private SGHLDANode globalTreeRoot;
    private Restaurant<SGHLDATable, String, SGHLDANode>[] localRestaurants;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private int numTokens;
    private int numDocs;
    private int[][] docTokenCounts;
    private double[] uniform;
    private DirMult[] emptyModels;
    private TruncatedStickBreaking emptyStick;
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;
    private int numConverged;

    public void configure(String folder,
            int[][][] words, double[][] responses,
            int V, int L,
            double alpha,
            double rho,
            double gem_mean,
            double gem_sigma,
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
        this.hyperparams.add(gem_sigma);
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

        updatePrecomputedHyperparameters();

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

        numDocs = 0;
        numTokens = 0;
        docTokenCounts = new int[D][];
        for (int d = 0; d < D; d++) {
            numDocs += words[d].length;
            docTokenCounts[d] = new int[words[d].length];
            for (int t = 0; t < words[d].length; t++) {
                numTokens += this.words[d][t].length;
                docTokenCounts[d][t] = words[d][t].length;
            }
        }

        logln("--- V = " + V);
        logln("--- # groups (D) = " + D); // number of groups
        logln("--- # documents = " + numDocs);
        logln("--- # tokens = " + numTokens);

        if (!debug) {
            System.err.close();
        }
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

    public void setSupervised(boolean s) {
        this.supervised = s;
    }

    public boolean isSupervised() {
        return this.supervised;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_SGHLDA")
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
            logln(printLocalRestaurants());
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
        this.globalTreeRoot = new SGHLDANode(iter, rootIndex, rootLevel, dmModel, regParam, null, HAS_PSEUDOCHILD);

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<SGHLDATable, String, SGHLDANode>();
        }

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }

        this.emptyStick = new TruncatedStickBreaking(L, hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
    }

    private void initializeDataStructure() {
        z = new int[D][][];
        x = new int[D][][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length][];
            x[d] = new int[words[d].length][];
            for (int t = 0; t < words[d].length; t++) {
                z[d][t] = new int[words[d][t].length];
                this.x[d][t] = new int[words[d][t].length];
            }
        }
    }

    private void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializePresetAssignments();
                break;

            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        if (K == 0) // this is not set
        {
            K = 50;
        }
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        int[][] flattenWords = new int[D][];
        for (int d = 0; d < D; d++) {
            int numDocTokens = 0;
            for (int t = 0; t < words[d].length; t++) {
                numDocTokens += words[d][t].length;
            }

            int count = 0;
            flattenWords[d] = new int[numDocTokens];
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    flattenWords[d][count++] = words[d][t][n];
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
                lda.outputTopicTopWords(new File(this.folder, "lda-topwords-" + K + ".txt"), 15);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // create K separate paths
        SGHLDANode[] initLeaves = new SGHLDANode[K];
        for (int k = 0; k < K; k++) {
            SGHLDANode node = globalTreeRoot;
            for (int l = 0; l < L - 1; l++) {
                node = createNode(node);
            }
            initLeaves[k] = node;
        }

        for (int d = 0; d < D; d++) {
            // initialize tables
            for (int k = 0; k < K; k++) {
                SGHLDATable table = new SGHLDATable(iter, k, initLeaves[k], d, L,
                        hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
                localRestaurants[d].addTable(table);
                addTableToPath(initLeaves[k]);
            }

            // assign customers to tables
            int count = 0;
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    z[d][t][n] = ldaZ[d][count++];
                    localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
                    sampleLevelForToken(d, t, n, !REMOVE, !OBSERVED);
                }
            }
        }

        // remove any empty tables
        for (int d = 0; d < D; d++) {
            for (int k = 0; k < K; k++) {
                SGHLDATable table = localRestaurants[d].getTable(k);
                if (table.isEmpty()) {
                    removeTableFromPath(table.getContent());
                    localRestaurants[d].removeTable(k);
                }
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
            int nDocs = Integer.parseInt(reader.readLine());
            ldaZ = new int[nDocs][];
            for (int d = 0; d < nDocs; d++) {
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
                            + "\t # tables: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                }
            }

            numTableAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;
            numConverged = 0;

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        this.sampleTableLevelForToken(d, t, n, REMOVE, OBSERVED, EXTEND);
                    }
                }

                for (SGHLDATable table : this.localRestaurants[d].getTables()) {
                    this.samplePathForTable(d, table.getIndex(), REMOVE, ADD, OBSERVED);
                }
            }

            if (isSupervised()) {
                optimize();
            }

            if (verbose && isSupervised()) {
                double[][] trPredResponses = getRegressionValues();
                RegressionEvaluation eval = new RegressionEvaluation(
                        MiscUtils.flatten2DArray(responses),
                        MiscUtils.flatten2DArray(trPredResponses));
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
            }

            if (iter >= BURN_IN) {
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
            logln(printGlobalTree());
//            logln(printLocalRestaurants());
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
    private void addTableToPath(SGHLDANode leafNode) {
        SGHLDANode node = leafNode;
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
    private SGHLDANode removeTableFromPath(SGHLDANode leafNode) {
        SGHLDANode retNode = leafNode;
        SGHLDANode node = leafNode;
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

    private SGHLDANode[] addObservationsToPath(SGHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SGHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
        return path;
    }

    private SGHLDANode[] removeObservationsFromPath(SGHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SGHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
        return path;
    }

    private void removeObservationsFromNode(SGHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    private void addObservationsToNode(SGHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    private SGHLDANode addObservationToNodeOnPath(SGHLDANode leafNode, int level, int observation) {
        SGHLDANode[] path = getPathFromNode(leafNode);
        path[level].getContent().increment(observation);
        return path[level];
    }

    private SGHLDANode removeObservationFromNodeOnPath(SGHLDANode leafNode, int level, int observation) {
        SGHLDANode[] path = getPathFromNode(leafNode);
        path[level].getContent().decrement(observation);
        return path[level];
    }

    private SGHLDANode createNewPath(SGHLDANode internalNode) {
        SGHLDANode node = internalNode;
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
    private SGHLDANode createNode(SGHLDANode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmm = new DirMult(V, betas[level], uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SGHLDANode child;
        if (level == L - 1) // leaf node
        {
            child = new SGHLDANode(iter, nextChildIndex, level, dmm, regParam, parent, !HAS_PSEUDOCHILD);
        } else {
            child = new SGHLDANode(iter, nextChildIndex, level, dmm, regParam, parent, HAS_PSEUDOCHILD);
        }
        return parent.addChild(nextChildIndex, child);
    }

    /**
     * Sample the level assignment of a token given its table assignment.
     *
     * @param d The group index
     * @param t The document index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     * @param observed Whether the response variable is observed
     */
    private void sampleLevelForToken(int d, int t, int n, boolean remove, boolean observed) {
        int curObs = words[d][t][n];
        SGHLDATable curTable = localRestaurants[d].getTable(z[d][t][n]);
        SGHLDANode[] curPath = getPathFromNode(curTable.getContent());

        if (remove) {
            curTable.decrement(t, x[d][t][n]);
            curPath[x[d][t][n]].getContent().decrement(curObs);
        }

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            double stickLogprior = curTable.getStick().getLogProbability(l);
            double wordLlh = curPath[l].getContent().getLogLikelihood(curObs);
            double resLlh = 0.0;
            if (observed) {
                // compute resllh here
            }
            double logprob = stickLogprior + wordLlh + resLlh;
            logprobs[l] = logprob;
        }

        x[d][t][n] = SamplerUtils.logMaxRescaleSample(logprobs);
        curTable.increment(t, x[d][t][n]);
        curPath[x[d][t][n]].getContent().increment(curObs);
    }

    /**
     * Block sample both table and level assignments for a token
     *
     * @param d Group index
     * @param t Document index
     * @param n Token index
     * @param remove Whether the current assignments should be removed
     * @param resObserved Whether the response variable is observed
     */
    private void sampleTableLevelForToken(int d, int t, int n,
            boolean remove,
            boolean resObserved,
            boolean extend) {
        int curObs = words[d][t][n];
        SGHLDATable curTable = localRestaurants[d].getTable(z[d][t][n]);

        if (remove) {
            localRestaurants[d].removeCustomerFromTable(getTokenId(t, n), z[d][t][n]);
            if (curTable.isEmpty()) {
                removeTableFromPath(curTable.getContent());
                localRestaurants[d].removeTable(curTable.getIndex());
            }
            curTable.decrement(t, x[d][t][n]);
            removeObservationFromNodeOnPath(curTable.getContent(), x[d][t][n], curObs);
        }

        double regParamSum = 0.0;
        int tokenCount = 0;
        if (resObserved) {
            regParamSum = getRegressionSum(d, t);
            tokenCount = words[d][t].length;
        }

        // debug
//        logln("iter = " + iter + ". d = " + d + ". t = " + t + ". n = " + n);

        // existing tables
        ArrayList<String> tableLevelList = new ArrayList<String>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SGHLDATable table : localRestaurants[d].getTables()) {
            SGHLDANode[] path = getPathFromNode(table.getContent());
            double logPrior = Math.log(table.getNumCustomers());
            for (int l = 0; l < L; l++) {
                double stickLogPrior = table.getStick().getLogProbability(l);
                double wordLlh = path[l].getContent().getLogLikelihood(curObs);
                double resLlh = 0.0;
                if (isSupervised() && resObserved) {
                    double mean = (regParamSum + path[l].getRegressionParameter()) / tokenCount;
                    resLlh = StatUtils.logNormalProbability(responses[d][t], mean, sqrtRho);
                }
                double logProb = logPrior + stickLogPrior + wordLlh + resLlh;
                tableLevelList.add(table.getIndex() + "_" + l);
                logProbs.add(logProb);

                // debubg
//                logln("table " + table.toString() + ". level = " + l);
//                StringBuilder str = new StringBuilder();
//                str.append(" logPrior: " + MiscUtils.formatDouble(logPrior))
//                    .append(" stickLogPrior: " + MiscUtils.formatDouble(stickLogPrior))
//                    .append(" wordLlh: " + MiscUtils.formatDouble(wordLlh))
//                    .append(" resLlh: " + MiscUtils.formatDouble(resLlh))
//                    .append(" --->>> logprob = " + MiscUtils.formatDouble(logProb) + "\n");
//                logln(str.toString());
            }
        }

        // new table
        if (extend) { // only considering new table during training time
            HashMap<SGHLDANode, Double> nodeLogPriors = new HashMap<SGHLDANode, Double>();
            this.computeNodeLogPrior(nodeLogPriors, globalTreeRoot, 0.0);

            HashMap<SGHLDANode, Double> nodeWordLogLikelihoods = new HashMap<SGHLDANode, Double>();
            this.computeNodeWordLogLikelihood(nodeWordLogLikelihoods, globalTreeRoot, curObs);

            // debug
            if (nodeLogPriors.size() != nodeWordLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of elements mismatch");
            }

            HashMap<SGHLDANode, Double> nodeResLogLikelihoods = new HashMap<SGHLDANode, Double>();
            if (isSupervised() && resObserved) {
                this.computeNodeResponseLogLikelihood(nodeResLogLikelihoods, globalTreeRoot,
                        responses[d][t], regParamSum, tokenCount);

                if (nodeLogPriors.size() != nodeResLogLikelihoods.size()) {
                    throw new RuntimeException("Numbers of elements mismatch");
                }
            }

            double logPrior = Math.log(hyperparams.get(ALPHA));
            double[] marginals = computeMarginals(nodeLogPriors, nodeWordLogLikelihoods, nodeResLogLikelihoods, resObserved);
            for (int l = 0; l < L; l++) {
                double stickLogPrior = emptyStick.getLogProbability(l);
                double marginal = marginals[l];
                double logProb = logPrior + stickLogPrior + marginal;
                tableLevelList.add(PSEUDO_TABLE_INDEX + "_" + l);
                logProbs.add(logProb);

                // debug
//                StringBuilder str = new StringBuilder();
//                str.append("new table " + ". level = " + l)
//                    .append("--- logPrior: " + MiscUtils.formatDouble(logPrior))
//                    .append("--- stickLogPrior: " + MiscUtils.formatDouble(stickLogPrior))
//                    .append("--- marginal: " + MiscUtils.formatDouble(marginal))
//                    .append("--->>> logprob: " + MiscUtils.formatDouble(logProb) + "\n");
//                logln(str.toString());
            }
        }

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        String[] tableLevel = tableLevelList.get(sampledIndex).split("_");
        int tableIndex = Integer.parseInt(tableLevel[0]);
        int level = Integer.parseInt(tableLevel[1]);

        // debug
//        logln(">>> sample: table index = " + tableIndex + ". level = " + level + "\n\n");

        SGHLDATable newTable;
        if (tableIndex == PSEUDO_TABLE_INDEX) { // create new table
            int newTableIndex = localRestaurants[d].getNextTableIndex();
            newTable = new SGHLDATable(iter, newTableIndex, null, d, L,
                    hyperparams.get(GEM_MEAN), hyperparams.get(GEM_SCALE));
            localRestaurants[d].addTable(newTable);
        } else {
            newTable = localRestaurants[d].getTable(tableIndex);
        }

        if (curTable != null && newTable.getIndex() != curTable.getIndex()) {
            numTokenAssignmentsChange++;
        }

        z[d][t][n] = newTable.getIndex();
        x[d][t][n] = level;
        localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
        newTable.increment(t, x[d][t][n]);

        if (newTable.getContent() == null) { // new table
            samplePathForTable(d, newTable.getIndex(), !REMOVE, ADD, OBSERVED);
        } else {
            addObservationToNodeOnPath(newTable.getContent(), x[d][t][n], curObs);
        }
    }

    /**
     * Sample a path for a table
     *
     * @param d Restaurant index
     * @param tableIndex Table index
     * @param remove Whether the current assignment should be removed
     * @param add Whether the new assignment should be added
     * @param resObserved Whether the response variable is observed
     */
    private void samplePathForTable(
            int d,
            int tableIndex,
            boolean remove,
            boolean add,
            boolean resObserved) {
        SGHLDATable curTable = localRestaurants[d].getTable(tableIndex);
        SGHLDANode curLeaf = curTable.getContent();

        boolean condition = remove;
//        if(condition)
//            logln("Start d = " + d + ". table index = " + tableIndex + ". # tables = " + localRestaurants[d].getNumTables());

        // current observations
        HashMap<Integer, Integer>[] obsCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            obsCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (String customer : curTable.getCustomers()) {
            int[] parsedCustomer = parseTokenId(customer);
            int obs = words[d][parsedCustomer[0]][parsedCustomer[1]];
            int level = x[d][parsedCustomer[0]][parsedCustomer[1]];

            Integer count = obsCountPerLevel[level].get(obs);
            if (count == null) {
                obsCountPerLevel[level].put(obs, 1);
            } else {
                obsCountPerLevel[level].put(obs, count + 1);
            }
        }

        // data likelihood for new nodes at each level
        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) // skip the root
        {
            dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(obsCountPerLevel[l]);
        }

        if (remove) {
            removeObservationsFromPath(curLeaf, obsCountPerLevel);
            removeTableFromPath(curLeaf);
        }

        // debug
//        System.out.println("After removing");
//        System.out.println(printGlobalTree());
//        System.out.println(printLocalRestaurants());


        HashMap<SGHLDANode, Double> pathLogPriors = new HashMap<SGHLDANode, Double>();
        computePathLogPrior(pathLogPriors, globalTreeRoot, 0.0);

        HashMap<SGHLDANode, Double> pathWordLlh = new HashMap<SGHLDANode, Double>();
        computePathWordLogLikelihood(pathWordLlh, globalTreeRoot, obsCountPerLevel, dataLlhNewTopic, 0.0);

        if (pathLogPriors.size() != pathWordLlh.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        HashMap<SGHLDANode, Double> pathResLlhs = new HashMap<SGHLDANode, Double>();
        if (isSupervised() && resObserved) {
            HashMap<Integer, SparseCount> curTableDocLevelCounts = curTable.getDocumentLevelCounts();
            HashMap<Integer, Double> docPreSums = new HashMap<Integer, Double>();
            HashMap<Integer, Double> docPassingValues = new HashMap<Integer, Double>();

            for (int docIdx : curTableDocLevelCounts.keySet()) {
                double docPreSum = 0.0;
                for (SGHLDATable otherTable : localRestaurants[d].getTables()) {
                    if (otherTable.equals(curTable)) {
                        continue;
                    }
                    SparseCount docOtherTableLevelCount = otherTable.getDocumentCount(docIdx);
                    if (docOtherTableLevelCount == null) // if no token of this document is assigned to this table
                    {
                        continue;
                    }
                    SGHLDANode[] otherPath = getPathFromNode(otherTable.getContent());
                    for (int l = 0; l < L; l++) {
                        docPreSum += otherPath[l].getRegressionParameter()
                                * docOtherTableLevelCount.getCount(l);
                    }
                }
                docPreSums.put(docIdx, docPreSum);
                docPassingValues.put(docIdx, 0.0);
            }

            // debug
//            System.out.println("# docs = " + docPreSums.size());
//            for(int docIdx : docPreSums.keySet()){
//                System.out.println("--- " + docIdx 
//                        + ". " + curTableDocLevelCounts.get(docIdx).toString()
//                        + ". presum = " + docPreSums.get(docIdx));
//            }

//            computePathResponseLogLikelihood(d, pathResLlhs, globalTreeRoot,
//                    docPreSums, curTableDocLevelCounts, docPassingValues);
            pathResLlhs = computePathResponseLogLikelihood(d, docPreSums, curTableDocLevelCounts);

            // debug
//            for(SGHLDANode path : pathResLlhs.keySet()){
//                System.out.println(path.getPathString() + "\t" + pathResLlhs.get(path));
//            }

            if (pathLogPriors.size() != pathResLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }
        }

        // sample path
        ArrayList<SGHLDANode> pathList = new ArrayList<SGHLDANode>();
        ArrayList<Double> logProbs = new ArrayList<Double>();
        for (SGHLDANode path : pathLogPriors.keySet()) {
            double lp = pathLogPriors.get(path) + pathWordLlh.get(path);
            if (isSupervised() && resObserved) {
                lp += pathResLlhs.get(path);
            }

            pathList.add(path);
            logProbs.add(lp);

            // debug
//            logln("path " + path 
//                    + ". logprior = " + MiscUtils.formatDouble(pathLogPriors.get(path))
//                    + ". word llh = " + MiscUtils.formatDouble(pathWordLlh.get(path))
//                    + ". res llh = " + MiscUtils.formatDouble(pathResLlhs.get(path))
//                    + "\t\t logprob = " + MiscUtils.formatDouble(lp));
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logProbs);
        SGHLDANode newLeaf = pathList.get(sampledIndex);

        // debug
//        logln(">>> sampled index = " + sampledIndex + ". newLeaf: " + newLeaf.toString() + "\n");

        // debug
        if (curLeaf == null || curLeaf.equals(newLeaf)) {
            numTableAssignmentsChange++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (newLeaf.getLevel() < L - 1) {
            newLeaf = this.createNewPath(newLeaf);
        }

        // update 
        curTable.setContent(newLeaf);

        if (add) {
            addTableToPath(newLeaf);
            addObservationsToPath(newLeaf, obsCountPerLevel);
        }

//        if(condition)
//            logln("End d = " + d + ". table index = " + tableIndex + "\n");
    }

    private void optimize() {
        ArrayList<SGHLDANode> flattenTree = flattenTree();
        int numNodes = flattenTree.size();

        // store the regression parameters
        HashMap<SGHLDATable, int[]> nodeIndices = new HashMap<SGHLDATable, int[]>();
        for (int d = 0; d < D; d++) {
            for (SGHLDATable table : localRestaurants[d].getTables()) {
                SGHLDANode[] path = getPathFromNode(table.getContent());
                int[] pathIndices = new int[path.length];
                for (int i = 0; i < pathIndices.length; i++) {
                    pathIndices[i] = flattenTree.indexOf(path[i]);
                }
                nodeIndices.put(table, pathIndices);
            }
        }

        // current regression parameters
        double[] regParams = new double[numNodes];
        double[] priorMeans = new double[numNodes];
        double[] priorStdvs = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            SGHLDANode node = flattenTree.get(i);
            regParams[i] = node.getRegressionParameter();
            priorMeans[i] = mus[node.getLevel()];
            priorStdvs[i] = sqrtSigmas[node.getLevel()];
        }

        // design matrix
        double[] responseArray = new double[numDocs];
        double[][] designMatrix = new double[numDocs][numNodes];
        int idx = 0;
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                responseArray[idx] = responses[d][t];

                int[] counts = new int[numNodes];
                for (int n = 0; n < words[d][t].length; n++) {
                    SGHLDATable table = localRestaurants[d].getTable(z[d][t][n]);
                    int nodeIdx = nodeIndices.get(table)[x[d][t][n]];
                    counts[nodeIdx]++;
                }

                for (int i = 0; i < numNodes; i++) {
                    designMatrix[idx][i] = (double) counts[i] / words[d][t].length;
                }

                idx++;
            }
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responseArray,
                sqrtRho,
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

        if (converged) // if the optimization converges
        {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < numNodes; i++) {
            double newEta = optimizable.getParameter(i);
            flattenTree.get(i).setRegressionParameter(newEta);
        }
    }

    /**
     * Compute the log prior of all possible paths in the tree. The probability
     * associated with an internal node represents the probability of a new path
     * from the root to the given node and to a new node
     *
     * @param pathLogProbs Hash table to store the results
     * @param curNode The current node
     * @param parentLogProb The log prior passed from the parent node
     */
    private void computePathLogPrior(
            HashMap<SGHLDANode, Double> pathLogProbs,
            SGHLDANode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
            newWeight += logGammas[curNode.getLevel()] - logNorm;

            // debug
//            System.out.println("--- lognorm = " + MiscUtils.formatDouble(logNorm)
//                    + ". log gamma = " + MiscUtils.formatDouble(logGammas[curNode.getLevel()]));

            for (SGHLDANode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(pathLogProbs, child, childWeight);
            }
        }
        pathLogProbs.put(curNode, newWeight);
    }

    /**
     * Compute the word log likelihood of each path given a set of observations
     * with their level assignments
     *
     * @param pathLogLikelihoods Hash table to store results
     * @param curNode The current node
     * @param obsPerLevel The observations per level
     * @param dataLlhNewTopic The likelihood of new node per level
     * @param parentLlh Passing value from parent node
     */
    private void computePathWordLogLikelihood(
            HashMap<SGHLDANode, Double> pathLogLikelihoods,
            SGHLDANode curNode,
            HashMap<Integer, Integer>[] obsPerLevel,
            double[] dataLlhNewTopic,
            double parentLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(obsPerLevel[level]);

        // populate to child nodes
        for (SGHLDANode child : curNode.getChildren()) {
            computePathWordLogLikelihood(pathLogLikelihoods, child, obsPerLevel,
                    dataLlhNewTopic, parentLlh + nodeDataLlh);
        }

        // store the data llh from the root to this current node
        double storeDataLlh = parentLlh + nodeDataLlh;
        level++;
        while (level < L) // if this is an internal node, add llh of new child node
        {
            storeDataLlh += dataLlhNewTopic[level++];
        }
        pathLogLikelihoods.put(curNode, storeDataLlh);
    }

    private HashMap<SGHLDANode, Double> computePathResponseLogLikelihood(
            int d,
            HashMap<Integer, Double> docPreSums,
            HashMap<Integer, SparseCount> docLevelCounts) {
        HashMap<SGHLDANode, Double> resLlhs = new HashMap<SGHLDANode, Double>();
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();
            SGHLDANode[] path = getPathFromNode(node);

            // debug
//            System.out.println("\n\nnode: " + node.toString());
//            for(int l=0; l<path.length; l++)
//                System.out.println("p [" + l + "]: " + path[l].toString());

            for (int docIdx : docPreSums.keySet()) {
                double sum = docPreSums.get(docIdx);
                double var = hyperparams.get(RHO);
                int docTokenCount = docTokenCounts[d][docIdx];

                // debug
//                System.out.println("docIdx = " + docIdx
//                        + ". pre sum = " + sum
//                        + ". pre var = " + var
//                        + ". docTokenCount = " + docTokenCount);

                int level;
                for (level = 0; level < path.length; level++) {
                    sum += docLevelCounts.get(docIdx).getCount(level)
                            * path[level].getRegressionParameter();

                    // debug
//                    System.out.println("--- level " + level
//                            + ". count = " + docLevelCounts.get(docIdx).getCount(level)
//                            + ". reg = " + MiscUtils.formatDouble(path[level].getRegressionParameter())
//                            + ". cur sum = " + MiscUtils.formatDouble(sum));
                }

                while (level < L) {
                    int docTabLevelCount = docLevelCounts.get(docIdx).getCount(level);
                    sum += docTabLevelCount * mus[level];
                    var += Math.pow((double) docTabLevelCount / docTokenCount, 2) * sigmas[level];

                    // debug
//                    System.out.println("--- level " + level
//                            + ". count = " + docLevelCounts.get(docIdx).getCount(level)
//                            + ". reg = " + MiscUtils.formatDouble(mus[level])
//                            + ". cur sum = " + MiscUtils.formatDouble(sum));

                    level++;
                }

                double mean = sum / docTokenCount;
                double resLlh = StatUtils.logNormalProbability(responses[d][docIdx], mean, Math.sqrt(var));

                // debug
//                System.out.println(">>> mean = " + MiscUtils.formatDouble(mean) 
//                        + ". var = " + MiscUtils.formatDouble(var)
//                        + ". response: " + responses[d][docIdx]
//                        + ". resLlh = " + MiscUtils.formatDouble(resLlh));

                resLlhs.put(node, resLlh);
            }

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return resLlhs;
    }

//    private void computePathResponseLogLikelihood(
//            int d, 
//            HashMap<SGHLDANode, Double> pathResLlhs,
//            SGHLDANode curNode,
//            HashMap<Integer, Double> docPreSums,
//            HashMap<Integer, SparseCount> docLevelCounts,
//            HashMap<Integer, Double> passingValues){
//        
//        int level = curNode.getLevel();
//        for(int docIdx : docPreSums.keySet()){
//            double curTableLevelVal = docLevelCounts.get(docIdx).getCount(level)
//                    * curNode.getRegressionParameter();
//            double val = passingValues.get(docIdx) + curTableLevelVal;
//            passingValues.put(docIdx, val);
//        }
//        
//        for(SGHLDANode child : curNode.getChildren())
//            computePathResponseLogLikelihood(d, pathResLlhs, child, 
//                    docPreSums, docLevelCounts, passingValues);
//        
//        double storeResLlh = 0;
//        for(int docIdx : docPreSums.keySet()){
//            double mean = docPreSums.get(docIdx) + passingValues.get(docIdx);
//            double var = hyperparams.get(RHO);
//            int docTokenCount = docTokenCounts[d][docIdx];
//            level = curNode.getLevel() + 1;
//            while(level < L){
//                int docTabLevelCount = docLevelCounts.get(docIdx).getCount(level);
//                mean += docTabLevelCount * mus[level];
//                var += Math.pow((double)docTabLevelCount / docTokenCount, 2) * sigmas[level];
//                
//                level ++;
//            }
//            mean /= docTokenCount;
//            storeResLlh += StatisticsUtils.logNormalProbability(responses[d][docIdx], mean, Math.sqrt(var));
//        }
//        pathResLlhs.put(curNode, storeResLlh);
//    }
    /**
     * Compute the log prior of all possible nodes in the tree. The probability
     * associated with each node is the probability of the path from the root to
     * the given node.
     *
     * @param nodeLogProbs Hash table to store results
     * @param curNode The current node
     * @param parentLogProb The log probability passed from the parent node
     */
    private void computeNodeLogPrior(
            HashMap<SGHLDANode, Double> nodeLogProbs,
            SGHLDANode curNode,
            double parentLogProb) {
        nodeLogProbs.put(curNode, parentLogProb);

        if (!isLeafNode(curNode)) {
            int curLevel = curNode.getLevel();
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curLevel]);

            // pseudo child
            double pseudoChildLogProb = logGammas[curLevel] - logNorm;
            nodeLogProbs.put(curNode.getPseudoChild(), pseudoChildLogProb);

            for (SGHLDANode child : curNode.getChildren()) {
                double childLogProb = Math.log(child.getNumCustomers()) - logNorm;
                computeNodeLogPrior(nodeLogProbs, child, childLogProb);
            }
        }
    }

    /**
     * Compute the log likelihood of all nodes in the tree given an observation
     *
     * @param nodeLogLikelihoods Hash table to store results
     * @param curNode The current node
     * @param observations The observation
     */
    private void computeNodeWordLogLikelihood(
            HashMap<SGHLDANode, Double> nodeLogLikelihoods,
            SGHLDANode curNode,
            int observation) {
        double curNodeLlh = curNode.getContent().getLogLikelihood(observation);
        nodeLogLikelihoods.put(curNode, curNodeLlh);

        if (!isLeafNode(curNode)) {
            // pseudo child
            double pseudoChildLlh = emptyModels[curNode.getLevel()].getLogLikelihood(observation);
            nodeLogLikelihoods.put(curNode.getPseudoChild(), pseudoChildLlh);

            for (SGHLDANode child : curNode.getChildren()) {
                computeNodeWordLogLikelihood(nodeLogLikelihoods, child, observation);
            }
        }
    }

    private void computeNodeResponseLogLikelihood(
            HashMap<SGHLDANode, Double> nodeResLlh,
            SGHLDANode curNode,
            double response,
            double regParamSum,
            int tokenCount) {
        double mean = (regParamSum + curNode.getRegressionParameter()) / tokenCount;
        double resLlh = StatUtils.logNormalProbability(response, mean, sqrtRho);
        nodeResLlh.put(curNode, resLlh);

        if (!isLeafNode(curNode)) {
            int childLevel = curNode.getLevel() + 1;
            double pseudoMean = (regParamSum + mus[childLevel]) / tokenCount;
            double pseudoVar = sigmas[childLevel] / (tokenCount * tokenCount) + hyperparams.get(RHO);
            double pseudoLlh = StatUtils.logNormalProbability(response, pseudoMean, Math.sqrt(pseudoVar));
            nodeResLlh.put(curNode.getPseudoChild(), pseudoLlh);

            for (SGHLDANode child : curNode.getChildren()) {
                computeNodeResponseLogLikelihood(nodeResLlh, child, response, regParamSum, tokenCount);
            }
        }
    }

    /**
     * Compute the marginal
     */
    private double[] computeMarginals(
            HashMap<SGHLDANode, Double> pathLogPriors,
            HashMap<SGHLDANode, Double> pathWordLogLikelihoods,
            HashMap<SGHLDANode, Double> pathResLogLikelihoods,
            boolean resObserved) {
        double[] marginals = new double[L];
        for (SGHLDANode node : pathLogPriors.keySet()) {
            int level = node.getLevel();
            double logprior = pathLogPriors.get(node);
            double loglikelihood = pathWordLogLikelihoods.get(node);

            double lp = logprior + loglikelihood;
            if (isSupervised() && resObserved) {
                lp += pathResLogLikelihoods.get(node);
            }

            if (marginals[level] == 0.0) {
                marginals[level] = lp;
            } else {
                marginals[level] = SamplerUtils.logAdd(marginals[level], lp);
            }
        }
        return marginals;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    private SGHLDANode[] getPathFromNode(SGHLDANode node) {
        SGHLDANode[] path = new SGHLDANode[node.getLevel() + 1];
        SGHLDANode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
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

    private boolean isLeafNode(SGHLDANode node) {
        return node.getLevel() == L - 1;
    }

    private ArrayList<SGHLDANode> flattenTree() {
        ArrayList<SGHLDANode> flattenTree = new ArrayList<SGHLDANode>();
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();
            flattenTree.add(node);
            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return flattenTree;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();

        // global
        int[] nodeCounts = new int[L];
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();
            nodeCounts[node.getLevel()]++;
            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> global: ").append(MiscUtils.arrayToString(nodeCounts)).append("\n");

        // local
        int[] numTables = new int[D];
        for (int d = 0; d < D; d++) {
            numTables[d] = localRestaurants[d].getNumTables();
        }
        str.append(">>> local: # tables: min = ").append(StatUtils.min(numTables))
                .append(". max = ").append(StatUtils.max(numTables))
                .append(". avg = ").append(StatUtils.mean(numTables))
                .append(". total = ").append(StatUtils.sum(numTables))
                .append("\n");

        return str.toString();
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);

        int totalObs = 0;

        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    //                    .append("\t").append(MiscUtils.arrayToString(node.getContent().getCounts()))
                    .append("\n");

            totalObs += node.getContent().getCountSum();

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # customers = ").append(globalTreeRoot.getNumCustomers())
                .append("\n");
        return str.toString();
    }

    public String printLocalRestaurants() {
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            logln("restaurant d = " + d
                    + ". # tables: " + localRestaurants[d].getNumTables()
                    + ". # total customers: " + localRestaurants[d].getTotalNumCustomers());
            for (SGHLDATable table : localRestaurants[d].getTables()) {
                logln("--- table: " + table.toString());
            }
            System.out.println();
        }
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLgprob = 0.0;
        double regParamLgprob = 0.0;
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood();

            if (isSupervised()) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        mus[node.getLevel()], sqrtSigmas[node.getLevel()]);
            }

            if (!isLeafNode(node)) {
                treeLgprob += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double restLgprob = 0.0;
        double stickLgprob = 0.0;
        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            restLgprob += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));

            for (SGHLDATable table : localRestaurants[d].getTables()) {
                stickLgprob += table.getStick().getLogLikelihood();
            }

            if (isSupervised()) {
                for (int t = 0; t < words[d].length; t++) {
                    double mean = getRegressionSum(d, t) / words[d][t].length;
                    resLlh += StatUtils.logNormalProbability(responses[d][t], mean, sqrtRho);
                }
            }
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLgprob)
                + ". rest = " + MiscUtils.formatDouble(restLgprob)
                + ". stick = " + MiscUtils.formatDouble(stickLgprob)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLgprob + restLgprob + stickLgprob + regParamLgprob + resLlh;
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
        double treeLgprob = 0.0;
        double regParamLgprob = 0.0;
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);

            if (isSupervised()) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        newMus[node.getLevel()], Math.sqrt(newSigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLgprob += node.getLogJointProbability(newGammas[node.getLevel()]);
            }

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double restLgprob = 0.0;
        double stickLgprob = 0.0;
        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            restLgprob += localRestaurants[d].getJointProbabilityAssignments(tParams.get(ALPHA));

            for (SGHLDATable table : localRestaurants[d].getTables()) {
                stickLgprob += table.getStick().getLogLikelihood(tParams.get(GEM_MEAN), tParams.get(GEM_SCALE));
            }

            if (isSupervised()) {
                for (int t = 0; t < words[d].length; t++) {
                    double mean = getRegressionSum(d, t) / words[d][t].length;
                    resLlh += StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(tParams.get(RHO)));
                }
            }
        }

        double llh = wordLlh + treeLgprob + restLgprob + stickLgprob + regParamLgprob + resLlh;
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

        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();
            node.getContent().setConcentration(betas[node.getLevel()]);
            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l].setConcentration(betas[l + 1]);
        }

        for (int d = 0; d < D; d++) {
            for (SGHLDATable table : localRestaurants[d].getTables()) {
                table.getStick().setMean(hyperparams.get(GEM_MEAN));
                table.getStick().setScale(hyperparams.get(GEM_SCALE));
            }
        }

        this.emptyStick.setMean(hyperparams.get(GEM_MEAN));
        this.emptyStick.setScale(hyperparams.get(GEM_SCALE));

        this.updatePrecomputedHyperparameters();
    }

    @Override
    public void validate(String msg) {
        int totalNumObs = 0;
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            totalNumObs += node.getContent().getCountSum();

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        if (totalNumObs != numTokens) {
            throw new RuntimeException(msg + ". Total numbers of observations mismatch. "
                    + totalNumObs + " vs. " + numTokens);
        }

        int totalNumTables = 0;
        for (int d = 0; d < D; d++) {
            totalNumTables += localRestaurants[d].getNumTables();

            for (SGHLDATable table : localRestaurants[d].getTables()) {
                table.validate(msg);
            }
        }

        if (totalNumTables != globalTreeRoot.getNumCustomers()) {
            throw new RuntimeException(msg + ". Total numbers of tables mismatch");
        }

        // sparse counts
        for (int d = 0; d < D; d++) {
            int trueGroupTokenCount = 0;
            for (int t = 0; t < words[d].length; t++) {
                trueGroupTokenCount += words[d][t].length;
            }

            int testGroupTokenCount = 0;
            for (SGHLDATable table : localRestaurants[d].getTables()) {
                HashMap<Integer, SparseCount> levelCounts = table.getDocumentLevelCounts();
                for (SparseCount sc : levelCounts.values()) {
                    testGroupTokenCount += sc.getCountSum();
                }
            }

            if (trueGroupTokenCount != testGroupTokenCount) {
                throw new RuntimeException(msg + ". Numbers of group tokens mismatch. "
                        + trueGroupTokenCount + " vs. " + testGroupTokenCount);
            }
        }
    }

    private double getRegressionSum(int d, int t) {
        double regSum = 0.0;
        for (SGHLDATable table : localRestaurants[d].getTables()) {
            SparseCount levelCount = table.getDocumentCount(t);
            if (levelCount == null) // this document does not have any tokens assigned to this table
            {
                continue;
            }
            SGHLDANode[] path = getPathFromNode(table.getContent());
            for (int l : levelCount.getIndices()) {
                regSum += levelCount.getCount(l) * path[l].getRegressionParameter();
            }
        }
        return regSum;
    }

    /**
     * Predict the response values using the current model
     */
    public double[][] getRegressionValues() {
        double[][] regValues = new double[D][];
        for (int d = 0; d < D; d++) {
            regValues[d] = new double[responses[d].length];
            for (int t = 0; t < responses[d].length; t++) {
                regValues[d][t] = getRegressionSum(d, t) / words[d][t].length;
            }
        }
        return regValues;
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
    public double[][] regressExistingGroups(int[][][] newWords, double[][] newResponses, String filepath) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[][]> predResponsesList = new ArrayList<double[][]>();
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            double[][] predResponses = regressExistingGroups(reportFolderpath
                    + filename, newWords, newResponses);
            predResponsesList.add(predResponses);

            RegressionEvaluation eval = new RegressionEvaluation(
                    MiscUtils.flatten2DArray(responses),
                    MiscUtils.flatten2DArray(predResponses));
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();

            // output results
            if (i == 0) {
                writer.write("Model");
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getName());
                }
                writer.write("\n");
            }
            writer.write(filename);
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");

            if (verbose) {
                logln("Model from " + reportFolderpath + filename);
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();

            }
        }
        writer.close();

        // average predicted response over different models
        double[][] finalPredResponses = new double[D][];
        for (int d = 0; d < D; d++) {
            finalPredResponses[d] = new double[words[d].length];
            for (int t = 0; t < finalPredResponses[d].length; t++) {
                double sum = 0.0;
                for (int i = 0; i < predResponsesList.size(); i++) {
                    sum += predResponsesList.get(i)[d][t];
                }
                finalPredResponses[d][t] = sum / predResponsesList.size();
            }
        }
        return finalPredResponses;
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data using a specific model.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
    private double[][] regressExistingGroups(String stateFile, int[][][] newWords, double[][] newResponses) {
        if (newWords.length != D) {
            throw new RuntimeException("Number of test documents does not match");
        }

        if (verbose) {
            logln("Perform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = newResponses; // for evaluation
        D = words.length;

        numDocs = 0;
        numTokens = 0;
        docTokenCounts = new int[D][];
        for (int d = 0; d < D; d++) {
            numDocs += words[d].length;
            docTokenCounts[d] = new int[words[d].length];
            for (int t = 0; t < words[d].length; t++) {
                numTokens += this.words[d][t].length;
                docTokenCounts[d][t] = words[d][t].length;
            }
        }

        // debug
        logln("--- # groups = " + D);
        logln("--- # documents = " + numDocs);
        logln("--- # tokens = " + numTokens);

        // initialize structure
        initializeDataStructure();

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    sampleTableLevelForToken(d, t, n, !REMOVE, !OBSERVED, !EXTEND);
                }
            }
        }

        // iterate
        ArrayList<double[][]> predResponsesList = new ArrayList<double[][]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        sampleTableLevelForToken(d, t, n, REMOVE, !OBSERVED, !EXTEND);
                    }
                }
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[][] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);

                if (verbose) {
                    logln("state file: " + stateFile
                            + ". iter = " + iter
                            + ". llh = " + getLogLikelihood());

                    RegressionEvaluation eval = new RegressionEvaluation(
                            MiscUtils.flatten2DArray(responses),
                            MiscUtils.flatten2DArray(predResponses));
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
        }

        // averaging prediction responses over time
        double[][] finalPredResponses = new double[D][];
        for (int d = 0; d < D; d++) {
            finalPredResponses[d] = new double[words[d].length];
            for (int t = 0; t < finalPredResponses[d].length; t++) {
                double sum = 0.0;
                for (int i = 0; i < predResponsesList.size(); i++) {
                    sum += predResponsesList.get(i)[d][t];
                }
                finalPredResponses[d][t] = sum / predResponsesList.size();
            }
        }
        return finalPredResponses;
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                modelStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                for (SGHLDATable table : localRestaurants[d].getTables()) {
                    modelStr.append(table.getIndex()).append("\n");
                    modelStr.append(TruncatedStickBreaking.output(table.getStick())).append("\n");
                    modelStr.append(table.getContent().getPathString()).append("\n");
                }
            }

            Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SGHLDANode node = stack.pop();

                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");

                for (SGHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
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

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        assignStr.append(d)
                                .append(":").append(t)
                                .append(":").append(n)
                                .append("\t").append(x[d][t][n])
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

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        // initialize
        this.initializeModelStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

        // local restaurants
        HashMap<String, String> tablePathStrMap = new HashMap<String, String>();
        for (int d = 0; d < D; d++) {
            int numTables = Integer.parseInt(reader.readLine().split("\t")[1]);
            for (int i = 0; i < numTables; i++) {
                int tableIndex = Integer.parseInt(reader.readLine());
                TruncatedStickBreaking stick = TruncatedStickBreaking.input(reader.readLine());
                String nodePath = reader.readLine();

//                int nDocs = Integer.parseInt(reader.readLine());
//                HashMap<Integer, SparseCount> docLevelCounts = new HashMap<Integer, SparseCount>();
//                for(int t=0; t<nDocs; t++){
//                    String[] sline = reader.readLine().split(":::");
//                    int docIdx = Integer.parseInt(sline[0]);
//                    SparseCount sc = SparseCount.input(sline[1]);
//                    docLevelCounts.put(docIdx, sc);
//                }

                SGHLDATable table = new SGHLDATable(iter, tableIndex, null, d, stick, new HashMap<Integer, SparseCount>());
                this.localRestaurants[d].addTable(table);
                tablePathStrMap.put(table.getTableId(), nodePath);

//                String[] sline = reader.readLine().trim().split("\t");
//                for(int j=0; j<sline.length; j++)
//                    this.localRestaurants[d].addCustomerToTable(sline[j], tableIndex);
            }
        }

        // global tree
        HashMap<String, SGHLDANode> nodeMap = new HashMap<String, SGHLDANode>();
        HashMap<SGHLDANode, Integer> nodeNumCusts = new HashMap<SGHLDANode, Integer>(); // for debug
        String line;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int numCusts = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SGHLDANode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
            int nodeLevel = pathIndices.length - 1;
            SGHLDANode node = new SGHLDANode(iter, nodeIndex,
                    nodeLevel, dmm, regParam, parent, (nodeLevel == L - 1 ? false : true));

            if (node.getLevel() == 0) {
                globalTreeRoot = node;
            }

            if (parent != null) {
                parent.addChild(node.getIndex(), node);
            }

            nodeMap.put(pathStr, node);
            nodeNumCusts.put(node, numCusts);
        }
        reader.close();

        // connect table -> node
        for (int d = 0; d < D; d++) {
            for (SGHLDATable table : localRestaurants[d].getTables()) {
                String path = tablePathStrMap.get(table.getTableId());
                SGHLDANode leafNode = nodeMap.get(path);
                table.setContent(leafNode);
                addTableToPath(leafNode);
            }
        }

        // debug
        for (SGHLDANode node : nodeNumCusts.keySet()) {
            if (node.getNumCustomers() != nodeNumCusts.get(node)) {
                throw new RuntimeException("Numbers of customers mismatch in node " + node.toString()
                        + ". " + node.getNumCustomers() + " vs. " + nodeNumCusts.get(node));
            }
        }

        // update inactive children list
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();
            if (!isLeafNode(node)) {
                node.fillInactiveChildIndices();
                for (SGHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        // update inactive tables
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d].fillInactiveTableIndices();
        }
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    String[] sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + t + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][t][n] = Integer.parseInt(sline[1]);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    String[] sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + t + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    x[d][t][n] = Integer.parseInt(sline[1]);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    SGHLDATable table = localRestaurants[d].getTable(z[d][t][n]);
                    localRestaurants[d].addCustomerToTable(getTokenId(t, n), table.getIndex());
                    SparseCount levelCount = table.getDocumentCount(t);
                    if (levelCount == null) {
                        levelCount = new SparseCount();
                    }
                    levelCount.increment(x[d][t][n]);
                    table.setDocumentCount(t, levelCount);
                }
            }
        }

        reader.close();
    }

    public void outputTopicTopWords(String outputFile, int numWords)
            throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
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
            System.out.println("Outputing topic coherence to file " + filepath);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);

        Stack<SGHLDANode> stack = new Stack<SGHLDANode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            SGHLDANode node = stack.pop();

            for (SGHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getNumCustomers()
                    + "\t" + node.getContent().getCountSum()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }

        writer.close();
    }

    public void diagnose(String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int d = 0; d < D; d++) {
            for (SGHLDATable table : this.localRestaurants[d].getTables()) {
                double[] distribution = table.getContent().getContent().getDistribution();
                int[] topic = SamplerUtils.getSortedTopic(distribution);

                writer.write(d + ": " + table.toString() + "\n");
                writer.write("\t");
                for (int i = 0; i < 15; i++) {
                    writer.write(wordVocab.get(topic[i]) + ", ");
                }
                writer.write("\n");

                int[] counts = new int[V];
                for (String customer : table.getCustomers()) {
                    int[] parsedId = parseTokenId(customer);
                    counts[words[d][parsedId[0]][parsedId[1]]]++;
                }
                ArrayList<RankingItem<Integer>> rankList = new ArrayList<RankingItem<Integer>>();
                for (int v = 0; v < V; v++) {
                    rankList.add(new RankingItem<Integer>(v, counts[v]));
                }
                Collections.sort(rankList);
                writer.write("\t");
                for (int i = 0; i < 15; i++) {
                    writer.write(wordVocab.get(rankList.get(i).getObject()) + ":"
                            + (int) rankList.get(i).getPrimaryValue() + ", ");
                }
                writer.write("\n\n");
            }
        }
        writer.close();
    }

    public static void main(String[] args) {
        try {
            test();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void test() throws Exception {
        int V = 4;
        double alpha = 1;
        double rho = 0.5;
        double gem_mean = 0.3;
        double gem_scale = 50;

        int L = 3;
        double[] betas = {1 * V, 0.5 * V, 0.1 * V};
        double[] gammas = {1, 1};
        double[] mus = {0.0, 0.0, 0.0};
        double[] sigmas = {0.0001, 0.5, 2};

        InitialState initState = InitialState.PRESET;
        boolean paramOpt = false;

        DirMult rootModel = new DirMult(V, betas[0], 1.0 / V);
        SGHLDANode root = new SGHLDANode(0, 0, 0, rootModel, 0.0, null, true);

        DirMult node00Model = new DirMult(V, betas[1], 1.0 / V);
        SGHLDANode node00 = new SGHLDANode(0, 0, 1, node00Model, -1.0, root, true);
        root.addChild(0, node00);

        DirMult node01Model = new DirMult(V, betas[1], 1.0 / V);
        SGHLDANode node01 = new SGHLDANode(0, 1, 1, node01Model, 1.0, root, true);
        root.addChild(1, node01);

        DirMult node000Model = new DirMult(V, betas[2], 1.0 / V);
        SGHLDANode node000 = new SGHLDANode(0, 0, 2, node000Model, -2.0, node00, false);
        node00.addChild(0, node000);

        DirMult node001Model = new DirMult(V, betas[2], 1.0 / V);
        SGHLDANode node001 = new SGHLDANode(0, 1, 2, node001Model, -0.5, node00, false);
        node00.addChild(1, node001);

        DirMult node010Model = new DirMult(V, betas[2], 1.0 / V);
        SGHLDANode node010 = new SGHLDANode(0, 0, 2, node010Model, 2, node01, false);
        node01.addChild(0, node010);

        SGHLDASampler sampler = new SGHLDASampler();
        sampler.setVerbose(true);
        sampler.setDebug(true);
        sampler.setLog(false);
        sampler.setReport(false);
        sampler.setSupervised(true);

        int D = 4;
        int N = 9;
        int[][][] words = new int[D][2][N];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < N; n++) {
                words[d][0][n] = d;
                words[d][1][n] = (d + 1) % V;
            }
        }
        double[][] responses = new double[D][2];
        for (int d = 0; d < D; d++) {
            if (d < D / 2) {
                responses[d][0] = -1;
                responses[d][1] = -1;
            } else {
                responses[d][0] = 1;
                responses[d][1] = 1;
            }
        }

        sampler.configure(null, words, responses, V, L,
                alpha, rho, gem_mean, gem_scale,
                betas,
                gammas,
                mus,
                sigmas,
                initState, paramOpt,
                2, 10, 3);
        sampler.initializeModelStructure();
        sampler.globalTreeRoot = root;

        sampler.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            sampler.localRestaurants[d] = new Restaurant<SGHLDATable, String, SGHLDANode>();
            SGHLDATable table0 = new SGHLDATable(0, 0, node000, d, L, gem_mean, gem_scale);
            sampler.localRestaurants[d].addTable(table0);
            sampler.addTableToPath(node000);

            SGHLDATable table1 = new SGHLDATable(0, 1, node001, d, L, gem_mean, gem_scale);
            sampler.localRestaurants[d].addTable(table1);
            sampler.addTableToPath(node001);

            SGHLDATable table2 = new SGHLDATable(0, 2, node010, d, L, gem_mean, gem_scale);
            sampler.localRestaurants[d].addTable(table2);
            sampler.addTableToPath(node010);
        }


        sampler.initializeDataStructure();
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < 2; t++) {
                for (int n = 0; n < N; n++) {
                    int tabIdx = d % 3;
                    SGHLDATable table = sampler.localRestaurants[d].getTable(tabIdx);

                    sampler.z[d][t][n] = tabIdx;
                    sampler.x[d][t][n] = n % L;
                    sampler.localRestaurants[d].addCustomerToTable(sampler.getTokenId(t, n), sampler.z[d][t][n]);
                    table.increment(t, sampler.x[d][t][n]);
                    sampler.addObservationToNodeOnPath(table.getContent(), sampler.x[d][t][n], words[d][t][n]);
                }
            }
        }
        sampler.validate("");

        System.out.println(sampler.printGlobalTree());
        System.out.println(sampler.printLocalRestaurants() + "\n");

        sampler.samplePathForTable(0, 0, true, true, true);
    }
}

class SGHLDANode extends TreeNode<SGHLDANode, DirMult> {

    private final int born;
    private int numCustomers;
    private double regression;
    private SGHLDANode pseudoChild;

    SGHLDANode(int iter, int index, int level, DirMult content, double regParam,
            SGHLDANode parent, boolean hasPseudoChild) {
        super(index, level, content, parent);
        this.born = iter;
        this.numCustomers = 0;
        this.regression = regParam;
        if (hasPseudoChild) {
            this.pseudoChild = new SGHLDANode(iter, SGHLDASampler.PSEUDO_NODE_INDEX, level + 1, null, 0.0, this, false);
        }
    }

    public int getIterationCreated() {
        return this.born;
    }

    double getLogJointProbability(double gamma) {
        ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
        for (SGHLDANode child : this.getChildren()) {
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

    public SGHLDANode getPseudoChild() {
        return pseudoChild;
    }

    public void setPseudoChild(SGHLDANode pseudoChild) {
        this.pseudoChild = pseudoChild;
    }

    public boolean isEmpty() {
        return this.numCustomers == 0;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", ").append(born)
                .append(", #ch = ").append(getNumChildren())
                .append(", #c = ").append(getNumCustomers())
                .append(", #o = ").append(getContent().getCountSum())
                .append(", reg = ").append(MiscUtils.formatDouble(regression))
                .append("]");
        return str.toString();
    }
}

class SGHLDATable extends FullTable<String, SGHLDANode> {

    private final int born;
    final int restIndex;
    private TruncatedStickBreaking stick;
    private HashMap<Integer, SparseCount> docLevelCounts;

    SGHLDATable(int iter, int index, SGHLDANode content, int restId,
            int numLevels, double gemMean, double gemScale) {
        super(index, content);
        this.born = iter;
        this.restIndex = restId;
        this.customers = new ArrayList<String>();
        this.stick = new TruncatedStickBreaking(numLevels, gemMean, gemScale);
        this.docLevelCounts = new HashMap<Integer, SparseCount>();
    }

    SGHLDATable(int iter, int index, SGHLDANode content, int restId,
            TruncatedStickBreaking stick, HashMap<Integer, SparseCount> indCounts) {
        super(index, content);
        this.born = iter;
        this.restIndex = restId;
        this.customers = new ArrayList<String>();
        this.stick = stick;
        this.docLevelCounts = indCounts;
    }

    public int getIterationCreated() {
        return this.born;
    }

    public TruncatedStickBreaking getStick() {
        return this.stick;
    }

    public HashMap<Integer, SparseCount> getDocumentLevelCounts() {
        return this.docLevelCounts;
    }

    public SparseCount getDocumentCount(int d) {
        return this.docLevelCounts.get(d);
    }

    public void setDocumentCount(int d, SparseCount sc) {
        this.docLevelCounts.put(d, sc);
    }

    public void increment(int docIndex, int level) {
        this.stick.increment(level);
        SparseCount levelCount = this.docLevelCounts.get(docIndex);
        if (levelCount == null) {
            levelCount = new SparseCount();
        }
        levelCount.increment(level);
        this.docLevelCounts.put(docIndex, levelCount);
    }

    public void decrement(int docIndex, int level) {
        this.stick.decrement(level);
        SparseCount levelCount = this.docLevelCounts.get(docIndex);
        levelCount.decrement(level);
        if (levelCount.isEmpty()) {
            this.docLevelCounts.remove(docIndex);
        }
    }

    public String getTableId() {
        return restIndex + ":" + index;
    }

    @Override
    public int getNumCustomers() {
        return this.stick.getCountSum();
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
        SGHLDATable r = (SGHLDATable) (obj);

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
                .append(", ").append(MiscUtils.arrayToString(stick.getCounts()))
                .append(", ").append(docLevelCounts.size())
                .append("]")
                .append(" >> ").append(getContent() == null ? "null" : getContent().toString());
        return str.toString();
    }

    public void validate(String msg) {
        this.stick.validate(msg);
    }
}

class NodeSparseCount {

    private HashMap<SGHLDANode, Integer> counts;
    private int countSum;

    public NodeSparseCount() {
        this.counts = new HashMap<SGHLDANode, Integer>();
        this.countSum = 0;
    }

    public Set<SGHLDANode> getNodeSet() {
        return counts.keySet();
    }

    public int getCountSum() {
        return this.countSum;
    }

    public int getCount(SGHLDANode node) {
        Integer count = this.counts.get(node);
        if (count == null) {
            return 0;
        } else {
            return count;
        }
    }

    public void setCount(SGHLDANode node, int count) {
        if (count < 0) {
            throw new RuntimeException("Setting a negative count. " + count);
        }
        int curCount = this.getCount(node);
        this.counts.put(node, count);
        this.countSum += count - curCount;
    }

    public void changeCount(SGHLDANode node, int delta) {
        int count = getCount(node);
        this.setCount(node, count + delta);
    }

    public void increment(SGHLDANode node) {
        Integer count = this.counts.get(node);
        if (count == null) {
            this.counts.put(node, 1);
        } else {
            this.counts.put(node, count + 1);
        }
        this.countSum++;
    }

    public void decrement(SGHLDANode node) {
        Integer count = this.counts.get(node);
        if (count == null) {
            for (SGHLDANode obs : this.counts.keySet()) {
                System.out.println(obs + ": " + this.counts.get(obs));
            }
            throw new RuntimeException("Removing observation that does not exist " + node);
        }
        if (count == 1) {
            this.counts.remove(node);
        } else {
            this.counts.put(node, count - 1);
        }
        this.countSum--;

        if (counts.get(node) != null && this.counts.get(node) < 0) {
            throw new RuntimeException("Negative count for observation " + node
                    + ". count = " + this.counts.get(node));
        }
        if (countSum < 0) {
            throw new RuntimeException("Negative count sumze " + countSum);
        }
    }

    public boolean isEmpty() {
        return this.countSum == 0;
    }
}