/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.backup;

import sampler.supervised.objective.GaussianIndLinearRegObjective;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.TreeNode;
import sampling.util.SparseCount;
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
public class SentSHLDASampler extends AbstractSampler {

    public static final int PSEUDO_NODE_INDEX = -1;
    public static final int RHO = 0;
    public static final int MEAN = 1;
    public static final int SCALE = 2;
    protected boolean supervised = true;
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;
    protected double[] sigmas;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int regressionLevel;
    protected int[][][] words;  // [D] [S_d] [N_ds]: doc-sent words
    protected double[] responses; // [D]: document observations
    private int[][][] z; // level assignments for each token
    private SentSHLDANode[][] c; // path assignments for each sentence
    private TruncatedStickBreaking[] doc_level_distr;
    private SparseCount[][] sent_level_count;
    private SentSHLDANode word_hier_root;
    private DirMult[] emptyModels;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private double[] uniform;
    private int tokenCount;
    private int[] docTokenCounts;
    private int numChangePath;
    private int numChangeLevel;
    private int existingPathCount;
    private int totalPathCount;
    private int optimizeCount = 0;
    private int convergeCount = 0;

    public void configure(String folder, int[][][] words, double[] y,
            int V, int L,
            double mean, // GEM mean
            double scale, // GEM scale
            double[] betas, // Dirichlet hyperparameter for distributions over words
            double[] gammas, // hyperparameter for nCRP
            double[] mus, // mean of Gaussian for regression parameters
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double rho, // standard deviation of Gaussian for document observations
            int regressLevel,
            AbstractSampler.InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;
        this.responses = y;
        if (this.responses == null) {
            this.supervised = false;
        }

        this.L = L;
        this.V = V;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;
        this.mus = mus;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(rho);
        this.hyperparams.add(mean);
        this.hyperparams.add(scale);

        for (int l = 0; l < betas.length; l++) {
            this.hyperparams.add(betas[l]);
        }

        for (int i = 0; i < gammas.length; i++) {
            this.hyperparams.add(gammas[i]);
        }

        for (int i = 0; i < mus.length; i++) {
            this.hyperparams.add(mus[i]);
        }

        for (int i = 0; i < sigmas.length; i++) {
            this.hyperparams.add(sigmas[i]);
        }

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.regressionLevel = regressLevel;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.tokenCount = 0;
        for (int d = 0; d < D; d++) {
            this.tokenCount += words[d].length;
        }

        this.uniform = new double[V];
        for (int i = 0; i < V; i++) {
            uniform[i] = 1.0 / V;
        }

        this.docTokenCounts = new int[D];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                this.docTokenCounts[d] += words[d][s].length;
            }
        }

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gamms must have length " + (this.L - 1)
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

        if (!debug) {
            System.err.close();
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- max level:\t" + L);
            logln("--- reg level:\t" + regressLevel);
            logln("--- GEM mean:\t" + hyperparams.get(MEAN));
            logln("--- GEM scale:\t" + hyperparams.get(SCALE));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- reg mus:\t" + MiscUtils.arrayToString(mus));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- response rho:\t" + hyperparams.get(RHO));
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
                .append(supervised ? "_Sent-SHLDA" : "_Sent-HLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_LVL-").append(L)
                .append("_RL-").append(regressionLevel)
                .append("_GEM-M-").append(formatter.format(hyperparams.get(MEAN)))
                .append("_GEM-S-").append(formatter.format(hyperparams.get(SCALE)))
                .append("_RHO-").append(formatter.format(hyperparams.get(RHO)));

        int count = SCALE + 1;
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
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_s");
        for (int i = 0; i < sigmas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }

        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
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

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

        int rootLevel = 0;
        int rootIndex = 0;
        DirMult dmModel = new DirMult(V, betas[rootLevel], uniform);
        double regParam = SamplerUtils.getGaussian(mus[rootLevel], sigmas[rootLevel]);
        this.word_hier_root = new SentSHLDANode(iter, rootIndex, rootLevel, dmModel, regParam, null);

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }
    }

    protected void initializeDataStructure() {
        this.doc_level_distr = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            this.doc_level_distr[d] = new TruncatedStickBreaking(L, hyperparams.get(MEAN), hyperparams.get(SCALE));
        }

        this.sent_level_count = new SparseCount[D][];
        for (int d = 0; d < D; d++) {
            this.sent_level_count[d] = new SparseCount[words[d].length];
            for (int s = 0; s < words[d].length; s++) {
                this.sent_level_count[d][s] = new SparseCount();
            }
        }

        this.c = new SentSHLDANode[D][];
        this.z = new int[D][][];
        for (int d = 0; d < D; d++) {
            c[d] = new SentSHLDANode[words[d].length];
            z[d] = new int[words[d].length][];
            for (int s = 0; s < words[d].length; s++) {
                z[d][s] = new int[words[d][s].length];
            }
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }

        if (verbose) {
            logln("--- Done initialization. "
                    + "Llh = " + this.getLogLikelihood()
                    + "\t" + this.getCurrentState());
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        // initialize path assignments
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                SentSHLDANode node = word_hier_root;
                for (int l = 0; l < L - 1; l++) {
                    node.incrementNumCustomers();
                    node = this.createNode(node); // create a new path for each document
                }
                node.incrementNumCustomers();
                c[d][s] = node;

                // forward sample levels
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelAssignments(d, s, n, !REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
                }

                // resample path
                if (d > 0) {
                    samplePathAssignments(d, s, REMOVE, ADD, !OBSERVED, EXTEND);
                }

                // resampler levels
                for (int n = 0; n < words[d][s].length; n++) {
                    sampleLevelAssignments(d, s, n, REMOVE, ADD, REMOVE, ADD, !OBSERVED);
                }
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

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
                            + "\t llh = " + loglikelihood
                            + "\t" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t" + getCurrentState());
                }
            }

            numChangePath = 0;
            numChangeLevel = 0;
            existingPathCount = 0;
            totalPathCount = 0;
            optimizeCount = 0;
            convergeCount = 0;
            numChangePath = 0;
            numChangeLevel = 0;

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    samplePathAssignments(d, s, REMOVE, ADD, OBSERVED, EXTEND);

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelAssignments(d, s, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                    }
                }
            }

            if (supervised) {
                optimize();
            }

            if (verbose && supervised) {
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

            if (debug) {
                validate("iter " + iter);
            }

            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    if (verbose) {
                        logln("*** *** Optimizing hyperparameters by slice sampling ...");
                        logln("*** *** cur param:" + MiscUtils.listToString(hyperparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }

                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        logln("*** *** new param:" + MiscUtils.listToString(sparams));
                        logln("*** *** new llh = " + this.getLogLikelihood());
                    }
                }
            }

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
    private void addCustomerToPath(SentSHLDANode leafNode) {
        SentSHLDANode node = leafNode;
        while (node != null) {
            node.incrementNumCustomers();
            node = node.getParent();
        }
    }

    /**
     * Remove a customer from a path. A path is specified by the pointer to its
     * leaf node. If the given node is not a leaf node, an exception will be
     * thrown. The number of customers at each node on the path will be
     * decremented. If the number of customers at a node is 0, the node will be
     * removed.
     *
     * @param leafNode The leaf node of the path
     * @return Return the node that specifies the path that the leaf node is
     * removed from. If a lower-level node has no customer, it will be removed
     * and the lowest parent node on the path that has non-zero number of
     * customers will be returned.
     */
    private SentSHLDANode removeCustomerFromPath(SentSHLDANode leafNode) {
        SentSHLDANode retNode = leafNode;
        SentSHLDANode node = leafNode;
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
     * Remove an observation from a node.
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void removeObservation(int observation, int level, SentSHLDANode leafNode) {
        SentSHLDANode node = getNode(level, leafNode);
        node.getContent().decrement(observation);
    }

    private void removeObservationsFromPath(SentSHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SentSHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            removeObservationsFromNode(path[l], observations[l]);
        }
    }

    private void removeObservationsFromNode(SentSHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, -count);
        }
    }

    /**
     * Add an observation to a node
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void addObservation(int observation, int level, SentSHLDANode leafNode) {
        SentSHLDANode node = getNode(level, leafNode);
        node.getContent().increment(observation);
    }

    private void addObservationsToPath(SentSHLDANode leafNode, HashMap<Integer, Integer>[] observations) {
        SentSHLDANode[] path = getPathFromNode(leafNode);
        for (int l = 0; l < L; l++) {
            addObservationsToNode(path[l], observations[l]);
        }
    }

    private void addObservationsToNode(SentSHLDANode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            node.getContent().changeCount(obs, count);
        }
    }

    /**
     * Create a new child of a parent node
     *
     * @param parent The parent node
     * @return The newly created child node
     */
    private SentSHLDANode createNode(SentSHLDANode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmModel = new DirMult(V, betas[level], uniform);
        double regParam = SamplerUtils.getGaussian(mus[level], sigmas[level]);
        SentSHLDANode child = new SentSHLDANode(iter, nextChildIndex, level, dmModel, regParam, parent);
        return parent.addChild(nextChildIndex, child);
    }

    private void samplePathAssignments(int d, int s,
            boolean remove,
            boolean add,
            boolean observed,
            boolean extend) {
        HashMap<Integer, Integer>[] docTypeCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            docTypeCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[d][s].length; n++) {
            int type = words[d][s][n];
            int level = z[d][s][n];
            Integer count = docTypeCountPerLevel[level].get(type);
            if (count == null) {
                docTypeCountPerLevel[level].put(type, 1);
            } else {
                docTypeCountPerLevel[level].put(type, count + 1);
            }
        }

        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) // skip the root
        {
            dataLlhNewTopic[l] = emptyModels[l - 1].getLogLikelihood(docTypeCountPerLevel[l]);
        }

        if (remove) {
            removeObservationsFromPath(c[d][s], docTypeCountPerLevel);
            removeCustomerFromPath(c[d][s]);
        }

        HashMap<SentSHLDANode, Double> pathLogPriors = new HashMap<SentSHLDANode, Double>();
        computePathLogPrior(pathLogPriors, word_hier_root, 0.0);

        HashMap<SentSHLDANode, Double> pathWordLlhs = new HashMap<SentSHLDANode, Double>();
        computePathWordLogLikelihood(pathWordLlhs, word_hier_root,
                docTypeCountPerLevel, dataLlhNewTopic, 0.0);

        if (pathLogPriors.size() != pathWordLlhs.size()) {
            throw new RuntimeException("Numbers of paths mismatch");
        }

        HashMap<SentSHLDANode, Double> pathResLlhs = new HashMap<SentSHLDANode, Double>();
        if (supervised && observed) {
            pathResLlhs = computePathResponseLogLikelihood(d, s);

            if (pathLogPriors.size() != pathResLlhs.size()) {
                throw new RuntimeException("Numbers of paths mismatch");
            }
        }

        // sample path
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<SentSHLDANode> pathList = new ArrayList<SentSHLDANode>();
        for (SentSHLDANode path : pathLogPriors.keySet()) {
            if (!extend && !isLeafNode(path)) // during test time, fix the tree
            {
                continue;
            }

            double lp = pathLogPriors.get(path) + pathWordLlhs.get(path);
            if (supervised && observed) {
                lp += pathResLlhs.get(path);
            }
            pathList.add(path);
            logprobs.add(lp);
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        SentSHLDANode newPath = pathList.get(sampledIndex);

        if (newPath.getLevel() < L - 1) {
            newPath = this.createNewPath(newPath);
        }

        c[d][s] = newPath;

        if (add) {
            addCustomerToPath(c[d][s]);
            addObservationsToPath(c[d][s], docTypeCountPerLevel);
        }
    }

    private void sampleLevelAssignments(int d, int s, int n,
            boolean removeLevelDist, boolean addLevelDist,
            boolean removeWordHier, boolean addWordHier,
            boolean observed) {

        // decrement 
        if (removeLevelDist) {
            doc_level_distr[d].decrement(z[d][s][n]);
            sent_level_count[d][s].decrement(z[d][s][n]);
        }

        if (removeWordHier) {
            removeObservation(words[d][s][n], z[d][s][n], c[d][s]);
        }

        double preSum = 0.0;
        if (observed) {
            for (int i = 0; i < words[d].length; i++) {
                double[] pathRegParams = getRegressionPath(c[d][i]);
                for (int l = 0; l < L; l++) {
                    preSum += pathRegParams[l] * sent_level_count[d][i].getCount(l);
                }
            }
        }

        double[] logprobs = new double[L];
        double[] curPathRegParams = getRegressionPath(c[d][s]);
        for (int l = 0; l < L; l++) {
            // sampling equation
            SentSHLDANode node = this.getNode(l, c[d][s]);
            double logprior = doc_level_distr[d].getLogProbability(l);
            double wordLlh = node.getContent().getLogLikelihood(words[d][s][n]);
            double lp = logprior + wordLlh;

            if (observed) {
                double sum = preSum + curPathRegParams[l];
                double mean = sum / docTokenCounts[d];
                double resLlh = StatUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));

                lp += resLlh;
                logprobs[l] = lp;
            }
        }

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);
        if (z[d][s][n] != sampledL) {
            numChangeLevel++;
        }

        // update and increment
        z[d][s][n] = sampledL;

        if (addLevelDist) {
            doc_level_distr[d].increment(z[d][s][n]);
            sent_level_count[d][s].increment(z[d][s][n]);
        }

        if (addWordHier) {
            this.addObservation(words[d][s][n], z[d][s][n], c[d][s]);
        }
    }

    private void optimizeNew() {
        try {
            ArrayList<SentSHLDANode> flatTree = flattenTree(word_hier_root);
            int numNodes = flatTree.size();


        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void optimize() {
        ArrayList<SentSHLDANode> flatTree = flattenTree(word_hier_root);
        int numNodes = flatTree.size();

        // current regression parameters
        double[] regParams = new double[numNodes];
        double[] priorMeans = new double[numNodes];
        double[] priorStdvs = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            SentSHLDANode node = flatTree.get(i);
            regParams[i] = node.getRegressionParameter();
            priorMeans[i] = mus[node.getLevel()];
            priorStdvs[i] = Math.sqrt(sigmas[node.getLevel()]);
        }

        double[][] designMatrix = new double[D][numNodes];
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                SentSHLDANode[] path = getPathFromNode(c[d][s]);
                for (int l = 0; l < L; l++) {
                    int nodeIdx = flatTree.indexOf(path[l]);
                    int count = sent_level_count[d][s].getCount(l);
                    designMatrix[d][nodeIdx] += count;
                }
            }
            for (int i = 0; i < numNodes; i++) {
                designMatrix[d][i] /= docTokenCounts[d];
            }
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses,
                Math.sqrt(hyperparams.get(RHO)),
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

        optimizeCount++;

        // if the number of observations is less than or equal to the number of parameters
        if (converged) {
            convergeCount++;
        }

        // update regression parameters
        for (int i = 0; i < flatTree.size(); i++) {
            flatTree.get(i).setRegressionParameter(optimizable.getParameter(i));
        }
    }

    private void computePathLogPrior(
            HashMap<SentSHLDANode, Double> nodeLogProbs,
            SentSHLDANode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);
            newWeight += Math.log(gammas[curNode.getLevel()]) - logNorm;

            for (SentSHLDANode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }

        }
        nodeLogProbs.put(curNode, newWeight);
    }

    private void computePathWordLogLikelihood(
            HashMap<SentSHLDANode, Double> nodeDataLlhs,
            SentSHLDANode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();
        double nodeDataLlh = curNode.getContent().getLogLikelihood(docTokenCountPerLevel[level]);

        // populate to child nodes
        for (SentSHLDANode child : curNode.getChildren()) {
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

    private HashMap<SentSHLDANode, Double> computePathResponseLogLikelihood(int d, int s) {
        HashMap<SentSHLDANode, Double> resLlhs = new HashMap<SentSHLDANode, Double>();

        double preSum = 0.0; // sum of reg from other sentences in the document
        for (int i = 0; i < words[d].length; i++) {
            if (i == s) {
                continue;
            }
            double[] pathRegs = getRegressionPath(c[d][i]);
            for (int l = 0; l < L; l++) {
                preSum += pathRegs[l] * sent_level_count[d][i].getCount(l);
            }
        }

        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            SentSHLDANode[] path = getPathFromNode(node);
            double addSum = 0.0;
            double var = hyperparams.get(RHO);
            int level;
            for (level = 0; level < path.length; level++) {
                addSum += path[level].getRegressionParameter() * sent_level_count[d][s].getCount(level);
            }
            while (level < L) {
                int levelCount = sent_level_count[d][s].getCount(level);
                addSum += levelCount * mus[level];
                var += Math.pow((double) levelCount / docTokenCounts[d], 2) * sigmas[level];
                level++;
            }

            double mean = (preSum + addSum) / docTokenCounts[d];
            double resLlh = StatUtils.logNormalProbability(responses[d], mean, Math.sqrt(var));
            resLlhs.put(node, resLlh);

            for (SentSHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        return resLlhs;
    }

    /**
     * Flatten a subtree given the root of the subtree
     *
     * @param subtreeRoot The subtree's root
     */
    private ArrayList<SentSHLDANode> flattenTree(SentSHLDANode subtreeRoot) {
        ArrayList<SentSHLDANode> flatSubtree = new ArrayList<SentSHLDANode>();
        Queue<SentSHLDANode> queue = new LinkedList<SentSHLDANode>();
        queue.add(subtreeRoot);
        while (!queue.isEmpty()) {
            SentSHLDANode node = queue.poll();
            flatSubtree.add(node);
            for (SentSHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }
        return flatSubtree;
    }

    /**
     * Return a path from the root to a given node
     *
     * @param node The given node
     * @return An array containing the path
     */
    private SentSHLDANode[] getPathFromNode(SentSHLDANode node) {
        SentSHLDANode[] path = new SentSHLDANode[node.getLevel() + 1];
        SentSHLDANode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    private boolean isLeafNode(SentSHLDANode node) {
        return node.getLevel() == L - 1;
    }

    private SentSHLDANode createNewPath(SentSHLDANode internalNode) {
        SentSHLDANode node = internalNode;
        for (int l = internalNode.getLevel(); l < L - 1; l++) {
            node = this.createNode(node);
        }
        return node;
    }

    /**
     * Get a node at a given level on a path on the tree. The path is determined
     * by its leaf node.
     *
     * @param level The level that the node is at
     * @param leafNode The leaf node of the path
     */
    private SentSHLDANode getNode(int level, SentSHLDANode leafNode) {
        if (!isLeafNode(leafNode)) {
            throw new RuntimeException("Exception while getting node. The given "
                    + "node is not a leaf node");
        }
        int curLevel = leafNode.getLevel();
        SentSHLDANode curNode = leafNode;
        while (curLevel != level) {
            curNode = curNode.getParent();
            curLevel--;
        }
        return curNode;
    }

    private SentSHLDANode getNode(int[] parsedPath) {
        SentSHLDANode node = word_hier_root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    /**
     * Get an array containing all the regression parameters along a path. The
     * path is specified by a leaf node.
     *
     * @param leafNode The leaf node
     */
    private double[] getRegressionPath(SentSHLDANode leafNode) {
        if (leafNode.getLevel() != L - 1) {
            throw new RuntimeException("Node " + leafNode.toString() + " is not a leaf node");
        }
        double[] regPath = new double[L];
        int level = leafNode.getLevel();
        SentSHLDANode curNode = leafNode;
        while (curNode != null) {
            regPath[level--] = curNode.getRegressionParameter();
            curNode = curNode.getParent();
        }
        return regPath;
    }

    private double[] getRegressionValues() {
        double[] regValues = new double[D];
        for (int d = 0; d < D; d++) {
            double sum = 0.0;
            for (int s = 0; s < words[d].length; s++) {
                double[] regParams = getRegressionPath(c[d][s]);
                for (int l = 0; l < L; l++) {
                    sum += regParams[l] * sent_level_count[d][s].getCount(l);
                }
            }
            regValues[d] = sum / docTokenCounts[d];
        }
        return regValues;
    }

    @Override
    public String getCurrentState() {
        double[] levelCount = new double[L];
        Queue<SentSHLDANode> queue = new LinkedList<SentSHLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            SentSHLDANode node = queue.poll();
            levelCount[node.getLevel()]++;
            // add children to the queue
            for (SentSHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(").append(levelCount[l]).append(") ");
        }
        return str.toString();
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
        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            for (SentSHLDANode child : node.getChildren()) {
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

        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            for (SentSHLDANode child : node.getChildren()) {
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
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            str.append(d).append(": ").append(MiscUtils.arrayToString(doc_level_distr[d].getCounts())).append("\n");
            for (int s = 0; s < words[d].length; s++) {
                str.append("--- ").append(s).append(": ")
                        .append(sent_level_count[d][s].toString())
                        .append(" -> ").append(c[d][s].getPathString())
                        .append("\n");
            }
            str.append("\n");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        writer.write(str.toString());
        writer.close();
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        double treeLogProb = 0.0;
        double regParamLgprob = 0.0;
        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood();

            if (supervised) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        mus[node.getLevel()], Math.sqrt(sigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(gammas[node.getLevel()]);
            }

            for (SentSHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += doc_level_distr[d].getLogLikelihood();
            if (supervised) {
                resLlh += StatUtils.logNormalProbability(responses[d],
                        regValues[d], Math.sqrt(hyperparams.get(RHO)));
            }
        }

        logln("^^^ word-llh = " + MiscUtils.formatDouble(wordLlh)
                + ". tree = " + MiscUtils.formatDouble(treeLogProb)
                + ". stick = " + MiscUtils.formatDouble(stickLgprob)
                + ". reg param = " + MiscUtils.formatDouble(regParamLgprob)
                + ". response = " + MiscUtils.formatDouble(resLlh));

        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        int count = SCALE + 1;
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
        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            wordLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);

            if (supervised) {
                regParamLgprob += StatUtils.logNormalProbability(node.getRegressionParameter(),
                        newMus[node.getLevel()], Math.sqrt(newSigmas[node.getLevel()]));
            }

            if (!isLeafNode(node)) {
                treeLogProb += node.getLogJointProbability(newGammas[node.getLevel()]);
            }

            for (SentSHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        double stickLgprob = 0.0;
        double resLlh = 0.0;
        double[] regValues = getRegressionValues();
        for (int d = 0; d < D; d++) {
            stickLgprob += doc_level_distr[d].getLogLikelihood(tParams.get(MEAN), tParams.get(SCALE));
            if (supervised) {
                resLlh += StatUtils.logNormalProbability(responses[d], regValues[d], Math.sqrt(tParams.get(RHO)));
            }
        }
        double llh = wordLlh + treeLogProb + stickLgprob + regParamLgprob + resLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
        this.hyperparams = new ArrayList<Double>();
        for (double param : tParams) {
            this.hyperparams.add(param);
        }

        int count = SCALE + 1;
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

        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();
            node.getContent().setConcentration(betas[node.getLevel()]);
            for (SentSHLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l].setConcentration(betas[l + 1]);
        }

        for (int d = 0; d < D; d++) {
            doc_level_distr[d].setMean(hyperparams.get(MEAN));
            doc_level_distr[d].setScale(hyperparams.get(SCALE));
        }
    }

    @Override
    public void validate(String msg) {
        validateModel(msg);

        validateAssignments(msg);
    }

    private void validateAssignments(String msg) {
        for (int d = 0; d < D; d++) {
            doc_level_distr[d].validate(msg);
        }

        HashMap<SentSHLDANode, Integer> leafCustCounts = new HashMap<SentSHLDANode, Integer>();
        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                Integer count = leafCustCounts.get(c[d][s]);
                if (count == null) {
                    leafCustCounts.put(c[d][s], 1);
                } else {
                    leafCustCounts.put(c[d][s], count + 1);
                }
            }
        }

        for (SentSHLDANode node : leafCustCounts.keySet()) {
            if (node.getNumCustomers() != leafCustCounts.get(node)) {
                throw new RuntimeException(msg + ". Numbers of customers mismach.");
            }
        }

        for (int d = 0; d < D; d++) {
            for (int l = 0; l < L; l++) {
                int levelCountStick = doc_level_distr[d].getCount(l);
                int levelCountSparse = 0;
                for (int s = 0; s < words[d].length; s++) {
                    levelCountSparse += sent_level_count[d][s].getCount(l);
                }

                if (levelCountStick != levelCountSparse) {
                    throw new RuntimeException(msg + ". Counts at level " + l
                            + " in document " + d + " mismatch");
                }
            }
        }
    }

    private void validateModel(String msg) {
        Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SentSHLDANode node = stack.pop();

            if (!isLeafNode(node)) {
                int numChildCusts = 0;
                for (SentSHLDANode child : node.getChildren()) {
                    stack.add(child);
                    numChildCusts += child.getNumCustomers();
                }

                if (numChildCusts != node.getNumCustomers()) {
                    throw new RuntimeException(msg + ". Numbers of customers mismatch."
                            + " " + numChildCusts + " vs. " + node.getNumCustomers());
                }
            }

            if (this.isLeafNode(node) && node.isEmpty()) {
                throw new RuntimeException(msg + ". Leaf node " + node.toString()
                        + " is empty");
            }
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model string
            StringBuilder modelStr = new StringBuilder();
            Stack<SentSHLDANode> stack = new Stack<SentSHLDANode>();
            stack.add(word_hier_root);
            while (!stack.isEmpty()) {
                SentSHLDANode node = stack.pop();

                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getIterationCreated()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\n");
                modelStr.append(node.getRegressionParameter()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");

                for (SentSHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignment string
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    for (int n = 0; n < words[d][s].length; n++) {
                        assignStr.append(d)
                                .append(":").append(s)
                                .append(":").append(n)
                                .append("\t").append(z[d][s][n])
                                .append("\n");
                    }
                }
            }

            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    assignStr.append(d)
                            .append(":").append(s)
                            .append("\t").append(c[d][s].getPathString())
                            .append("\n");
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
        HashMap<String, SentSHLDANode> nodeMap = new HashMap<String, SentSHLDANode>();
        String line;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int iterCreated = Integer.parseInt(reader.readLine());
            int numCustomers = Integer.parseInt(reader.readLine());
            double regParam = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SentSHLDANode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
            int nodeLevel = pathIndices.length - 1;
            SentSHLDANode node = new SentSHLDANode(iterCreated, nodeIndex,
                    nodeLevel, dmm, regParam, parent);

            node.changeNumCustomers(numCustomers);

            if (node.getLevel() == 0) {
                word_hier_root = node;
            }

            if (parent != null) {
                parent.addChild(node.getIndex(), node);
            }

            nodeMap.put(pathStr, node);
        }
        reader.close();

        validateModel("Loading model " + filename);
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
            for (int s = 0; s < words[d].length; s++) {
                for (int n = 0; n < words[d][s].length; n++) {
                    String[] sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + s + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][s][n] = Integer.parseInt(sline[1]);
                }
            }
        }

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != d) {
                    throw new RuntimeException("Mismatch");
                }
                String pathStr = sline[1];
                SentSHLDANode node = getNode(parseNodePath(pathStr));
                c[d][s] = node;
            }
        }
        reader.close();

        validateAssignments("Load assignments from " + filename);
    }

    public double[] regressNewDocuments(
            int[][][] newWords,
            double[] newResponses,
            String filepath) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolderpath);
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            if (!filename.contains("zip")) {
                continue;
            }

            double[] predResponses = regressNewDocuments(reportFolderpath
                    + filename, newWords, newResponses);
            predResponsesList.add(predResponses);

            RegressionEvaluation eval = new RegressionEvaluation(
                    responses, predResponses);
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

    private double[] regressNewDocuments(String stateFile, int[][][] newWords, double[] newResponses) {
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

        // initialize structure
        initializeDataStructure();

        for (int d = 0; d < D; d++) {
            for (int s = 0; s < words[d].length; s++) {
                // initialize levels
                for (int n = 0; n < words[d][s].length; n++) {
                    z[d][s][n] = rand.nextInt(L);
                    doc_level_distr[d].increment(z[d][s][n]);
                    sent_level_count[d][s].increment(z[d][s][n]);
                }

                // initialize paths
                samplePathAssignments(d, s, !REMOVE, ADD, !OBSERVED, !EXTEND);
            }
        }

        // iterate
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int s = 0; s < words[d].length; s++) {
                    samplePathAssignments(d, s, REMOVE, ADD, !OBSERVED, !EXTEND);

                    for (int n = 0; n < words[d][s].length; n++) {
                        sampleLevelAssignments(d, s, n, REMOVE, ADD, REMOVE, ADD, !OBSERVED);
                    }
                }

            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);

                if (verbose) {
                    logln("state file: " + stateFile
                            + ". iter = " + iter
                            + ". llh = " + getLogLikelihood());

                    RegressionEvaluation eval = new RegressionEvaluation(
                            responses, predResponses);
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
}

class SentSHLDANode extends TreeNode<SentSHLDANode, DirMult> {

    private final int born;
    private int numCustomers;
    private double regression;

    SentSHLDANode(int iter, int index, int level, DirMult content,
            double regParam, SentSHLDANode parent) {
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
        for (SentSHLDANode child : this.getChildren()) {
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
                .append(", ").append(born)
                .append(", #ch = ").append(getNumChildren())
                .append(", #c = ").append(getNumCustomers())
                .append(", #o = ").append(getContent().getCountSum())
                .append(", reg = ").append(MiscUtils.formatDouble(regression))
                .append("]");
        return str.toString();
    }
}