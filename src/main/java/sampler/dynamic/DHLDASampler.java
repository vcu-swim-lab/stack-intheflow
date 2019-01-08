package sampler.dynamic;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampling.likelihood.LogisticNormal;
import sampling.likelihood.TruncatedStickBreaking;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class DHLDASampler extends AbstractSampler {

    public static final boolean AHEAD = true;
    public static final String SEPARATOR = "#";
    public static final int NUM_TOPWORDS = 15;
    public static final int Q = 5; // number of samples from prior
    public static final int SIGMA = 0; // variance in the Gaussian transition 
    public static final int MEAN = 1; // mean of the GEM distribution
    public static final int SCALE = 2; // scale of the GEM distribution
    public static final int LAMBDA = 3; // decay factor
    protected double[] betas; // level-specific pseudo-counts for topics' word distribution
    protected double[] gammas; // level-specific parameters for the nCRP
    private double[] logGammas; // precomputed log gammas
    protected int T;
    protected int V;
    protected int L;
    protected int delta;
    protected int[][][] words; // [T x D_t x N_{td}]
    protected TruncatedStickBreaking[][] docLevelDists; // GEMs
    protected DNCRPNode[] dynamicRoots; // dnCRPs
    protected int[][][] z; // level assignments for each token
    protected DNCRPNode[][] c; // path assignments for each document
    protected LogisticNormal emptyModel;
//    protected HashMap<DNCRPNode, Double>[] jointPathLogProbs;
    private double[] zeros;
    private double[] sigmaSquares;
    private int numDocsChangePath;
    private int numTokensChangeLevel;

    public void configure(
            String folder,
            int V,
            int L,
            int delta,
            int[][][] words,
            double sigma,
            double mean,
            double scale,
            double lambda,
            double[] betas,
            double[] gammas,
            InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;

        this.V = V;
        this.L = L;
        this.T = this.words.length;
        this.delta = delta;
        if (this.delta > this.T) {
            logln("*** [WARNING]: The window size " + this.delta
                    + " is greater than the number of time epochs " + this.T);
            this.delta = this.T;
        }

        this.betas = betas;
        this.gammas = gammas;

        this.logGammas = new double[this.gammas.length];
        for (int i = 0; i < this.gammas.length; i++) {
            this.logGammas[i] = Math.log(this.gammas[i]);
        }

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(sigma);
        this.hyperparams.add(mean);
        this.hyperparams.add(scale);
        this.hyperparams.add(lambda);
        for (int l = 0; l < betas.length; l++) {
            this.hyperparams.add(betas[l]);
        }
        for (int i = 0; i < gammas.length; i++) {
            this.hyperparams.add(gammas[i]);
        }

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- sigma:\t" + hyperparams.get(SIGMA));
            logln("--- GEM mean:\t" + hyperparams.get(MEAN));
            logln("--- GEM scale:\t" + hyperparams.get(SCALE));
            logln("--- delta:\t" + delta);
            logln("--- lambda:\t" + hyperparams.get(LAMBDA));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }

        if (!debug) {
            System.err.close();
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_DHLDA-")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_S-").append(formatter.format(hyperparams.get(SIGMA)))
                .append("_GEM-M-").append(formatter.format(hyperparams.get(MEAN)))
                .append("_GEM-S-").append(formatter.format(hyperparams.get(SCALE)))
                .append("_D-").append(delta)
                .append("_LD-").append(formatter.format(hyperparams.get(LAMBDA)));

        int count = LAMBDA + 1;
        str.append("_b");
        for (int i = 0; i < betas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_g");
        for (int i = 0; i < gammas.length; i++) {
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

        initializeHierarchies();

        initializeAssignments();
    }

    protected void initializeHierarchies() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

        zeros = new double[V];
        Arrays.fill(zeros, 0.0);

        sigmaSquares = new double[V];
        Arrays.fill(sigmaSquares, hyperparams.get(SIGMA) * hyperparams.get(SIGMA));

        // stick breaking prior over levels
        docLevelDists = new TruncatedStickBreaking[T][];
        for (int t = 0; t < T; t++) {
            docLevelDists[t] = new TruncatedStickBreaking[words[t].length];
            for (int d = 0; d < words[t].length; d++) {
                docLevelDists[t][d] = new TruncatedStickBreaking(L, hyperparams.get(MEAN), hyperparams.get(SCALE));
            }
        }

        // initial all roots
        dynamicRoots = new DNCRPNode[T];
        // --- for the 1st time point
        LogisticNormal ln0 = new LogisticNormal(V, zeros, sigmaSquares);
        ln0.sampleFromPrior();
        dynamicRoots[0] = new DNCRPNode(0, 0, ln0, null, null, null);
        dynamicRoots[0].createPseudoChildNode();

        // empty model 
        this.emptyModel = new LogisticNormal(V, zeros, sigmaSquares);

        // allocate memory
        this.c = new DNCRPNode[T][];
        for (int t = 0; t < T; t++) {
            this.c[t] = new DNCRPNode[words[t].length];
        }

        this.z = new int[T][][];
        for (int t = 0; t < T; t++) {
            this.z[t] = new int[words[t].length][];
            for (int d = 0; d < words[t].length; d++) {
                this.z[t][d] = new int[words[t][d].length];
            }
        }
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        // initialize assignments for the 1st tree
        initializeFirstEpoch();

        // debug
        if (verbose) {
            logln("--- --- Initialized t = 0:\n" + printTree(0));
        }

        // from the 2nd tree onwards
        for (int t = 1; t < T; t++) {
            // steps:
            // 1. replicate the previous tree
            initializeFollowingEpoch(t);

            for (int d = 0; d < words[t].length; d++) {
                // 2. create a new path, sample level
                DNCRPNode node = this.dynamicRoots[t];
                for (int l = 1; l < L; l++) {
                    node.incrementNumCustomers();
                    node = createChild(node);
                }
                node.incrementNumCustomers();
                c[t][d] = node;
                for (int n = 0; n < words[t][d].length; n++) {
                    sampleLevelAssignment(t, d, n, !REMOVE);
                }

                // 3. sample path
                samplePathAssignment(t, d, OBSERVED, REMOVE, !AHEAD);

                // 4. resample the level
                for (int n = 0; n < words[t][d].length; n++) {
                    sampleLevelAssignment(t, d, n, REMOVE);
                }
            }
            // run forward filtering to update new chains
            this.initializeTopics(t);

            if (verbose) {
                logln("--- --- Initialized t = " + t + ":\n" + printTree(t));
            }
        }

        if (verbose) {
            logln("--- Done initialization");
            logln("--- Current state:\n" + getCurrentState());
        }

        if (debug) {
            this.validate("Done initializing assignment");
        }
    }

    // -------------------- Initialize first tree ------------------------------
    private void initializeFirstEpoch() {
        if (verbose) {
            logln("--- --- Initializing first epoch ...");
        }

        int t = 0;
        int D0 = words[t].length;
        for (int d = 0; d < D0; d++) {
            // create a brand-new path for each token
            DNCRPNode node = this.dynamicRoots[t];
            for (int l = 1; l < L; l++) {
                node.incrementNumCustomers();
                node = createChild(node);
            }
            node.incrementNumCustomers();
            c[t][d] = node;

            // randomly assign to levels
            for (int n = 0; n < words[t][d].length; n++) {
                sampleLevelAssignmentFirstEpoch(d, n, !REMOVE);
            }

            if (d > 0) {
                samplePathAssignmentFirstEpoch(d, REMOVE);
            }

            for (int n = 0; n < words[t][d].length; n++) {
                sampleLevelAssignmentFirstEpoch(d, n, REMOVE);
            }
        }

        int numSteps = 3;
        for (int i = 0; i < numSteps; i++) {
            logln("--- --- --- i = " + i);

            for (int d = 0; d < D0; d++) {
                samplePathAssignmentFirstEpoch(d, REMOVE);

                for (int n = 0; n < words[t][d].length; n++) {
                    sampleLevelAssignmentFirstEpoch(d, n, REMOVE);
                }
            }
        }

        // perform a forward filtering pass to update 
        this.initializeTopics(t);
    }

    /**
     * Sample level assignments to initialize the first tree. The likelihood
     * function used here is the Dirichlet Multinomial
     *
     * @param d The document index
     * @param n The token index
     * @param remove Whether this token should be removed from the structure
     */
    private void sampleLevelAssignmentFirstEpoch(int d, int n, boolean remove) {
        int t = 0;

        if (remove) {
            docLevelDists[t][d].decrement(z[t][d][n]);
            getNodeOnPath(c[t][d], z[t][d][n]).getContent().decrement(words[t][d][n]);
        }

        double[] logprobs = new double[L];
        DNCRPNode node = c[t][d];
        while (node != null) {
            int l = node.getLevel();
            logprobs[l] = docLevelDists[t][d].getLogProbability(l);
            logprobs[l] += Math.log(node.getContent().getCount(words[t][d][n]) + betas[node.getLevel()])
                    - Math.log(node.getContent().getCountSum() + betas[node.getLevel()] * V);
            node = node.getParent();
        }
        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);

        z[t][d][n] = sampledL;
        docLevelDists[t][d].increment(z[t][d][n]);
        getNodeOnPath(c[t][d], z[t][d][n]).getContent().increment(words[t][d][n]);
    }

    /**
     * Sample the path assignments to initialize the first tree. The likelihood
     * function used here is the Dirichlet Multinomial
     *
     * @param d The document index
     * @param remove Whether this document should be removed from the structure
     */
    private void samplePathAssignmentFirstEpoch(int d, boolean remove) {
        int t = 0;
        // per-level token counts
        HashMap<Integer, Integer>[] docTypeCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            docTypeCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[t][d].length; n++) {
            Integer count = docTypeCountPerLevel[z[t][d][n]].get(words[t][d][n]);
            if (count == null) {
                docTypeCountPerLevel[z[t][d][n]].put(words[t][d][n], 1);
            } else {
                docTypeCountPerLevel[z[t][d][n]].put(words[t][d][n], count + 1);
            }
        }

        if (remove) {
            this.removeCustomerFromPath(c[t][d]);
            this.removeObservationsFromPath(c[t][d], docTypeCountPerLevel);
        }

        // compute the log prior probabilities
        HashMap<DNCRPNode, Double> nodeLogPriors = new HashMap<DNCRPNode, Double>();
        double rootPseudoCount = 0.0;
        this.dynamicRoots[t].setNumPseudoCustomers(rootPseudoCount);
        this.computePathLogPrior(nodeLogPriors, this.dynamicRoots[t], 0.0);

        // compute log likelihood
        HashMap<DNCRPNode, Double> nodeLogLikelihoods = new HashMap<DNCRPNode, Double>();
        computePathLogLikelihoodFirstEpoch(nodeLogLikelihoods, this.dynamicRoots[t], docTypeCountPerLevel, 0.0);

        // debug
        if (nodeLogPriors.size() != nodeLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of nodes are different");
        }

        // sample
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<DNCRPNode> nodelist = new ArrayList<DNCRPNode>();
        for (DNCRPNode node : nodeLogPriors.keySet()) {
            nodelist.add(node);

            double logprob = nodeLogPriors.get(node) + nodeLogLikelihoods.get(node);
            logprobs.add(logprob);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        DNCRPNode sampledNode = nodelist.get(sampledIndex);

        if (sampledNode.getLevel() < L - 1) // sample an internal node
        {
            sampledNode = createPath(sampledNode);
        }

        // update
        c[t][d] = sampledNode;
        this.addCustomerToPath(t, c[t][d]);
        this.addObservationsToPath(c[t][d], docTypeCountPerLevel);
    }

    private void computePathLogLikelihoodFirstEpoch(
            HashMap<DNCRPNode, Double> nodeLogLikelihoods,
            DNCRPNode curNode,
            HashMap<Integer, Integer>[] docLevelTokenCounts,
            double parentLlh) {
        int level = curNode.getLevel();
        double beta = betas[level];

        double nodeLlh = getDirMultLogLikelihood(curNode.getContent().getCounts(),
                curNode.getContent().getCountSum(),
                beta, beta * V, docLevelTokenCounts[level]);

        for (DNCRPNode child : curNode.getChildren()) {
            computePathLogLikelihoodFirstEpoch(nodeLogLikelihoods, child, docLevelTokenCounts, parentLlh + nodeLlh);
        }

        double storeLlh = parentLlh + nodeLlh;
        level++;
        while (level < L) {
            storeLlh += getDirMultLogLikelihood(new int[V], 0, beta, beta * V, docLevelTokenCounts[level]);
            level++;
        }
        nodeLogLikelihoods.put(curNode, storeLlh);
    }

    private double getDirMultLogLikelihood(int[] counts, int countSum,
            double prior, double sumPrior,
            HashMap<Integer, Integer> obsCounts) {
        double llh = 0.0;
        int j = 0;
        for (int obs : obsCounts.keySet()) {
            for (int i = 0; i < obsCounts.get(obs); i++) {
                llh += Math.log(counts[obs] + prior + i)
                        - Math.log(sumPrior + countSum + j);
                j++;
            }
        }
        return llh;
    }
    // -------------------- Done initializing first tree -----------------------

    /**
     * Initialize a tree at 2nd time epoch onwards. This is done by copying all
     * nodes from the previous tree which have positive number of pseudo
     * documents
     *
     * @param t The time epoch
     */
    private void initializeFollowingEpoch(int t) {
        if (verbose) {
            logln("--- Initializing epoch " + t);
        }

        LogisticNormal ln0 = new LogisticNormal(V,
                dynamicRoots[t - 1].getContent().getMean(),
                dynamicRoots[t - 1].getContent().getVariance());
        ln0.sampleFromPrior();
        dynamicRoots[t] = new DNCRPNode(dynamicRoots[t - 1].getIndex(),
                dynamicRoots[t - 1].getLevel(), ln0, null, dynamicRoots[t - 1], null);
        dynamicRoots[t - 1].setPosNode(dynamicRoots[t]);
        dynamicRoots[t].setNumPseudoCustomers(computeNumPseudoCustomers(dynamicRoots[t].getPreNode()));

        Stack<DNCRPNode> preNodesStack = new Stack<DNCRPNode>();
        preNodesStack.add(this.dynamicRoots[t - 1]);

        while (!preNodesStack.isEmpty()) {
            DNCRPNode preNode = preNodesStack.pop();
            DNCRPNode curNode = preNode.getPosNode();

            for (DNCRPNode preChild : preNode.getChildren()) {
                // ignore child node with 0 pseudo customer
                double numPseudoCusts = computeNumPseudoCustomers(preChild);
                if (numPseudoCusts == 0) {
                    continue;
                }

                // add this child node to the stack
                preNodesStack.add(preChild);

                // create a corresponding node if number of pseudo documents > 0
                LogisticNormal ln = new LogisticNormal(V,
                        preChild.getContent().getMean(),
                        preChild.getContent().getVariance());
                ln.sampleFromPrior();
                DNCRPNode curChild = new DNCRPNode(preChild.getIndex(), preChild.getLevel(), ln, curNode, preChild, null);
                preChild.setPosNode(curChild);
                curChild.setNumPseudoCustomers(numPseudoCusts);
                curNode.addChild(curChild.getIndex(), curChild);
            }

            // update the list of inactive child indices
            curNode.fillInactiveChildIndices();
        }
    }

    /**
     * Create a child node of a parent node
     *
     * @param parentNode The given parent node
     */
    private DNCRPNode createChild(DNCRPNode parentNode) {
        int childIndex = parentNode.getNextChildIndex();
        LogisticNormal lnModel = new LogisticNormal(V, zeros, sigmaSquares);
        lnModel.sampleFromPrior();
        DNCRPNode childNode = new DNCRPNode(childIndex, parentNode.getLevel() + 1, lnModel, parentNode, null, null);
        childNode.createPseudoChildNode();
        parentNode.addChild(childIndex, childNode);

        // create child node in future trees
        DNCRPNode parentPosNode = parentNode.getPosNode();
        LogisticNormal curModel = lnModel;
        DNCRPNode curChildNode = childNode;
        int t = 1;
        while (parentPosNode != null && t <= delta) {
            int posChildIndex = parentPosNode.getNextChildIndex();
            LogisticNormal posModel = new LogisticNormal(V, curModel.getMean(), curModel.getVariance());

            DNCRPNode childPosNode = new DNCRPNode(posChildIndex, childNode.getLevel(), posModel, parentPosNode, curChildNode, null);
            curChildNode.setPosNode(childPosNode);
            curChildNode.createPseudoChildNode();
            parentPosNode.addChild(posChildIndex, childPosNode);

            curChildNode = curChildNode.getPosNode();
            curModel = posModel;
            parentPosNode = parentPosNode.getPosNode();

            t++;
        }

        return childNode;
    }

    /**
     * Create a new path from a given node to the leaf level
     *
     * @param internalNode The given internal node
     */
    private DNCRPNode createPath(DNCRPNode internalNode) {
        DNCRPNode node = internalNode;
        for (int l = internalNode.getLevel(); l < L - 1; l++) {
            node = createChild(node);
        }
        return node;
    }

    /**
     * Get a node at a given level on a path represented by the leaf node
     *
     * @param leafNode The leaf node
     * @param targetLevel The level
     */
    private DNCRPNode getNodeOnPath(DNCRPNode leafNode, int targetLevel) {
        DNCRPNode node = leafNode;
        while (node.getLevel() != targetLevel) {
            node = node.getParent();
        }
        return node;
    }

    /**
     * Remove a customer from a path represented by a leaf node. IF the node
     * become empty after the removal, remove it from the tree
     *
     * @param leafNode The leaf node
     */
    private DNCRPNode removeCustomerFromPath(DNCRPNode leafNode) {
        DNCRPNode retNode = leafNode;
        DNCRPNode node = leafNode;
        while (node != null) {
            // decrement number of actual customers at this node
            node.decrementNumCustomers();

            // decrement the number of pseudo customers ahead
            this.updateFutureNodeNumPseudoCustomers(node);

            // remove the node if there is no customers (both actual and pseudo)
            if (node.getNumActualCustomers() + node.getNumPseudoCustomers() == 0) {
                retNode = node.getParent();
//                node.getParent().removeChild(node.getIndex());
                removeNode(node);
            }

            node = node.getParent();
        }
        return retNode;
    }

    /**
     * Increment the number of customers of a path
     *
     * @param leafNode The leaf node representing the path
     */
    private void addCustomerToPath(int t, DNCRPNode leafNode) {
        DNCRPNode node = leafNode;
        while (node != null) {
            // increment the number of
            node.incrementNumCustomers();

            // update the number of pseudo customer ahead
            this.updateFutureNodeNumPseudoCustomers(node);

            node = node.getParent();
        }
//        this.updateJointPathLogProbability(jointPathLogProbs[t], leafNode);
    }

    /**
     * Update the number of pseudo customers of future nodes of a given node
     *
     * @param A node, whose future nodes need to be updated
     */
    private void updateFutureNodeNumPseudoCustomers(DNCRPNode node) {
        DNCRPNode posNode = node.getPosNode();
        int t = 1;
        while (posNode != null && t <= delta) {
            double numPseudoCusts = computeNumPseudoCustomers(posNode.getPreNode());
            posNode.setNumPseudoCustomers(numPseudoCusts);

            // remove if this is empty
            if (posNode.getNumPseudoCustomers() + posNode.getNumActualCustomers() == 0) {
                removeNode(posNode);
            }
//                posNode.getParent().removeChild(posNode.getIndex());

            posNode = posNode.getPosNode();
            t++;
        }
    }

    private void removeNode(DNCRPNode node) {
        node.getParent().removeChild(node.getIndex());
        if (node.getPreNode() != null) {
            node.getPreNode().setPosNode(null);
        }
        if (node.getPosNode() != null) {
            node.getPosNode().setPreNode(null);
        }
    }

    /**
     * Removing a set of observations from a path
     *
     * @param leafNode The leaf node representing the path
     * @param tokenCountsPerLevel The set of observations
     */
    private void removeObservationsFromPath(DNCRPNode leafNode, HashMap<Integer, Integer>[] tokenCountsPerLevel) {
        DNCRPNode node = leafNode;
        while (node != null) {
            node.removeObservations(tokenCountsPerLevel[node.getLevel()]);
            node = node.getParent();
        }
    }

    /**
     * Adding a set of observations to a path
     *
     * @param leafNode The leaf node representing the path
     * @param tokenCountsPerLevel The set of observations
     */
    private void addObservationsToPath(DNCRPNode leafNode, HashMap<Integer, Integer>[] tokenCountsPerLevel) {
        DNCRPNode node = leafNode;
        while (node != null) {
            node.addObservations(tokenCountsPerLevel[node.getLevel()]);
            node = node.getParent();
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }

        logLikelihoods = new ArrayList<Double>();

        int numDocs = 0;
        int numTokens = 0;
        for (int t = 0; t < T; t++) {
            numDocs += words[t].length;
            for (int d = 0; d < words[t].length; d++) {
                numTokens += words[t][d].length;
            }
        }

        for (iter = 0; iter < MAX_ITER; iter++) {

            System.out.println();
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t num docs change = " + numDocsChangePath + " / " + numDocs
                            + "\t num tokens change = " + numTokensChangeLevel + " / " + numTokens
                            + "\nCurrent state:\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t num docs change = " + numDocsChangePath + " / " + numDocs
                            + "\t num tokens change = " + numTokensChangeLevel + " / " + numTokens
                            + "\nCurrent state:\n" + getCurrentState());
                }
            }

            // reset
            numDocsChangePath = 0;
            numTokensChangeLevel = 0;

            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    samplePathAssignment(t, d, OBSERVED, REMOVE, AHEAD);

                    for (int n = 0; n < words[t][d].length; n++) {
                        sampleLevelAssignment(t, d, n, REMOVE);
                    }
                }
            }

            // forward-filtering and backward-smoothing to update topics
            updateTopics();

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
        }
    }

    /**
     * Sample the level assignment for a token
     *
     * @param t The time epoch
     * @param d The document index
     * @param n The token index
     * @param remove Whether the token should be removed from the structure
     */
    private void sampleLevelAssignment(int t, int d, int n, boolean remove) {
        if (remove) {
            docLevelDists[t][d].decrement(z[t][d][n]);
            getNodeOnPath(c[t][d], z[t][d][n]).getContent().decrement(words[t][d][n]);
        }

        double[] logprobs = new double[L];
        DNCRPNode node = c[t][d];
        while (node != null) {
            int l = node.getLevel();
            logprobs[l] = docLevelDists[t][d].getLogProbability(l)
                    + node.getContent().getLogLikelihood(words[t][d][n]);

            // debug
//            logln("l = " + l 
//                    + ". lp = " + MiscUtils.formatDouble(docLevelDists[t][d].getLogProbability(l))
//                    + ". llh = " + MiscUtils.formatDouble(node.getContent().getLogLikelihood(words[t][d][n]))
//                    );

            node = node.getParent();
        }
        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledL == logprobs.length) {
            System.out.println("t = " + t + ". d = " + d + ". n = " + n + ". " + c[t][d].toString());
            System.out.println("logprobs: " + MiscUtils.arrayToString(logprobs));
            throw new RuntimeException("Sampling out-of-bound");
        }

        if (z[t][d][n] != sampledL) {
            numTokensChangeLevel++;
        }

        z[t][d][n] = sampledL;
        docLevelDists[t][d].increment(z[t][d][n]);
        getNodeOnPath(c[t][d], z[t][d][n]).getContent().increment(words[t][d][n]);
    }

    /**
     * Sample the path assignment for a document
     *
     * @param t The time epoch
     * @param d The document index
     * @param remove Whether the current assignment should be removed
     * @param lookAhead Whether the assignments in the next few epochs should be
     * considered
     */
    private void samplePathAssignment(int t, int d,
            boolean observed,
            boolean remove,
            boolean lookAhead) {
        // per-level token counts
        HashMap<Integer, Integer>[] docTypeCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            docTypeCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[t][d].length; n++) {
            Integer count = docTypeCountPerLevel[z[t][d][n]].get(words[t][d][n]);
            if (count == null) {
                docTypeCountPerLevel[z[t][d][n]].put(words[t][d][n], 1);
            } else {
                docTypeCountPerLevel[z[t][d][n]].put(words[t][d][n], count + 1);
            }
        }

        // compute the current joint probabilities of all nodes in the future windows
        HashMap<DNCRPNode, Double>[] curNodeJointLogProbs = new HashMap[delta];
        for (int i = 0; i < delta; i++) {
            curNodeJointLogProbs[i] = new HashMap<DNCRPNode, Double>();
            this.computeJointPathLogProbability(curNodeJointLogProbs[i], dynamicRoots[t], 0.0);
        }

        // remove the current document assignment
        DNCRPNode curNode = null;
        if (remove) {
            curNode = this.removeCustomerFromPath(c[t][d]);
            this.removeObservationsFromPath(c[t][d], docTypeCountPerLevel);
        }

        // compute the log prior probabilities
        HashMap<DNCRPNode, Double> nodeLogPriors = new HashMap<DNCRPNode, Double>();
        double rootPseudoCount = computeNumPseudoCustomers(this.dynamicRoots[t].getPreNode());
        this.dynamicRoots[t].setNumPseudoCustomers(rootPseudoCount);
        this.computePathLogPrior(nodeLogPriors, this.dynamicRoots[t], 0.0);

        // compute log likelihood
        HashMap<DNCRPNode, Double> nodeLogLikelihoods = new HashMap<DNCRPNode, Double>();
        if (observed) {
            computePathLogLikelihood(nodeLogLikelihoods, this.dynamicRoots[t], docTypeCountPerLevel, 0.0);

            // debug
            if (nodeLogPriors.size() != nodeLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of nodes are different");
            }
        }

        // compute the "look-\Delta-step-ahead" probability
        HashMap<DNCRPNode, Double> nodeLogTransProbs = new HashMap<DNCRPNode, Double>();
        if (lookAhead) {
            // compute 
            nodeLogTransProbs = this.computeNodeLogTransitionProbabilities(t);

            if (nodeLogPriors.size() != nodeLogTransProbs.size()) {
                throw new RuntimeException("Numbers of nodes are different");
            }
        }

        // sample
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<DNCRPNode> nodelist = new ArrayList<DNCRPNode>();
        for (DNCRPNode node : nodeLogPriors.keySet()) {
            nodelist.add(node);

            double logprob = nodeLogPriors.get(node);

            if (observed) {
                logprob += nodeLogLikelihoods.get(node);
            }

            if (lookAhead) {
                logprob += nodeLogTransProbs.get(node);
            }

            if (observed && lookAhead && debug) {
                logln("t = " + t + ". node " + node.toString()
                        + ". cur node " + c[t][d].getPathString() + " ---> " + logprob
                        + "\n\tlog prior = " + MiscUtils.formatDouble(nodeLogPriors.get(node))
                        + "\n\tlog lh    = " + MiscUtils.formatDouble(nodeLogLikelihoods.get(node))
                        + "\n\tlog trans = " + MiscUtils.formatDouble(nodeLogTransProbs.get(node)));
            }

            logprobs.add(logprob);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledIndex == logprobs.size()) {
            logln("Logprobs: " + MiscUtils.listToString(logprobs));
            logln("Nodes: " + nodelist.toString());
            throw new RuntimeException("Sampling out-of-bound");
        }

        DNCRPNode sampledNode = nodelist.get(sampledIndex);

        if (!sampledNode.equals(curNode)) {
            numDocsChangePath++;
        }

        if (sampledNode.getLevel() < L - 1) // sample an internal node
        {
            sampledNode = createPath(sampledNode);
        }

        // update
        c[t][d] = sampledNode;
        this.addCustomerToPath(t, c[t][d]);
        this.addObservationsToPath(c[t][d], docTypeCountPerLevel);
    }

    public void testComputeLogTransProbs() throws Exception {
        for (int t = 0; t < T; t++) {
            logln("t = " + t + ". " + this.printTree(t));

            HashMap<DNCRPNode, Double> nodeLogTransProbs = new HashMap<DNCRPNode, Double>();
            computeJointPathLogProbability(nodeLogTransProbs, dynamicRoots[t], 0.0);
            for (DNCRPNode node : nodeLogTransProbs.keySet()) {
                System.out.println(node.toString() + "\t"
                        + nodeLogTransProbs.get(node));
                System.out.println();
            }
        }
    }

    /**
     * Compute the cumulative joint probability along a path
     *
     * @param logprobs The joint assignment log probability of each node
     * @param node The lowest node to represent the path
     */
    private double getCumulativePossibleJointPathProbability(
            HashMap<DNCRPNode, HashMap<Integer, Double>> logprobs, DNCRPNode node) {
        int childIndex = node.getIndex();
        DNCRPNode parent = node.getParent();
        double cummLogProb = 0.0;
        while (parent != null) {
            // debug
            if (logprobs == null) {
                System.out.println("logprobs null");
            } else if (logprobs.get(parent) == null) {
                System.out.println("logprobs get parent");
            } else if (logprobs.get(parent).get(childIndex) == null) {
                System.out.println("logprobs childIndex null");
                System.out.println(parent.toString() + ". childIndex = " + childIndex + ". node = " + node);
                System.out.println("backward: " + getBackwardChain(node));
                System.out.println("forward: " + getForwardChain(node));

                for (DNCRPNode cNode : logprobs.keySet()) {
                    System.out.println(cNode.toString());
                    HashMap<Integer, Double> map = logprobs.get(cNode);
                    for (int index : map.keySet()) {
                        System.out.println("--- " + index + ". " + map.get(index));
                    }
                }
            }


            cummLogProb += logprobs.get(parent).get(childIndex);

            childIndex = parent.getIndex();
            parent = parent.getParent();
        }
        return cummLogProb;
    }

    /**
     * Compute the log transition probabilities for a given time epoch. This
     * will return a hash table containing - Key: The set of nodes in the tree
     * at the given time epoch. Each node represents a possible path to assign a
     * document. - Value: The log transition probability of assigning a document
     * to each possible path
     *
     * @param Time epoch
     */
    private HashMap<DNCRPNode, Double> computeNodeLogTransitionProbabilities(int t) {
        int maxDelta = Math.min(delta, T - 1 - t);

        // compute the current joint log probabilities 
        HashMap<DNCRPNode, Double>[] curJointLogProbs = new HashMap[maxDelta];
        for (int i = 0; i < maxDelta; i++) {
            curJointLogProbs[i] = new HashMap<DNCRPNode, Double>();
            this.computeJointPathLogProbability(curJointLogProbs[i], dynamicRoots[t + i + 1], 0.0);
        }

        // compute the possible joint path log probabilities
        HashMap<DNCRPNode, HashMap<Integer, Double>>[] newJointLogProbs = new HashMap[maxDelta];
        for (int i = 0; i < maxDelta; i++) {
            newJointLogProbs[i] = this.computePossibleJointPathLogProbability(dynamicRoots[t + i + 1], i + 1);
        }

        // debug
        for (int i = 0; i < maxDelta; i++) {
            if (curJointLogProbs[i].size() != newJointLogProbs[i].size()) {
                throw new RuntimeException("Sizes mismatch");
            }
        }

        // compute the transition probability for this path
        HashMap<DNCRPNode, Double> nodeLogTransProbs = new HashMap<DNCRPNode, Double>();
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();

            double nodeLogTranProb = 0.0;
            DNCRPNode curNode = node.getPosNode();
            int i = 1;
            while (curNode != null && i <= maxDelta) {
                DNCRPNode parentNode = curNode.getParent();
                if (isLeafNode(curNode)) {
                    double pathOldLogProb = curJointLogProbs[i - 1].get(parentNode);
                    double pathNewLogProb = getCumulativePossibleJointPathProbability(newJointLogProbs[i - 1], curNode);
                    nodeLogTranProb += pathNewLogProb - pathOldLogProb;
//                    logln("leaf: " + curNode + "\t" + pathOldLogProb + ". " + pathNewLogProb + ". " + (pathNewLogProb - pathOldLogProb));
                } else {
                    double pathOldLogProb = curJointLogProbs[i - 1].get(curNode);
                    double pseudoLogProb = newJointLogProbs[i - 1].get(curNode).get(DNCRPNode.PSEUDO_CHILD_INDEX);
                    double pathNewLogProb = pseudoLogProb + getCumulativePossibleJointPathProbability(newJointLogProbs[i - 1], curNode);
                    nodeLogTranProb += pathNewLogProb - pathOldLogProb;
//                    logln("non-leaf: " + curNode + "\t" + pathOldLogProb + ". " + pathNewLogProb + ". " + (pathNewLogProb - pathOldLogProb));
                }

                curNode = curNode.getPosNode();
                i++;
            }
            nodeLogTransProbs.put(node, nodeLogTranProb);

            // add to passingLogProb
            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return nodeLogTransProbs;
    }

    /**
     * Compute the possible joint log probability of all nodes in a tree if a
     * customer is added to any node in a tree delta epochs before
     *
     * @param root The root of the tree
     * @param delta The time difference
     */
    private HashMap<DNCRPNode, HashMap<Integer, Double>> computePossibleJointPathLogProbability(DNCRPNode root, int delta) {
        HashMap<DNCRPNode, HashMap<Integer, Double>> nodeLogProbs = new HashMap<DNCRPNode, HashMap<Integer, Double>>();
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();

            if (!isLeafNode(node)) {
                HashMap<Integer, Double> nodeProbMap = computePossibleJointAssignmentLogProbability(node, delta);
                nodeLogProbs.put(node, nodeProbMap);

                for (DNCRPNode child : node.getChildren()) {
                    stack.add(child);
                    computeJointAssignmentLogProbability(node);
                }
            }
        }
        return nodeLogProbs;
    }

    /**
     * Compute the joint assignment log probability of an internal node if a
     * customer is added to a child of the delta-previous node of this node.
     *
     * @param node The current node
     * @param delta The time difference
     */
    private HashMap<Integer, Double> computePossibleJointAssignmentLogProbability(DNCRPNode node, int delta) {
        HashMap<Integer, Double> possJointAssgnLogProbs = new HashMap<Integer, Double>();

        double addPseudoCount = Math.exp(-(double) delta / hyperparams.get(LAMBDA));

        HashMap<Integer, Double> curVals = new HashMap<Integer, Double>();
        HashMap<Integer, Double> newVals = new HashMap<Integer, Double>();
        double curSum = 0.0;
        for (DNCRPNode child : node.getChildren()) {
            double childPseudoCount = child.getNumPseudoCustomers();
            int childActualCount = child.getNumActualCustomers();
            double curVal = computeLogPseudoFactorial(childPseudoCount, childActualCount, logGammas[node.getLevel()]);
            curVals.put(child.getIndex(), curVal);

            double newVal = computeLogPseudoFactorial(childPseudoCount + addPseudoCount, childActualCount, logGammas[node.getLevel()]);
            newVals.put(child.getIndex(), newVal);

            curSum += curVal;
        }
        newVals.put(DNCRPNode.PSEUDO_CHILD_INDEX, logGammas[node.getLevel()]);

        double nodePseudoCount = node.getNumPseudoCustomers();
        int nodeActualCount = node.getNumActualCustomers();
        double logDen = computeLogPseudoFactorial(nodePseudoCount + addPseudoCount + gammas[node.getLevel()],
                nodeActualCount, logGammas[node.getLevel()]);

        for (DNCRPNode child : node.getChildren()) {
            int childIndex = child.getIndex();
            possJointAssgnLogProbs.put(childIndex, curSum
                    - curVals.get(childIndex) + newVals.get(childIndex) - logDen);
        }
        possJointAssgnLogProbs.put(DNCRPNode.PSEUDO_CHILD_INDEX, curSum
                + newVals.get(DNCRPNode.PSEUDO_CHILD_INDEX) - logDen);

        return possJointAssgnLogProbs;
    }

    /**
     * Recursively compute the joint path log probabilities
     *
     * // TODO: this should be precomputed and updated as necessary
     *
     * @param nodeLogProbs Hash table to store the result. Set of keys are all
     * internal nodes
     * @param curNode The current node
     * @param parentLogProb The log probability from the parent
     */
    private void computeJointPathLogProbability(
            HashMap<DNCRPNode, Double> nodeLogProbs,
            DNCRPNode curNode,
            double parentLogProb) {
        if (!isLeafNode(curNode)) {
            double logprob = computeJointAssignmentLogProbability(curNode);

            nodeLogProbs.put(curNode, parentLogProb + logprob);

            for (DNCRPNode child : curNode.getChildren()) {
                computeJointPathLogProbability(nodeLogProbs, child, parentLogProb + logprob);
            }
        }
    }

    private void updateJointPathLogProbability(
            HashMap<DNCRPNode, Double> jointPathLogProb,
            DNCRPNode leafNode) {
        ArrayList<DNCRPNode> path = new ArrayList<DNCRPNode>();
        DNCRPNode node = leafNode;
        while (node != null) {
            path.add(node);
            node = node.getParent();
        }
        double passingLogProb = 0.0;
        for (int i = path.size() - 1; i > 0; i--) { // exclude leaf
            double logprob = computeJointAssignmentLogProbability(path.get(i));
            passingLogProb += logprob;
            jointPathLogProb.put(path.get(i), passingLogProb);
        }
    }

    /**
     * Compute the joint assignment log probability of a given node. This can be
     * viewed as a Chinese restaurant, but at the beginning there are non-empty
     * tables, occupied by pseudo counts.
     *
     * @param node The given node
     */
    private double computeJointAssignmentLogProbability(DNCRPNode node) {
        double logNum = 0.0;
        for (DNCRPNode child : node.getChildren()) {
            double childPseudoCount = child.getNumPseudoCustomers();
            int childActualCount = child.getNumActualCustomers();
            logNum += computeLogPseudoFactorial(childPseudoCount, childActualCount, logGammas[node.getLevel()]);
        }

        double nodePseudoCount = node.getNumPseudoCustomers();
        int nodeActualCount = node.getNumActualCustomers();
        double logDen = computeLogPseudoFactorial(nodePseudoCount + gammas[node.getLevel()],
                nodeActualCount, logGammas[node.getLevel()]);

        double logprob = logNum - logDen;
        return logprob;
    }

    /**
     * Compute the log of: a * (a+1) * ... * (a+b-1).
     *
     * If a = 0, this will return the log of: \gamma * 1 * 2 * ... * (b-1)
     *
     * TODO: cached this value to speed up. a is a function of \Delta integers.
     *
     * @param a Existing number
     * @param b Additional number
     */
    private double computeLogPseudoFactorial(double a, int b, double loggamma) {
        double logvalue = 0.0;
        for (int i = 0; i < b; i++) {
            if (a + i == 0) {
                logvalue += loggamma;
            } else {
                logvalue += Math.log(a + i);
            }
        }
        return logvalue;
    }

    /**
     * Compute the log prior probabilities for all path in a tree
     *
     * @param nodeLogPriors The hash table to store the results
     * @param curNode The current node
     * @param curLogPrior The log prior of the current node
     */
    private void computePathLogPrior(HashMap<DNCRPNode, Double> nodeLogPriors,
            DNCRPNode curNode, double parentLogPrior) {
        double curLogPrior = parentLogPrior;

        if (!isLeafNode(curNode)) {
            double lognorm = Math.log(curNode.getNumActualCustomers()
                    + curNode.getNumPseudoCustomers() + gammas[curNode.getLevel()]);
            // for each child
            for (DNCRPNode child : curNode.getChildren()) {
                double childLogPrior = Math.log(child.getNumActualCustomers() + child.getNumPseudoCustomers()) - lognorm;
                computePathLogPrior(nodeLogPriors, child, parentLogPrior + childLogPrior);
            }

            // new path
            curLogPrior += this.logGammas[curNode.getLevel()] - lognorm;
        }
        nodeLogPriors.put(curNode, curLogPrior);
    }

    /**
     * Recursively compute the log likelihood for all path in a tree
     *
     * @param nodeLogLikelihoods Hash table to store result
     * @param curNode The current node
     * @param docLevelTokenCounts The per-level observations of this documents.
     * This implicitly contains the information about the level assignments.
     *
     * @param parentLlh The log likelihood passed on from the parent node
     */
    private void computePathLogLikelihood(
            HashMap<DNCRPNode, Double> nodeLogLikelihoods,
            DNCRPNode curNode,
            HashMap<Integer, Integer>[] docLevelTokenCounts,
            double parentLlh) {

        int level = curNode.getLevel();
        double nodeLlh = computeLogLikelihood(curNode, docLevelTokenCounts[level]);
        for (DNCRPNode child : curNode.getChildren()) {
            computePathLogLikelihood(nodeLogLikelihoods, child, docLevelTokenCounts, parentLlh + nodeLlh);
        }

        double storeLlh = parentLlh + nodeLlh;
        level++;
        while (level < L) {
            storeLlh += computeLogLikelihoodForNewNode(docLevelTokenCounts[level++]);
        }
        nodeLogLikelihoods.put(curNode, storeLlh);
    }

    /**
     * Compute the log likelihood for a node node.
     *
     * If there is no customers currently being assigned to this node, sample a
     * new topic from Gaussian(pre-node-mean, sigma^2 * I)
     *
     * @param node The current existing empty node
     * @param obsCounts The observation counts
     */
    private double computeLogLikelihood(DNCRPNode node, HashMap<Integer, Integer> obsCounts) {
        double llh;
        if (node.getIndex() == DNCRPNode.PSEUDO_CHILD_INDEX) // draw Q samples and average
        {
            llh = computeLogLikelihoodForNewNode(obsCounts);
        } else {
            // we don't need to draw 1 sample since when a new node is created 
            // from a previous node, we already sample from prior
            llh = node.getContent().getLogLikelihood(obsCounts);
        }
        return llh;
    }

    /**
     * Compute the log likelihood of a new node given the observations. Due to
     * the non-conjugacy of the Logistic-Normal prior, there is no closed form
     * for computing this.
     *
     * This is done by sampling Q samples from Normal(0, sigma^2 * I), computing
     * the log likelihood of each sample and averaging over the Q log
     * likelihoods
     *
     * Note that if this step slows down the sample significantly, the samples
     * should be pre-sampled and fixed for all trees.
     *
     * @param obsCounts The observation counts
     *
     */
    private double computeLogLikelihoodForNewNode(HashMap<Integer, Integer> obsCounts) {
        double llh = 0.0;
        for (int q = 0; q < Q; q++) {
            this.emptyModel.sampleFromPrior();
            llh += this.emptyModel.getLogLikelihood(obsCounts);
        }
        return llh / Q;
    }

    /**
     * Compute the number of pseudo customers starting from a certain
     *
     * @param node The first node in the past
     */
    private double computeNumPseudoCustomers(DNCRPNode node) {
        DNCRPNode preNode = node;
        int t = 1;
        double pseudo = 0.0;
        while (preNode != null && t <= delta) {
            pseudo += preNode.getNumActualCustomers() * Math.exp(-(double) t / hyperparams.get(LAMBDA));
            preNode = preNode.getPreNode();
            t++;
        }
        return pseudo;
    }

    /**
     * Update the topics of all nodes. This is done by performing for all chains
     * a forward filtering pass and followed by a backward smoothing pass.
     */
    private void updateTopics() {
        for (int t = 0; t < T; t++) {
            ArrayList<ArrayList<DNCRPNode>> chains = getForwardChains(t);
            for (ArrayList<DNCRPNode> chain : chains) {
                forwardSingleChain(chain, zeros, sigmaSquares);

                backwardSingleChain(chain);
            }
        }
    }

    /**
     * Initialize the topics of a newly created tree
     *
     * @param t The time epoch
     */
    private void initializeTopics(int t) {
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();
            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] priorMean = zeros;
            double[] priorVar = sigmaSquares;
            if (node.getPreNode() != null) {
                priorMean = node.getPreNode().getContent().getMean();
                priorVar = node.getPreNode().getContent().getVariance();
            }

            ArrayList<DNCRPNode> chain = new ArrayList<DNCRPNode>();
            chain.add(node);

//            while(node != null){
//                chain.add(node);
//                node = node.getPosNode();
//            }
//            System.out.println("chain size = " + chain.size());

            // update the topic of this node using previous node's mean and variance
            // if this node is just born, use mean and variance from the prior
            forwardSingleChain(chain, priorMean, priorVar);
        }
    }

    /**
     * Perform backward smoothing for a single chain
     *
     * @param node The chain
     */
    private void backwardSingleChain(ArrayList<DNCRPNode> nodes) {
        if (debug) {
            logln("--- backward smoothing chain: " + nodes.toString());
        }
        int tau = nodes.size() - 1;
        double[] posMean = nodes.get(tau).getContent().getMean();
        double[] posVar = nodes.get(tau).getContent().getVariance();

        for (int t = tau - 1; t >= 0; t--) {
            double[] curMean = nodes.get(t).getContent().getMean();
            double[] curVar = nodes.get(t).getContent().getVariance();
            double[] tempK = new double[V];
            for (int v = 0; v < V; v++) {
                tempK[v] = curVar[v] / (curVar[v] + this.sigmaSquares[v]);
            }

            for (int v = 0; v < V; v++) {
                double newMean = curMean[v] + tempK[v] * (posMean[v] - curMean[v]);
                nodes.get(t).getContent().setMean(v, newMean);
                double newVar = curVar[v] + tempK[v] * posVar[v] * tempK[v] - tempK[v] * curVar[v];
                nodes.get(t).getContent().setVariance(v, newVar);
            }
        }
    }

    /**
     * Perform forward filtering for a single chain of nodes
     *
     * @param nodes The chain
     */
    private void forwardSingleChain(ArrayList<DNCRPNode> nodes,
            double[] priorMean, double[] priorVar) {
        if (debug) {
            logln("--- forward filtering chain: " + nodes.toString());
        }

//        double[] preMean = zeros.clone();
//        double[] preVar = sigmaSquares.clone();
        double[] preMean = priorMean.clone();
        double[] preVar = priorVar.clone();

        int numConverged = 0;
        for (int t = 0; t < nodes.size(); t++) {
            StateObjective objective = new StateObjective(preMean, preVar, nodes.get(t).getContent().getSparseCounts());
            Optimizer optimizer = new LimitedMemoryBFGS(objective);
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

            if (converged) {
                numConverged++;
            }

            for (int i = 0; i < V; i++) {
                nodes.get(t).getContent().setMean(i, objective.getParameter(i));
            }

            // compute diagonal approximation of the Hessian
            double[] exps = new double[V];
            double sumExp = 0.0;
            for (int i = 0; i < V; i++) {
                exps[i] = Math.exp(nodes.get(t).getContent().getMean(i));
                sumExp += exps[i];
            }

            for (int i = 0; i < V; i++) {
                double prob = exps[i] / sumExp;
                double negHess =
                        1.0 / preVar[i]
                        + nodes.get(t).getContent().getCountSum() * prob * (1 - prob);
                nodes.get(t).getContent().setVariance(i, 1.0 / negHess);

//                logln("i = " + i 
//                        + ". exps = " + MiscUtils.formatDouble(exps[i])
//                        + ". preVar = " + MiscUtils.formatDouble(preVar[i])
//                        + ". prob = " + MiscUtils.formatDouble(prob)
//                        + ". negH = " + MiscUtils.formatDouble(negHess)
//                        + ". ---> " + MiscUtils.formatDouble(1.0 / negHess));
            }

            // debug
//            logln("---> node: " + nodes.get(t).toString());
//            logln(MiscUtils.arrayToString(nodes.get(t).getContent().getVariance()) + "\n");

            // update 
            nodes.get(t).getContent().updateDistribution();
            for (int i = 0; i < V; i++) {
                preMean[i] = nodes.get(t).getContent().getMean(i);
                preVar[i] = nodes.get(t).getContent().getVariance(i) + sigmaSquares[0];
            }
        }
    }

    /**
     * Get all chains starting at a specific epoch
     *
     * @param t Time epoch
     */
    public ArrayList<ArrayList<DNCRPNode>> getForwardChains(int t) {
        ArrayList<ArrayList<DNCRPNode>> chains = new ArrayList<ArrayList<DNCRPNode>>();
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();

            if (node.getPreNode() == null) {
                ArrayList<DNCRPNode> chain = new ArrayList<DNCRPNode>();
                DNCRPNode tempNode = node;
                while (tempNode != null) {
                    chain.add(tempNode);
                    tempNode = tempNode.getPosNode();
                }
                chains.add(chain);
            }

            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return chains;
    }

    public DNCRPNode getNode(int t, String nodepath) {
        String[] sline = nodepath.split(":");
        DNCRPNode curNode = dynamicRoots[t];
        for (int i = 1; i < sline.length; i++) {
            curNode = curNode.getChild(Integer.parseInt(sline[i]));
        }
        return curNode;
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;

        for (int t = 0; t < T; t++) {
            Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
            stack.add(dynamicRoots[t]);
            while (!stack.isEmpty()) {
                DNCRPNode node = stack.pop();

                double nodeLogPrior = 0.0;
                if (!isLeafNode(node)) {
                    nodeLogPrior = computeJointAssignmentLogProbability(node);
                }

                double nodeLlh = node.getContent().getLogLikelihood();

                llh += nodeLogPrior + nodeLlh;

                for (DNCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        double llh = 0.0;

        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
        for (int t = 0; t < T; t++) {
            int trueNumObs = 0;
            for (int d = 0; d < words[t].length; d++) {
                trueNumObs += words[t][d].length;
            }

            int numObsOnTrees = 0;
            Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
            stack.add(dynamicRoots[t]);
            while (!stack.isEmpty()) {
                DNCRPNode node = stack.pop();

                numObsOnTrees += node.getContent().getCountSum();

                if (node.getNumActualCustomers() + node.getNumPseudoCustomers() == 0) {
                    throw new RuntimeException(msg + ". t = " + t + ". node " + node.toString());
                }
                for (DNCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            if (numObsOnTrees != trueNumObs) {
                throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                        + numObsOnTrees + " vs. " + trueNumObs);
            }

            int numObsOnStick = 0;
            for (int d = 0; d < words[t].length; d++) {
                numObsOnStick += docLevelDists[t][d].getCountSum();
            }
            if (numObsOnStick != trueNumObs) {
                throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                        + numObsOnStick + " vs. " + trueNumObs);
            }
        }
    }

    private ArrayList<DNCRPNode> getForwardChain(DNCRPNode node) {
        ArrayList<DNCRPNode> chain = new ArrayList<DNCRPNode>();
        DNCRPNode temp = node;
        while (temp != null) {
            chain.add(temp);
            temp = temp.getPosNode();
        }
        return chain;
    }

    private ArrayList<DNCRPNode> getBackwardChain(DNCRPNode node) {
        ArrayList<DNCRPNode> chain = new ArrayList<DNCRPNode>();
        DNCRPNode temp = node;
        while (temp != null) {
            chain.add(temp);
            temp = temp.getPreNode();
        }
        return chain;
    }

    private boolean isLeafNode(DNCRPNode node) {
        return node.getLevel() == L - 1;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        for (int t = 0; t < T; t++) {
            str.append(t).append("\t").append(printTreeStructure(t)).append("\n");
        }
        return str.toString();
    }

    /**
     * Print all trees
     */
    private String printTrees() {
        StringBuilder str = new StringBuilder();
        for (int t = 0; t < T; t++) {
            str.append("t = ").append(t).append("\n").append(printTree(t)).append("\n");
        }
        return str.toString();
    }

    /**
     * Print a tree
     *
     * @param t Time epoch
     */
    private String printTree(int t) {
        StringBuilder str = new StringBuilder();
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    //                    .append("\t").append(MiscUtils.arrayToString(node.getContent().getDistribution()))
                    //                    .append("\t").append(MiscUtils.arrayToString(node.getContent().getCounts()))
                    .append("\n");

            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        return str.toString();
    }

    /**
     * Print the number of nodes at each level of a tree
     *
     * @param t Time epoch
     */
    private String printTreeStructure(int t) {
        int[] levelCount = new int[L];
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();
            levelCount[node.getLevel()]++;
            // add children to the queue
            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(").append(levelCount[l]).append(") ");
        }
        return str.toString();
    }

    public void outputChains(String filepath) throws Exception {
        logln("--- Outputing chains to " + filepath);

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < T; t++) {
            ArrayList<ArrayList<DNCRPNode>> chains = getForwardChains(t);

            for (ArrayList<DNCRPNode> chain : chains) {
                int i = t;
                for (DNCRPNode node : chain) {
                    writer.write(i++ + " " + getNodeString(node) + "\n");
                }
                writer.write("\n\n\n");
            }
        }
        writer.close();
    }

    public ArrayList<double[]> getTopics() {
        ArrayList<double[]> topics = new ArrayList<double[]>();
        for (int t = 0; t < T; t++) {
            Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
            stack.add(this.dynamicRoots[t]);
            while (!stack.isEmpty()) {
                DNCRPNode node = stack.pop();

                if (node.getLevel() != 0) {
                    topics.add(node.getContent().getDistribution());
                }

                for (DNCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }
        return topics;
    }

    public String getTopWordsHierarchy(int t) {
        StringBuilder str = new StringBuilder();
        Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
        stack.add(this.dynamicRoots[t]);
        while (!stack.isEmpty()) {
            DNCRPNode node = stack.pop();

            for (DNCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }

            // debug
            if (node.getContent().getDistribution() == null) {
                throw new RuntimeException(node.toString());
            }

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(getNodeString(node));
            str.append("\n\n");
        }

        return str.toString();
    }

    private String getNodeString(DNCRPNode node) {
        StringBuilder str = new StringBuilder();
        str.append(node.getPathString())
                .append(" (").append(node.getNumActualCustomers())
                .append("; ").append(MiscUtils.formatDouble(node.getNumPseudoCustomers()))
                .append("; ").append(node.getContent().getCountSum())
                .append("; ").append(node.getPreNodePathString())
                .append("; ").append(node.getPosNodePathString())
                .append(")");
        String[] topWords = getTopWords(node.getContent().getDistribution(), wordVocab, NUM_TOPWORDS);
        for (String topWord : topWords) {
            str.append(" ").append(topWord);
        }
        return str.toString();
    }

    private String[] getTopWords(double[] distribution, ArrayList<String> vocab, int numWords) {
        ArrayList<RankingItem<String>> topicSortedVocab = IOUtils.getSortedVocab(distribution, vocab);
        String[] topWords = new String[numWords];
        for (int i = 0; i < numWords; i++) {
            topWords[i] = topicSortedVocab.get(i).getObject();
        }
        return topWords;
    }

    public void outputTopWords(String filepath) throws Exception {
        logln("--- Outputing top word hierarchies to " + filepath);

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < T; t++) {
            writer.write(t + "\n\n" + getTopWordsHierarchy(t) + "\n");
        }
        writer.close();
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            StringBuilder str = new StringBuilder();

            // dynamic trees
            for (int t = 0; t < T; t++) {
                Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
                stack.add(this.dynamicRoots[t]);
                while (!stack.isEmpty()) {
                    DNCRPNode node = stack.pop();

                    // write node
                    str.append(node.getPathString())
                            .append("\t").append(node.getNumActualCustomers())
                            .append("\t").append(node.getNumPseudoCustomers())
                            .append("\t").append(node.getPreNodePathString())
                            .append("\t").append(node.getPosNodePathString())
                            .append("\n");

                    // prior mean
                    for (int i = 0; i < V; i++) {
                        str.append(node.getContent().getPriorMean(i)).append("\t");
                    }
                    str.append("\n");

                    // prior variance
                    for (int i = 0; i < V; i++) {
                        str.append(node.getContent().getPriorVariance(i)).append("\t");
                    }
                    str.append("\n");

                    // mean
                    for (int i = 0; i < V; i++) {
                        str.append(node.getContent().getMean(i)).append("\t");
                    }
                    str.append("\n");

                    // variance
                    for (int i = 0; i < V; i++) {
                        str.append(node.getContent().getVariance(i)).append("\t");
                    }
                    str.append("\n");

                    // observations
                    for (int obs : node.getContent().getUniqueObservations()) {
                        str.append(obs).append(":").append(node.getContent().getCount(obs)).append("\t");
                    }
                    str.append("\n");

                    // recursive call
                    for (DNCRPNode child : node.getChildren()) {
                        stack.add(child);
                    }
                }

                str.append(SEPARATOR).append("\n");
            }

            // document-specific level distribution
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    str.append(t).append(":").append(d);
                    for (int l = 0; l < L; l++) {
                        str.append("\t").append(docLevelDists[t][d].getCount(l));
                    }
                    str.append("\n");
                }
            }

            // document path assignments
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    str.append(t).append(":").append(d).append("\t").append(c[t][d].getPathString()).append("\n");
                }
            }

            // token level assignment
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    for (int n = 0; n < words[t][d].length; n++) {
                        str.append(t).append(":").append(d).append(":").append(n).append("\t").append(z[t][d][n]).append("\n");
                    }
                }
            }

            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);
            ZipEntry e = new ZipEntry(filename + ".txt");
            writer.putNextEntry(e);
            byte[] data = str.toString().getBytes();
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
            logln("--- Inputing state from " + filepath);
        }

        this.initializeHierarchies();

        try {
            ZipFile zipFile = new ZipFile(filepath);
            ZipEntry entry = zipFile.entries().nextElement();
            InputStream input = zipFile.getInputStream(entry);
            BufferedReader reader = new BufferedReader(new InputStreamReader(input, "UTF-8"));

            String line;
            String[] sline;
            HashMap<String, DNCRPNode>[] nodeMap = new HashMap[T];
            HashMap<String, String>[] nodePreNodeMap = new HashMap[T];
            HashMap<String, String>[] nodePosNodeMap = new HashMap[T];

            // load trees
            for (int t = 0; t < T; t++) {
                nodeMap[t] = new HashMap<String, DNCRPNode>();
                nodePreNodeMap[t] = new HashMap<String, String>();
                nodePosNodeMap[t] = new HashMap<String, String>();

                String info = reader.readLine();
                String priorMean = reader.readLine();
                String priorVariance = reader.readLine();
                String mean = reader.readLine();
                String variance = reader.readLine();
                String nodeObservation = reader.readLine();

                this.dynamicRoots[t] = inputSingleNode(info, priorMean,
                        priorVariance, mean, variance, nodeObservation,
                        nodePreNodeMap[t], nodePosNodeMap[t], nodeMap[t]);
                nodeMap[t].put(dynamicRoots[t].getPathString(), this.dynamicRoots[t]);

                while (!(line = reader.readLine()).equals(SEPARATOR)) {
                    info = line;
                    priorMean = reader.readLine();
                    priorVariance = reader.readLine();
                    mean = reader.readLine();
                    variance = reader.readLine();
                    nodeObservation = reader.readLine();

                    DNCRPNode node = inputSingleNode(info, priorMean,
                            priorVariance, mean, variance, nodeObservation,
                            nodePreNodeMap[t], nodePosNodeMap[t], nodeMap[t]);
                    nodeMap[t].put(node.getPathString(), node);
                }
            }

            // load level distributions
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    line = reader.readLine();
                    sline = line.split("\t");
                    if (!(t + ":" + d).equals(sline[0])) {
                        throw new RuntimeException("Read line mismatched. " + t + ":" + d + ". " + line);
                    }

                    int[] counts = new int[L];
                    for (int l = 0; l < L; l++) {
                        counts[l] = Integer.parseInt(sline[l + 1]);
                    }
                    docLevelDists[t][d].setCounts(counts);
                }
            }

            // load document path assignments
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    line = reader.readLine();
                    sline = line.split("\t");
                    if (!(t + ":" + d).equals(sline[0])) {
                        throw new RuntimeException("Read line mismatched. " + t + ":" + d + ". " + line);
                    }
                    c[t][d] = nodeMap[t].get(sline[1]);
                }
            }

            // load token level assignments
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < words[t].length; d++) {
                    for (int n = 0; n < words[t][d].length; n++) {
                        line = reader.readLine();
                        sline = line.split("\t");
                        if (!(t + ":" + d + ":" + n).equals(sline[0])) {
                            throw new RuntimeException("Read line mismatched. " + t + ":" + d + ":" + n + ". " + line);
                        }
                        z[t][d][n] = Integer.parseInt(sline[1]);
                    }
                }
            }
            reader.close();
            input.close();
            zipFile.close();

            // update pre and pos node
            for (int t = 0; t < T; t++) {
                for (String nodePath : nodePreNodeMap[t].keySet()) {
                    String preNodePath = nodePreNodeMap[t].get(nodePath);
                    if (!preNodePath.equals(DNCRPNode.EMPTY_NODE_PATH)) {
                        DNCRPNode node = nodeMap[t].get(nodePath);
                        DNCRPNode preNode = nodeMap[t - 1].get(preNodePath);
                        node.setPreNode(preNode);
                    }
                }

                for (String nodePath : nodePosNodeMap[t].keySet()) {
                    String posNodePath = nodePosNodeMap[t].get(nodePath);
                    if (!posNodePath.equals(DNCRPNode.EMPTY_NODE_PATH)) {
                        DNCRPNode node = nodeMap[t].get(nodePath);
                        DNCRPNode posNode = nodeMap[t + 1].get(posNodePath);
                        node.setPosNode(posNode);
                    }
                }
            }

            // update inactive children list
            for (int t = 0; t < T; t++) {
                Stack<DNCRPNode> stack = new Stack<DNCRPNode>();
                stack.add(dynamicRoots[t]);
                while (!stack.isEmpty()) {
                    DNCRPNode node = stack.pop();
                    if (!isLeafNode(node)) {
                        node.fillInactiveChildIndices(); // be careful!!!
                        for (DNCRPNode child : node.getChildren()) {
                            stack.add(child);
                        }
                    }
                }
            }

            if (verbose) {
                for (int t = 0; t < T; t++) {
                    logln("t = " + t + "\t" + this.printTreeStructure(t));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private DNCRPNode inputSingleNode(
            String info,
            String priorMean,
            String priorVariance,
            String mean,
            String variance,
            String nodeObservation,
            HashMap<String, String> nodePreNodeMap,
            HashMap<String, String> nodePosNodeMap,
            HashMap<String, DNCRPNode> nodeMap) {
        // create root node
        String[] sline = info.split("\t");
        String pathStr = sline[0];
        int numActCusts = Integer.parseInt(sline[1]);
        double numPseudoCusts = Double.parseDouble(sline[2]);
        String preNodePathStr = sline[3];
        String posNodePathStr = sline[4];

        nodePreNodeMap.put(pathStr, preNodePathStr);
        nodePosNodeMap.put(pathStr, posNodePathStr);

        // prior mean
        sline = priorMean.split("\t");
        double[] pMean = new double[sline.length];
        for (int i = 0; i < pMean.length; i++) {
            pMean[i] = Double.parseDouble(sline[i]);
        }

        // prior variance
        sline = priorVariance.split("\t");
        double[] pVar = new double[sline.length];
        for (int i = 0; i < pVar.length; i++) {
            pVar[i] = Double.parseDouble(sline[i]);
        }

        LogisticNormal lnModel = new LogisticNormal(V, pMean, pVar);

        // mean
        sline = mean.split("\t");
        for (int i = 0; i < sline.length; i++) {
            lnModel.setMean(i, Double.parseDouble(sline[i]));
        }

        // variance
        sline = variance.split("\t");
        for (int i = 0; i < sline.length; i++) {
            lnModel.setVariance(i, Double.parseDouble(sline[i]));
        }

        lnModel.updateDistribution();

        // observations
        if (!nodeObservation.isEmpty()) {
            sline = nodeObservation.split("\t");
            for (int i = 0; i < sline.length; i++) {
                int obs = Integer.parseInt(sline[i].split(":")[0]);
                int count = Integer.parseInt(sline[i].split(":")[1]);
                lnModel.changeCount(obs, count);
            }
        }

        int lastColonIndex = pathStr.lastIndexOf(":");
        DNCRPNode parent = null;
        if (lastColonIndex != -1) {
            parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
        }

        String[] pathIndices = pathStr.split(":");
        DNCRPNode node = new DNCRPNode(Integer.parseInt(pathIndices[pathIndices.length - 1]),
                pathIndices.length - 1, lnModel, parent, null, null);
        if (parent != null) {
            parent.addChild(node.getIndex(), node);
        }

        node.setNumPseudoCustomers(numPseudoCusts);
        node.changeNumCustomers(numActCusts);

        return node;
    }

    public static void main(String[] args) {
        try {
            testStateIO();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testStateIO() throws Exception {
        int T = 3;
        int D = 50;
        int N = 5;
        int V = 10;
        int[][][] observations = new int[T][D][N];
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < N; n++) {
                    observations[t][d][n] = rand.nextInt(V);
                }
            }
        }

        DHLDASampler sampler = new DHLDASampler();
        sampler.setDebug(true);
        sampler.setVerbose(true);
        InitialState initState = InitialState.RANDOM;
        boolean paramOpt = false;
        String prefix = "";
        sampler.setPrefix(prefix);

        int L = 3;
        double sigma = 1.0;
        double gem_mean = 0.3;
        double gem_scale = 50;
        int delta = 2;
        double lambda = 1.0;
        double[] betas = {2, 0.5, 0.1};
        double[] gammas = {2, 1};
        int burn_in = 2;
        int max_iters = 10;
        int sample_lag = 2;

        sampler.configure(null, V, L, delta, observations,
                sigma, gem_mean, gem_scale,
                lambda, betas, gammas,
                initState, paramOpt,
                burn_in, max_iters, sample_lag);
        sampler.initialize();
        System.out.println("After initializing");
        System.out.println(sampler.printTrees());


        HashMap<DNCRPNode, Double> nodeLogPrior = new HashMap<DNCRPNode, Double>();
        sampler.computePathLogPrior(nodeLogPrior, sampler.dynamicRoots[0], 0.0);
        for (DNCRPNode node : nodeLogPrior.keySet()) {
            System.out.println(node.toString() + "\t" + nodeLogPrior.get(node));
        }

//        sampler.iterate();
//        System.out.println("After iterating");
//        System.out.println(sampler.printTrees());
    }
}
