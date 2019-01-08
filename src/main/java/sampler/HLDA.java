package sampler;

import core.AbstractSampler;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import sampling.likelihood.DirMult;
import sampling.likelihood.TruncatedStickBreaking;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 *
 * @author vietan
 */
public class HLDA extends AbstractSampler {

    public static final int MEAN = 0; // GEM mean
    public static final int SCALE = 1;  // GEM scale
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int[][] words;  // words
    private int[][] z; // level assignments
    private HLDANode[] c; // path assignments
    private TruncatedStickBreaking[] doc_level_distr;
    private HLDANode word_hier_root;
    private double[] uniform;
    private int numChangePath;
    private int numChangeLevel;

    public void configure(String folder, int[][] words,
            int V, int L,
            double m, double pi, double[] betas, double[] gammas,
            AbstractSampler.InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;

        this.L = L;
        this.V = V;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(m);
        this.hyperparams.add(pi);

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

        this.paramOptimized = paramOpt;
        this.initState = initState;
        this.prefix = initState.toString();
        this.setName();

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gamms must have length " + (this.L - 1)
                    + ". Current length = " + this.gammas.length);
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- tree height:\t" + L);
            logln("--- m:\t" + MiscUtils.formatDouble(m));
            logln("--- pi:\t" + MiscUtils.formatDouble(pi));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- lag:\t" + LAG);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_hLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_LVL-").append(L);

        str.append("_m-").append(formatter.format(hyperparams.get(MEAN)));
        str.append("_pi-").append(formatter.format(hyperparams.get(SCALE)));
        int count = SCALE + 1;
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

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeHierarchies() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

        doc_level_distr = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            doc_level_distr[d] = new TruncatedStickBreaking(L, hyperparams.get(MEAN), hyperparams.get(SCALE));
        }

        uniform = new double[V];
        for (int i = 0; i < V; i++) {
            uniform[i] = 1.0 / V;
        }
        DirMult dmModel = new DirMult(V, betas[0], uniform);
        this.word_hier_root = new HLDANode(iter, 0, 0, dmModel, null);
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
        }

        if (verbose) {
            logln("--- Done initialization. Llh = " + this.getLogLikelihood()
                    + "\t" + this.getCurrentState());
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        c = new HLDANode[D];
        z = new int[D][];

        // initialize path assignments
        for (int d = 0; d < D; d++) {
            HLDANode node = word_hier_root;
            for (int l = 0; l < L - 1; l++) {
                node.incrementNumCustomers();
                node = this.createNode(node); // create a new path for each document
            }
            node.incrementNumCustomers();
            c[d] = node;

            // forward sample levels
            z[d] = new int[words[d].length];
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, !REMOVE);
            }

            // resample path
            if (d > 0) {
                samplePathAssignments(d, REMOVE);
            }

            // resampler levels
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, REMOVE);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            System.out.println();
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t topic count: " + getCurrentState()
                            + "\t #paths changed: " + numChangePath
                            + "\t #levels changed: " + numChangeLevel);
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t topic count: " + getCurrentState()
                            + "\t #paths changed: " + numChangePath
                            + "\t #levels changed: " + numChangeLevel);
                }
            }

            numChangePath = 0;
            numChangeLevel = 0;

            for (int d = 0; d < D; d++) {
                samplePathAssignments(d, REMOVE);

                for (int n = 0; n < words[d].length; n++) {
                    sampleLevelAssignments(d, n, REMOVE);
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
        }
    }

    /**
     * Sample the path assignment c[d] for a document given other documents'
     * path assignments.
     *
     * Here, we need to consider all possible paths - existing paths, each of
     * which is represented by a leaf node - novel paths, each of which is
     * represented by an internal node
     *
     * For each path c_d, we need to compute two things - The prior on c_d
     * implied by the nested CRP: P(c_d | c_{-d}) - The likelihood of the data
     * given a particular choice of c_d: P(w_m | c, w_{-m}, z)
     *
     * @param d The index of the document
     */
    private void samplePathAssignments(int d, boolean remove) {
        // remove the current document from the current path, and remove tokens
        // from the corresponding multinomials
        HLDANode curPathNode = null;
        if (remove) {
            curPathNode = this.removeCustomerFromPath(c[d]);
            for (int n = 0; n < words[d].length; n++) {
                this.removeObservation(words[d][n], z[d][n], c[d]);
            }
        }

        // compute log probability of each path which is represented by either
        // a leaf node (existing path) or an internal node (novel path)
        // P(c_d | c_{-d})
        HashMap<HLDANode, Double> nodeLogPriors = new HashMap<HLDANode, Double>();
        computePathLogPrior(nodeLogPriors, word_hier_root, 0.0);

        // compute data log likelihood for each path P(w_m | c, w_{-m}, z)
        // --- 1. Store the word counts of the current document at each level
        // based on {z_{dn}}
        HashMap<Integer, Integer>[] docTypeCountPerLevel = new HashMap[L];
        for (int l = 0; l < L; l++) {
            docTypeCountPerLevel[l] = new HashMap<Integer, Integer>();
        }
        for (int n = 0; n < words[d].length; n++) {
            Integer count = docTypeCountPerLevel[z[d][n]].get(words[d][n]);
            if (count == null) {
                docTypeCountPerLevel[z[d][n]].put(words[d][n], 1);
            } else {
                docTypeCountPerLevel[z[d][n]].put(words[d][n], count + 1);
            }
        }

        // --- 2. Compute the data likelihood for a new path at a given level.
        // Since a symmetric Dirichlet prior is used for all nodes, only one
        // likelihood per level is sufficient. If different priors are used,
        // we need to compute this likelihood for each internal node (each
        // representing a novel path)
        double[] dataLlhNewTopic = new double[L];
        for (int l = 1; l < L; l++) { // skip the root
            HashMap<Integer, Integer> docTokenCount = docTypeCountPerLevel[l];
            int j = 0;
            for (int type : docTokenCount.keySet()) {
                for (int i = 0; i < docTokenCount.get(type); i++) {
                    dataLlhNewTopic[l] += Math.log(betas[l] / V + i) - Math.log(betas[l] + j);
                    j++;
                }
            }
        }

        // --- 3. Compute the data likelihood for all possible paths
        HashMap<HLDANode, Double> nodeDataLlhs = new HashMap<HLDANode, Double>();
        computePathLogLikelihood(nodeDataLlhs, word_hier_root,
                docTypeCountPerLevel, dataLlhNewTopic, 0.0);

        if (nodeLogPriors.size() != nodeDataLlhs.size()) {
            throw new RuntimeException("Numbers of nodes mismatch");
        }

        // sample path
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<HLDANode> nodeLists = new ArrayList<HLDANode>();
        int index = 0;
        for (HLDANode node : nodeLogPriors.keySet()) {
            nodeLists.add(node);
            logprobs.add(nodeLogPriors.get(node) + nodeDataLlhs.get(node));

            // debug
//            logln("iter = " + iter 
//                    + ". d = " + d
//                    + ". index = " + index
//                    + ". node = " + node.toString()
//                    + ". logprior = " + MiscUtils.formatDouble(nodeLogPriors.get(node)) // path prior
//                    + ". datallh = " + MiscUtils.formatDouble(nodeDataLlhs.get(node)) // data likelihood
//                    + ". total = " + MiscUtils.formatDouble(nodeLogPriors.get(node) // path prior
//                                        + nodeDataLlhs.get(node) // data likelihood
//                    )
//                    );
            index++;
        }
        int sampledIndex = SamplerUtils.logMinRescaleSample(logprobs);
        HLDANode node = nodeLists.get(sampledIndex);

//        logln("---> samplerIndex = " + sampledIndex + ". node = " + node.toString());

        if (node.getLevel() < L - 1) // pick an internal node
        {
            node = this.getNewLeaf(node);
        }

        if (curPathNode != null && !nodeLists.get(sampledIndex).equals(curPathNode)) {
            numChangePath++;
        }

        // add this document to the new sampled path and add tokens to the 
        // corresponding multinomials on the path
        c[d] = node;
        this.addCustomerToPath(c[d]);
        for (int n = 0; n < words[d].length; n++) {
            this.addObservation(words[d][n], z[d][n], c[d]);
        }
    }

    /**
     * Compute the data log likelihood for all possible paths. The set of all
     * possible paths corresponds to the union of - the set of existing paths
     * through the tree, each represented by a leaf - the set of possible novel
     * paths, each represented by an internal node
     *
     * @param nodeDataLlhs A Hashtable which maps each node in the tree (which
     * represent a possible path) to its data log likelihood
     * @param curNode The current node
     * @param docTokenCountPerLevel L Hashtables, each contains the token counts
     * of a document at the corresponding level
     * @param dataLlhNewTopic L-length array which contains the data log
     * likelihood for new path at each level. Here each level share 1 value
     * since all multinomials are assumed to be generated from a shared
     * symmetric Dirichlet prior
     * @param parentDataLlh The data log likelihood passed from the parent node
     */
    private void computePathLogLikelihood(
            HashMap<HLDANode, Double> nodeDataLlhs,
            HLDANode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();

        // compute the data log likelihood at the current node
        double nodeDataLlh = getWordObsLogLikelihoodFromNode(curNode, docTokenCountPerLevel[level]);

        // populate to child nodes
        for (HLDANode child : curNode.getChildren()) {
            computePathLogLikelihood(nodeDataLlhs, child, docTokenCountPerLevel,
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

    /**
     * Compute the log likelihood of a set of word observations given a node
     *
     * @param curNode The node (which contains a multinomial over the
     * vocabulary)
     * @param docTokenCount A table storing the count of each word type
     */
    private double getWordObsLogLikelihoodFromNode(HLDANode curNode,
            HashMap<Integer, Integer> docTokenCount) {
        double nodeDataLlh = 0;
        int level = curNode.getLevel();
        int j = 0;
        for (int type : docTokenCount.keySet()) {
            for (int i = 0; i < docTokenCount.get(type); i++) {
                nodeDataLlh += Math.log(betas[level] * curNode.getContent().getCenterElement(type)
                        + curNode.getContent().getCount(type) + i)
                        - Math.log(betas[level] + curNode.getContent().getCountSum() + j);
                j++;
            }
        }
        return nodeDataLlh;
    }

    /**
     * Recursively compute the log probability of each path in the tree given
     * the path assignments. The set of all possible paths corresponds to the
     * union of - the set of existing paths through the tree, each represented
     * by a leaf - the set of possible novel paths, each represented by an
     * internal node
     *
     * @param nodeLogProbs A Hashtable to map each node in the tree (which
     * represents a possible path) with its corresponding log probability
     * @param curNode The current node in the recursive function
     * @param parentLogProb The log probability from the parent node that is
     * passed to the child node
     */
    private void computePathLogPrior(
            HashMap<HLDANode, Double> nodeLogProbs,
            HLDANode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);

            for (HLDANode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }
            newWeight += Math.log(gammas[curNode.getLevel()]) - logNorm;
        }
        nodeLogProbs.put(curNode, newWeight);
    }

    private void sampleLevelAssignments(int d, int n, boolean remove) {
        if (remove) {
            doc_level_distr[d].decrement(z[d][n]);
            this.removeObservation(words[d][n], z[d][n], c[d]);
        }

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            HLDANode node = this.getNode(l, c[d]);
            logprobs[l] =
                    doc_level_distr[d].getLogProbability(l)
                    + node.getContent().getLogLikelihood(words[d][n]);
        }
        int sampledL = SamplerUtils.logMinRescaleSample(logprobs);

        if (z[d][n] != sampledL) {
            numChangeLevel++;
        }

        z[d][n] = sampledL;
        doc_level_distr[d].increment(z[d][n]);
        this.addObservation(words[d][n], z[d][n], c[d]);
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    private void addCustomerToPath(HLDANode leafNode) {
        HLDANode node = leafNode;
        while (node != null) {
            node.incrementNumCustomers();
            node = node.getParent();
        }
    }

    /**
     * Remove an observation from a node.
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void removeObservation(int observation, int level, HLDANode leafNode) {
        HLDANode node = getNode(level, leafNode);
        node.getContent().decrement(observation);
    }

    /**
     * Add an observation to a node
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void addObservation(int observation, int level, HLDANode leafNode) {
        HLDANode node = getNode(level, leafNode);
        node.getContent().increment(observation);
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
    private HLDANode removeCustomerFromPath(HLDANode leafNode) {
        HLDANode retNode = leafNode;
        HLDANode node = leafNode;
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
     * Create a new child of a parent node
     *
     * @param parent The parent node
     * @return The newly created child node
     */
    private HLDANode createNode(HLDANode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmModel = new DirMult(V, betas[level], uniform);
        HLDANode child = new HLDANode(iter, nextChildIndex, level, dmModel, parent);
        return parent.addChild(nextChildIndex, child);
    }

    private boolean isLeafNode(HLDANode node) {
        return node.getLevel() == L - 1;
    }

    private HLDANode getNewLeaf(HLDANode internalNode) {
        HLDANode node = internalNode;
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
    private HLDANode getNode(int level, HLDANode leafNode) {
        if (!isLeafNode(leafNode)) {
            throw new RuntimeException("Exception while getting node. The given "
                    + "node is not a leaf node");
        }
        int curLevel = leafNode.getLevel();
        HLDANode curNode = leafNode;
        while (curLevel != level) {
            curNode = curNode.getParent();
            curLevel--;
        }
        return curNode;
    }

    @Override
    public double getLogLikelihood() {
//        double docLevelLogProb = 0;
//        for(int d=0; d<D; d++)
//            docLevelLogProb += doc_level_distr[d].getL();

        double logWordLikelihood = 0.0;
        double logAssgnProb = 0.0;

        Queue<HLDANode> queue = new LinkedList<HLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            HLDANode node = queue.poll();
            logWordLikelihood += node.getContent().getLogLikelihood();

            if (!this.isLeafNode(node)) {
                logAssgnProb += getLogJointProbability(node);

                // add children to the queue
                for (HLDANode child : node.getChildren()) {
                    queue.add(child);
                }
            }
        }

        return logWordLikelihood
                //                + docLevelLogProb 
                + logAssgnProb;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            doc_level_distr[d].validate(msg);
        }

        int totalNumObsAssigned = 0;
        Stack<HLDANode> stack = new Stack<HLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            HLDANode node = stack.pop();
            totalNumObsAssigned += node.getContent().getCountSum();

            for (HLDANode child : node.getChildren()) {
                stack.add(child);
            }
        }


        // check the total number of observations
        int totalNumObs = 0;
        for (int d = 0; d < D; d++) {
            totalNumObs += words[d].length;
        }
        if (totalNumObs != totalNumObsAssigned) {
            throw new RuntimeException("Total number of observations mismatched. "
                    + totalNumObs + " vs. " + totalNumObsAssigned);
        }
    }

    private double getLogJointProbability(HLDANode node) {
        ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
        for (HLDANode child : node.getChildren()) {
            numChildrenCusts.add(child.getNumCustomers());
        }
        return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gammas[node.getLevel()]);
    }

    @Override
    public String getCurrentState() {
        int[] custCountPerLevel = new int[L];
        int[] obsCountPerLevel = new int[L];

        Queue<HLDANode> queue = new LinkedList<HLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            HLDANode node = queue.poll();
            custCountPerLevel[node.getLevel()]++;
            obsCountPerLevel[node.getLevel()] += node.getContent().getCountSum();

            // add children to the queue
            for (HLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(")
                    .append(custCountPerLevel[l])
                    .append(", ").append(obsCountPerLevel[l])
                    .append(")\t");
        }
        str.append("total obs: ").append(StatUtils.sum(obsCountPerLevel));
        return str.toString();
    }

    @Override
    public void outputState(String filepath) {
        throw new RuntimeException("This function is not supported at the moment");
    }

    @Override
    public void inputState(String filepath) {
        throw new RuntimeException("This function is not supported at the moment");
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
        Stack<HLDANode> stack = new Stack<HLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            HLDANode node = stack.pop();

            for (HLDANode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }

            String[] topWords = node.getTopWords(wordVocab, numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getNumCustomers())
                    .append("; ").append(node.getContent().getCountSum())
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

        Stack<HLDANode> stack = new Stack<HLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            HLDANode node = stack.pop();

            for (HLDANode child : node.getChildren()) {
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

    class HLDANode extends TreeNode<HLDANode, DirMult> {

        private final int born;
        int numCustomers;
        HLDANode pseudoChild;

        public HLDANode(int iter, int index, int level, DirMult content, HLDANode parent) {
            super(index, level, content, parent);
            this.born = iter;
            this.numCustomers = 0;
            this.pseudoChild = null;
        }

        public int getIterationCreated() {
            return this.born;
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

        public HLDANode getPseudoChild() {
            return pseudoChild;
        }

        public void setPseudoChild(HLDANode pseudoChild) {
            this.pseudoChild = pseudoChild;
        }

        public boolean isEmpty() {
            return this.numCustomers == 0;
        }

        public String[] getTopWords(ArrayList<String> vocab, int numWords) {
            ArrayList<RankingItem<String>> topicSortedVocab = IOUtils.getSortedVocab(content.getDistribution(), vocab);
            String[] topWords = new String[numWords];
            for (int i = 0; i < numWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }
    }
}
