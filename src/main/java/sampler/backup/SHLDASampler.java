package sampler.backup;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
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
public class SHLDASampler extends AbstractSampler {

    public static final int RHO = 0;
    public static final int MEAN = 1;
    public static final int SCALE = 2;
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // DP
    protected double[] mus;
    protected double[] sigmas;
    protected int L; // level of hierarchies
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int regressionLevel;
    protected int[][] words;  // words
    protected double[] y; // [D]: document observations
    private int[][] z; // level assignments
    private SHLDANode[] c; // path assignments
    private TruncatedStickBreaking[] doc_level_distr;
    private SHLDANode word_hier_root;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private double[] uniform;
    private int numChangePath;
    private int numChangeLevel;
    private int existingPathCount;
    private int totalPathCount;
    private int optimizeCount = 0;
    private int convergeCount = 0;
    private int numTokens = 0;

    public void configure(String folder, int[][] words, double[] y,
            int V, int L,
            double mean, // GEM mean
            double scale, // GEM scale
            double[] betas, // Dirichlet hyperparameter for distributions over words
            double[] gammas, // hyperparameter for nCRP
            double[] mus, // mean of Gaussian for regression parameters
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double rho, // standard deviation of Gaussian for document observations
            int regressLevel,
            InitialState initState, boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.words = words;
        this.y = y;

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
                .append("_SHLDA-")
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

        numTokens = 0;
        doc_level_distr = new TruncatedStickBreaking[D];
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
            doc_level_distr[d] = new TruncatedStickBreaking(L, hyperparams.get(MEAN), hyperparams.get(SCALE));
        }

        uniform = new double[V];
        for (int i = 0; i < V; i++) {
            uniform[i] = 1.0 / V;
        }
        DirMult dmModel = new DirMult(V, betas[0], uniform);
        word_hier_root = new SHLDANode(0, 0, dmModel, null);
    }

    protected void initializeDataStructure() {
        c = new SHLDANode[D];
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                this.initializeRandomAssignments();
                break;
//            case FORWARD :
//                this.initializeForwardAssignments();
//                break;
//            case PRESET :
//                this.initializePresetAssignments();
//                break;
        }

        if (verbose) {
            logln("--- Done initialization. Llh = " + this.getLogLikelihood()
                    + "\t" + this.getCurrentState());
//            logln("--- " + word_hier.printTreeStructure());
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        // initialize path assignments
        for (int d = 0; d < D; d++) {
            SHLDANode node = word_hier_root;
            for (int l = 0; l < L - 1; l++) {
                node.incrementNumCustomers();
                node = this.createNode(node); // create a new path for each document
            }
            node.incrementNumCustomers();
            c[d] = node;

            // forward sample levels
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, !REMOVE, ADD, !REMOVE, ADD, OBSERVED);
            }

            // resample path
            if (d > 0) {
                samplePathAssignments(d, REMOVE, ADD, OBSERVED);
            }

            // resampler levels
            for (int n = 0; n < words[d].length; n++) {
                sampleLevelAssignments(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
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
            existingPathCount = 0;
            totalPathCount = 0;
            optimizeCount = 0;
            convergeCount = 0;
            numChangePath = 0;
            numChangeLevel = 0;

//            RegressionEvaluation eval;
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t topic count: " + getCurrentState() //                            + "\t #paths changed: " + numChangePath
                            //                            + "\t #levels changed: " + numChangeLevel
                            );
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t topic count: " + getCurrentState() //                            + "\t #paths changed: " + numChangePath
                            //                            + "\t #levels changed: " + numChangeLevel
                            );
                }
            }

            numChangePath = 0;
            numChangeLevel = 0;

            for (int d = 0; d < D; d++) {
                samplePathAssignments(d, REMOVE, ADD, OBSERVED);

                for (int n = 0; n < words[d].length; n++) {
                    sampleLevelAssignments(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            updateRegressionParameters();

//            if(verbose){
//                logln("--- --- # paths changed: " + numChangePath
//                        + " (" + ((double)numChangePath / D) + ")"
//                        + ". # levels changed: " + numChangeLevel
//                        + " (" + ((double)numChangeLevel / numTokens + ")"));
//                logln("--- --- # optimized: " + optimizeCount
//                        + ". # converged: " + convergeCount
//                        + ". convergence ratio: " + (double)convergeCount / optimizeCount);
//                logln("--- --- # existing paths: " + existingPathCount
//                        + ". # paths: " + totalPathCount
//                        + ". existing ratio: " + (double)existingPathCount / totalPathCount
//                        + "\n");
//                
//                double[] trPredResponses = getRegressionValues();
//                eval = new RegressionEvaluation(y, trPredResponses);
//                eval.computeCorrelationCoefficient();
//                eval.computeMeanSquareError();
//                eval.computeRSquared();
//                ArrayList<Measurement> measurements = eval.getMeasurements();
//                for(Measurement measurement : measurements)
//                    logln("--- --- --- " + measurement.getName() + ":\t" + measurement.getValue());
//            }


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

    private void samplePathAssignments(int d, boolean remove, boolean add, boolean observed) {
        SHLDANode curPathNode = null;

        if (remove) {
            curPathNode = this.removeCustomerFromPath(c[d]);
            for (int n = 0; n < words[d].length; n++) {
                this.removeObservation(words[d][n], z[d][n], c[d]);
            }
        }

        // compute log probability of each path which is represented by either
        // a leaf node (existing path) or an internal node (novel path)
        // P(c_d | c_{-d})
        HashMap<SHLDANode, Double> nodeLogPriors = new HashMap<SHLDANode, Double>();
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
        HashMap<SHLDANode, Double> nodeDataLlhs = new HashMap<SHLDANode, Double>();
        computePathLogLikelihood(nodeDataLlhs, word_hier_root,
                docTypeCountPerLevel, dataLlhNewTopic, 0.0);

        // Compute the likelihood of document-specific observations 
        // P(y_d | c_d, z_d, \eta)
        HashMap<SHLDANode, Double> pathRegMean = new HashMap<SHLDANode, Double>();
        HashMap<SHLDANode, Double> pathObsLlhs = new HashMap<SHLDANode, Double>();
        if (observed) {
            // --- a. Compute the current empirical level distribution of this document
            double[] empiricalLevelDistr = doc_level_distr[d].getEmpiricalDistribution();
            // --- b. Compute the Gaussian mean for each path
            computePathRegressionMean(pathRegMean, word_hier_root, empiricalLevelDistr, 0.0);
            // --- c. Compute the actual Gaussian log likelihood of y[d] given the mean
            for (SHLDANode node : pathRegMean.keySet()) {
                double mean = pathRegMean.get(node);
                double normLlh = StatUtils.logNormalProbability(y[d], mean, hyperparams.get(RHO));
                pathObsLlhs.put(node, normLlh);
            }
        }

        // sample path
        ArrayList<Double> logprobs = new ArrayList<Double>();
        ArrayList<SHLDANode> nodeLists = new ArrayList<SHLDANode>();
        int index = 0;
        for (SHLDANode node : nodeLogPriors.keySet()) {
            // only sample among existing path if we are sampling for new document            
            if (!observed && !this.isLeafNode(node)) {
                continue;
            }

            double samplingValue = nodeLogPriors.get(node)
                    + nodeDataLlhs.get(node);
            if (observed) {
                samplingValue += pathObsLlhs.get(node);
            }
            nodeLists.add(node);
            logprobs.add(samplingValue);
            index++;
        }
        int sampledIndex = SamplerUtils.logMinRescaleSample(logprobs);
        SHLDANode node = nodeLists.get(sampledIndex);

        // update statistics for debugging
        if (this.isLeafNode(node)) {
            existingPathCount++;
        }
        totalPathCount++;
        if (curPathNode == null || !nodeLists.get(sampledIndex).equals(curPathNode)) {
            numChangePath++;
        }

        // if pick an internal node, create the path from the internal node to leave
        if (node.getLevel() < L - 1) {
            node = this.getNewLeaf(node);
        }

        // add this document to the new sampled path and add tokens to the 
        // corresponding multinomials on the path
        c[d] = node;

        if (add) {
            this.addCustomerToPath(c[d]);
            for (int n = 0; n < words[d].length; n++) {
                this.addObservation(words[d][n], z[d][n], c[d]);
            }
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
            HashMap<SHLDANode, Double> nodeDataLlhs,
            SHLDANode curNode,
            HashMap<Integer, Integer>[] docTokenCountPerLevel,
            double[] dataLlhNewTopic,
            double parentDataLlh) {

        int level = curNode.getLevel();

        // compute the data log likelihood at the current node
        double nodeDataLlh = getWordObsLogLikelihoodFromNode(curNode, docTokenCountPerLevel[level]);

        // populate to child nodes
        for (SHLDANode child : curNode.getChildren()) {
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

        // debug
//        logln("--- iter = " + iter
//                + ". node = " + curNode.getPathString()
//                + ". parentLlh = " + parentDataLlh
//                + ". curnodeLlh = " + nodeDataLlh
//                + ". storedLlh = " + storeDataLlh);

        nodeDataLlhs.put(curNode, storeDataLlh);
    }

    /**
     * Compute the log likelihood of a set of word observations given a node
     *
     * @param curNode The node (which contains a multinomial over the
     * vocabulary)
     * @param docTokenCount A table storing the count of each word type
     */
    private double getWordObsLogLikelihoodFromNode(SHLDANode curNode,
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
            HashMap<SHLDANode, Double> nodeLogProbs,
            SHLDANode curNode,
            double parentLogProb) {
        double newWeight = parentLogProb;
        if (!isLeafNode(curNode)) {
            double logNorm = Math.log(curNode.getNumCustomers() + gammas[curNode.getLevel()]);

            for (SHLDANode child : curNode.getChildren()) {
                double childWeight = parentLogProb + Math.log(child.getNumCustomers()) - logNorm;
                computePathLogPrior(nodeLogProbs, child, childWeight);
            }
            newWeight += Math.log(gammas[curNode.getLevel()]) - logNorm;
        }
        nodeLogProbs.put(curNode, newWeight);
    }

    /**
     * Compute the regression mean of each path in the tree
     *
     * @param nodeRegMean Table to store the computed regression mean
     * @param curNode The current node
     * @param empiricalLevelDistr The empirical level distribution of the
     * current document
     * @param parentRegMean The regression mean passed on from the parent node
     */
    private void computePathRegressionMean(
            HashMap<SHLDANode, Double> nodeRegMean,
            SHLDANode curNode,
            double[] empiricalLevelDistr,
            double parentRegMean) {
        int level = curNode.getLevel();
        double prod = curNode.getRegressionParameter() * empiricalLevelDistr[level];
        double passOnRegMean = parentRegMean + prod;

        // from pseudo children
        SHLDANode node = curNode.getPseudoChild();
        while (node != null) {
            prod += node.getRegressionParameter() * empiricalLevelDistr[node.getLevel()];
            node = node.getPseudoChild();
        }
        nodeRegMean.put(curNode, parentRegMean + prod); // store in the Hashtable

        for (SHLDANode child : curNode.getChildren()) {
            computePathRegressionMean(nodeRegMean, child, empiricalLevelDistr, passOnRegMean);
        }
    }

    /**
     * Sample the level assignment z[d][n] for a token.
     *
     * @param d The document index
     * @param n The word position in the document
     * @param remove Whether this token should be removed from the current state
     */
    private void sampleLevelAssignments(int d, int n,
            boolean removeLevelDist, boolean addLevelDist,
            boolean removeWordHier, boolean addWordHier,
            boolean observed) {
        // decrement 
        if (removeLevelDist) {
            doc_level_distr[d].decrement(z[d][n]);
        }
        if (removeWordHier) {
            this.removeObservation(words[d][n], z[d][n], c[d]);
        }
        double[] pathRegressionParams = this.getRegressionPath(c[d]);

        double[] logprobs = new double[L];
        for (int l = 0; l < L; l++) {
            // sampling equation
            SHLDANode node = this.getNode(l, c[d]);
            logprobs[l] = doc_level_distr[d].getLogProbability(l)
                    + node.getContent().getLogLikelihood(words[d][n]);

            if (observed) {
                // compute new empirical distribution if this token is assigned to level l
                doc_level_distr[d].increment(l);
                double[] empDist = doc_level_distr[d].getEmpiricalDistribution();
                doc_level_distr[d].decrement(l);

                // compute regression mean of the path
                double mean = 0.0;
                for (int k = regressionLevel; k < L; k++) {
                    mean += pathRegressionParams[k] * empDist[k];
                }
                logprobs[l] += StatUtils.logNormalProbability(y[d], mean, hyperparams.get(RHO));
            }
        }

        int sampledL = SamplerUtils.logMaxRescaleSample(logprobs);
        if (z[d][n] != sampledL) {
            numChangeLevel++;
        }

        // update and increment
        z[d][n] = sampledL;

        if (addLevelDist) {
            doc_level_distr[d].increment(z[d][n]);
        }
        if (addWordHier) {
            this.addObservation(words[d][n], z[d][n], c[d]);
        }
    }

    /**
     * Update the regression parameters using L-BFGS
     */
    private void updateRegressionParameters() {
        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            // update for all subtrees having the root node at the regressLevel
            if (node.getLevel() == regressionLevel) {
                optimizeRegressionParametersSubtree(node);
            }

            // resample the regression parameters of pseudo nodes from prior
            SHLDANode pseudoChild = node.getPseudoChild();
            while (pseudoChild != null) {
                pseudoChild.getContent().sampleFromPrior();
                pseudoChild = pseudoChild.getPseudoChild();
            }

            // recurse
            for (SHLDANode child : node.getChildren()) {
                queue.add(child);
            }
        }
    }

    /**
     * Update the regression parameters for all nodes in a subtree. The
     * regression parameters are updated by optimizing the posterior (i.e.,
     * Maximum A Posteriori (MAP) estimation). Optimization is done using L-BFGS
     * (via the implementation of L-BFGS in Mallet)
     *
     * @param root The root node of the subtree
     */
    private void optimizeRegressionParametersSubtree(SHLDANode root) {
        ArrayList<SHLDANode> flatTree = this.flattenTree(root);

        // map a node in the subtree to its index in the flattened tree (i.e. 1D array)
        HashMap<SHLDANode, Integer> nodeMap = new HashMap<SHLDANode, Integer>();
        for (int i = 0; i < flatTree.size(); i++) {
            nodeMap.put(flatTree.get(i), i);
        }

        ArrayList<double[]> designList = new ArrayList<double[]>();
        ArrayList<Double> responseList = new ArrayList<Double>();
        for (int d = 0; d < D; d++) {
            if (!nodeMap.containsKey(c[d])) // if this subtree does not contain this document
            {
                continue;
            }
            SHLDANode[] docPath = this.getPathFromNode(c[d]);
            double[] empiricalProb = doc_level_distr[d].getEmpiricalDistribution();

            double[] row = new double[flatTree.size()];
            for (int l = root.getLevel(); l < L; l++) {
                int nodeIndex = nodeMap.get(docPath[l]);
                row[nodeIndex] = empiricalProb[l];
            }
            designList.add(row);
            responseList.add(y[d]);
        }

        double[][] designMatrix = new double[designList.size()][flatTree.size()];
        for (int i = 0; i < designList.size(); i++) {
            designMatrix[i] = designList.get(i);
        }

        double[] response = new double[responseList.size()];
        for (int i = 0; i < response.length; i++) {
            response[i] = responseList.get(i);
        }

        double[] curParams = new double[flatTree.size()];
        for (int i = 0; i < flatTree.size(); i++) {
            curParams[i] = flatTree.get(i).getRegressionParameter();
        }

        // get the arrays of mus and sigmas corresponding to the list of flatten nodes
        double[] flattenMus = new double[flatTree.size()];
        double[] flattenSigmas = new double[flatTree.size()];
        for (int i = 0; i < flatTree.size(); i++) {
            int nodeLevel = flatTree.get(i).getLevel();
            flattenMus[i] = this.mus[nodeLevel];
            flattenSigmas[i] = this.sigmas[nodeLevel];
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                curParams, designMatrix, response,
                hyperparams.get(RHO),
                flattenMus, flattenSigmas);
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
            // update regression parameters
            for (int i = 0; i < flatTree.size(); i++) {
                flatTree.get(i).setRegressionParameter(optimizable.getParameter(i));
            }

            convergeCount++;
        }
    }

    /**
     * Flatten a subtree given the root of the subtree
     *
     * @param subtreeRoot The subtree's root
     */
    private ArrayList<SHLDANode> flattenTree(SHLDANode subtreeRoot) {
        ArrayList<SHLDANode> flatSubtree = new ArrayList<SHLDANode>();
        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(subtreeRoot);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            flatSubtree.add(node);
            for (SHLDANode child : node.getChildren()) {
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
    private SHLDANode[] getPathFromNode(SHLDANode node) {
        SHLDANode[] path = new SHLDANode[node.getLevel() + 1];
        SHLDANode curNode = node;
        int l = node.getLevel();
        while (curNode != null) {
            path[l--] = curNode;
            curNode = curNode.getParent();
        }
        return path;
    }

    /**
     * Get an array containing all the regression parameters along a path. The
     * path is specified by a leaf node.
     *
     * @param leafNode The leaf node
     */
    private double[] getRegressionPath(SHLDANode leafNode) {
        if (leafNode.getLevel() != L - 1) {
            throw new RuntimeException("Node " + leafNode.toString() + " is not a leaf node");
        }
        double[] regPath = new double[L];
        int level = leafNode.getLevel();
        SHLDANode curNode = leafNode;
        while (curNode != null) {
            regPath[level--] = curNode.getRegressionParameter();
            curNode = curNode.getParent();
        }
        return regPath;
    }

    /**
     * Add a customer to a path. A path is specified by the pointer to its leaf
     * node. If the given node is not a leaf node, an exception will be thrown.
     * The number of customers at each node on the path will be incremented.
     *
     * @param leafNode The leaf node of the path
     */
    private void addCustomerToPath(SHLDANode leafNode) {
        SHLDANode node = leafNode;
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
    private void removeObservation(int observation, int level, SHLDANode leafNode) {
        SHLDANode node = getNode(level, leafNode);
        node.getContent().decrement(observation);
    }

    /**
     * Add an observation to a node
     *
     * @param observation The observation to be added
     * @param level The level of the node
     * @param leafNode The leaf node of the path
     */
    private void addObservation(int observation, int level, SHLDANode leafNode) {
        SHLDANode node = getNode(level, leafNode);
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
    private SHLDANode removeCustomerFromPath(SHLDANode leafNode) {
        SHLDANode retNode = leafNode;
        SHLDANode node = leafNode;
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
    private SHLDANode createNode(SHLDANode parent) {
        int nextChildIndex = parent.getNextChildIndex();
        int level = parent.getLevel() + 1;
        DirMult dmModel = new DirMult(V, betas[level], uniform);
        SHLDANode child = new SHLDANode(nextChildIndex, level, dmModel,
                //mus[level], sigmas[level], 
                parent);
        return parent.addChild(nextChildIndex, child);
    }

    private boolean isLeafNode(SHLDANode node) {
        return node.getLevel() == L - 1;
    }

    private SHLDANode getNewLeaf(SHLDANode internalNode) {
        SHLDANode node = internalNode;
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
    private SHLDANode getNode(int level, SHLDANode leafNode) {
        if (!isLeafNode(leafNode)) {
            throw new RuntimeException("Exception while getting node. The given "
                    + "node is not a leaf node");
        }
        int curLevel = leafNode.getLevel();
        SHLDANode curNode = leafNode;
        while (curLevel != level) {
            curNode = curNode.getParent();
            curLevel--;
        }
        return curNode;
    }

    @Override
    public String getCurrentState() {
        double[] levelCount = new double[L];
        Queue<SHLDANode> queue = new LinkedList<SHLDANode>();
        queue.add(word_hier_root);
        while (!queue.isEmpty()) {
            SHLDANode node = queue.poll();
            levelCount[node.getLevel()]++;
            // add children to the queue
            for (SHLDANode child : node.getChildren()) {
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
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (SHLDANode child : node.getChildren()) {
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

        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getNumCustomers()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }

        writer.close();
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;

        double logWordLikelihood = 0.0;
        double logAssgnProb = 0.0;
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();
            logWordLikelihood += node.getContent().getLogLikelihood();

            if (!this.isLeafNode(node)) {
                logAssgnProb += getLogJointProbability(node);
                for (SHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        double logObsLikelihood = 0.0;
        double logStickProb = 0.0;

        return llh;
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        // this is not correct
        try {
            // model string
            StringBuilder modelStr = new StringBuilder();
            Stack<SHLDANode> stack = new Stack<SHLDANode>();
            stack.add(word_hier_root);
            while (!stack.isEmpty()) {
                SHLDANode node = stack.pop();

                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getNumCustomers()).append("\t");
                modelStr.append(DirMult.output(node.getContent())).append("\n");

                for (SHLDANode child : node.getChildren()) {
                    stack.add(child);
                }
            }

            // assignment string
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(d)
                            .append(":").append(n)
                            .append("\t").append(z[d][n])
                            .append("\n");
                }
            }

            for (int d = 0; d < D; d++) {
                assignStr.append(d)
                        .append("\t").append(c[d].getPathString())
                        .append("\n");
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
        HashMap<String, SHLDANode> nodeMap = new HashMap<String, SHLDANode>();
        String line;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int numCustomers = Integer.parseInt(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SHLDANode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            SHLDANode node = new SHLDANode(Integer.parseInt(pathIndices[pathIndices.length - 1]),
                    pathIndices.length - 1, dmm, parent);

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
            for (int n = 0; n < words[d].length; n++) {
                String[] sline = reader.readLine().split("\t");
                if (!sline[0].equals(d + ":" + n)) {
                    throw new RuntimeException("Mismatch");
                }
                z[d][n] = Integer.parseInt(sline[1]);
            }
        }

        for (int d = 0; d < D; d++) {
            String[] sline = reader.readLine().split("\t");
            if (Integer.parseInt(sline[0]) != d) {
                throw new RuntimeException("Mismatch");
            }
            String pathStr = sline[1];
            SHLDANode node = getNode(parseNodePath(pathStr));
            c[d] = node;
        }
        reader.close();
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    private SHLDANode getNode(int[] parsedPath) {
        SHLDANode node = word_hier_root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    private double getLogJointProbability(SHLDANode node) {
        ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
        for (SHLDANode child : node.getChildren()) {
            numChildrenCusts.add(child.getNumCustomers());
        }
        return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gammas[node.getLevel()]);
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

        // make sure there is no path having no customers
        int totalNumObs = 0;
        Stack<SHLDANode> stack = new Stack<SHLDANode>();
        stack.add(word_hier_root);
        while (!stack.isEmpty()) {
            SHLDANode node = stack.pop();

            totalNumObs += node.getContent().getCountSum();

            for (SHLDANode child : node.getChildren()) {
                stack.add(child);
            }

            if (this.isLeafNode(node) && node.isEmpty()) {
                throw new RuntimeException(msg + ". Leaf node " + node.toString()
                        + " is empty");
            }
        }

        if (numTokens != totalNumObs) {
            throw new RuntimeException(msg + ". Numbers of observations mismatch."
                    + " " + numTokens + " vs. " + totalNumObs);
        }
    }
}

class SHLDANode extends TreeNode<SHLDANode, DirMult> {

    private int numCustomers;
    private SHLDANode pseudoChild;
//    private double mu; // mean of the Gaussian distribution
//    private double sigma; // standard deviation of the Gaussian distrbution
    private double regression; // regression parameter

    public SHLDANode(int index, int level, DirMult content,
            //            double mu, double sigma,
            SHLDANode parent) {
        super(index, level, content, parent);
//        this.mu = mu;
//        this.sigma = sigma;
        this.numCustomers = 0;
        this.pseudoChild = null;
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

    public SHLDANode getPseudoChild() {
        return pseudoChild;
    }

    public void setPseudoChild(SHLDANode pseudoChild) {
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