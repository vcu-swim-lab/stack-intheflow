package sampler.labeled.hierarchy;

import cc.mallet.types.Dirichlet;
import cc.mallet.util.Randoms;
import core.AbstractSampler;
import data.LabelTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.labeled.LabeledLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import taxonomy.AbstractTaxonomyBuilder;
import taxonomy.BetaTreeBuilder;
import taxonomy.MSTBuilder;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class L2H extends AbstractSampler {

    public static Randoms randoms = new Randoms(1);
    public static final int INSIDE = 0;
    public static final int OUTSIDE = 1;
    // hyperparameter indices
    public static final int ALPHA = 0; // concentration parameter
    public static final int BETA = 1; // concentration parameter
    public static final int A_0 = 2;
    public static final int B_0 = 3;
    // inputs
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int V;    // vocab size
    protected int L;    // number of unique labels
    public int getL() {
        return L;
    }

    protected int D;    // number of documents
    // graph
    private SparseVector[] inWeights; // the weights of in-edges for each nodes
    // tree
    private AbstractTaxonomyBuilder treeBuilder;
    private Node root;
    private Node[] nodes;
    // latent variables
    private int[][] x;
    private int[][] z;
    private DirMult[] docSwitches;
    private SparseCount[] docLabelCounts;
    private Set<Integer>[] docMaskes;
    // configurations
    private PathAssumption pathAssumption;
    private boolean treeUpdated;
    private boolean sampleExact = false;
    // internal
    private HashMap<Integer, Set<Integer>> labelDocIndices;
    // information
    private ArrayList<String> labelVocab;
    private int numAccepts; // number of sampled nodes accepted
    private int[] labelFreqs;
    private double[] switchPrior;

    public L2H() {
        this.basename = "L2H";
    }

    public L2H(String basename) {
        this.basename = basename;
    }

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public void configure(L2H sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(A_0),
                sampler.hyperparams.get(B_0),
                sampler.treeBuilder,
                sampler.treeUpdated,
                sampler.sampleExact,
                sampler.initState,
                sampler.pathAssumption,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
        this.setWordVocab(sampler.wordVocab);
        this.setLabelVocab(sampler.labelVocab);
    }

    public void configure(String folder,
            int V,
            double alpha,
            double beta,
            double a0, double b0,
            AbstractTaxonomyBuilder treeBuilder,
            boolean treeUp,
            boolean sampleExact,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;

        this.treeBuilder = treeBuilder;
        this.labelVocab = treeBuilder.getLabelVocab();

        this.L = labelVocab.size();
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.hyperparams.add(a0);
        this.hyperparams.add(b0);

        this.switchPrior = new double[2];
        this.switchPrior[INSIDE] = a0;
        this.switchPrior[OUTSIDE] = b0;

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.treeUpdated = treeUp;
        this.sampleExact = sampleExact;

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.pathAssumption = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- a0:\t" + MiscUtils.formatDouble(a0));
            logln("--- b0:\t" + MiscUtils.formatDouble(b0));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- path assumption:\t" + pathAssumption);
            logln("--- tree builder:\t" + treeBuilder.getName());
            logln("--- updating tree?\t" + treeUpdated);
            logln("--- exact sampling?\t" + sampleExact);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_opt-").append(this.paramOptimized)
                .append("_").append(this.pathAssumption);
        for (double hp : this.hyperparams) {
            str.append("-").append(MiscUtils.formatDouble(hp));
        }
        str.append("-").append(treeBuilder.getName());
        str.append("-").append(treeUpdated);
        str.append("-").append(sampleExact);
        this.name = str.toString();
    }
    
    public Node[] getNodes() {
    	return nodes;
    }

    /**
     * Set training data.
     *
     * @param docIndices Indices of selected documents
     * @param words Document words
     * @param labels Document labels
     */
    public void train(ArrayList<Integer> docIndices, int[][] words, int[][] labels) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                docIndices.add(dd);
            }
        }

        this.D = docIndices.size();
        this.words = new int[D][];
        this.labels = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            this.words[ii] = words[dd];
            this.labels[ii] = labels[dd];
        }
        this.labelDocIndices = new HashMap<Integer, Set<Integer>>();
        for (int d = 0; d < D; d++) {
            for (int ll : labels[d]) {
                Set<Integer> docs = this.labelDocIndices.get(ll);
                if (docs == null) {
                    docs = new HashSet<Integer>();
                }
                docs.add(d);
                this.labelDocIndices.put(ll, docs);
            }
        }

        int emptyDocCount = 0;
        this.numTokens = 0;
        this.labelFreqs = new int[L];
        for (int d = 0; d < D; d++) {
            if (labels[d].length == 0) {
                emptyDocCount++;
                continue;
            }
            this.numTokens += words[d].length;
            for (int ii = 0; ii < labels[d].length; ii++) {
                labelFreqs[labels[d][ii]]++;
            }
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # empty documents:\t" + emptyDocCount);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    public void test(int[][] newWords) {
        this.words = newWords;
        this.labels = null;
        this.D = this.words.length;

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    protected String getLabelString(int labelIdx) {
        return this.labelVocab.get(labelIdx) + " (" + this.labelFreqs[labelIdx] + ")";
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
            logln("Tree:\n" + this.printTree(10));
            logln("Tree structure:\n" + this.printTreeStructure());
        }
    }

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        this.nodes = new Node[L];
        this.root = new Node(treeBuilder.getTreeRoot().getContent(), 0, 0,
                new SparseCount(), null);
        Stack<TreeNode<TreeNode, Integer>> stack = new Stack<TreeNode<TreeNode, Integer>>();
        stack.add(treeBuilder.getTreeRoot());
        while (!stack.isEmpty()) {
            TreeNode<TreeNode, Integer> node = stack.pop();
            for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                stack.add(child);
            }

            int labelIdx = node.getContent();

            // parents
            Node gParent = null;
            if (!node.isRoot()) {
                TreeNode<TreeNode, Integer> parent = node.getParent();
                int parentLabelIdx = parent.getContent();
                gParent = nodes[parentLabelIdx];
            }

            // global node
            Node gNode = new Node(labelIdx,
                    node.getIndex(),
                    node.getLevel(),
                    new SparseCount(),
                    gParent);
            nodes[labelIdx] = gNode;
            if (gParent != null) {
                gParent.addChild(gNode.getIndex(), gNode);
            }
            if (node.isRoot()) {
                root = gNode;
            }
        }
        estimateEdgeWeights(); // estimate edge weights
    }

    public void estimateEdgeWeights() {
        this.inWeights = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            this.inWeights[ll] = new SparseVector();
        }
        int[] labelFreq = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int l : labels[dd]) {
                labelFreq[l]++;
            }
        }
        int maxLabelFreq = StatUtils.max(labelFreq);

        // pair frequencies
        for (int dd = 0; dd < D; dd++) {
            int[] docLabels = labels[dd];
            for (int ii = 0; ii < docLabels.length; ii++) {
                for (int jj = 0; jj < docLabels.length; jj++) {
                    if (ii == jj) {
                        continue;
                    }
                    double weight = this.inWeights[docLabels[jj]].get(docLabels[ii]);
                    this.inWeights[docLabels[jj]].set(docLabels[ii], weight + 1.0);
                }
            }
        }

        // root weights
        for (int l = 0; l < L; l++) {
            int lFreq = labelFreq[l];
            for (int ii : inWeights[l].getIndices()) {
                double weight = inWeights[l].get(ii) / lFreq;
                inWeights[l].set(ii, weight);
            }

            double selfWeight = (double) lFreq / maxLabelFreq;
            inWeights[l].set(L - 1, selfWeight);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }
        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];

        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
            this.x[d] = new int[words[d].length];
            this.docSwitches[d] = new DirMult(new double[]{hyperparams.get(A_0),
                hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();
            if (labels != null) { // if labels are given during training time
                updateMaskes(d);
            }
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- --- Initializing assignments. " + initState + " ...");
        }
        switch (initState) {
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
            logln("--- --- 1. Estimating topics using Labeled LDA");
            logln("--- --- 2. Forward sampling using the estimated topics");
        }
        int lda_burnin = 50;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        int lda_repInterval = 10;
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        LabeledLDA llda = new LabeledLDA();
        llda.setDebug(debug);
        llda.setVerbose(verbose);
        llda.setLog(false);
        llda.configure(folder,
                V, L, lda_alpha, lda_beta, InitialState.RANDOM, false,
                lda_burnin, lda_maxiter, lda_samplelag, lda_repInterval);
        llda.train(null, words, labels);
        try {
            File lldaZFile = new File(llda.getSamplerFolderPath(), basename + ".zip");
            if (lldaZFile.exists()) {
                llda.inputState(lldaZFile, true, false);
            } else {
                IOUtils.createFolder(llda.getSamplerFolderPath());
                llda.sample();
                llda.outputState(lldaZFile, true, false);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while initializing topics "
                    + "with Labeled LDA");
        }
        setLog(true);

        DirMult[] tps = llda.getTopicWordDistributions();
        for (int ll = 0; ll < L; ll++) {
            nodes[ll].topic = tps[ll].getDistribution();
        }

        if (verbose) {
            logln("--- Initializing assignments by sampling ...");
        }
        long eTime;
        if (sampleExact) {
            eTime = sampleXZsExact(!REMOVE, ADD, !REMOVE, ADD);
        } else {
            eTime = sampleXZsMH(!REMOVE, ADD, !REMOVE, ADD);
        }
        if (verbose) {
            logln("--- --- Elapsed time: " + eTime);
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        if (report) {
            if (this.wordVocab == null) {
                throw new RuntimeException("The word vocab has not been assigned yet");
            }

            if (this.labelVocab == null) {
                throw new RuntimeException("The label vocab has not been assigned yet");
            }
            IOUtils.createFolder(reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            // sampling x's and z's
            long sampleXZTime;
            if (sampleExact) {
                sampleXZTime = sampleXZsExact(REMOVE, ADD, REMOVE, ADD);
            } else {
                sampleXZTime = sampleXZsMH(REMOVE, ADD, REMOVE, ADD);
            }

            // sampling topics
            long sampleTopicTime = sampleTopics();

            // updating tree
            long updateTreeTime = 0;
            if (treeUpdated) {
                updateTreeTime = updateTree();
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t # tokens changed: " + numTokensChanged
                        + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                        + "\t # accepts: " + numAccepts
                        + " (" + MiscUtils.formatDouble((double) numAccepts / L) + ")"
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
                logln("--- Elapsed time: sXZs: " + sampleXZTime
                        + "\tsTopic: " + sampleTopicTime
                        + "\tuTree: " + updateTreeTime);
                System.out.println();
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, getIteratedStateFile()), true, false);
                outputGlobalTree(new File(reportFolderPath, getIteratedTopicFile()), 20);
            }
        }

        if (report) {
            outputState(new File(reportFolderPath, getIteratedStateFile()), true, false);
            outputGlobalTree(new File(reportFolderPath, getIteratedTopicFile()), 20);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Update the structure of the tree. This is done by
     *
     * (1) proposing a new parent node for each node in the tree using the edge
     * weights of the background graph,
     *
     * (2) accepting or rejecting the proposed new parent using
     * Metropolis-Hastings.
     */
    private long updateTree() {
        long sTime = System.currentTimeMillis();

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("--- Updating tree ...");
        }

        numAccepts = 0;
        for (int ll = 0; ll < L; ll++) {

            Node node = nodes[ll];
            if (node.isRoot()) {
                continue;
            }

            // current and new parents
            Node currentParent = node.getParent();
            Node proposeParent = proposeParent(node);

            if (proposeParent.equals(node.getParent())) { // if the same node, move on
                numAccepts++;
                continue;
            }

            Set<Integer> subtreeDocs = getSubtreeDocumentIndices(node);
            Set<Integer> subtreeNodes = node.getSubtree();

            // current x & z log prob
            double curXLogprob = 0.0;
            double curZLogprob = 0.0;
            for (int d : subtreeDocs) {
                curXLogprob += docSwitches[d].getLogLikelihood();
                curZLogprob += computeDocLabelLogprob(docLabelCounts[d], docMaskes[d]);
            }

            // phi
            double curPhiLogprob = computeWordLogprob(node, currentParent);
            double newPhiLogprob = computeWordLogprob(node, proposeParent);
            double newXLogprob = 0.0;
            double newZLogprob = 0.0;
            HashMap<Integer, Set<Integer>> proposedMasks = new HashMap<Integer, Set<Integer>>();
            for (int d : subtreeDocs) {
                Set<Integer> proposedMask = getProposedMask(d, node.id, subtreeNodes, proposeParent);
                newZLogprob += computeDocLabelLogprob(docLabelCounts[d], proposedMask);
                proposedMasks.put(d, proposedMask);

                int[] proposedSwitchCount = new int[2];
                for (int n = 0; n < words[d].length; n++) {
                    if (proposedMask.contains(z[d][n])) {
                        proposedSwitchCount[INSIDE]++;
                    } else {
                        proposedSwitchCount[OUTSIDE]++;
                    }
                }
                newXLogprob += SamplerUtils.computeLogLhood(proposedSwitchCount,
                        words[d].length, switchPrior);
            }

            double curLogprob = curPhiLogprob + curXLogprob + curZLogprob;
            double newLogprob = newPhiLogprob + newXLogprob + newZLogprob;
            double mhRatio = Math.exp(newLogprob - curLogprob);

            if (rand.nextDouble() < mhRatio) {
                numAccepts++;

                // update parent
                currentParent.removeChild(node.getIndex());
                int newIndex = proposeParent.getNextChildIndex();
                node.setIndex(newIndex);
                proposeParent.addChild(newIndex, node);
                node.setParent(proposeParent);

                // update level of nodes in the subtree
                for (int n : subtreeNodes) {
                    nodes[n].setLevel(nodes[n].getLevel()
                            - currentParent.getLevel()
                            + proposeParent.getLevel());
                }

                // remove current switch assignments
                for (int d : subtreeDocs) {
                    docMaskes[d] = proposedMasks.get(d);
                    for (int n = 0; n < words[d].length; n++) {
                        docSwitches[d].decrement(x[d][n]); // decrement

                        // update
                        if (docMaskes[d].contains(z[d][n])) {
                            x[d][n] = INSIDE;
                        } else {
                            x[d][n] = OUTSIDE;
                        }
                        docSwitches[d].increment(x[d][n]); // increment
                    }
                }
            }

            if (debug) {
                validate("Update node " + ll + ". " + node.toString());
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    private double computeWordLogprob(Node node, Node parent) {
        SparseCount obs = new SparseCount();
        for (int v : node.getContent().getIndices()) {
            obs.changeCount(v, node.getContent().getCount(v));
        }
        for (int v : node.pseudoCounts.getIndices()) {
            obs.changeCount(v, node.pseudoCounts.getCount(v));
        }
        return SamplerUtils.computeLogLhood(obs, parent.topic, hyperparams.get(BETA));
    }

    /**
     * Compute the log probability of the topic assignments of a document given
     * a candidate set.
     *
     * @param docLabelCount Store the number of times tokens in this document
     * assigned to each topic
     * @param docMask The candidate set
     */
    private double computeDocLabelLogprob(SparseCount docLabelCount, Set<Integer> docMask) {
        double priorVal = hyperparams.get(ALPHA);
        double logGammaPriorVal = SamplerUtils.logGammaStirling(priorVal);

        double insideLp = 0.0;
        insideLp += SamplerUtils.logGammaStirling(priorVal * docMask.size());
        insideLp -= docMask.size() * logGammaPriorVal;

        double outsideLp = 0.0;
        outsideLp += SamplerUtils.logGammaStirling(priorVal * (L - docMask.size()));
        outsideLp -= (L - docMask.size()) * logGammaPriorVal;

        int insideCountSum = 0;
        int outsideCountSum = 0;
        for (int ll : docLabelCount.getIndices()) {
            int count = docLabelCount.getCount(ll);

            if (docMask.contains(ll)) {
                insideLp += SamplerUtils.logGammaStirling(count + priorVal);
                insideCountSum += count;
            } else {
                outsideLp += SamplerUtils.logGammaStirling(count + priorVal);
                outsideCountSum += count;
            }
        }

        insideLp -= SamplerUtils.logGammaStirling(insideCountSum + priorVal * docMask.size());
        outsideLp -= SamplerUtils.logGammaStirling(outsideCountSum + priorVal * (L - docMask.size()));

        double logprob = insideLp + outsideLp;
        return logprob;
    }

    /**
     * Return the set of documents whose label set contains any label in the
     * subtree rooted at a given node.
     *
     * @param node The root of the subtree
     */
    private Set<Integer> getSubtreeDocumentIndices(Node node) {
        Set<Integer> docIdx = new HashSet<Integer>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(node);
        while (!stack.isEmpty()) {
            Node n = stack.pop();

            for (Node c : n.getChildren()) {
                stack.add(c);
            }

            for (int d : this.labelDocIndices.get(n.id)) {
                docIdx.add(d);
            }
        }
        return docIdx;
    }

    /**
     * Return the set of mask node if the subtree root node become a child of
     * the a proposed parent node.
     *
     * @param d Document index
     * @param subtreeRoot The ID of the root of the subtree
     * @param subtree Set of nodes in the subtree
     * @param proposedParent The proposed parent node
     */
    private Set<Integer> getProposedMask(int d,
            int subtreeRoot,
            Set<Integer> subtree,
            Node proposedParent) {
        Set<Integer> ppMask = new HashSet<Integer>();
        boolean insideSubtree = false;
        for (int label : labels[d]) {
            Node n = nodes[label];

            // if this label is inside the subtree, add all nodes from the label
            // node to the subtree root to the mask
            if (subtree.contains(label)) {
                while (n.id != subtreeRoot) {
                    ppMask.add(n.id);
                    n = n.getParent();
                }
                ppMask.add(subtreeRoot);
                insideSubtree = true;
            } // if this label is outside the subtree, all all nodes from the label
            // node to the root as usual
            else {
                while (n != null) {
                    ppMask.add(n.id);
                    n = n.getParent();
                }
            }
        }

        // if there is any label inside the subtree, add nodes from the proposed
        // label to the root
        if (insideSubtree) {
            Node n = nodes[proposedParent.id];
            while (n != null) {
                ppMask.add(n.id);
                n = n.getParent();
            }
        }
        return ppMask;
    }

    /**
     * Propose a parent node for a given node by sampling from the prior weights
     * (the MLE conditional probabilities)
     *
     * @param node A node
     */
    private Node proposeParent(Node node) {
        // sort candidate parents
        ArrayList<RankingItem<Node>> rankCandNodes = new ArrayList<RankingItem<Node>>();
        for (int idx : inWeights[node.id].getIndices()) {
            Node candNode = nodes[idx];
            if (node.isDescendent(candNode)) {
                continue;
            }
            double weight = inWeights[node.id].get(idx);
            rankCandNodes.add(new RankingItem<Node>(nodes[idx], weight));
        }
        Collections.sort(rankCandNodes);

        // sample from a limited set
        ArrayList<Node> candNodes = new ArrayList<Node>();
        ArrayList<Double> candWeights = new ArrayList<Double>();
        int numCands = Math.min(10, rankCandNodes.size());
        for (int ii = 0; ii < numCands; ii++) {
            RankingItem<Node> rankNode = rankCandNodes.get(ii);
            candNodes.add(rankNode.getObject());
            candWeights.add(rankNode.getPrimaryValue());
        }
        int sampledIdx = SamplerUtils.scaleSample(candWeights);
        Node sampledNode = candNodes.get(sampledIdx);
        return sampledNode;
    }

    /**
     * Sample x and z together for all documents.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private long sampleXZsExact(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        numTokensChanged = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleXZExact(d, n, removeFromModel, addToModel, removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample x and z together for a token.
     *
     * @param d
     * @param n
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private void sampleXZExact(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            nodes[z[d][n]].getContent().decrement(words[d][n]);
            nodes[z[d][n]].removeToken(d, n);
        }
        if (removeFromData) {
            docSwitches[d].decrement(x[d][n]);
            docLabelCounts[d].decrement(z[d][n]);
        }

        double[] logprobs = new double[L];
        for (int ll = 0; ll < L; ll++) {
            boolean inside = docMaskes[d].contains(ll);
            double xLlh;
            double zLlh;
            double wLlh = Math.log(nodes[ll].topic[words[d][n]]);

            if (inside) {
                xLlh = Math.log(docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size()));
            } else {
                xLlh = Math.log(docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0));
                zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(OUTSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size())));
            }
            logprobs[ll] = xLlh + zLlh + wLlh;
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledZ != z[d][n]) {
            numTokensChanged++;
        }
        z[d][n] = sampledZ;
        if (docMaskes[d].contains(z[d][n])) {
            x[d][n] = INSIDE;
        } else {
            x[d][n] = OUTSIDE;
        }

        if (addToModel) {
            nodes[z[d][n]].getContent().increment(words[d][n]);
            nodes[z[d][n]].addToken(d, n);
        }

        if (addToData) {
            docSwitches[d].increment(x[d][n]);
            docLabelCounts[d].increment(z[d][n]);
        }
    }

    /**
     * Sample x and z. This is done by first sampling x, and given the value of
     * x, sample z. This is an approximation of sampleXZsExact.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private long sampleXZsMH(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        numTokensChanged = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                sampleXZMH(d, n, removeFromModel, addToModel, removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample x and z. This is done by first sampling x, and given the value of
     * x, sample z. This is an approximation of sampleXZsExact.
     *
     * @param d
     * @param n
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    private void sampleXZMH(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromModel) {
            nodes[z[d][n]].getContent().decrement(words[d][n]);
            nodes[z[d][n]].removeToken(d, n);
        }
        if (removeFromData) {
            docSwitches[d].decrement(x[d][n]);
            docLabelCounts[d].decrement(z[d][n]);
        }

        // propose
        int pX;
        if (!docMaskes[d].isEmpty()) {
            double[] ioLogProbs = new double[2];
            ioLogProbs[INSIDE] = docSwitches[d].getCount(INSIDE) + hyperparams.get(A_0);
            ioLogProbs[OUTSIDE] = docSwitches[d].getCount(OUTSIDE) + hyperparams.get(B_0);
            pX = SamplerUtils.scaleSample(ioLogProbs);
        } else { // if candidate set is empty
            pX = OUTSIDE;
        }
        int pZ = proposeZ(d, n, pX);

        // compute MH ratio: accept all for now
        if (pZ != z[d][n]) {
            numTokensChanged++;
        }
        z[d][n] = pZ;
        x[d][n] = pX;

        // accept or reject
        if (docMaskes[d].contains(z[d][n])) {
            x[d][n] = INSIDE;
        } else {
            x[d][n] = OUTSIDE;
        }

        if (addToModel) {
            nodes[z[d][n]].getContent().increment(words[d][n]);
            nodes[z[d][n]].addToken(d, n);
        }

        if (addToData) {
            docSwitches[d].increment(x[d][n]);
            docLabelCounts[d].increment(z[d][n]);
        }
    }

    /**
     * Sample a node for a token given the binary indicator x.
     *
     * @param d Document index
     * @param n Token index
     * @param pX Binary indicator specifying whether the given token should be
     * assigned to a node in the candidate set or not.
     */
    private int proposeZ(int d, int n, int pX) {
//        ArrayList<Integer> indices = new ArrayList<Integer>();
//        ArrayList<Double> logprobs = new ArrayList<Double>();
        Double[] logProbsArray = new Double[L];
        Integer[] indexArray = new Integer[L];
        int idx = 0;
        if (pX == INSIDE) {
            for (int ll : docMaskes[d]) {
                double zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * docMaskes[d].size()));
                double wLlh = Math.log(nodes[ll].topic[words[d][n]]);
                // logprobs.add(zLlh + wLlh);
                // indices.add(ll);
                logProbsArray[idx] = zLlh + wLlh;
                indexArray[idx ++] = ll;
            }
        } else {
            for (int ll = 0; ll < L; ll++) {
                if (docMaskes[d].contains(ll)) {
                    continue;
                }
                double zLlh = Math.log((docLabelCounts[d].getCount(ll) + hyperparams.get(ALPHA))
                        / (docSwitches[d].getCount(INSIDE) + hyperparams.get(ALPHA) * (L - docMaskes[d].size())));
                double wLlh = Math.log(nodes[ll].topic[words[d][n]]);
                // logprobs.add(zLlh + wLlh);
                // indices.add(ll);
                logProbsArray[idx] = zLlh + wLlh;
                indexArray[idx ++] = ll;
            }
        }
        ArrayList<Double> logprobs = new ArrayList<Double>(Arrays.asList(Arrays.copyOfRange(logProbsArray, 0, idx)));
        ArrayList<Integer> indices = new ArrayList<Integer>(Arrays.asList(Arrays.copyOfRange(indexArray, 0, idx)));
        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
        return indices.get(sampledIdx);
    }

    /**
     * Sample topics (distributions over words) in the tree. This is done by (1)
     * performing a bottom-up smoothing to compute the pseudo-counts from
     * children for each node, and (2) top-down sampling to get the topics.
     */
    private long sampleTopics() {
        if (verbose && iter % REP_INTERVAL == 0) {
            logln("--- Sampling topics ...");
        }
        long sTime = System.currentTimeMillis();
        // get all leaves of the tree
        ArrayList<Node> leaves = new ArrayList<Node>();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getChildren().isEmpty()) {
                leaves.add(node);
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // bottom-up smoothing to compute pseudo-counts from children
        Queue<Node> queue = new LinkedList<Node>();
        for (Node leaf : leaves) {
            queue.add(leaf);
        }
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            Node parent = node.getParent();
            if (!node.isRoot() && !queue.contains(parent)) {
                queue.add(parent);
            }
            if (node.isLeaf()) {
                continue;
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
        queue = new LinkedList<Node>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            for (Node child : node.getChildren()) {
                queue.add(child);
            }
            node.sampleTopic();
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Update the set of candidate labels for a document d. This set is defined
     * based on the set of actual document labels and the topic tree structure.
     *
     * @param d Document index
     */
    private void updateMaskes(int d) {
        if (labels[d].length > 0) {
            this.docMaskes[d] = new HashSet<Integer>();
            for (int label : labels[d]) {
                Node node = nodes[label];
                while (node != null) {
                    docMaskes[d].add(node.id);
                    node = node.getParent();
                }
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int ll = 0; ll < L; ll++) {
            wordLlh += nodes[ll].getWordLogLikelihood();
        }

        double docSwitchesLlh = 0.0;
        for (int dd = 0; dd < D; dd++) {
            docSwitchesLlh += docSwitches[dd].getLogLikelihood();
        }

        double docLabelLlh = 0.0;
        for (int dd = 0; dd < D; dd++) {
            // inside
            if (!docMaskes[dd].isEmpty()) {
                int[] insideCounts = new int[docMaskes[dd].size()];
                int insideCountSum = 0;
                int ii = 0;
                for (int ll : docMaskes[dd]) {
                    insideCounts[ii++] = docLabelCounts[dd].getCount(ll);
                    insideCountSum += docLabelCounts[dd].getCount(ll);
                }
                double insideLlh = SamplerUtils.computeLogLhood(insideCounts,
                        insideCountSum, hyperparams.get(ALPHA));
                docLabelLlh += insideLlh;
            }

            // outside
            int[] outsideCounts = new int[L - docMaskes[dd].size()];
            int outsideCountSum = 0;
            int ii = 0;
            for (int ll = 0; ll < L; ll++) {
                if (docMaskes[dd].contains(ll)) {
                    continue;
                }
                outsideCounts[ii++] = docLabelCounts[dd].getCount(ll);
                outsideCountSum += docLabelCounts[dd].getCount(ll);
            }
            double outsideLlh = SamplerUtils.computeLogLhood(outsideCounts,
                    outsideCountSum, hyperparams.get(ALPHA));
            docLabelLlh += outsideLlh;
        }

        double treeLp = 0.0;
        for (int ll = 0; ll < L; ll++) {
            Node node = nodes[ll];
            if (node.isRoot()) {
                continue;
            }
            treeLp += Math.log(inWeights[ll].get(node.getParent().id));
        }

        logln(">>> >>> word: " + MiscUtils.formatDouble(wordLlh)
                + ". switch: " + MiscUtils.formatDouble(docSwitchesLlh)
                + ". label: " + MiscUtils.formatDouble(docLabelLlh)
                + ". tree: " + MiscUtils.formatDouble(treeLp));
        return wordLlh + docSwitchesLlh + docLabelLlh + treeLp;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> tParams) {
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath();
    }

    @Override
    public void validate(String msg) {
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        int numNodes = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            numNodes++;
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.validate(msg);
        }

        if (numNodes != L) {
            throw new RuntimeException(msg + ". Number of connected nodes: "
                    + numNodes + ". L = " + L);
        }

        for (int d = 0; d < D; d++) {
            docSwitches[d].validate(msg);
            docLabelCounts[d].validate(msg);

            if (labels[d].length > 0) {
                HashSet<Integer> tempDocMask = new HashSet<Integer>();
                for (int label : labels[d]) {
                    Node node = nodes[label];
                    while (node != null) {
                        tempDocMask.add(node.id);
                        node = node.getParent();
                    }
                }

                if (tempDocMask.size() != docMaskes[d].size()) {
                    for (int ll : labels[d]) {
                        System.out.println("label " + ll + "\t" + nodes[ll].toString());
                    }
                    System.out.println();
                    for (int ii : tempDocMask) {
                        System.out.println("true " + ii + "\t" + nodes[ii].toString());
                    }
                    System.out.println();
                    for (int ii : docMaskes[d]) {
                        System.out.println("actu " + ii + "\t" + nodes[ii].toString());
                    }
                    throw new RuntimeException(msg + ". Mask sizes mismatch. "
                            + tempDocMask.size() + " vs. " + docMaskes[d].size()
                            + " in document " + d);
                }
            }
        }
    }

    @Override
    public void outputState(String filepath) {
        outputState(filepath, true, true);
    }

    /**
     * Output current state.
     *
     * @param filepath Output file
     * @param outputModel Whether to output the model
     * @param outputData Whether to output the assignments
     */
    public void outputState(File filepath, boolean outputModel, boolean outputData) {
        this.outputState(filepath.getAbsolutePath(), outputModel, outputData);
    }

    /**
     * Output current state.
     *
     * @param filepath Output file
     * @param outputModel Whether to output the model
     * @param outputData Whether to output the assignments
     */
    public void outputState(String filepath, boolean outputModel, boolean outputData) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
            logln("--- --- Outputing model? " + outputModel);
            logln("--- --- Outputing assignments? " + outputData);
        }
        try {
            // model
            String modelStr = null;
            if (outputModel) {
                StringBuilder mStr = new StringBuilder();
                for (int k = 0; k < nodes.length; k++) {
                    Node node = nodes[k];
                    mStr.append(k)
                            .append("\t").append(node.getIndex())
                            .append("\t").append(node.getLevel())
                            .append("\n");
                    for (int v = 0; v < node.topic.length; v++) {
                        mStr.append(node.topic[v]).append("\t");
                    }
                    mStr.append("\n");
                }

                for (int k = 0; k < nodes.length; k++) {
                    mStr.append(k);
                    if (nodes[k].getParent() == null) {
                        mStr.append("\t-1\n");
                    } else {
                        mStr.append("\t").append(nodes[k].getParent().id).append("\n");
                    }
                }

                for (int k = 0; k < nodes.length; k++) {
                    mStr.append(k).append("\t")
                            .append(SparseVector.output(inWeights[k])).append("\n");
                }
                modelStr = mStr.toString();
            }

            // data
            String assignStr = null;
            if (outputData) {
                StringBuilder asgnS = new StringBuilder();
                for (int d = 0; d < D; d++) {
                    asgnS.append(d).append("\n");
                    asgnS.append(DirMult.output(docSwitches[d])).append("\n");
                    asgnS.append(SparseCount.output(docLabelCounts[d])).append("\n");
                    for (int n = 0; n < words[d].length; n++) {
                        asgnS.append(z[d][n]).append("\t");
                    }
                    asgnS.append("\n");
                    for (int n = 0; n < words[d].length; n++) {
                        asgnS.append(x[d][n]).append("\t");
                    }
                    asgnS.append("\n");
                }
                assignStr = asgnS.toString();
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr, assignStr);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing state to "
                    + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        inputState(filepath, true, true);
    }

    /**
     * Input model state.
     *
     * @param filepath Output file
     * @param inputModel Whether to input the model
     * @param inputData Whether to input the assignments
     */
    public void inputState(File filepath, boolean inputModel, boolean inputData) {
        this.inputState(filepath.getAbsolutePath(), inputModel, inputData);
    }

    /**
     * Input model state.
     *
     * @param filepath Output file
     * @param inputModel Whether to input the model
     * @param inputData Whether to input the assignments
     */
    public void inputState(String filepath, boolean inputModel, boolean inputData) {
        if (verbose) {
            logln("--- Inputing state to " + filepath);
            logln("--- --- Inputing model? " + inputModel);
            logln("--- --- Inputing assignments? " + inputData);
        }
        try {
            if (inputModel) {
                inputModel(filepath);
            }
            if (inputData) {
                inputAssignments(filepath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing from " + filepath);
        }
    }

    /**
     * Input learned model.
     *
     * @param zipFilepath Input file
     */
    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            // initialize
            this.nodes = new Node[L];
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);

            for (int k = 0; k < L; k++) {
                // node ids
                String[] sline = reader.readLine().split("\t");
                int id = Integer.parseInt(sline[0]);
                if (id != k) {
                    throw new RuntimeException("Mismatch");
                }
                int idx = Integer.parseInt(sline[1]);
                int level = Integer.parseInt(sline[2]);
                nodes[k] = new Node(id, idx, level, null, null);

                // node topic
                sline = reader.readLine().split("\t");
                double[] topic = new double[V];
                if (sline.length != V) {
                    throw new RuntimeException("Mismatch: " + sline.length + " vs. " + V);
                }
                for (int v = 0; v < V; v++) {
                    topic[v] = Double.parseDouble(sline[v]);
                }
                nodes[k].topic = topic;
            }

            // tree structure
            for (int k = 0; k < L; k++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != k) {
                    throw new RuntimeException("Mismatch");
                }
                int parentId = Integer.parseInt(sline[1]);
                if (parentId == -1) {
                    root = nodes[k];
                } else {
                    nodes[k].setParent(nodes[parentId]);
                    nodes[parentId].addChild(nodes[k].getIndex(), nodes[k]);
                }
            }
            for (int k = 0; k < L; k++) {
                nodes[k].fillInactiveChildIndices();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }

        if (verbose) {
            logln("--- Model loaded.\n" + printTreeStructure());
        }
    }

    /**
     * Input assignments.
     *
     * @param zipFilepath Input file
     */
    private void inputAssignments(String zipFilepath) {
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
                docSwitches[d] = DirMult.input(reader.readLine());
                docLabelCounts[d] = SparseCount.input(reader.readLine());
                String[] sline = reader.readLine().trim().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("Mismatch");
                }
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
                sline = reader.readLine().trim().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("Mismatch");
                }
                for (int n = 0; n < words[d].length; n++) {
                    x[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Print out the structure (number of observations at each level) of the
     * global tree.
     *
     * @return
     */
    public String printTreeStructure() {
        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        SparseCount nodeCountPerLevel = new SparseCount(); // nodes
        SparseCount obsCountPerLevel = new SparseCount();  // observation

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            // node
            nodeCountPerLevel.increment(node.getLevel());
            // observation
            if (node.getContent() != null) {
                obsCountPerLevel.changeCount(node.getLevel(),
                        node.getContent().getCountSum());
            }
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        for (int level : nodeCountPerLevel.getSortedIndices()) {
            double tokenRatio = (double) obsCountPerLevel.getCount(level)
                    / nodeCountPerLevel.getCount(level);
            str.append(">>> level ").append(level)
                    .append(". n: ").append(nodeCountPerLevel.getCount(level))
                    .append(". o: ").append(obsCountPerLevel.getCount(level))
                    .append(" (").append(MiscUtils.formatDouble(tokenRatio))
                    .append(")").append("\n");
        }
        str.append(">>> >>> # nodes: ").append(nodeCountPerLevel.getCountSum()).append("\n");
        str.append(">>> >>> # obs  : ").append(obsCountPerLevel.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * Print out the global tree.
     *
     * @param numTopWords Number of top words shown for each topic
     * @return
     */
    public String printTree(int numTopWords) {
        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        int totalObs = 0;
        int numNodes = 0;
        SparseCount nodeCountPerLevel = new SparseCount(); // nodes
        SparseCount obsCountPerLevel = new SparseCount();  // observations

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            numNodes++;
            // node
            nodeCountPerLevel.increment(node.getLevel());

            // observation
            if (node.getContent() != null) {
                obsCountPerLevel.changeCount(node.getLevel(),
                        node.getContent().getCountSum());
                totalObs += node.getContent().getCountSum();
            }

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }

            String[] topWords = null;
            if (node.topic != null) {
                topWords = getTopWords(node.topic, numTopWords);
            }

            str.append(node.toString()).append(", ")
                    .append(getLabelString(node.id))
                    .append(" ").append(node.getContent() == null ? "" : node.getContent().getCountSum())
                    .append(" ").append(topWords == null ? "" : Arrays.toString(topWords))
                    .append("\n\n");

            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append(">>> # observations = ").append(totalObs)
                .append("\n>>> # nodes = ").append(numNodes)
                .append("\n");
        for (int level : nodeCountPerLevel.getSortedIndices()) {
            str.append(">>> level ").append(level)
                    .append(". n: ").append(nodeCountPerLevel.getCount(level))
                    .append(". o: ").append(obsCountPerLevel.getCount(level))
                    .append("\n");
        }
        return str.toString();
    }

    /**
     * Write global tree to output file.
     *
     * @param outputFile
     * @param numTopWords
     */
    public void outputGlobalTree(File outputFile, int numTopWords) {
        if (verbose) {
            logln("Outputing global tree to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(this.printTree(numTopWords));
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    // ******************* Start prediction ************************************
    /**
     * Sample test documents in parallel.
     *
     * @param newWords Test document
     * @param iterPredFolder Folder to store predictions using different models
     * @param sampler The configured sampler
     * @param initPredictions Initial predictions from TF-IDF
     * @param topK The number of nearest neighbors to be initially included in
     * the candidate set
     */
    public static void parallelTest(int[][] newWords, File iterPredFolder, L2H sampler,
            double[][] initPredictions, int topK) {
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

                // folder contains multiple samples during test using a learned model
                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");

                L2HTestRunner runner = new L2HTestRunner(sampler,
                        newWords, stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath(),
                        initPredictions, topK);
                Thread thread = new Thread(runner);
                threads.add(thread);
            }
            runThreads(threads); // run MAX_NUM_PARALLEL_THREADS threads at a time
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }

    /**
     * Perform sampling for test documents to predict labels.
     *
     * @param stateFile File containing learned model
     * @param newWords Test document
     * @param outputResultFile
     * @param initPredictions
     * @param topK
     */
    public void sampleNewDocuments(String stateFile,
            int[][] newWords,
            String outputResultFile,
            double[][] initPredictions,
            int topK) {
        if (verbose) {
            System.out.println();
            logln("Perform prediction using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        // test data
        this.words = newWords;
        this.labels = null;
        this.D = this.words.length;

        // initialize data structure
        this.z = new int[D][];
        this.x = new int[D][];
        this.docSwitches = new DirMult[D];
        this.docLabelCounts = new SparseCount[D];
        this.docMaskes = new Set[D];

        for (int d = 0; d < D; d++) {
            this.z[d] = new int[words[d].length];
            this.x[d] = new int[words[d].length];
            this.docSwitches[d] = new DirMult(
                    new double[]{hyperparams.get(A_0), hyperparams.get(B_0)});
            this.docLabelCounts[d] = new SparseCount();
            this.docMaskes[d] = new HashSet<Integer>();

            // Hui Chen: Set<Integer> cands = getCandidates(initPredictions[d], topK); -> 2 statements
            // int actualK = topK <= initPredictions[d].length ? topK : initPredictions[d].length;
            // Set<Integer> cands = getCandidates(initPredictions[d], actualK);
            Set<Integer> cands = getCandidates(initPredictions[d], topK);
            for (int label : cands) {
                Node node = nodes[label];
                while (node != null) {
                    docMaskes[d].add(node.id);
                    node = node.getParent();
                }
            }
        }

        // initialize: sampling using global distribution over labels
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
        double[][] predictedScores = new double[D][L - 1]; // exclude root
        int count = 0;
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId()
                        + "\t" + getSamplerFolderPath());
            }
            if (iter == 0) {
                sampleXZsMH(!REMOVE, !ADD, !REMOVE, ADD);
            } else {
                sampleXZsMH(!REMOVE, !ADD, REMOVE, ADD);
            }

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                for (int dd = 0; dd < D; dd++) {
                    for (int ll = 0; ll < L - 1; ll++) {
                        predictedScores[dd][ll]
                                += (double) docLabelCounts[dd].getCount(ll) / words[dd].length;
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
            for (int ll = 0; ll < L - 1; ll++) {
                predictedScores[dd][ll] /= count;
            }
        }
        PredictionUtils.outputSingleModelClassifications(new File(outputResultFile),
                predictedScores);
    }

    /**
     * TODO: add other ways to get the candidate set.
     */
    private Set<Integer> getCandidates(double[] scores, int topK) {
        Set<Integer> cands = new HashSet<Integer>();
        ArrayList<RankingItem<Integer>> docRankLabels = MiscUtils.getRankingList(scores);
        System.out.println("scores.length: " + scores.length + " docRankLabels: " + docRankLabels.size());
        for (int ii = 0; ii < topK; ii++) {
            cands.add(docRankLabels.get(ii).getObject());
        }
        return cands;
    }
    // ******************* End prediction **************************************

    public class Node extends TreeNode<Node, SparseCount> {

        final int id;
        double[] topic;
        SparseCount pseudoCounts;
        HashMap<Integer, ArrayList<Integer>> assignedTokens;

        Node(int id, int index, int level,
                SparseCount content,
                Node parent) {
            super(index, level, content, parent);
            this.id = id;
            this.pseudoCounts = new SparseCount();
            this.assignedTokens = new HashMap<Integer, ArrayList<Integer>>();
        }
        
        public int getId() {
        	return id;
        }

        public Set<Integer> getSubtree() {
            Set<Integer> subtree = new HashSet<Integer>();
            Stack<Node> stack = new Stack<Node>();
            stack.add(this);
            while (!stack.isEmpty()) {
                Node n = stack.pop();
                for (Node c : n.getChildren()) {
                    stack.add(c);
                }
                subtree.add(n.id);
            }
            return subtree;
        }

        public Set<Integer> getAssignedDocuments() {
            return this.assignedTokens.keySet();
        }

        public HashMap<Integer, ArrayList<Integer>> getAssignedTokens() {
            return this.assignedTokens;
        }

        public void addToken(int d, int n) {
            ArrayList<Integer> docAssignedTokens = this.assignedTokens.get(d);
            if (docAssignedTokens == null) {
                docAssignedTokens = new ArrayList<Integer>();
            }
            docAssignedTokens.add(n);
            this.assignedTokens.put(d, docAssignedTokens);
        }

        public void removeToken(int d, int n) {
            this.assignedTokens.get(d).remove(Integer.valueOf(n));
            if (this.assignedTokens.get(d).isEmpty()) {
                this.assignedTokens.remove(d);
            }
        }

        public double[] getTopic() {
            return this.topic;
        }

        public void setTopic(double[] t) {
            this.topic = t;
        }

        /**
         * Return true if the given node is in the subtree rooted at this node
         *
         * @param node The given node to check
         */
        public boolean isDescendent(Node node) {
            Node temp = node;
            while (temp != null) {
                if (temp.equals(this)) {
                    return true;
                }
                temp = temp.parent;
            }
            return false;
        }

        public void getPseudoCountsFromChildren() {
            if (pathAssumption == PathAssumption.MINIMAL) {
                this.getPseudoCountsFromChildrenMin();
            } else if (pathAssumption == PathAssumption.MAXIMAL) {
                this.getPseudoCountsFromChildrenMax();
            } else {
                throw new RuntimeException("Path assumption " + pathAssumption
                        + " is not supported.");
            }
        }

        /**
         * Propagate the observations from all children nodes to this node using
         * minimal path assumption, which means for each observation type v, a
         * child node will propagate a value of 1 if it contains v, and 0
         * otherwise.
         */
        public void getPseudoCountsFromChildrenMin() {
            this.pseudoCounts = new SparseCount();
            for (Node child : this.getChildren()) {
                SparseCount childObs = child.getContent();
                for (int obs : childObs.getIndices()) {
                    this.pseudoCounts.increment(obs);
                }
            }
        }

        /**
         * Propagate the observations from all children nodes to this node using
         * maximal path assumption, which means that each child node will
         * propagate its full observation vector.
         */
        public void getPseudoCountsFromChildrenMax() {
            this.pseudoCounts = new SparseCount();
            for (Node child : this.getChildren()) {
                SparseCount childObs = child.getContent();
                for (int obs : childObs.getIndices()) {
                    this.pseudoCounts.changeCount(obs, childObs.getCount(obs));
                }
            }
        }

        /**
         * Sample topic. This applies since the topic of a node is modeled as a
         * drawn from a Dirichlet distribution with the mean vector is the topic
         * of the node's parent and scaling factor gamma.
         *
         * @param beta Topic smoothing parameter
         * @param gamma Dirichlet-Multinomial chain parameter
         */
        public void sampleTopic() {
            double[] meanVector = new double[V];
            Arrays.fill(meanVector, hyperparams.get(BETA) / V); // to prevent zero count

            // from parent
            if (!this.isRoot()) {
                double[] parentTopic = parent.topic;
                for (int v = 0; v < V; v++) {
                    meanVector[v] += parentTopic[v] * hyperparams.get(BETA);
                }
            } else {
                for (int v = 0; v < V; v++) {
                    meanVector[v] += hyperparams.get(BETA) / V;
                }
            }

            // current observations
            SparseCount observations = this.content;
            for (int obs : observations.getIndices()) {
                meanVector[obs] += observations.getCount(obs);
            }

            // from children
            for (int obs : this.pseudoCounts.getIndices()) {
                meanVector[obs] += this.pseudoCounts.getCount(obs);
            }

            Dirichlet dir = new Dirichlet(meanVector);
            double[] ts = dir.nextDistribution();

            if (debug) {
                for (int v = 0; v < V; v++) {
                    if (ts[v] == 0) {
                        throw new RuntimeException("Zero probability");
                    }
                }
            }
            this.setTopic(ts);
        }

        public double getWordLogLikelihood() {
            double llh = 0.0;
            for (int w : getContent().getIndices()) {
                llh += getContent().getCount(w) * Math.log(topic[w]);
            }
            return llh;
        }

        public void validate(String msg) {
            int numTokens = 0;
            for (int dd : this.assignedTokens.keySet()) {
                numTokens += this.assignedTokens.get(dd).size();
            }

            if (numTokens != this.getContent().getCountSum()) {
                throw new RuntimeException(msg + ". Mismatch: " + numTokens
                        + " vs. " + this.getContent().getCountSum());
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[")
                    .append(id).append(", ")
                    .append(getPathString())
                    .append(", #c = ").append(getChildren().size())
                    .append(", #o = ").append(getContent().getCountSum())
                    .append("]");
            return str.toString();
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + L2H.class.getName() + " -help";
    }

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("format-file", "Formatted file name");
            addOption("output", "Output folder");

            // sampling configurations
            addSamplingOptions();

            // model parameters
            addOption("K", "Number of topics");
            addOption("numTopwords", "Number of top words per topic");
            addOption("min-label-freq", "Minimum label frequency");

            // model hyperparameters
            addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for topic distributions");
            addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                    + "for word distributions");
            addOption("a0", "a0");
            addOption("b0", "b0");
            addOption("path", "Path assumption");
            addOption("tree-init", "Tree initialization type");

            options.addOption("train", false, "Training");
            options.addOption("tree", false, "Whether the tree is updated or not");
            options.addOption("paramOpt", false, "Whether hyperparameter "
                    + "optimization using slice sampling is performed");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            runModel();
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    private static void runModel() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String formatFolder = cmd.getOptionValue("format-folder");
        String outputFolder = cmd.getOptionValue("output");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int minLabelFreq = CLIUtils.getIntegerArgument(cmd, "min-label-freq", 100);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 2);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 10);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 1000);
        double a0 = CLIUtils.getDoubleArgument(cmd, "a0", 90);
        double b0 = CLIUtils.getDoubleArgument(cmd, "b0", 10);
        boolean treeUpdate = cmd.hasOption("tree");
        boolean sampleExact = cmd.hasOption("exact");

        PathAssumption pathAssumption;
        String path = CLIUtils.getStringArgument(cmd, "path", "max");
        switch (path) {
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "antoniak":
                pathAssumption = PathAssumption.ANTONIAK;
                break;
            default:
                throw new RuntimeException(path + " path assumption is not"
                        + " supported. Use min or max.");
        }

        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        if (verbose) {
            //
            System.out.println("--- folder\t" + outputFolder);
            //System.out.println("--- label vocab:\t" + L); // knew it after loading data
            // System.out.println("--- word vocab:\t" + V);
            System.out.println("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            System.out.println("--- beta:\t" + MiscUtils.formatDouble(beta));
            System.out.println("--- a0:\t" + MiscUtils.formatDouble(a0));
            System.out.println("--- b0:\t" + MiscUtils.formatDouble(b0));
            System.out.println("--- burn-in:\t" + burnIn);
            System.out.println("--- max iter:\t" + maxIters);
            System.out.println("--- sample lag:\t" + sampleLag);
            //System.out.println("--- paramopt:\t" + paramOptimized);
            //System.out.println("--- initialize:\t" + initState);
            System.out.println("--- path assumption:\t" + pathAssumption);
            //System.out.println("--- tree builder:\t" + treeBuilder.getName());
            //System.out.println("--- updating tree?\t" + treeUpdated);
            System.out.println("--- exact sampling?\t" + sampleExact);
            //
            System.out.println("\nLoading formatted data ...");
        }
        LabelTextDataset data = new LabelTextDataset(datasetName);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(formatFolder);
        data.filterLabelsByFrequency(minLabelFreq);
        data.prepareTopicCoherence(numTopWords);

        int V = data.getWordVocab().size();
        int K = data.getLabelVocab().size();

        boolean paramOpt = cmd.hasOption("paramOpt");
        InitialState initState = InitialState.PRESET;

        String treeInit = CLIUtils.getStringArgument(cmd, "tree-init", "mst");
        AbstractTaxonomyBuilder treeBuilder;
        switch (treeInit) {
            case "mst":
                treeBuilder = new MSTBuilder(data.getLabels(), data.getLabelVocab());
                break;
            case "beta":
                double treeAlpha = CLIUtils.getDoubleArgument(cmd, "tree-alpha", 100);
                double treeA = CLIUtils.getDoubleArgument(cmd, "tree-a", 0.1);
                double treeB = CLIUtils.getDoubleArgument(cmd, "tree-b", 0.1);
                treeBuilder = new BetaTreeBuilder(data.getLabels(), data.getLabelVocab(),
                        treeAlpha, treeA, treeB);
                break;
            default:
                throw new RuntimeException(treeInit + " not supported");
        }

        File builderFolder = new File(outputFolder, treeBuilder.getName());
        IOUtils.createFolder(builderFolder);
        File treeFile = new File(builderFolder, "tree.txt");
        File labelVocFile = new File(builderFolder, "labels.voc");
        if (treeFile.exists()) {
            treeBuilder.inputTree(treeFile);
            treeBuilder.inputLabelVocab(labelVocFile);
        } else {
            treeBuilder.buildTree();
            treeBuilder.outputTree(treeFile);
            treeBuilder.outputLabelVocab(labelVocFile);
        }
        treeBuilder.outputTreeTemp(new File(treeFile + "-temp"));

        L2H sampler = new L2H();
        sampler.setReport(true);
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(data.getWordVocab());
        sampler.setLabelVocab(data.getLabelVocab());

        sampler.configure(outputFolder,
                data.getWordVocab().size(),
                alpha, beta,
                a0, b0,
                treeBuilder,
                treeUpdate,
                sampleExact,
                initState,
                pathAssumption,
                paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        sampler.train(null, data.getWords(), data.getLabels());
        sampler.initialize();
        sampler.iterate();
        sampler.outputGlobalTree(new File(samplerFolder, TopWordFile), numTopWords);
    }
}

class L2HTestRunner implements Runnable {

    L2H sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;
    double[][] initPredidctions;
    int topK;

    public L2HTestRunner(L2H sampler,
            int[][] newWords,
            String stateFile,
            String outputFile,
            double[][] initPreds,
            int topK) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
        this.initPredidctions = initPreds;
        this.topK = topK;
    }

    @Override
    public void run() {
        L2H testSampler = new L2H();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFile,
                    initPredidctions, topK);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
