/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.backup;

import cc.mallet.types.Dirichlet;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import sampler.LDA;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.FullTable;
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
public class RCRPSampler extends AbstractSampler {

    public static final int SELF_INDEX = -2;
    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    public static final String SEPARATOR = "#";
    public static final int ALPHA = 0; // concentration for local DPs
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // global tree's concentrations
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int L;
    protected int K;
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z; // local table index
    private RCRPNode globalTreeRoot;
    private Restaurant<RCRPTable, Integer, RCRPNode>[] localRestaurants;
    private int totalNumObservations = 0;
    private double[] uniform;
//    private DirichletMultinomialModel[] emptyModels;
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;

    public void configure(String folder,
            int[][] words,
            int V, int L,
            double alpha,
            double[] betas,
            double[] gammas,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;

        this.V = V;
        this.L = L;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }
        for (double gamma : gammas) {
            this.hyperparams.add(gamma);
        }

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

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

        for (int d = 0; d < D; d++) {
            totalNumObservations += this.words[d].length;
        }
        logln("--- D = " + D);
        logln("--- V = " + V);
        logln("--- # observations = " + totalNumObservations);
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_RCRP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)));
        int count = ALPHA + 1;
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

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- --- Done initializing. " + getCurrentState());
            logln(printGlobalTree());
        }
    }

    private void initializeModelStructure() {
//        DirichletMultinomialModel dmModel = new DirichletMultinomialModel(V, betas[0], uniform);
        SparseCount count = new SparseCount();
        Dirichlet dir = new Dirichlet(betas[0] * V, uniform);
        this.globalTreeRoot = new RCRPNode(0, 0, count, dir.nextDistribution(), null);

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<RCRPTable, Integer, RCRPNode>();
        }

//        this.emptyModels = new DirichletMultinomialModel[L-1];
//        for(int l=0; l<emptyModels.length; l++)
//            this.emptyModels[l] = new DirichletMultinomialModel(V, betas[l+1], uniform);
    }

    private void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    protected void initializeAssignments() {
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

        lda.configure(null, words, V, K, lda_alpha, lda_beta, initState,
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

        for (int d = 0; d < D; d++) {
            logln("Initialize d = " + d);

            // create tables
            for (int k = 0; k < K; k++) {
                RCRPTable table = new RCRPTable(k, null, d);
                this.localRestaurants[d].addTable(table);
            }

            // add customers to tables
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = ldaZ[d][n];
                this.localRestaurants[d].addCustomerToTable(n, z[d][n]);
            }

            // assign tables with global nodes
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (RCRPTable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }
                this.sampleNodeForTable(d, table.getIndex(), !REMOVE);
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

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t # tokens change = " + numTokenAssignmentsChange
                            + "\t # tables change = " + numTableAssignmentsChange
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + loglikelihood
                            + "\t # tokens change = " + numTokenAssignmentsChange
                            + "\t # tables change = " + numTableAssignmentsChange
                            + "\n" + getCurrentState());
                }
            }

            numTableAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    this.sampleTableForToken(d, n, REMOVE);
                }

                for (RCRPTable table : this.localRestaurants[d].getTables()) {
                    this.sampleNodeForTable(d, table.getIndex(), REMOVE);
                }
            }

            updateTopics();

            if (report && iter % LAG == 0) {
                try {
                    IOUtils.createFolder(folder + getSamplerFolder() + ReportFolder);
                    outputTopicTopWords(folder + getSamplerFolder() + ReportFolder + "iter-" + iter + "-topwords.txt", 20);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
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
                logln("Validating iter = " + iter);
                this.validate("Iteration " + iter);
            }
            System.out.println();
        }
    }

    /**
     * Remove a customer from a table. If the table is empty after the removal
     * remove the table from the restaurant and also remove the assignment from
     * the global tree. If the node in the global tree is empty, remove it.
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param customer The customer index
     */
    private void removeCustomerFromTable(int d, int tableIndex, int n) {
        RCRPTable table = this.localRestaurants[d].getTable(tableIndex);

        // remove the customer
        this.localRestaurants[d].removeCustomerFromTable(n, tableIndex);

        if (table.isEmpty()) { // remove table if empty
            removeTable(d, table, null);
        }
    }

    /**
     * Remove a table from the structure. If the table is empty, remove it from
     * the corresponding restaurant
     *
     * @param d The restaurant index
     * @param table The table to be removed
     */
    private void removeTable(int d, RCRPTable table, HashMap<Integer, Integer> observations) {
        RCRPNode node = table.getContent();
        node.removeCustomer(table);

        if (observations != null) {
            removeObservations(node, observations);
        }

        if (table.isEmpty()) {
            this.localRestaurants[d].removeTable(table.getIndex());
        }

        if (node.isEmpty()) {
            ArrayList<RCRPTable> tables = removeNode(node);

            // resample for removed tables
            for (RCRPTable t : tables) {
                sampleNodeForTable(t.restIndex, t.getIndex(), !REMOVE);
            }
        }
    }

    /**
     * Remove a node from the global tree. If the node is an internal node,
     * remove all nodes in the subtree rooted at the given node.
     *
     * @param node The node to be removed
     * @return The set of tables currently assigned to the subtree rooted at the
     * given node
     */
    private ArrayList<RCRPTable> removeNode(RCRPNode node) {
        if (!node.isEmpty()) {
            throw new RuntimeException("Removing a non-empty table");
        }

        // get the list of all customers currently assigned to the subtree rooted at this node
        ArrayList<RCRPTable> tables = new ArrayList<RCRPTable>();
        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(node);
        while (!stack.isEmpty()) {
            RCRPNode curNode = stack.pop();
            for (RCRPTable t : curNode.getCustomers()) {
                tables.add(t);
            }
            for (RCRPNode child : curNode.getChildren()) {
                stack.add(child);
            }
        }

        if (tables.size() != node.getNumPathCustomers()) {
            throw new RuntimeException("Numbers mismatch");
        }

        // decrease the number of customers on path from root to this node
        node.changeNumPathCustomers(-tables.size());

        // remove the observations of the subtree
        removeObservations(node, node.getContent().getObservations());

        // remove node from its parent
        if (!node.isRoot()) {
            node.getParent().removeChild(node.getIndex());
        } else {
            node.removeAllChilren();
        }

        return tables;
    }

    /**
     * Add a set of observations to the path from the root to a given node
     *
     * @param node The ending node of the path
     * @param observations The set of observations to be added
     */
    private void addObservations(RCRPNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
//            node.counts.changeCount(obs, count);

            RCRPNode tempNode = node;
            while (tempNode != null) {
                tempNode.getContent().changeCount(obs, count);
                tempNode = tempNode.getParent();
            }
        }
    }

    /**
     * Add an observations to the path from the root to a given node
     *
     * @param node The ending node of the path
     * @param observation The observation to be added
     */
    private void addObservation(RCRPNode node, int observation) {
//        node.counts.increment(observation);
        RCRPNode tempNode = node;
        while (tempNode != null) {
            tempNode.getContent().increment(observation);
            tempNode = tempNode.getParent();
        }
    }

    /**
     * Remove a set of observations from the path from the root to a given node
     *
     * @param node The ending node of the path
     * @param observations The set of observations to be removed
     */
    private void removeObservations(RCRPNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
//            node.counts.changeCount(obs, -count);
            RCRPNode tempNode = node;
            while (tempNode != null) {
                tempNode.getContent().changeCount(obs, -count);
                tempNode = tempNode.getParent();
            }
        }
    }

    /**
     * Remove an observations from the path from the root to a given node
     *
     * @param node The ending node of the path
     * @param observation The observation to be added
     */
    private void removeObservation(RCRPNode node, int observation) {
//        node.counts.decrement(observation);
        RCRPNode tempNode = node;
        while (tempNode != null) {
            tempNode.getContent().decrement(observation);
            tempNode = tempNode.getParent();
        }
    }

    /**
     * Create a child node of a global node
     *
     * @param parentNode The parent node
     * @return The newly created child node
     */
    private RCRPNode createGlobalNode(RCRPNode parentNode) {
        int childIndex = parentNode.getNextChildIndex();
        int childLevel = parentNode.getLevel() + 1;
//        DirichletMultinomialModel llhModel = new DirichletMultinomialModel(V, betas[childLevel], uniform);
        SparseCount count = new SparseCount();
        Dirichlet dir = new Dirichlet(betas[childLevel], parentNode.getTopic());
        RCRPNode childNode = new RCRPNode(childIndex, childLevel, count, dir.nextDistribution(), parentNode);
        parentNode.addChild(childIndex, childNode);
        return childNode;
    }

    /**
     * Sample a table for a token
     *
     * @param d The document index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     */
    private void sampleTableForToken(int d, int n, boolean remove) {
        int curObs = words[d][n];
        int curTableIndex = z[d][n];

        if (remove) {
            removeObservation(this.localRestaurants[d].getTable(curTableIndex).getContent(), words[d][n]);
            removeCustomerFromTable(d, curTableIndex, n);
        }

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        // for existing tables
        for (RCRPTable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());
            double logprob =
                    Math.log(table.getNumCustomers())
                    + table.getContent().computeLogLikelihood(curObs);
//                    + table.getContent().getContent().getLogLikelihood(curObs);
            logprobs.add(logprob);

            // debug
//            logln((logprobs.size()-1)
//                    + "\t" + d + ":" + n + ":" + words[d][n]
//                    + "\t" + MiscUtils.formatDouble(Math.log(table.getNumCustomers()))
//                    + "\t" + MiscUtils.formatDouble(table.getContent().getContent().getLogLikelihood(curObs))
//                    + "\t" + MiscUtils.formatDouble(logprob)
//                    + "\t\t\t" + table.toString()
//                    + "\t" + table.getContent().toString());
        }

        // for new tables
        tableIndices.add(PSEUDO_TABLE_INDEX);

        // compute the log priors
        HashMap<String, Double> nodeLogPriors = new HashMap<String, Double>();
        computeNodeLogPriors(nodeLogPriors, globalTreeRoot, 0.0);

        // compute the log likelihoods
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        observations.put(curObs, 1);
        HashMap<String, Double> nodeLogLikelihoods = new HashMap<String, Double>();
        computeLogLikelihoods(nodeLogLikelihoods, globalTreeRoot, observations);

        // combine
        double marginalLogLh = 0.0;
        for (String nodePath : nodeLogPriors.keySet()) {
            double lp = nodeLogPriors.get(nodePath) + nodeLogLikelihoods.get(nodePath);
            if (marginalLogLh == 0.0) {
                marginalLogLh = lp;
            } else {
                marginalLogLh = SamplerUtils.logAdd(marginalLogLh, lp);
            }
        }

        double logprob =
                Math.log(hyperparams.get(ALPHA))
                + marginalLogLh;
        logprobs.add(logprob);

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int tableIndex = tableIndices.get(sampledIndex);

        // debug
//        logln((logprobs.size()-1)
//                + "\t" + d + ":" + n + ":" + words[d][n]
//                + "\t" + MiscUtils.formatDouble(Math.log(hyperparams.get(ALPHA)))
//                + "\t" + MiscUtils.formatDouble(marginalLogLh)
//                + "\t" + MiscUtils.formatDouble(logprob)
//                );
//        logln("---> index = " + sampledIndex + ". " + tableIndex + "\n");

        if (curTableIndex != tableIndex) {
            numTokenAssignmentsChange++;
        }

        RCRPTable table;
        if (tableIndex == PSEUDO_TABLE_INDEX) {
            String globalNodePath = sampleNode(nodeLogPriors, nodeLogLikelihoods);
            RCRPNode globalNode;
            if (globalNodePath.contains("-1")) { // create a new node
                RCRPNode parentGlobalNode = getGlobalNode(parseParentNodePath(globalNodePath));
                globalNode = createGlobalNode(parentGlobalNode);
            } else {
                globalNode = getGlobalNode(parseNodePath(globalNodePath));
            }

            // sample a global node by recursive call
//            RCRPNode globalNode = sampleNode(globalTreeRoot, observations);

            // create a new local table
            int newTableIndex = this.localRestaurants[d].getNextTableIndex();
            table = new RCRPTable(newTableIndex, globalNode, d);

            globalNode.addCustomer(table);
            this.localRestaurants[d].addTable(table);
        } else {
            table = this.localRestaurants[d].getTable(tableIndex);
        }

        // update
        z[d][n] = table.getIndex();
        this.localRestaurants[d].addCustomerToTable(n, z[d][n]);
        addObservation(table.getContent(), curObs);
    }

    /**
     * Sample a global node for a table
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param remove Whether the current assignment should be removed
     */
    private void sampleNodeForTable(int d, int tableIndex, boolean remove) {
        RCRPTable table = this.localRestaurants[d].getTable(tableIndex);
        RCRPNode curNode = table.getContent();

        // current observations assigned to this table
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        for (int c : table.getCustomers()) {
            Integer count = observations.get(words[d][c]);
            if (count == null) {
                observations.put(words[d][c], 1);
            } else {
                observations.put(words[d][c], count + 1);
            }
        }

        // in case of the first table assignment
        if (globalTreeRoot.isEmpty()) {
            globalTreeRoot.addCustomer(table);
            table.setContent(globalTreeRoot);
            addObservations(globalTreeRoot, observations);
            return;
        }

        if (remove) {
            removeTable(d, table, observations);
        }

        // compute the log priors
        HashMap<String, Double> nodeLogPriors = new HashMap<String, Double>();
        computeNodeLogPriors(nodeLogPriors, globalTreeRoot, 0.0);

        // compute the log likelihoods
        HashMap<String, Double> nodeLogLikelihoods = new HashMap<String, Double>();
        computeLogLikelihoods(nodeLogLikelihoods, globalTreeRoot, observations);

        if (nodeLogPriors.size() != nodeLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of nodes mismatch");
        }

        // sample node
        String globalNodePath = sampleNode(nodeLogPriors, nodeLogLikelihoods);

        // debug
//        for(String nodepath : nodeLogPriors.keySet()){
//            logln(nodepath
//                    + ". lp = " + MiscUtils.formatDouble(nodeLogPriors.get(nodepath))
//                    + ". llh = " + MiscUtils.formatDouble(nodeLogLikelihoods.get(nodepath))
//                    + ". sum = " + MiscUtils.formatDouble(nodeLogPriors.get(nodepath) + nodeLogLikelihoods.get(nodepath)));
//        }
//        logln("---> " + globalNodePath + "\n");

        RCRPNode globalNode;
        if (globalNodePath.contains("-1")) { // create a new node
            RCRPNode parentGlobalNode = getGlobalNode(parseParentNodePath(globalNodePath));
            globalNode = createGlobalNode(parentGlobalNode);
        } else {
            globalNode = getGlobalNode(parseNodePath(globalNodePath));
        }

//        RCRPNode globalNode = sampleNode(globalTreeRoot, observations);

        if (!globalNode.equals(curNode)) {
            numTableAssignmentsChange++;
        }

        // update
        table.setContent(globalNode);
        globalNode.addCustomer(table);
        addObservations(globalNode, observations);
    }

    private RCRPNode sampleNode(RCRPNode curNode, HashMap<Integer, Integer> observations) {
        if (isLeafNode(curNode)) {
            return curNode;
        }

        // debug
//        logln("Sample node: cur node = " + curNode.toString());

        ArrayList<Integer> nodeIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        nodeIndices.add(SELF_INDEX);
        double curLogprob = Math.log(curNode.getNumNodeCustomers())
                + curNode.computeLogLikelihood(observations);
//                + curNode.getContent().getLogLikelihood(observations);
        logprobs.add(curLogprob);

        if (!isLeafNode(curNode)) {
            nodeIndices.add(PSEUDO_NODE_INDEX);
            double pseudoLogprob = Math.log(gammas[curNode.getLevel()]);
            Dirichlet dir = new Dirichlet(betas[curNode.getLevel() + 1], curNode.getTopic());
            double[] newTopic = dir.nextDistribution();
            pseudoLogprob += computeLogLikelihood(newTopic, observations);
//                    + emptyModels[curNode.getLevel()].getLogLikelihood(observations);
            logprobs.add(pseudoLogprob);

            // debug 
//            logln("--- pseudo. lp = " + MiscUtils.formatDouble(Math.log(gammas[curNode.getLevel()]))
//                + ". llh = " + MiscUtils.formatDouble(emptyModels[curNode.getLevel()].getLogLikelihood(observations)));

            for (RCRPNode child : curNode.getChildren()) {
                double childLogprob = Math.log(child.getNumPathCustomers())
                        + child.computeLogLikelihood(observations);
//                        + child.getContent().getLogLikelihood(observations);
                nodeIndices.add(child.getIndex());
                logprobs.add(childLogprob);

                // debug
//                logln("--- child. lp = " + MiscUtils.formatDouble(Math.log(child.getNumPathCustomers()))
//                        + ". llh = " + MiscUtils.formatDouble(child.getContent().getLogLikelihood(observations))
//                        + ".\t\t" + child.toString());
            }
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int nodeIndex = nodeIndices.get(sampledIndex);

        // debug 
//        logln("--- self. lp = " + MiscUtils.formatDouble(Math.log(curNode.getNumNodeCustomers()))
//                + ". llh = " + MiscUtils.formatDouble(curNode.getContent().getLogLikelihood(observations))
//                + "\t\t" + curNode.toString());
//        logln(">>> >>> sampler index = " + sampledIndex + ". " + nodeIndex + "\n\n");        

        if (nodeIndex == SELF_INDEX) {
            return curNode;
        } else if (nodeIndex == PSEUDO_NODE_INDEX) {
            return createGlobalNode(curNode);
        } else {
            return sampleNode(curNode.getChild(nodeIndex), observations);
        }
    }

    public double computeLogLikelihood(double[] topic, HashMap<Integer, Integer> observations) {
        double llh = 0.0;
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            llh += count * Math.log(topic[obs]);
        }
        return llh;
    }

    /**
     * Sample a global node given precomputed log priors and log likelihoods
     *
     * @param nodeLogPriors Log priors
     * @param nodeLlhs Log likelihood
     * @return The path to the sampled node
     */
    private String sampleNode(
            HashMap<String, Double> nodeLogPriors,
            HashMap<String, Double> nodeLlhs) {
        ArrayList<String> nodePaths = new ArrayList<String>();
        ArrayList<Double> nodeLogProbs = new ArrayList<Double>();
        for (String nodePath : nodeLogPriors.keySet()) {
            nodePaths.add(nodePath);
            double logprob = nodeLogPriors.get(nodePath) + nodeLlhs.get(nodePath);
            nodeLogProbs.add(logprob);

//            logln(nodePath
//                    + ". " + nodeLogPriors.get(nodePath)
//                    + ". " + nodeLlhs.get(nodePath)
//                    + ". " + logprob
//                    );
        }
//        logln("");

        int sampledIndex = SamplerUtils.logMaxRescaleSample(nodeLogProbs);
        String sampledNodePath = nodePaths.get(sampledIndex);
        return sampledNodePath;
    }

    private void updateTopics() {
        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            double[] prior = new double[V];
            if (isRootNode(node)) {
                for (int v = 0; v < V; v++) {
                    double val = node.getContent().getCount(v) + betas[node.getLevel()] * uniform[v];
                    prior[v] = val;
                }
            } else {
                for (int v = 0; v < V; v++) {
                    double val = node.getContent().getCount(v) + betas[node.getLevel()] * node.getTopicElement(v);
                    prior[v] = val;
                }
            }

            Dirichlet dir = new Dirichlet(prior);
            node.setTopic(dir.nextDistribution());

            for (RCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
    }

    /**
     * Recursively compute the log likelihoods of each node in the global tree
     * given a set of observations.
     *
     * @param nodeLlhs The hash table to store the result
     * @param curNode The current node
     * @param observations The set of observations
     */
    private void computeLogLikelihoods(HashMap<String, Double> nodeLlhs,
            RCRPNode curNode,
            HashMap<Integer, Integer> observations) {
//        double curNodeLlh = curNode.getContent().getLogLikelihood(observations);
        double curNodeLlh = curNode.computeLogLikelihood(observations);
        nodeLlhs.put(curNode.getPathString(), curNodeLlh);

        if (!this.isLeafNode(curNode)) {
//            double[] pseudoPrior = new double[V];
//            for(int v=0; v<V; v++)
//                pseudoPrior[v] = betas[curNode.getLevel()] / V * (curNode.getContent().getCount(v) + 
//                        curNode.getContent().getConcentration() * curNode.getContent().getCenterElement(v));
//            double pseudoChildLlh = SamplerUtils.computeLogLhood(obsCounts, obsCountSum, pseudoPrior);

//            double pseudoChildLlh = this.emptyModels[curNode.getLevel()].getLogLikelihood(observations);
//            System.out.println(curNode.toString());
//            System.out.println(MiscUtils.arrayToString(curNode.getTopic()));

            Dirichlet dir = new Dirichlet(betas[curNode.getLevel() + 1], curNode.getTopic());
            double[] newTopic = dir.nextDistribution();
            double pseudoChildLlh = computeLogLikelihood(newTopic, observations);
            nodeLlhs.put(curNode.getPseudoChildPathString(), pseudoChildLlh);

            for (RCRPNode child : curNode.getChildren()) {
                computeLogLikelihoods(nodeLlhs, child, observations);
            }
        }
    }

    /**
     * Recursively compute the log priors of all possible assignments (including
     * existing and new nodes) of the global tree.
     *
     * @param logPriors The hash table to store the result
     * @param curNode The current node
     * @param passingLogProb The log probability passed from the parent node
     */
    private void computeNodeLogPriors(
            HashMap<String, Double> logPriors,
            RCRPNode curNode,
            double passingLogProb) {

        if (curNode.getNumPathCustomers() == 0) {
            throw new RuntimeException("Empty path. " + curNode.toString());
        }

        double curNodeLogProb;
        if (this.isLeafNode(curNode)) {
            curNodeLogProb = 0.0;
        } else {
            double normalizer = Math.log(curNode.getNumPathCustomers() + gammas[curNode.getLevel()]);
            curNodeLogProb = Math.log(curNode.getNumNodeCustomers()) - normalizer;

            double pseudoChildLogProb = Math.log(gammas[curNode.getLevel()]) - normalizer;
            logPriors.put(curNode.getPseudoChildPathString(), passingLogProb + pseudoChildLogProb);

            for (RCRPNode child : curNode.getChildren()) {
                double childLogProb = Math.log(child.getNumPathCustomers()) - normalizer;
                computeNodeLogPriors(logPriors, child, passingLogProb + childLogProb);
            }
        }
        logPriors.put(curNode.getPathString(), passingLogProb + curNodeLogProb);
    }

    private boolean isRootNode(RCRPNode node) {
        return node.getLevel() == 0;
    }

    private boolean isLeafNode(RCRPNode node) {
        return node.getLevel() == L - 1;
    }

    public int[] parseParentNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length - 1];
        for (int i = 0; i < ss.length - 1; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    private RCRPNode getGlobalNode(int[] parsedPath) {
        RCRPNode node = this.globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    @Override
    public String getCurrentState() {
        double[] levelCount = new double[L];
        Queue<RCRPNode> queue = new LinkedList<RCRPNode>();
        queue.add(this.globalTreeRoot);
        while (!queue.isEmpty()) {
            RCRPNode node = queue.poll();
            levelCount[node.getLevel()]++;
            for (RCRPNode child : node.getChildren()) {
                queue.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(").append(levelCount[l]).append(") ");
        }

        int[] numTables = new int[D];
        for (int d = 0; d < D; d++) {
            numTables[d] = localRestaurants[d].getNumTables();
        }
        String s = "\t# tables: min = " + StatUtils.min(numTables)
                + ". max = " + StatUtils.max(numTables)
                + ". avg = " + MiscUtils.formatDouble(StatUtils.mean(numTables))
                + ". total = " + StatUtils.sum(numTables);
        str.append(s);

        return str.toString();
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(globalTreeRoot);

        int totalCus = 0;

        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    //                    .append("\t").append(MiscUtils.arrayToString(node.getContent().getCounts()))
                    .append("\n");

            totalCus += node.getNumNodeCustomers();

            for (RCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("# observations = ").append(globalTreeRoot.getContent().getCountSum())
                .append("\n# customers = ").append(totalCus)
                .append("\n");
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double obsLlh = 0.0;
        double treeLogProb = 0.0;

        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            // not correct
            obsLlh += node.computeLogLikelihood();
//            obsLlh += node.getContent().getLogLikelihood();
//            obsLlh += node.computeLogLikelihood();

            if (!isLeafNode(node)) {
                treeLogProb += getLogJointProbability(node, gammas[node.getLevel()]);

                for (RCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        double restLogProb = 0.0;
        for (int d = 0; d < D; d++) {
            restLogProb += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));
        }

        double llh = obsLlh + treeLogProb + restLogProb;

        if (verbose) {
            logln("*** obs: " + MiscUtils.formatDouble(obsLlh)
                    + ". tree: " + MiscUtils.formatDouble(treeLogProb)
                    + ". restaurants: " + MiscUtils.formatDouble(restLogProb));
        }

        return llh;
    }

    private double getLogJointProbability(RCRPNode node, double gamma) {
        ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
        for (RCRPNode child : node.getChildren()) {
            numChildrenCusts.add(child.getNumPathCustomers());
        }
        numChildrenCusts.add(node.getNumNodeCustomers());
        return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gamma);
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        return 0.0;
//        double newAlpha = tParams.get(ALPHA);
//        int count = ALPHA;
//        double[] newBetas = new double[L];
//        for(int l=0; l<L; l++)
//            newBetas[l] = tParams.get(count++);
//        double[] newGammas = new double[L-1];
//        for(int l=0; l<newGammas.length; l++)
//            newGammas[l] = tParams.get(count++);
//        
//        double obsLlh = 0.0;
//        double treeLogProb = 0.0;
//        
//        Stack<RCRPNode> stack = new Stack<RCRPNode>();
//        stack.add(globalTreeRoot);
//        while(!stack.isEmpty()){
//            RCRPNode node = stack.pop();
//            
//            obsLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);
//            
//            if(!isLeafNode(node)){
//                treeLogProb += getLogJointProbability(node, newGammas[node.getLevel()]);
//                
//                for(RCRPNode child : node.getChildren())
//                    stack.add(child);
//            }
//        }
//        
//        double restLogProb = 0.0;
//        for(int d=0; d<D; d++)
//            restLogProb += localRestaurants[d].getJointProbabilityAssignments(newAlpha);
//        
//        double llh = obsLlh + treeLogProb + restLogProb;
//        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
//        int count = ALPHA;
//        for(int l=0; l<L; l++)
//            betas[l] = newParams.get(count++);
//        for(int l=0; l<gammas.length; l++)
//            gammas[l] = newParams.get(count++);
//        
//        Stack<RCRPNode> stack = new Stack<RCRPNode>();
//        stack.add(globalTreeRoot);
//        while(!stack.isEmpty()){
//            RCRPNode node = stack.pop();
//            
//            node.getContent().setConcentration(betas[node.getLevel()]);
//                
//            for(RCRPNode child : node.getChildren())
//                stack.add(child);
//        }
//        
//        this.hyperparams = new ArrayList<Double>();
//        for(double param : newParams)
//            this.hyperparams.add(param);
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            int docNumTokens = 0;
            for (RCRPTable table : this.localRestaurants[d].getTables()) {
                table.validate(msg);
                docNumTokens += table.getNumCustomers();
            }

            if (docNumTokens != words[d].length) {
                throw new RuntimeException(msg + ". Numbers of tokens mismatch. "
                        + docNumTokens + " vs. " + words[d].length);
            }
        }

        if (globalTreeRoot.getContent().getCountSum() != totalNumObservations) {
            throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                    + globalTreeRoot.getContent().getCountSum() + " vs. " + totalNumObservations);
        }

        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            if (node.isEmpty()) {
                throw new RuntimeException(msg + ". Empty node. " + node.toString());
            }

            node.validate(msg);

//            totalNodeObs += node.counts.getCountSum();

            int curNodeNumObs = node.getContent().getCountSum();
            for (RCRPTable table : node.customers) {
                curNodeNumObs -= table.getNumCustomers();
            }

            if (!isLeafNode(node)) {
                int subtreeNumObs = 0;
                for (RCRPNode child : node.getChildren()) {
                    stack.add(child);

                    subtreeNumObs += child.getContent().getCountSum();
                }

                if (curNodeNumObs != subtreeNumObs) {
                    throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                            + node.toString()
                            + ". " + curNodeNumObs
                            + " vs. " + subtreeNumObs);
                }
            }
        }
    }

    @Override
    public void outputState(String filepath) {
        throw new RuntimeException("Not supported");
//        if(verbose)
//            logln("--- Outputing current state to " + filepath);
//        
//        try{
//            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
//            StringBuilder str = new StringBuilder();
//
//            // document-specific restaurants
//            for(int d=0; d<D; d++){
//                str.append(d).append("\t").append(localRestaurants[d].getNumTables()).append("\n");
//                for(RCRPTable table : this.localRestaurants[d].getTables()){
//                    str.append(table.getIndex())
//                            .append("\t").append(table.getContent().getPathString())
//                            .append("\n");
//                }
//            }
//
//            // tree
//            Stack<RCRPNode> stack = new Stack<RCRPNode>();
//            stack.add(globalTreeRoot);
//            while(!stack.isEmpty()){
//                RCRPNode node = stack.pop();
//
//                // write node
//                str.append(node.getPathString())
//                        .append("\t").append(node.getNumPathCustomers())
//                        .append("\n")
//                        ;
//
//                // mean of Dirichlet prior
//                for(int v=0; v<V; v++)
//                    str.append(node.getContent().getCenterElement(v)).append("\t");
//                str.append("\n");
//
//                // scale of Dirichlet prior
//                str.append(node.getContent().getConcentration()).append("\n");
//
//                // node's customers
//                for(RCRPTable table : node.getCustomers())
//                    str.append(table.getTableId()).append("\t");
//                str.append("\n");
//
//                // recursive call
//                for(RCRPNode child : node.getChildren())
//                    stack.add(child);
//            }
//            str.append(SEPARATOR).append("\n");
//
//            // document table assignments
//            for(int d=0; d<D; d++){
//                for(int n=0; n<words[d].length; n++)
//                    str.append(d).append(":").append(n).append("\t").append(z[d][n]).append("\n");
//            }
//
//            // output to a compressed file
//            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);
//            ZipEntry e = new ZipEntry(filename + ".txt");
//            writer.putNextEntry(e);
//            byte[] data = str.toString().getBytes();
//            writer.write(data, 0, data.length);
//            writer.closeEntry();
//            writer.close();
//        }
//        catch(Exception e){
//            e.printStackTrace();
//            System.exit(1);
//        }
    }

    @Override
    public void inputState(String filepath) {
        throw new RuntimeException("Not supported");
//        this.initializeModelStructure();
//        
//        this.initializeDataStructure();
//        
//        if(verbose)
//            logln("--- Reading state from " + filepath);
//        try{
//            ZipFile zipFile = new ZipFile(filepath);
//            ZipEntry entry = zipFile.entries().nextElement();
//            InputStream input = zipFile.getInputStream(entry);
//            BufferedReader reader = new BufferedReader(new InputStreamReader(input, "UTF-8"));
//
//            String line; String[] sline;
//
//            // read document-specific restaurants
//    //        HashMap<RCRPTable, String> tableNodeMap = new HashMap<RCRPTable, String>();
//            HashMap<String, RCRPTable> tableMap = new HashMap<String, RCRPTable>();
//            for(int d=0; d<D; d++){
//                sline = reader.readLine().split("\t");
//                int numTables = Integer.parseInt(sline[1]);
//                for(int i=0; i<numTables; i++){
//                    sline = reader.readLine().split("\t");
//                    RCRPTable table = new RCRPTable(Integer.parseInt(sline[0]), null, d);
//                    localRestaurants[d].addTable(table);
//
//    //                tableNodeMap.put(table, sline[1]);
//                    tableMap.put(table.getTableId(), table);
//                }
//            }
//
//            // read the tree
//            HashMap<String, RCRPNode> nodeMap = new HashMap<String, RCRPNode>();
//            HashMap<RCRPNode, Integer> nodeNumPathCustomers = new HashMap<RCRPNode, Integer>(); // for debug
//            while(!(line = reader.readLine()).equals(SEPARATOR)){
//                sline = line.split("\t");
//                String pathStr = sline[0];
//                int numPathCusts = Integer.parseInt(sline[1]);
//
//                // Dirichlet
//                sline = reader.readLine().split("\t");
//                double[] mean = new double[sline.length];
//                for(int v=0; v<mean.length; v++)
//                    mean[v] = Double.parseDouble(sline[v]);
//                double concentration = Double.parseDouble(reader.readLine());
//
//                DirichletMultinomialModel dmm = new DirichletMultinomialModel(V, concentration, mean);
//
//                // create node
//                int lastColonIndex = pathStr.lastIndexOf(":");
//                RCRPNode parent = null;
//                if(lastColonIndex != -1)
//                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
//
//                String[] pathIndices = pathStr.split(":");
//                RCRPNode node = new RCRPNode(Integer.parseInt(pathIndices[pathIndices.length-1]), 
//                    pathIndices.length-1, dmm, parent);
//
//                if(node.getLevel() == 0)
//                    globalTreeRoot = node;
//
//                if(parent != null)
//                    parent.addChild(node.getIndex(), node);
//
//                // customers (i.e., tables)
//                sline = reader.readLine().split("\t");
//                for(int i=0; i<sline.length; i++){
//                    RCRPTable table = tableMap.get(sline[i]);
//                    node.addCustomer(table);
//                    table.setContent(node);
//                }
//
//                nodeMap.put(pathStr, node);
//                nodeNumPathCustomers.put(node, numPathCusts);
//            }
//
//            // validate
//            for(RCRPNode node : nodeNumPathCustomers.keySet()){
//                if(node.getNumPathCustomers() != nodeNumPathCustomers.get(node))
//                    throw new RuntimeException("Numbers of customers on path mismatch. "
//                            + node.toString() + "\t"
//                            + node.getNumPathCustomers() + " vs. " + nodeNumPathCustomers.get(node));
//            }
//
//            // read assignments
//            for(int d=0; d<D; d++){
//                for(int n=0; n<words[d].length; n++){
//                    sline = reader.readLine().split("\t");
//                    if(!sline[0].equals(d + ":" + n))
//                        throw new RuntimeException("Mismatch");
//
//                    z[d][n] = Integer.parseInt(sline[1]);
//
//                    RCRPTable table = tableMap.get(d + ":" + z[d][n]);
//    //                table.addCustomer(n);
//    //                localRestaurants[d].incrementCustomer(table.getIndex());
//                    this.localRestaurants[d].addCustomerToTable(n, table.getIndex());                
//    //                table.getContent().getContent().increment(words[d][n]);
//                    addObservation(table.getContent(), words[d][n]);
//                }
//            }
//            reader.close();
//
//            // update inactive children list
//            Stack<RCRPNode> stack = new Stack<RCRPNode>();
//            stack.add(globalTreeRoot);
//            while(!stack.isEmpty()){
//                RCRPNode node = stack.pop();
//                if(!isLeafNode(node)){
//                    node.fillInactiveChildIndices();
//                    for(RCRPNode child : node.getChildren())
//                        stack.add(child);
//                }
//            }
//
//            // update inactive tables
//            for(int d=0; d<D; d++)
//                this.localRestaurants[d].fillInactiveTableIndices();
//
//            validate("Loading state from " + filepath);
//
//            if(verbose)
//                logln("--- loaded\n" + getCurrentState() + "\n" + printGlobalTree());
//        }
//        catch(Exception e){
//            e.printStackTrace();
//            System.exit(1);
//        }
    }

    public void diagnose(int[] labels, ArrayList<String> labelVoc) throws Exception {
        logln("Diagnosing ...");

        for (int d = 0; d < D; d++) {
            System.out.println("d = " + d
                    + ". # tokens: " + words[d].length
                    + ". label:" + labelVoc.get(labels[d])
                    + ". # tables: " + this.localRestaurants[d].getNumTables());
            for (RCRPTable table : this.localRestaurants[d].getTables()) {
                System.out.print("--- " + table.toString());
                for (int n : table.getCustomers()) {
                    System.out.print(", " + wordVocab.get(words[d][n]));
                }
                System.out.println();
            }
            System.out.println();
        }

        System.out.println(this.printGlobalTree());
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
        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            for (RCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            // skip leaf nodes that are empty
            if (isLeafNode(node) && node.getContent().getCountSum() == 0) {
                continue;
            }

            String[] topWords = getTopWords(node.getTopic(), numWords);
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.getNumNodeCustomers())
                    .append("; ").append(node.getNumPathCustomers())
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

        Stack<RCRPNode> stack = new Stack<RCRPNode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            RCRPNode node = stack.pop();

            for (RCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getTopic();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(node.getPathString()
                    + "\t" + node.getNumNodeCustomers()
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
            for (RCRPTable table : this.localRestaurants[d].getTables()) {
                double[] distribution = table.getContent().getTopic();
                int[] topic = SamplerUtils.getSortedTopic(distribution);

                writer.write(d + ": " + table.toString() + "\n");
                writer.write("\t");
                for (int i = 0; i < 15; i++) {
                    writer.write(wordVocab.get(topic[i]) + ", ");
                }
                writer.write("\n");

                int[] counts = new int[V];
                for (int n : table.getCustomers()) {
                    counts[words[d][n]]++;
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
            testInitialization();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testInitialization() {
        int D = 20;
        int N = 1000;
        int V = 10;
        int[][] words = new int[D][N];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < N; n++) {
                words[d][n] = d % V;
            }
        }

        int L = 3;
        double alpha = 0.5;
        double[] betas = {V, 0.1 * V, 0.01 * V};
        double[] gammas = {1, 0.5};
        RCRPSampler sampler = new RCRPSampler();
        sampler.setVerbose(true);
        sampler.setDebug(true);
        sampler.setLog(false);
        boolean paramOpt = true;
        sampler.setK(V);

        sampler.configure(null, words, V, L, alpha, betas, gammas, InitialState.PRESET, paramOpt, 2, 10, 2);
//        sampler.initializeHierarchies();
//        sampler.initializeAssignments();
//        System.out.println(sampler.printGlobalTree());

        sampler.sample();
        System.out.println();
        System.out.println(sampler.printGlobalTree());
    }
}

class RCRPNode extends TreeNode<RCRPNode, SparseCount> {

    ArrayList<RCRPTable> customers;
    int numPathCustomers; // number of customers on the path from root to this node (including customers in the subtree)
    private double[] topic;

    public RCRPNode(int index, int level, SparseCount content, double[] topic, RCRPNode parent) {
        super(index, level, content, parent);
        this.numPathCustomers = 0;
        this.customers = new ArrayList<RCRPTable>();
        this.topic = topic;
    }

    public int getDimension() {
        return this.topic.length;
    }

    public double getTopicElement(int index) {
        return topic[index];
    }

    public void setTopicElement(int index, double value) {
        this.topic[index] = value;
    }

    public void setTopic(double[] phi) {
        this.topic = phi;
    }

    public double[] getTopic() {
        return this.topic;
    }

    public double computeLogLikelihood(int obs) {
        return Math.log(topic[obs]);
    }

    public double computeLogLikelihood(HashMap<Integer, Integer> observations) {
        double llh = 0.0;
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            llh += count * Math.log(topic[obs]);
        }
        return llh;
    }

    public double computeLogLikelihood(int[] obsCounts) {
        double llh = 0.0;
        for (int v = 0; v < obsCounts.length; v++) {
            if (obsCounts[v] > 0) {
                llh += obsCounts[v] * Math.log(topic[v]);
            }
        }
        return llh;
    }

//    double computeLogLikelihood(int[] obsCounts, int obsCountSum){
////        if(this.parent == null)
////            return this.content.getLogLikelihood(observations);
//        int dim = content.getDimension();
//        
//        double[] dist = new double[dim];
//        for(int v=0; v<dim; v++){
//            dist[v] = content.getCount(v) + 
//                    content.getConcentration() / dim * (parent.getContent().getCount(v) 
//                    + parent.getContent().getConcentration() * parent.getContent().getCenterElement(v));
//        }
//        return SamplerUtils.computeLogLhood(obsCounts, obsCountSum, dist);
//    }
    public double computeLogLikelihood() {
        int[] curNodeCounts = new int[getDimension()];
        for (int v = 0; v < getDimension(); v++) {
            curNodeCounts[v] = content.getCount(v);
            for (RCRPNode node : getChildren()) {
                curNodeCounts[v] -= node.getContent().getCount(v);
            }
        }

        double llh = computeLogLikelihood(curNodeCounts);
        return llh;
    }

    public ArrayList<RCRPTable> getCustomers() {
        return this.customers;
    }

    public void addCustomer(RCRPTable c) {
        this.customers.add(c);
        this.changeNumPathCustomers(1);
    }

    public void removeCustomer(RCRPTable c) {
        this.customers.remove(c);
        this.changeNumPathCustomers(-1);
    }

    public int getNumPathCustomers() {
        return numPathCustomers;
    }

    public int getNumNodeCustomers() {
        return this.customers.size();
    }

    public String getPseudoChildPathString() {
        return this.getPathString() + ":" + RCRPSampler.PSEUDO_NODE_INDEX;
    }

    public void changeNumPathCustomers(int delta) {
        RCRPNode node = this;
        while (node != null) {
            node.numPathCustomers += delta;
            if (node.numPathCustomers < 0) {
                throw new RuntimeException("Negative count. " + node.toString());
            }
            node = node.getParent();
        }
    }

    public boolean isEmpty() {
        return this.getNumNodeCustomers() == 0;
    }

    public void validate(String str) {
        int sumChildrentPathNumCustomers = 0;
        for (RCRPNode child : this.getChildren()) {
            sumChildrentPathNumCustomers += child.getNumPathCustomers();
        }
        if (sumChildrentPathNumCustomers + this.getNumNodeCustomers() != this.numPathCustomers) {
            throw new RuntimeException(str + ". Numbers of customers mismatch. "
                    + (sumChildrentPathNumCustomers + this.getNumNodeCustomers())
                    + " vs. " + numPathCustomers
                    + ". " + this.toString());
        }

        if (this.numPathCustomers < this.getNumNodeCustomers()) {
            throw new RuntimeException(str + ". Invalid number of customers");
        }

        int totalChildrenObs = 0;
        for (RCRPNode child : this.getChildren()) {
            totalChildrenObs += child.getContent().getCountSum();
        }

        if (this.isEmpty()) {
            throw new RuntimeException(str + ". Empty node: " + this.toString());
        }

//        if(totalChildrenObs + counts.getCountSum() != this.getContent().getCountSum())
//            throw new RuntimeException(str + ". Numbers of observations mismatch"
//                    + ". " + totalChildrenObs
//                    + ". " + counts.getCountSum()
//                    + ". " + this.toString());
//        this.counts.validate(str);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", #ch = ").append(getChildren().size())
                .append(", #n = ").append(getNumNodeCustomers())
                .append(", #p = ").append(getNumPathCustomers())
                //                .append(", #o = ").append(counts.getCountSum())
                .append(", #o = ").append(getContent().getCountSum())
                .append("]");
        return str.toString();
    }
}

class RCRPTable extends FullTable<Integer, RCRPNode> {

    int restIndex;

    public RCRPTable(int index, RCRPNode content, int restId) {
        super(index, content);
        this.restIndex = restId;
    }

    public String getTableId() {
        return restIndex + ":" + index;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(restIndex).append("-").append(index)
                .append(". #c = ").append(getNumCustomers())
                .append("] -> ").append(content.getPathString());
        return str.toString();
    }

    public void validate(String msg) {
        if (this.customers.size() != this.getNumCustomers()) {
            throw new RuntimeException("Numbers of customers mismatch");
        }
    }
}