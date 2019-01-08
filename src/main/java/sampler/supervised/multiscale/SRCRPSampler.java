package sampler.supervised.multiscale;

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
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
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
public class SRCRPSampler extends AbstractSampler {

    public static final int PSEUDO_TABLE_INDEX = -1;
    public static final int PSEUDO_NODE_INDEX = -1;
    public static final int ALPHA = 0; // concentration for local DPs
    public static final int RHO = 1; // standard deviation for response variable
    public static final int MU = 2; // prior mean of root's mu
    public static final int SIGMA = 3; // prior variance of root's mu
    protected double[] betas;  // topics concentration parameter
    protected double[] gammas; // global tree's concentrations
    protected double[] sigmas; // prior variance over tree levels
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int L;
    protected int K;
    protected int[][][] words;  // [D] x [Td] x [Ndt]: words
    protected double[][] responses; // [D] x [Td]
    protected int[][][] z; // local table index
    protected SparseCount[][] turnCounts;
    private SRCRPNode globalTreeRoot;
    private Restaurant<SRCRPTable, String, SRCRPNode>[] localRestaurants;
    private MultiscaleStateSpace multiscaleModel;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private int totalNumObservations = 0;
    private double[] uniform;
    private int[] docNumWords;
    private DirMult[] emptyModels;
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;
    private int numConverged;

    public void configure(String folder,
            int[][][] words, double[][] responses,
            int V, int L,
            double alpha, double rho,
            double mu, double sigma,
            double[] sigmas,
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
        this.responses = responses;

        this.V = V;
        this.L = L;
        this.D = this.words.length;

        this.betas = betas;
        this.gammas = gammas;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(rho);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma);
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }
        for (double gamma : gammas) {
            this.hyperparams.add(gamma);
        }
        for (double s : sigmas) {
            this.hyperparams.add(s);
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

        // assert dimensions
        if (this.betas.length != this.L) {
            throw new RuntimeException("Vector betas must have length " + this.L
                    + ". Current length = " + this.betas.length);
        }
        if (this.gammas.length != this.L - 1) {
            throw new RuntimeException("Vector gamms must have length " + (this.L - 1)
                    + ". Current length = " + this.gammas.length);
        }
        if (this.sigmas.length != this.L) {
            throw new RuntimeException("Vector sigma must have length " + this.L
                    + ". Current length = " + this.sigmas.length);
        }

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        this.docNumWords = new int[D];
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                this.docNumWords[d] += words[d][t].length;
            }
        }

        int numTs = 0;
        for (int d = 0; d < D; d++) {
            numTs += words[d].length;
            for (int t = 0; t < words[d].length; t++) {
                totalNumObservations += this.words[d][t].length;
            }
        }
        logln("--- D = " + D);
        logln("--- V = " + V);
        logln("--- Total Ts = " + numTs);
        logln("--- # observations = " + totalNumObservations);

        if (!debug) {
            System.err.close();
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append(isSupervised() ? "_SRCRP" : "_RCRP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_ss-").append(formatter.format(hyperparams.get(SIGMA)));
        int count = SIGMA + 1;
        str.append("_b");
        for (int i = 0; i < betas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_g");
        for (int i = 0; i < gammas.length; i++) {
            str.append("-").append(formatter.format(hyperparams.get(count++)));
        }
        str.append("_s");
        for (int i = 0; i < sigmas.length; i++) {
            str.append("-").append(formatter.format(sigmas[i]));
        }
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public boolean isSupervised() {
        return this.responses != null;
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
        int rootLevel = 0;
        int rootIndex = 0;
        DirMult dmModel = new DirMult(V, betas[rootLevel], uniform);
        double rootMean = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA));
        this.globalTreeRoot = new SRCRPNode(rootIndex, rootLevel, dmModel, null, rootMean);

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<SRCRPTable, String, SRCRPNode>();
        }

        this.emptyModels = new DirMult[L - 1];
        for (int l = 0; l < emptyModels.length; l++) {
            this.emptyModels[l] = new DirMult(V, betas[l + 1], uniform);
        }

        this.multiscaleModel = new MultiscaleStateSpace(globalTreeRoot,
                hyperparams.get(MU), hyperparams.get(SIGMA), sigmas, L);
    }

    private void initializeDataStructure() {
        z = new int[D][][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length][];
            for (int t = 0; t < words[d].length; t++) {
                z[d][t] = new int[words[d][t].length];
            }
        }

        turnCounts = new SparseCount[D][];
        for (int d = 0; d < D; d++) {
            turnCounts[d] = new SparseCount[words[d].length];
            for (int t = 0; t < words[d].length; t++) {
                turnCounts[d][t] = new SparseCount();
            }
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

        for (int d = 0; d < D; d++) {
            // create tables
            for (int k = 0; k < K; k++) {
                SRCRPTable table = new SRCRPTable(k, null, d, Double.NaN);
                this.localRestaurants[d].addTable(table);
            }

            int count = 0;
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    z[d][t][n] = ldaZ[d][count++];
                    this.localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
                    turnCounts[d][t].increment(z[d][t][n]);
                }
            }

            // assign tables with global nodes
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }
                this.sampleNodeForTable(d, table.getIndex(), !REMOVE, !OBSERVED);
            }

            // remove empty table
            for (int tIndex : emptyTables) {
                this.localRestaurants[d].removeTable(tIndex);
            }
        }

        // debug
        logln("After assignment initialization\n" + printGlobalTree() + "\n");

        // optimize
        for (int d = 0; d < D; d++) {
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                SRCRPNode node = table.getContent();
                double mean = SamplerUtils.getGaussian(node.getMean(), sigmas[node.getLevel()]);
                table.setEta(mean);
            }

            optimize(d);
        }

        // debug
        logln("After optimization initialization\n" + printGlobalTree() + "\n");

        // update tree's parameters
        updateMultiscale();
//        updateTreeRegressionParameters();
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
                        this.sampleTableForToken(d, t, n, REMOVE, OBSERVED, ADD, EXTEND);
                    }
                }

                for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                    this.sampleNodeForTable(d, table.getIndex(), REMOVE, OBSERVED);
                }

                optimize(d);
            }

            updateMultiscale();
//            updateTreeRegressionParameters();

            if (verbose) {
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
     * Create a child node of a global node
     *
     * @param parentNode The parent node
     * @return The newly created child node
     */
    private SRCRPNode createGlobalNode(SRCRPNode parentNode) {
        int childIndex = parentNode.getNextChildIndex();
        int childLevel = parentNode.getLevel() + 1;
        DirMult llhModel = new DirMult(V, betas[childLevel], uniform);
        double mean = SamplerUtils.getGaussian(parentNode.getMean(), sigmas[parentNode.getLevel()]);
        SRCRPNode childNode = new SRCRPNode(childIndex, childLevel, llhModel, parentNode, mean);
        parentNode.addChild(childIndex, childNode);
        return childNode;
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
    private void removeCustomerFromTable(int d, int tableIndex, int t, int n) {
        SRCRPTable table = this.localRestaurants[d].getTable(tableIndex);

        // remove the customer
        this.localRestaurants[d].removeCustomerFromTable(getTokenId(t, n), tableIndex);
        removeObservation(table.getContent(), words[d][t][n]);

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
     * @param observations The set of observations currently assigned to the
     * table
     */
    private void removeTable(int d, SRCRPTable table, HashMap<Integer, Integer> observations) {
        SRCRPNode node = table.getContent();
        node.removeCustomer(table);

        if (observations != null) {
            removeObservations(node, observations);
        }

        if (table.isEmpty()) {
            this.localRestaurants[d].removeTable(table.getIndex());
        }

        if (node.isEmpty()) {
            ArrayList<SRCRPTable> tables = removeNode(node);

            // resample for removed tables
            for (SRCRPTable t : tables) {
                sampleNodeForTable(t.restIndex, t.getIndex(), !REMOVE, OBSERVED);
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
    private ArrayList<SRCRPTable> removeNode(SRCRPNode node) {
        if (!node.isEmpty()) {
            throw new RuntimeException("Removing a non-empty table");
        }

        // get the list of all customers currently assigned to the subtree rooted at this node
        ArrayList<SRCRPTable> tables = new ArrayList<SRCRPTable>();
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(node);
        while (!stack.isEmpty()) {
            SRCRPNode curNode = stack.pop();
            for (SRCRPTable t : curNode.getCustomers()) {
                tables.add(t);
            }
            for (SRCRPNode child : curNode.getChildren()) {
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
    private void addObservations(SRCRPNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            SRCRPNode tempNode = node;
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
    private void addObservation(SRCRPNode node, int observation) {
        SRCRPNode tempNode = node;
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
    private void removeObservations(SRCRPNode node, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            int count = observations.get(obs);
            SRCRPNode tempNode = node;
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
    private void removeObservation(SRCRPNode node, int observation) {
        SRCRPNode tempNode = node;
        while (tempNode != null) {
            tempNode.getContent().decrement(observation);
            tempNode = tempNode.getParent();
        }
    }

    /**
     * Sample a table for a token
     *
     * @param d The debate index
     * @param t The turn index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     * @param resObserved Whether the response variable is observed
     */
    private void sampleTableForToken(int d, int t, int n, boolean remove,
            boolean resObserved, boolean add, boolean extend) {
        int curObs = words[d][t][n];
        int curTableIndex = z[d][t][n];

        if (remove) {
            removeCustomerFromTable(d, curTableIndex, t, n);
            turnCounts[d][t].decrement(z[d][t][n]);
        }

        // precompute for when response variable is observed
        double weightedSum = 0.0;
        int numTokens = turnCounts[d][t].getCountSum() + 1;
        if (resObserved) {
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                weightedSum += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
            }
        }

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();
        // for existing tables
        for (SRCRPTable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());
            double logprior = Math.log(table.getNumCustomers());
            double wordLlh = table.getContent().getContent().getLogLikelihood(curObs);
            double logprob = logprior + wordLlh;

            if (resObserved) {
                double mean = (weightedSum + table.getEta()) / numTokens;
                double resLlh = StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(hyperparams.get(RHO)));
                logprob += resLlh;

                // debug
//                logln((logprobs.size()) 
//                    + ". " + d + ":" + t + ":" + n + ":" + words[d][t][n]
//                    + "\t lp: " + MiscUtils.formatDouble(logprior)
//                    + "\t wllh: " + MiscUtils.formatDouble(wordLlh)
//                    + "\t rllh: " + MiscUtils.formatDouble(resLlh)
//                    + "\t logprob = " + MiscUtils.formatDouble(logprob)
//                    + "\n" + table.toString()
//                    + "\t" + table.getContent().toString()
//                    + "\n"
//                    );
            }

            logprobs.add(logprob);
        }

        HashMap<String, Double> nodeLogPriors = null;
        HashMap<String, Double> nodeLogLikelihoods = null;
        HashMap<String, Double> nodeResLogLikelihoods = null;
        if (extend) {
            // for new tables
            tableIndices.add(PSEUDO_TABLE_INDEX);

            // --- compute the log priors
            nodeLogPriors = new HashMap<String, Double>();
            computeNodeLogPriors(nodeLogPriors, globalTreeRoot, 0.0);

            // --- compute the log likelihoods
            HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
            observations.put(curObs, 1);
            nodeLogLikelihoods = new HashMap<String, Double>();
            computeWordLogLikelihoods(nodeLogLikelihoods, globalTreeRoot, observations);

            // debug
            if (nodeLogPriors.size() != nodeLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of nodes mismatch");
            }

            // --- compute response log likelihoods
            nodeResLogLikelihoods = new HashMap<String, Double>();
            if (resObserved) {
                computeResponseLogLikelihoodsNewTable(nodeResLogLikelihoods,
                        globalTreeRoot, responses[d][t], weightedSum, numTokens);

                // debug
                if (nodeLogPriors.size() != nodeResLogLikelihoods.size()) {
                    throw new RuntimeException("Numbers of nodes mismatch");
                }
            }

            // combine
            double marginalLlh = 0.0;
            for (String nodePath : nodeLogPriors.keySet()) {
                double lp = nodeLogPriors.get(nodePath) + nodeLogLikelihoods.get(nodePath);
                if (resObserved) {
                    lp += nodeResLogLikelihoods.get(nodePath);
                }

                if (marginalLlh == 0.0) {
                    marginalLlh = lp;
                } else {
                    marginalLlh = SamplerUtils.logAdd(marginalLlh, lp);
                }
            }

            double pseudoLogPrior = Math.log(hyperparams.get(ALPHA));
            double logprob = pseudoLogPrior + marginalLlh;

            // debug
//            logln((logprobs.size()) 
//                    + ". " + d + ":" + t + ":" + n + ":" + words[d][t][n]
//                    + "\t lp: " + MiscUtils.formatDouble(pseudoLogPrior)
//                    + "\t wllh: " + MiscUtils.formatDouble(marginalLlh)
//                    + "\t logprob = " + MiscUtils.formatDouble(logprob)
//                    + "\n"
//                    );

            logprobs.add(logprob);
        }
        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int tableIndex = tableIndices.get(sampledIndex);

        // debug
//        logln("---> index = " + sampledIndex + ". " + tableIndex + "\n\n");

        if (curTableIndex != tableIndex) {
            numTokenAssignmentsChange++;
        }

        SRCRPTable table;
        if (tableIndex == PSEUDO_TABLE_INDEX) {
            // sample global node
            String globalNodePath = sampleNode(nodeLogPriors, nodeLogLikelihoods, nodeResLogLikelihoods, resObserved);
            SRCRPNode globalNode;
            if (globalNodePath.contains("-1")) { // create a new node 
                SRCRPNode parentGlobalNode = getGlobalNode(parseParentNodePath(globalNodePath));
                globalNode = createGlobalNode(parentGlobalNode);
            } else {
                globalNode = getGlobalNode(parseNodePath(globalNodePath));
            }

            int newTableIndex = this.localRestaurants[d].getNextTableIndex();
            double tempTableRegParam = SamplerUtils.getGaussian(globalNode.getMean(), sigmas[globalNode.getLevel()]);
            table = new SRCRPTable(newTableIndex, globalNode, d, tempTableRegParam);

            localRestaurants[d].addTable(table);
            globalNode.addCustomer(table);
        } else { // existing table
            table = this.localRestaurants[d].getTable(tableIndex);
        }

        // update
        z[d][t][n] = table.getIndex();
        turnCounts[d][t].increment(z[d][t][n]);

        if (add) {
            this.localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
            this.addObservation(table.getContent(), curObs);
        }
    }

    /**
     * Sample a node in the global tree for a table
     *
     * @param d The debate index
     * @param tableIndex The table index
     * @param remove Whether the current assignment should be removed
     * @param resObserved Whether the response variable is observed
     */
    private void sampleNodeForTable(int d, int tableIndex, boolean remove, boolean resObserved) {
        SRCRPTable table = this.localRestaurants[d].getTable(tableIndex);
        SRCRPNode curNode = table.getContent();

        boolean microDebug = false;

        // current observations assigned to this table
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        for (String c : table.getCustomers()) {
            int[] parsedId = parseTokenId(c);
            int type = words[d][parsedId[0]][parsedId[1]];
            Integer count = observations.get(type);
            if (count == null) {
                observations.put(type, 1);
            } else {
                observations.put(type, count + 1);
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
        computeWordLogLikelihoods(nodeLogLikelihoods, globalTreeRoot, observations);

        if (nodeLogPriors.size() != nodeLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of nodes mismatch");
        }

        HashMap<String, Double> nodeResLogLikelihoods = new HashMap<String, Double>();
        if (resObserved) {
            computeResponseLogLikelihoodsExistingTable(nodeResLogLikelihoods, globalTreeRoot, table.getEta());

            if (nodeLogPriors.size() != nodeResLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of nodes mismatch");
            }
        }

        // sample node
        String globalNodePath = sampleNode(nodeLogPriors, nodeLogLikelihoods, nodeResLogLikelihoods, resObserved);

        // debug
        if (microDebug) {
            ArrayList<RankingItem<String>> rankItems = new ArrayList<RankingItem<String>>();
            for (String nodePath : nodeLogLikelihoods.keySet()) {
                rankItems.add(new RankingItem<String>(nodePath,
                        nodeLogPriors.get(nodePath) + nodeLogLikelihoods.get(nodePath) + nodeResLogLikelihoods.get(nodePath)));
            }
            Collections.sort(rankItems);

            for (int i = 0; i < rankItems.size(); i++) {
                String nodePath = rankItems.get(i).getObject();
                SRCRPNode node = null;
                if (!nodePath.contains("-1")) {
                    node = getGlobalNode(parseNodePath(nodePath));
                }
                logln(nodePath
                        + ". " + MiscUtils.formatDouble(nodeLogPriors.get(nodePath))
                        + ". " + MiscUtils.formatDouble(nodeLogLikelihoods.get(nodePath))
                        + ". " + MiscUtils.formatDouble(nodeResLogLikelihoods.get(nodePath))
                        + ". " + MiscUtils.formatDouble(rankItems.get(i).getPrimaryValue())
                        + "\t" + (node == null ? "---" : node.toString()));
            }
            logln("---> " + globalNodePath + "\n");
        }

        if (curNode != null && !curNode.getPathString().equals(globalNodePath)) {
            numTableAssignmentsChange++;
        }

        SRCRPNode globalNode;
        if (globalNodePath.contains("-1")) { // create a new node
            SRCRPNode parentGlobalNode = getGlobalNode(parseParentNodePath(globalNodePath));
            globalNode = createGlobalNode(parentGlobalNode);
        } else {
            globalNode = getGlobalNode(parseNodePath(globalNodePath));
        }

        // update
        table.setContent(globalNode);
        globalNode.addCustomer(table);
        this.addObservations(globalNode, observations);
    }

    /**
     * Sample a global node given precomputed log priors and log likelihoods
     *
     * @param nodeLogPriors Log priors
     * @param nodeLlhs Word log likelihood
     * @param nodeResLlhs Response log likelihood
     * @param responseObserved Whether the response variable is observed
     * @return The path to the sampled node
     */
    private String sampleNode(
            HashMap<String, Double> nodeLogPriors,
            HashMap<String, Double> nodeLlhs,
            HashMap<String, Double> nodeResLlhs,
            boolean responseObserved) {
        ArrayList<String> nodePaths = new ArrayList<String>();
        ArrayList<Double> nodeLogProbs = new ArrayList<Double>();

        for (String nodePath : nodeLogPriors.keySet()) {
            nodePaths.add(nodePath);
            double logprob = nodeLogPriors.get(nodePath) + nodeLlhs.get(nodePath);

            if (responseObserved) {
                logprob += nodeResLlhs.get(nodePath);
            }

            nodeLogProbs.add(logprob);

            // debug
//            logln(nodePath
//                    + ".\t" + MiscUtils.formatDouble(nodeLogPriors.get(nodePath))
//                    + ". " + MiscUtils.formatDouble(nodeLlhs.get(nodePath))
//                    + ". " + (responseObserved ? MiscUtils.formatDouble( nodeResLlhs.get(nodePath)) : "--- ")
//                    + ". " + MiscUtils.formatDouble(logprob)
//                    );
        }

        int sampledIndex = SamplerUtils.logMaxRescaleSample(nodeLogProbs);
        String sampledNodePath = nodePaths.get(sampledIndex);

        // debug
//        logln("---> index = " + sampledIndex + ". " + sampledNodePath + "\n");

        return sampledNodePath;
    }

    private void updateMultiscale() {
        this.multiscaleModel.upwardFilter();
        this.multiscaleModel.downwardSmooth();
        this.multiscaleModel.update();
    }

    private void updateTreeRegressionParameters() {
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

//            double priorVar = hyperparams.get(SIGMA);
//            double priorMean = hyperparams.get(MU);
//            if(node.getLevel() > 0){
//                priorVar = sigmas[node.getLevel()-1];
//                priorMean = node.getParent().getMean();
//            }
//            double obsVar = sigmas[node.getLevel()];

            double priorVar = 1.0;
            double priorMean = 0.0;
            double obsVar = 0.5;

            double newPriorVar = priorVar * obsVar
                    / (node.getNumNodeCustomers() * priorVar + obsVar);
            double sumEtas = 0.0;
            for (SRCRPTable table : node.getCustomers()) {
                sumEtas += table.getEta();
            }
            double newPriorMean = newPriorVar * sumEtas / obsVar
                    + newPriorVar * priorMean / priorVar;

            double newMean = SamplerUtils.getGaussian(newPriorMean, newPriorVar);
            node.setMean(newMean);

            // debug
//            logln(">>> iter = " + iter
//                    + ". node: " + node.getPathString()
//                    + ". new v: " + MiscUtils.formatDouble(newPriorVar)
//                    + ". new m: " + MiscUtils.formatDouble(newPriorMean)
//                    + ". m: " + MiscUtils.formatDouble(newMean));

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
    }

    /**
     * Optimize the regression parameters at each table in a given restaurant
     *
     * @param d The restaurant index
     */
    private void optimize(int d) {
        int numTables = this.localRestaurants[d].getNumTables();

        double[] regParams = new double[numTables];
        double[] priorMeans = new double[numTables];
        double[] priorStdvs = new double[numTables];
        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        int count = 0;
        for (SRCRPTable table : this.localRestaurants[d].getTables()) {
            SRCRPNode node = table.getContent();
            tableIndices.add(table.getIndex());
            regParams[count] = table.getEta();
            priorMeans[count] = node.getMean();
            priorStdvs[count] = Math.sqrt(sigmas[node.getLevel()]);
            count++;
        }

        int Td = words[d].length;
        double[][] designMatrix = new double[Td][numTables];
        for (int t = 0; t < Td; t++) {
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                int idx = tableIndices.indexOf(table.getIndex());
                designMatrix[t][idx] = (double) turnCounts[d][t].getCount(table.getIndex())
                        / turnCounts[d][t].getCountSum();
            }
        }

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses[d],
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

        if (converged) // if the optimization converges
        {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < regParams.length; i++) {
            regParams[i] = optimizable.getParameter(i);
        }

        // upadte tables' etas
        for (int i = 0; i < regParams.length; i++) {
            int tableIndex = tableIndices.get(i);
            this.localRestaurants[d].getTable(tableIndex).setEta(regParams[i]);
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
    private void computeWordLogLikelihoods(HashMap<String, Double> nodeLlhs,
            SRCRPNode curNode,
            HashMap<Integer, Integer> observations) {
        double curNodeLlh = curNode.getContent().getLogLikelihood(observations);
        nodeLlhs.put(curNode.getPathString(), curNodeLlh);

        if (!this.isLeafNode(curNode)) {
            double pseudoChildLlh = this.emptyModels[curNode.getLevel()].getLogLikelihood(observations);
            nodeLlhs.put(curNode.getPseudoChildPathString(), pseudoChildLlh);

            for (SRCRPNode child : curNode.getChildren()) {
                computeWordLogLikelihoods(nodeLlhs, child, observations);
            }
        }
    }

    /**
     * Compute the log probability of assigning an existing table to a possible
     * node in the tree based on the response variable.
     *
     * @param resLlhs The hash table storing the result
     * @param curNode The current node in the recursive call
     * @param eta The regression parameter
     */
    private void computeResponseLogLikelihoodsExistingTable(
            HashMap<String, Double> resLlhs,
            SRCRPNode curNode,
            double eta) {
        double mean = curNode.getMean();
//        double var = curNode.getVariance();
        double var = sigmas[curNode.getLevel()];
        double curNodeResLlh = StatUtils.logNormalProbability(eta, mean, Math.sqrt(var));
        resLlhs.put(curNode.getPathString(), curNodeResLlh);

        // debug
//        logln("cur node: " + curNode.toString() 
//                + ". mean = " + MiscUtils.formatDouble(mean)
//                + ". var = " + MiscUtils.formatDouble(var)
//                + ". llh = " + MiscUtils.formatDouble(curNodeResLlh)
//                );

        if (!this.isLeafNode(curNode)) {
            double pseudoVar = var + this.sigmas[curNode.getLevel() + 1];
            double pseudoResLlh = StatUtils.logNormalProbability(eta, mean, Math.sqrt(pseudoVar));
            resLlhs.put(curNode.getPseudoChildPathString(), pseudoResLlh);

            // debug
//            logln("pseudo " + curNode.getPseudoChildPathString()
//                    + ". var = " + MiscUtils.formatDouble(pseudoVar)
//                    + ". llh = " + MiscUtils.formatDouble(pseudoResLlh));

            for (SRCRPNode child : curNode.getChildren()) {
                computeResponseLogLikelihoodsExistingTable(resLlhs, child, eta);
            }
        }
    }

    /**
     * Recursively compute the log probability of assigning a new table to a
     * possible node in the tree based on the response variable
     *
     * @param resLlhs The results
     * @param curNode The current node in the recursive call
     * @param response Value of the response variable
     * @param weightedSum The current weight sum
     * @param count Number of tokens
     */
    private void computeResponseLogLikelihoodsNewTable(
            HashMap<String, Double> resLlhs,
            SRCRPNode curNode,
            double response,
            double weightedSum,
            int count) {
        double mean = (weightedSum + curNode.getMean()) / count;
        double var = sigmas[curNode.getLevel()] / (count * count) + hyperparams.get(RHO);
        double curNodeResLlh = StatUtils.logNormalProbability(response, mean, Math.sqrt(var));
        resLlhs.put(curNode.getPathString(), curNodeResLlh);

        // debug
//        logln("cur node: " + curNode.toString() 
//                + ". sum = " + MiscUtils.formatDouble(sum)
//                + ". mean = " + MiscUtils.formatDouble(mean)
//                + ". var = " + MiscUtils.formatDouble(var)
//                + ". llh = " + MiscUtils.formatDouble(curNodeResLlh)
//                );

        if (!this.isLeafNode(curNode)) {
            double pseudoVar = var + sigmas[curNode.getLevel() + 1] / (count * count);
            double pseudoResLlh = StatUtils.logNormalProbability(response, mean, Math.sqrt(pseudoVar));
            resLlhs.put(curNode.getPseudoChildPathString(), pseudoResLlh);

            // debug
//            logln("pseudo " + curNode.getPseudoChildPathString()
//                    + ". var = " + MiscUtils.formatDouble(pseudoVar)
//                    + ". llh = " + MiscUtils.formatDouble(pseudoResLlh));

            for (SRCRPNode child : curNode.getChildren()) {
                computeResponseLogLikelihoodsNewTable(resLlhs, child, response, weightedSum, count);
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
            SRCRPNode curNode,
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

            for (SRCRPNode child : curNode.getChildren()) {
                double childLogProb = Math.log(child.getNumPathCustomers()) - normalizer;
                computeNodeLogPriors(logPriors, child, passingLogProb + childLogProb);
            }
        }
        logPriors.put(curNode.getPathString(), passingLogProb + curNodeLogProb);
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

    private boolean isLeafNode(SRCRPNode node) {
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

    private SRCRPNode getGlobalNode(int[] parsedPath) {
        SRCRPNode node = this.globalTreeRoot;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    @Override
    public String getCurrentState() {
        double[] levelCount = new double[L];
        Queue<SRCRPNode> queue = new LinkedList<SRCRPNode>();
        queue.add(this.globalTreeRoot);
        while (!queue.isEmpty()) {
            SRCRPNode node = queue.poll();
            levelCount[node.getLevel()]++;
            for (SRCRPNode child : node.getChildren()) {
                queue.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l < L; l++) {
            str.append(l).append("(").append(levelCount[l]).append(") ");
        }
        str.append("\n");

        int[] numTables = new int[D];
        for (int d = 0; d < D; d++) {
            numTables[d] = this.localRestaurants[d].getNumTables();
        }
        str.append("# tables: avg = ").append(StatUtils.mean(numTables))
                .append(". min = ").append(StatUtils.min(numTables))
                .append(". max = ").append(StatUtils.max(numTables))
                .append(". sum = ").append(StatUtils.sum(numTables));

        return str.toString();
    }

    public String printLocalRestaurants() {
        StringBuilder str = new StringBuilder();
        for (int d = 0; d < D; d++) {
            str.append(d).append(", ").append(localRestaurants[d].getNumTables()).append(":\t");
            for (SRCRPTable table : localRestaurants[d].getTables()) {
                str.append(table.getIndex())
                        .append(", ").append(table.getNumCustomers())
                        .append(", ").append(MiscUtils.formatDouble(table.getEta()))
                        .append("\t");
            }
            str.append("\n");
        }
        return str.toString();
    }

    public String printGlobalTree() {
        StringBuilder str = new StringBuilder();
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);

        int totalCus = 0;

        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString())
                    .append(", ").append(MiscUtils.formatDouble(node.getMean()))
                    //                    .append(", ").append(MiscUtils.formatDouble(node.getVariance()))
                    .append("\n");

            totalCus += node.getNumNodeCustomers();

            for (SRCRPNode child : node.getChildren()) {
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
        double wordsLlh = 0.0;
        double treeAssignment = 0.0;
        double treeMeans = 0.0;

        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            // this is not correct
            wordsLlh += node.getContent().getLogLikelihood();

            if (node.getLevel() != 0) {
                treeMeans += StatUtils.logNormalProbability(node.getMean(),
                        node.getParent().getMean(), Math.sqrt(sigmas[node.getLevel() - 1]));

//                logln("iter = " + iter 
//                        + ". node " + node.getPathString()
//                        + ". mean = " + MiscUtils.formatDouble(node.getMean())
//                        + ". parent m: " + MiscUtils.formatDouble(node.getParent().getMean())
//                        + ". parent v: " + MiscUtils.formatDouble(node.getParent().getVariance())
//                        + ". lp: " + MiscUtils.formatDouble(StatisticsUtils.logNormalProbability(node.getMean(), 
//                        node.getParent().getMean(), node.getParent().getVariance()))
//                        );
            } else {
                treeMeans += StatUtils.logNormalProbability(node.getMean(),
                        hyperparams.get(MU), Math.sqrt(hyperparams.get(SIGMA)));
            }

            if (!isLeafNode(node)) {
                treeAssignment += node.getLogJointProbability(gammas[node.getLevel()]);

                // recursive call
                for (SRCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        double restAssignment = 0.0;
        double restRegParam = 0.0;
        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            restAssignment += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA));

            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                restRegParam += StatUtils.logNormalProbability(table.getEta(),
                        table.getContent().getMean(), Math.sqrt(sigmas[table.getContent().getLevel()]));

                // debug
//                logln("iter = " + iter
//                        + ". table: " + table.getTableId()
//                        + ". table eta = " + MiscUtils.formatDouble(table.getEta())
//                        + ". mean = " + MiscUtils.formatDouble(table.getContent().getMean())
//                        + ". var = " + MiscUtils.formatDouble(table.getContent().getVariance())
//                        + ". resLlh = " + MiscUtils.formatDouble(StatisticsUtils.logNormalProbability(table.getEta(), 
//                        table.getContent().getMean(), Math.sqrt(table.getContent().getVariance())))
//                        );
            }

            for (int t = 0; t < words[d].length; t++) {
                double mean = 0.0;
                for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                    mean += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                mean /= turnCounts[d][t].getCountSum();
                resLlh += StatUtils.logNormalProbability(responses[d][t],
                        mean, Math.sqrt(hyperparams.get(RHO)));
            }
        }

        double llh = wordsLlh + treeAssignment + restAssignment
                + restRegParam + treeMeans + resLlh;

        if (verbose) {
            logln("*** obs: " + MiscUtils.formatDouble(wordsLlh)
                    + ". tree: " + MiscUtils.formatDouble(treeAssignment)
                    + ". restaurants: " + MiscUtils.formatDouble(restAssignment)
                    + ". reg-param: " + MiscUtils.formatDouble(restRegParam)
                    + ". tree-mean: " + MiscUtils.formatDouble(treeMeans)
                    + ". table-reg: " + MiscUtils.formatDouble(resLlh));
        }

        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        int count = SIGMA + 1;
        double[] newBetas = new double[betas.length];
        for (int i = 0; i < newBetas.length; i++) {
            newBetas[i] = tParams.get(count++);
        }
        double[] newGammas = new double[gammas.length];
        for (int i = 0; i < newGammas.length; i++) {
            newGammas[i] = tParams.get(count++);
        }
        double[] newSigmas = new double[sigmas.length];
        for (int i = 0; i < newSigmas.length; i++) {
            newSigmas[i] = tParams.get(count++);
        }

        double wordsLlh = 0.0;
        double treeAssignment = 0.0;
        double treeMeans = 0.0;
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            wordsLlh += node.getContent().getLogLikelihood(newBetas[node.getLevel()], uniform);

            if (node.getLevel() != 0) {
                treeMeans += StatUtils.logNormalProbability(node.getMean(),
                        node.getParent().getMean(), Math.sqrt(newSigmas[node.getLevel() - 1]));
            } else {
                treeMeans += StatUtils.logNormalProbability(node.getMean(),
                        tParams.get(MU), Math.sqrt(tParams.get(SIGMA)));
            }

            if (!isLeafNode(node)) {
                treeAssignment += node.getLogJointProbability(newGammas[node.getLevel()]);

                for (SRCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        double restAssignment = 0.0;
        double restRegParam = 0.0;
        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            restAssignment += localRestaurants[d].getJointProbabilityAssignments(tParams.get(ALPHA));

            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                restRegParam += StatUtils.logNormalProbability(table.getEta(),
                        table.getContent().getMean(), Math.sqrt(sigmas[table.getContent().getLevel()]));
            }

            for (int t = 0; t < words[d].length; t++) {
                double mean = 0.0;
                for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                    mean += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                mean /= turnCounts[d][t].getCountSum();
                resLlh += StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(tParams.get(RHO)));
            }
        }

        double llh = wordsLlh + treeAssignment + restAssignment
                + restRegParam + treeMeans + resLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        logln("Updating hyperparameters ...");
        logln("Before");
        logln("--- alpha = " + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
        logln("--- rho = " + MiscUtils.formatDouble(hyperparams.get(RHO)));
        logln("--- mu = " + MiscUtils.formatDouble(hyperparams.get(MU)));
        logln("--- sigma = " + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
        logln("--- betas = " + MiscUtils.arrayToString(betas));
        logln("--- gammas = " + MiscUtils.arrayToString(gammas));
        logln("--- sigmas = " + MiscUtils.arrayToString(sigmas));

        int count = SIGMA + 1;
        betas = new double[betas.length];
        for (int i = 0; i < betas.length; i++) {
            betas[i] = newParams.get(count++);
        }
        gammas = new double[gammas.length];
        for (int i = 0; i < gammas.length; i++) {
            gammas[i] = newParams.get(count++);
        }
        sigmas = new double[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            sigmas[i] = newParams.get(count++);
        }

        // update betas
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();
            node.getContent().setConcentration(betas[node.getLevel()]);
            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        this.hyperparams = new ArrayList<Double>();
        for (double param : newParams) {
            this.hyperparams.add(param);
        }

        logln("After");
        logln("--- alpha = " + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
        logln("--- rho = " + MiscUtils.formatDouble(hyperparams.get(RHO)));
        logln("--- mu = " + MiscUtils.formatDouble(hyperparams.get(MU)));
        logln("--- sigma = " + MiscUtils.formatDouble(hyperparams.get(SIGMA)));
        logln("--- betas = " + MiscUtils.arrayToString(betas));
        logln("--- gammas = " + MiscUtils.arrayToString(gammas));
        logln("--- sigmas = " + MiscUtils.arrayToString(sigmas));
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            int docNumTokens = 0;
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
                table.validate(msg);
                docNumTokens += table.getNumCustomers();
            }

            int trueDocNumTokens = 0;
            for (int t = 0; t < words[d].length; t++) {
                trueDocNumTokens += words[d][t].length;
            }

            if (docNumTokens != trueDocNumTokens) {
                throw new RuntimeException(msg + ". Numbers of tokens mismatch. "
                        + docNumTokens + " vs. " + trueDocNumTokens);
            }
        }

        if (globalTreeRoot.getContent().getCountSum() != totalNumObservations) {
            throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                    + globalTreeRoot.getContent().getCountSum() + " vs. " + totalNumObservations);
        }

        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            if (node.isEmpty()) {
                throw new RuntimeException(msg + ". Empty node. " + node.toString());
            }

            int curNodeNumObs = node.getContent().getCountSum();
            for (SRCRPTable table : node.customers) {
                curNodeNumObs -= table.getNumCustomers();
            }

            if (!isLeafNode(node)) {
                int subtreeNumObs = 0;
                for (SRCRPNode child : node.getChildren()) {
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
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            StringBuilder modelStr = new StringBuilder();

            // local restaurants
            for (int d = 0; d < D; d++) {
                modelStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                for (SRCRPTable table : localRestaurants[d].getTables()) {
                    modelStr.append(table.getIndex())
                            .append("\t").append(table.getEta());
                    for (String customer : table.getCustomers()) {
                        modelStr.append("\t").append(customer);
                    }
                    modelStr.append("\n");
                }
            }

            // global tree
            Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
            stack.add(globalTreeRoot);
            while (!stack.isEmpty()) {
                SRCRPNode node = stack.pop();

                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.getNumPathCustomers()).append("\n");
                modelStr.append(node.getMean()).append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");
                for (SRCRPTable table : node.getCustomers()) {
                    modelStr.append(table.getTableId()).append("\t");
                }
                modelStr.append("\n");

                for (SRCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }

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
        HashMap<String, SRCRPTable> tableMap = new HashMap<String, SRCRPTable>();

        // local restaurants
        for (int d = 0; d < D; d++) {
            int numTables = Integer.parseInt(reader.readLine().split("\t")[1]);
            for (int i = 0; i < numTables; i++) {
                String[] sline = reader.readLine().split("\t");
                int tableIndex = Integer.parseInt(sline[0]);
                double eta = Double.parseDouble(sline[1]);

                SRCRPTable table = new SRCRPTable(tableIndex, null, d, eta);
                this.localRestaurants[d].addTable(table);
                for (int j = 2; j < sline.length; j++) {
                    this.localRestaurants[d].addCustomerToTable(sline[j], tableIndex);
                }
                tableMap.put(table.getTableId(), table);
            }
        }

        // global tree
        HashMap<String, SRCRPNode> nodeMap = new HashMap<String, SRCRPNode>();
        HashMap<SRCRPNode, Integer> nodeNumPathCustomers = new HashMap<SRCRPNode, Integer>(); // for debug
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            String pathStr = line;
            int numPathCusts = Integer.parseInt(reader.readLine());
            double mean = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());

            // create node
            int lastColonIndex = pathStr.lastIndexOf(":");
            SRCRPNode parent = null;
            if (lastColonIndex != -1) {
                parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
            }

            String[] pathIndices = pathStr.split(":");
            SRCRPNode node = new SRCRPNode(Integer.parseInt(pathIndices[pathIndices.length - 1]),
                    pathIndices.length - 1, dmm, parent, mean);

            if (node.getLevel() == 0) {
                globalTreeRoot = node;
            }

            if (parent != null) {
                parent.addChild(node.getIndex(), node);
            }

            // customers (i.e., tables)
            sline = reader.readLine().split("\t");
            for (int i = 0; i < sline.length; i++) {
                SRCRPTable table = tableMap.get(sline[i]);
                node.addCustomer(table);
                table.setContent(node);
            }

            nodeMap.put(pathStr, node);
            nodeNumPathCustomers.put(node, numPathCusts);
        }

        // validate
        for (SRCRPNode node : nodeNumPathCustomers.keySet()) {
            if (node.getNumPathCustomers() != nodeNumPathCustomers.get(node)) {
                throw new RuntimeException("Numbers of customers on path mismatch. "
                        + node.toString() + "\t"
                        + node.getNumPathCustomers() + " vs. " + nodeNumPathCustomers.get(node));
            }
        }

        // update inactive children list
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();
            if (!isLeafNode(node)) {
                node.fillInactiveChildIndices();
                for (SRCRPNode child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        // update inactive tables
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d].fillInactiveTableIndices();
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
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    String[] sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + t + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][t][n] = Integer.parseInt(sline[1]);
                    turnCounts[d][t].increment(z[d][t][n]);
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
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            for (SRCRPNode child : node.getChildren()) {
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
                    .append(" (").append(node.getNumNodeCustomers())
                    .append("; ").append(node.getNumPathCustomers())
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.getMean()))
                    //                    .append("; ").append(MiscUtils.formatDouble(node.getVariance()))
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

        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(this.globalTreeRoot);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            double[] distribution = node.getContent().getDistribution();
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
            for (SRCRPTable table : this.localRestaurants[d].getTables()) {
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

    public double[][] getRegressionValues() {
        double[][] regValues = new double[D][];
        for (int d = 0; d < D; d++) {
            regValues[d] = new double[responses[d].length];
            for (int t = 0; t < responses[d].length; t++) {
                double sum = 0.0;
                for (SRCRPTable table : localRestaurants[d].getTables()) {
                    sum += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                regValues[d][t] = sum / words[d][t].length;

                // debug
//                logln("d = " + d + ". t = " + t 
//                        + ". count = " + words[d][t].length
//                        + " vs. " + turnCounts[d][t].getCountSum()
//                        + ". true: " + MiscUtils.formatDouble(responses[d][t])
//                        + ". pred: " + MiscUtils.formatDouble(regValues[d][t]))
//                        ;
//                for(SHDPTable table : localRestaurants[d].getTables()){
//                    if(turnCounts[d][t].getCount(table.getIndex()) > 0)
//                        logln("--- table " + table.getIndex()
//                            + ". " + MiscUtils.formatDouble(table.getEta())
//                            + ". " + table.getNumCustomers()
//                            + ". " + turnCounts[d][t].getCount(table.getIndex())
//                            );
//                }
//                System.out.println();
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

        // initialize structure
        initializeDataStructure();

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    sampleTableForToken(d, t, n, !REMOVE, !OBSERVED, ADD, !EXTEND);
                }
            }
        }

        // iterate
        ArrayList<double[][]> predResponsesList = new ArrayList<double[][]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        sampleTableForToken(d, t, n, REMOVE, !OBSERVED, ADD, !EXTEND);
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
}

class SRCRPNode extends TreeNode<SRCRPNode, DirMult> {

    ArrayList<SRCRPTable> customers;
    int numPathCustomers; // number of customers on the path from root to this node (including customers in the subtree)
    private double mean;

    SRCRPNode(int index, int level, DirMult content, SRCRPNode parent,
            double mean) {
        super(index, level, content, parent);
        this.numPathCustomers = 0;
        this.customers = new ArrayList<SRCRPTable>();
        this.mean = mean;
    }

    double getLogJointProbability(double gamma) {
        ArrayList<Integer> numChildrenCusts = new ArrayList<Integer>();
        for (SRCRPNode child : this.getChildren()) {
            numChildrenCusts.add(child.getNumPathCustomers());
        }
        numChildrenCusts.add(this.getNumNodeCustomers());
        return SamplerUtils.getAssignmentJointLogProbability(numChildrenCusts, gamma);
    }

//    double computeWordLogLikelihood(){
//        
//    }
    double getMean() {
        return mean;
    }

    void setMean(double mean) {
        this.mean = mean;
    }

    ArrayList<SRCRPTable> getCustomers() {
        return this.customers;
    }

    void addCustomer(SRCRPTable c) {
        this.customers.add(c);
        this.changeNumPathCustomers(1);
    }

    void removeCustomer(SRCRPTable c) {
        this.customers.remove(c);
        this.changeNumPathCustomers(-1);
    }

    int getNumPathCustomers() {
        return numPathCustomers;
    }

    int getNumNodeCustomers() {
        return this.customers.size();
    }

    String getPseudoChildPathString() {
        return this.getPathString() + ":" + SRCRPSampler.PSEUDO_NODE_INDEX;
    }

    void changeNumPathCustomers(int delta) {
        SRCRPNode node = this;
        while (node != null) {
            node.numPathCustomers += delta;
            if (node.numPathCustomers < 0) {
                throw new RuntimeException("Negative count. " + node.toString());
            }
            node = node.getParent();
        }
    }

    boolean isEmpty() {
        return this.getNumNodeCustomers() == 0;
    }

    void validate(String str) {
        int sumChildrentPathNumCustomers = 0;
        for (SRCRPNode child : this.getChildren()) {
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

        if (this.isEmpty()) {
            throw new RuntimeException(str + ". Empty node: " + this.toString());
        }
    }

    String[] getTopWords(ArrayList<String> vocab, int numWords) {
        ArrayList<RankingItem<String>> topicSortedVocab = IOUtils.getSortedVocab(content.getDistribution(), vocab);
        String[] topWords = new String[numWords];
        for (int i = 0; i < numWords; i++) {
            topWords[i] = topicSortedVocab.get(i).getObject();
        }
        return topWords;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", #ch = ").append(getChildren().size())
                .append(", #n = ").append(getNumNodeCustomers())
                .append(", #p = ").append(getNumPathCustomers())
                .append(", #o = ").append(getContent().getCountSum())
                .append(", m: ").append(MiscUtils.formatDouble(mean))
                //                .append(", v: ").append(MiscUtils.formatDouble(variance))
                .append("]");
        return str.toString();
    }
}

class SRCRPTable extends FullTable<String, SRCRPNode> {

    final int restIndex;
    private double eta;

    SRCRPTable(int index, SRCRPNode content, int restId, double eta) {
        super(index, content);
        this.restIndex = restId;
        this.customers = new ArrayList<String>();
        this.eta = eta;
    }

    public String getTableId() {
        return restIndex + ":" + index;
    }

    int getRestaurantIndex() {
        return this.restIndex;
    }

    double getEta() {
        return this.eta;
    }

    void setEta(double eta) {
        this.eta = eta;
    }

    void validate(String msg) {
        if (this.customers.size() != this.getNumCustomers()) {
            throw new RuntimeException("Numbers of customers mismatch");
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("(")
                .append(restIndex).append("-").append(index)
                .append(". #c: ").append(getNumCustomers())
                .append(", m: ").append(MiscUtils.formatDouble(eta))
                .append(")")
                .append(" -> ").append(content.toString());
        return str.toString();
    }
}

class MultiscaleStateSpace {

    int MEAN_INDEX = 0;
    int VARIANCE_INDEX = 1;
    SRCRPNode root;
    double rootPriorMean; // \mu^*
    double rootPriorVar; // \sigma^*
    int L;
    double[] priorVariances; // prior variance [L]
    double[] sumPriorVariances; // prior covariance: Sigma [L]
    double[] varianceRatios; // Sigma_parent / Sigma_child [L]
    HashMap<SRCRPNode, HashMap<Integer, double[]>> transValues;
    HashMap<SRCRPNode, double[]> filteredValues;
    HashMap<SRCRPNode, double[]> smoothedValues;

    public MultiscaleStateSpace(SRCRPNode root, double rootPMean, double rootPVar,
            double[] priorVars, int L) {
        this.root = root;
        this.rootPriorMean = rootPMean;
        this.rootPriorVar = rootPVar;
        this.priorVariances = priorVars;
        this.L = L;

        this.sumPriorVariances = new double[L];
        this.varianceRatios = new double[L]; // the first value is undefined

        this.sumPriorVariances[0] = this.priorVariances[0] + this.rootPriorVar;
        this.varianceRatios[0] = this.sumPriorVariances[0] / this.rootPriorVar;
        for (int l = 1; l < L; l++) {
            this.sumPriorVariances[l] = this.sumPriorVariances[l - 1] + this.priorVariances[l];
            this.varianceRatios[l] = this.sumPriorVariances[l - 1] / this.sumPriorVariances[l];
        }

        this.transValues = new HashMap<SRCRPNode, HashMap<Integer, double[]>>();
        this.filteredValues = new HashMap<SRCRPNode, double[]>();
        this.smoothedValues = new HashMap<SRCRPNode, double[]>();

        // debug
//        System.out.println("root prior mean = " + rootPMean);
//        System.out.println("root prior var = " + rootPVar);
//        System.out.println("prior vars = " + MiscUtils.arrayToString(priorVariances));
//        System.out.println("sum prior vars = " + MiscUtils.arrayToString(sumPriorVariances));
//        System.out.println("var ratios = " + MiscUtils.arrayToString(varianceRatios));
    }

    private double[] computePosterior(SRCRPNode node) {
        int level = node.getLevel();
        int numObs = node.getNumNodeCustomers();

        double parentSumVar = this.rootPriorVar;
        if (node.getLevel() > 0) {
            parentSumVar = this.sumPriorVariances[level - 1];
        }
        double curPriorVar = this.priorVariances[level];

        double var = parentSumVar * curPriorVar / (curPriorVar + numObs * parentSumVar);

        double sumObs = 0.0;
        for (SRCRPTable table : node.getCustomers()) {
            sumObs += table.getEta();
        }
        double mean = var * sumObs / curPriorVar + var * this.rootPriorMean / parentSumVar;

        // debug
//        ArrayList<Double> obs = new ArrayList<Double>();
//        for(SRCRPTable table : node.getCustomers())
//            obs.add(table.getEta());
//        System.out.println("--- --- Computing posterior: " + node.getPathString());
//        System.out.println("--- --- sumPriorVar = " + parentSumVar);
//        System.out.println("--- --- priorVar = " + curPriorVar);
//        System.out.println("--- --- mean = " + mean);
//        System.out.println("--- --- var = " + var);
//        System.out.println("--- --- emp mean = " + StatisticsUtils.mean(obs));

        double[] posterior = {mean, var};
        return posterior;
    }

    void upwardFilter() {
        // store all nodes in a stack
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        Stack<SRCRPNode> tempStack = new Stack<SRCRPNode>();
        tempStack.add(root);
        while (!tempStack.isEmpty()) {
            SRCRPNode node = tempStack.pop();
            stack.add(node);

            for (SRCRPNode child : node.getChildren()) {
                tempStack.add(child);
            }
        }

        // upward filter
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();
            int nodeLevel = node.getLevel();
            int childLevel = nodeLevel + 1;
            HashMap<Integer, double[]> nodeTransValues = new HashMap<Integer, double[]>();

            // compute the posterior
            double[] posterior = computePosterior(node); // child 0
            nodeTransValues.put(SRCRPSampler.PSEUDO_NODE_INDEX, posterior);

            // debug
//            System.out.println(node.getPathString() 
//                    + ": mean = " + MiscUtils.formatDouble(posterior[MEAN_INDEX])
//                    + ". var = " + MiscUtils.formatDouble(posterior[VARIANCE_INDEX]));

            // prediction from each child node
            for (SRCRPNode child : node.getChildren()) {
                double varRatio = this.varianceRatios[childLevel];
                double transMean = varRatio * filteredValues.get(child)[MEAN_INDEX];
                double transVar = varRatio * varRatio * filteredValues.get(child)[VARIANCE_INDEX]
                        + varRatio * this.priorVariances[nodeLevel];
                double[] childTransValues = {transMean, transVar};
                nodeTransValues.put(child.getIndex(), childTransValues);

                // debug
//                System.out.println("--- child " + child.getPathString()
//                        + ". var ratio = " + MiscUtils.formatDouble(varRatio)
//                        + ". transMean = " + MiscUtils.formatDouble(transMean)
//                        + ". transVar = " + MiscUtils.formatDouble(transVar)
//                        );
            }
            this.transValues.put(node, nodeTransValues);

            // compute filtered values (combine all children)
            double sumInverseVar = 1.0 / posterior[VARIANCE_INDEX];
            for (SRCRPNode child : node.getChildren()) {
                sumInverseVar += 1.0 / nodeTransValues.get(child.getIndex())[VARIANCE_INDEX];
            }
            double filteredVar = 1.0 / (sumInverseVar + (1 - node.getNumChildren()) / this.sumPriorVariances[nodeLevel]);

            double filteredMean = 0.0;
            for (double[] tValues : nodeTransValues.values()) {
                filteredMean += tValues[MEAN_INDEX] / tValues[VARIANCE_INDEX];
            }
            filteredMean *= filteredVar;
            double[] filteredValue = {filteredMean, filteredVar};

            // debug
//            System.out.println(">>> combine: filteredMean = " + MiscUtils.formatDouble(filteredMean)
//                    + ". filteredVar = " + MiscUtils.formatDouble(filteredVar)
//                    + "\n\n");


            this.filteredValues.put(node, filteredValue);
        }
    }

    void downwardSmooth() {
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(root);

        double rootSmoothedMean = this.filteredValues.get(root)[MEAN_INDEX];
        double rootSmoothedVar = this.filteredValues.get(root)[VARIANCE_INDEX];
        double[] rootSmoothedValue = {rootSmoothedMean, rootSmoothedVar};
        this.smoothedValues.put(root, rootSmoothedValue);

        // debug
//        System.out.println("Downward smoothing ...");
//        System.out.println("root smoothed mean = " + MiscUtils.formatDouble(rootSmoothedMean));
//        System.out.println("root smoothed var = "  + MiscUtils.formatDouble(rootSmoothedVar));

        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }

            if (node.equals(root)) {
                continue;
            }
            int nodeLevel = node.getLevel();
            double[] parentSmoothedVals = this.smoothedValues.get(node.getParent());
            double[] filteredVals = this.filteredValues.get(node);
            double[] transVals = this.transValues.get(node.getParent()).get(node.getIndex());

            double Jnode = this.varianceRatios[nodeLevel]
                    * filteredVals[VARIANCE_INDEX] / transVals[VARIANCE_INDEX];
            double smoothedMean = filteredVals[MEAN_INDEX]
                    + Jnode * (parentSmoothedVals[MEAN_INDEX] - transVals[MEAN_INDEX]);
            double smoothedVar = filteredVals[VARIANCE_INDEX]
                    + Jnode * Jnode * (parentSmoothedVals[VARIANCE_INDEX] - transVals[VARIANCE_INDEX]);
            double[] smoothedVals = {smoothedMean, smoothedVar};
            this.smoothedValues.put(node, smoothedVals);

            // debug
//            System.out.println(node.getPathString()
//                    + ". smoothed mean = " + MiscUtils.formatDouble(smoothedVals[MEAN_INDEX])
//                    + ". smoothed var = " + MiscUtils.formatDouble(smoothedVals[VARIANCE_INDEX]));
        }
    }

    void update() {
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();

            double[] smoothedVals = smoothedValues.get(node);
            double newMean = SamplerUtils.getGaussian(smoothedVals[MEAN_INDEX], smoothedVals[VARIANCE_INDEX]);
            node.setMean(newMean);
//            node.setVariance(smoothedVals[VARIANCE_INDEX]);

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
    }

    private void printTree(HashMap<SRCRPNode, Double> trueMeans) {
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();
            ArrayList<Double> obs = new ArrayList<Double>();
            for (SRCRPTable table : node.getCustomers()) {
                obs.add(table.getEta());
            }
            double avg = StatUtils.mean(obs);

            for (int i = 0; i < node.getLevel(); i++) {
                System.out.print("\t");
            }
            System.out.println("["
                    + node.getPathString()
                    + "]\tfilter " + MiscUtils.arrayToString(filteredValues.get(node))
                    + "\tsmooth " + MiscUtils.arrayToString(smoothedValues.get(node))
                    + "\ttrue-mean = " + MiscUtils.formatDouble(trueMeans.get(node))
                    + "\temp-mean = " + MiscUtils.formatDouble(avg)
                    + "\n");

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }
    }

    public static void main(String[] args) {
        testFilterSmooth();
    }

    private static void testFilterSmooth() {
        int L = 3;
        double priorRootMean = 0.0;
        double priorRootVar = 0.5;
        int numObs = 1000;
        double[] priorVariances = {0.1, 2.5, 5};
        HashMap<SRCRPNode, Double> trueMeans = new HashMap<SRCRPNode, Double>();

        SRCRPNode root = new SRCRPNode(0, 0, null, null, priorRootMean);
        generateObservations(root, numObs, priorVariances[0]);
        trueMeans.put(root, root.getMean());

        SRCRPNode node00 = createNode(0, 1, root, priorVariances[0]);
        generateObservations(node00, numObs, priorVariances[1]);
        trueMeans.put(node00, node00.getMean());

        SRCRPNode node01 = createNode(1, 1, root, priorVariances[0]);
        generateObservations(node01, numObs, priorVariances[1]);
        trueMeans.put(node01, node01.getMean());

        SRCRPNode node000 = createNode(0, 2, node00, priorVariances[1]);
        generateObservations(node000, numObs, priorVariances[2]);
        trueMeans.put(node000, node000.getMean());

        SRCRPNode node001 = createNode(1, 2, node00, priorVariances[1]);
        generateObservations(node001, numObs, priorVariances[2]);
        trueMeans.put(node001, node001.getMean());

        SRCRPNode node010 = createNode(0, 2, node01, priorVariances[1]);
        generateObservations(node010, numObs, priorVariances[2]);
        trueMeans.put(node010, node010.getMean());

        // print true tree
        Stack<SRCRPNode> stack = new Stack<SRCRPNode>();
        stack.add(root);
        while (!stack.isEmpty()) {
            SRCRPNode node = stack.pop();
            ArrayList<Double> obs = new ArrayList<Double>();
            for (SRCRPTable table : node.getCustomers()) {
                obs.add(table.getEta());
            }

            for (int i = 0; i < node.getLevel(); i++) {
                System.out.print("\t");
            }
            double stdv = StatUtils.standardDeviation(obs);
            System.out.println("["
                    + node.getPathString()
                    + ", " + MiscUtils.formatDouble(node.getMean())
                    //                    + ", " + MiscUtils.formatDouble(node.getVariance())
                    + "]\t"
                    + "\t" + MiscUtils.formatDouble(StatUtils.mean(obs))
                    + "\t" + MiscUtils.formatDouble(stdv * stdv)
                    //                    + "\t" + MiscUtils.listToString(obs)
                    + "\n");

            for (SRCRPNode child : node.getChildren()) {
                stack.add(child);
            }
        }

        MultiscaleStateSpace model = new MultiscaleStateSpace(root, priorRootMean, priorRootVar, priorVariances, L);
        model.upwardFilter();
        model.downwardSmooth();
        model.printTree(trueMeans);
    }

    private static SRCRPNode createNode(int index, int level, SRCRPNode parent, double meanVar) {
        double mean = SamplerUtils.getGaussian(parent.getMean(), meanVar);
        SRCRPNode node = new SRCRPNode(index, level, null, parent, mean);
        parent.addChild(index, node);
        return node;
    }

    private static void generateObservations(SRCRPNode node, int numObs, double obsVar) {
        double sum = 0.0;
        for (int i = 0; i < numObs; i++) {
            double obs = SamplerUtils.getGaussian(node.getMean(), obsVar);
            SRCRPTable table = new SRCRPTable(i, node, -1, obs);
            node.addCustomer(table);

            sum += obs;
        }

        System.out.println("---> emp mean = " + sum / numObs);
    }
}