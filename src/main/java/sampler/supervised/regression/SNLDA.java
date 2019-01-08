package sampler.supervised.regression;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.LabelTextDataset;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import optimization.RidgeLinearRegressionOptimizable;
import optimization.RidgeLogisticRegressionOptimizable;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.PredictionUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import util.evaluation.ClassificationEvaluation;

/**
 *
 * @author vietan
 */
public class SNLDA extends AbstractSampler {

    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    // hyperparameters for fixed-height tree
    protected double[] alphas;          // [L-1]
    protected double[] betas;           // [L]
    protected double[] pis;     // [L-1] mean of bias coins
    protected double[] gammas;    // [L-1] scale of bias coins
    protected double rho;
    protected double mu;
    protected double[] sigmas;

    // inputs
    protected int[][] words; // all words
    protected double[] responses; // [D] continous responses
    protected int[] labels; // [D] binary responses
    protected ArrayList<Integer> docIndices; // indices of docs under consideration
    protected int V;    // vocabulary size
    protected int[] Ks; // [L-1]: number of children per node at each level
    protected PathAssumption path;
    // derived
    protected int D; // number of documents
    protected int L;
    // latent
    Node[][] z;
    Node root;
    // internal
    private int numTokensAccepted;
    private double[] docMeans;
    private boolean isBinary;
    private Set<Integer> positives;
    private double uniform;
    private boolean isRooted;

    // cached probabilities computed at the first level
    private HashMap<Node, Double> cachedProbabilities;

    public SNLDA() {
        this.basename = "SNLDA";
    }

    public SNLDA(String bname) {
        this.basename = bname;
    }

    public void configure(SNLDA sampler) {
        this.isBinary = sampler.isBinary;
        if (this.isBinary) {
            this.configureBinary(sampler.folder,
                    sampler.V,
                    sampler.Ks,
                    sampler.alphas,
                    sampler.betas,
                    sampler.pis,
                    sampler.gammas,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.initState,
                    sampler.path,
                    sampler.paramOptimized,
                    sampler.isRooted,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        } else {
            this.configureContinuous(sampler.folder,
                    sampler.V,
                    sampler.Ks,
                    sampler.alphas,
                    sampler.betas,
                    sampler.pis,
                    sampler.gammas,
                    sampler.rho,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.initState,
                    sampler.path,
                    sampler.paramOptimized,
                    sampler.isRooted,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        }
    }

    public void configureBinary(String folder,
            int V, int[] Ks,
            double[] alphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            double mu,
            double[] sigmas,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt, boolean isRooted,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / this.V;
        this.Ks = Ks;
        this.L = this.Ks.length + 1;

        this.alphas = alphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.mu = mu;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.path = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.isBinary = true;
        this.isRooted = isRooted;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- pis:\t" + MiscUtils.arrayToString(pis));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
            logln("--- is rooted:\t" + this.isRooted);
        }

        validateInputHyperparameters();
    }

    public void configureContinuous(String folder,
            int V, int[] Ks,
            double[] alphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            double rho,
            double mu,
            double[] sigmas,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt, boolean isRooted,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / this.V;
        this.Ks = Ks;
        this.L = this.Ks.length + 1;

        this.alphas = alphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.rho = rho;
        this.mu = mu;
        this.sigmas = sigmas;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.path = pathAssumption;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.isBinary = false;
        this.isRooted = isRooted;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- pis:\t" + MiscUtils.arrayToString(pis));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
            logln("--- is rooted:\t" + this.isRooted);
        }

        validateInputHyperparameters();
    }

    private void validateInputHyperparameters() {
        if (L - 1 != Ks.length) {
            throw new MismatchRuntimeException(L - 1, Ks.length);
        }
        if (alphas.length != L - 1) {
            throw new MismatchRuntimeException(alphas.length, L - 1);
        }
        if (betas.length != L) {
            throw new MismatchRuntimeException(betas.length, L);
        }
        if (pis.length != L - 1) {
            throw new MismatchRuntimeException(pis.length, L - 1);
        }
        if (gammas.length != L - 1) {
            throw new MismatchRuntimeException(gammas.length, L - 1);
        }
        if (sigmas.length != L) {
            throw new MismatchRuntimeException(sigmas.length, L - 1);
        }
        if (!isBinary && rho == 0.0) {
            throw new RuntimeException("Rho should not be 0 when the response is continuous");
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename);
        str.append("_Ks");
        for (int K : Ks) {
            str.append("-").append(K);
        }
        str.append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG);
        str.append("_a");
        for (double la : alphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_p");
        for (double gm : pis) {
            str.append("-").append(MiscUtils.formatDouble(gm));
        }
        str.append("_g");
        for (double gs : gammas) {
            str.append("-").append(MiscUtils.formatDouble(gs));
        }
        str.append("_r-").append(MiscUtils.formatDouble(rho));
        str.append("_m-").append(MiscUtils.formatDouble(mu));
        str.append("_s");
        for (double s : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(s));
        }
        str.append("_opt-").append(this.paramOptimized);
        str.append("_bin-").append(this.isBinary);
        str.append("_path-").append(this.path);
        str.append("_root-").append(this.isRooted);
        this.name = str.toString();
    }

    protected double getAlpha(int l) {
        return this.alphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getPi(int l) {
        return this.pis[l];
    }

    protected double getGamma(int l) {
        return this.gammas[l];
    }

    protected double getSigma(int l) {
        return this.sigmas[l];
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath() + "\n"
                + printGlobalTreeSummary() + "\n";
    }

    public boolean isLeafNode(int level) {
        return level == L - 1;
    }

    public double[] getPredictedValues() {
        return docMeans;
    }

    /**
     * Setting up text data.
     *
     * @param docWords
     * @param docIndices
     */
    private void setupTextData(int[][] docWords, ArrayList<Integer> docIndices) {
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.numTokens = 0;
        this.D = this.docIndices.size();
        this.words = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = docWords[dd];
            this.numTokens += this.words[ii].length;
        }

        if (verbose) {
            logln("--- # all docs:\t" + words.length);
            logln("--- # selected docs:\t" + D);
            logln("--- # tokens:\t" + numTokens);
        }
    }

    /**
     * Set up continuous responses.
     *
     * @param docResponses
     */
    public void setContinuousResponses(double[] docResponses) {
        this.responses = new double[D];
        for (int ii = 0; ii < D; ii++) {
            this.responses[ii] = docResponses[this.docIndices.get(ii)];
        }
        if (verbose) {
            logln("--- continuous responses:");
            logln("--- --- mean\t" + MiscUtils.formatDouble(
                    StatUtils.mean(responses)));
            logln("--- --- stdv\t" + MiscUtils.formatDouble(
                    StatUtils.standardDeviation(responses)));
            int[] histogram = StatUtils.bin(responses, 10);
            for (int ii = 0; ii < histogram.length; ii++) {
                logln("--- --- " + ii + "\t" + histogram[ii]);
            }
        }
    }

    /**
     * Set up binary responses.
     *
     * @param docLabels
     */
    public void setBinaryResponses(int[] docLabels) {
        this.labels = new int[D];
        this.positives = new HashSet<Integer>();
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.labels[ii] = docLabels[dd];
            if (this.labels[ii] == POSITVE) {
                this.positives.add(ii);
            }
        }
        if (verbose) {
            logln("--- binary responses:");
            int posCount = this.positives.size();
            logln("--- --- # postive: " + posCount
                    + " (" + ((double) posCount / D) + ")");
            logln("--- --- # negative: " + (D - posCount));
        }
    }

    /**
     * Set up training data with continuous responses.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     * @param docResponses Continuous responses
     */
    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            double[] docResponses) {
        setupTextData(docWords, docIndices);
        setContinuousResponses(docResponses);
    }

    /**
     * Set up training data with binary responses.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     * @param docLabels Binary labels
     */
    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            int[] docLabels) {
        setupTextData(docWords, docIndices);
        setBinaryResponses(docLabels);
    }

    /**
     * Set up test data.
     *
     * @param docWords Test documents
     * @param docIndices Indices of test documents
     */
    public void test(int[][] docWords, ArrayList<Integer> docIndices) {
        setupTextData(docWords, docIndices);
    }

    /**
     * Set up test data.
     *
     * @param stateFile Input file storing trained model
     * @param testStateFile Output file to store assignments
     * @param predictionFile Output file to store predictions at different test
     * iterations using the given trained model
     * @return Prediction on all documents using the given model
     */
    public double[] sampleTest(File stateFile, File testStateFile, File predictionFile) {
        setTestConfigurations(BURN_IN / 2, MAX_ITER / 2, LAG / 2);
        if (stateFile == null) {
            stateFile = getFinalStateFile();
        }
        inputModel(stateFile.toString()); // input stored model
        initializeDataStructure(); // initialize data

        // store predictions at different test iterations
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();

        // sample topic assignments for test document
        for (iter = 0; iter < this.testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + ". current thread: " + Thread.currentThread().getId();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. " + str + "\n" + getCurrentState());
                }
            }

            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD);
            }

            // store prediction (on all documents) at a test iteration
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double[] predResponses = new double[D];
                System.arraycopy(docMeans, 0, predResponses, 0, D);
                predResponsesList.add(predResponses);

                if (responses != null) { // debug
                    evaluatePerformances();
                }
            }
        }

        // output state file containing the assignments for test documents
        if (testStateFile != null) {
            outputState(testStateFile.getAbsolutePath(), false, true);
        }

        // store predictions if necessary
        if (predictionFile != null) {
            PredictionUtils.outputSingleModelRegressions(predictionFile, predResponsesList);
        }

        // average over all stored predictions
        double[] predictions = new double[D];
        for (int dd = 0; dd < D; dd++) {
            for (double[] predResponses : predResponsesList) {
                predictions[dd] += predResponses[dd] / predResponsesList.size();
            }
        }
        return predictions;
    }

    @Override
    public void initialize() {
        initialize(null, null);
    }

    public void initialize(double[][] priorTopics, double[] initEtas) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;
        initializeModelStructure(priorTopics, initEtas);
        initializeDataStructure();
        initializeAssignments();
        updateEtas();

        if (verbose) {
            logln("--- Done initializing.\n" + printGlobalTree());
            logln("\n" + printGlobalTreeSummary() + "\n");
            getLogLikelihood();
        }

        outputTopicTopWords(new File(getSamplerFolderPath(), "init-" + TopWordFile), 20);
        validate("Initialized");
    }

    protected void initializeModelFirstLevelNodes(double[][] priorTopics, double[] initEtas) {
        int level = 1;
        for (int kk = 0; kk < Ks[0]; kk++) {
            // prior topic
            double[] prior;
            if (priorTopics == null) {
                prior = new double[V];
                Arrays.fill(prior, uniform);
            } else {
                prior = priorTopics[kk];
            }

            // initial eta
            double eta;
            if (initEtas != null) {
                eta = initEtas[kk];
            } else {
                eta = SamplerUtils.getGaussian(mu, getSigma(level));
            }

            // initialize
            DirMult topic = new DirMult(V, getBeta(1) * V, prior);
            Node node = new Node(iter, kk, level, topic, root, eta);
            this.root.addChild(kk, node);
        }
    }

    protected void initializeModelStructure(double[][] priorTopics, double[] initEtas) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        // initialize root node
        DirMult rootTopic = new DirMult(V, getBeta(0) * V, uniform);
        double rootEta = SamplerUtils.getGaussian(mu, getSigma(0));
        this.root = new Node(iter, 0, 0, rootTopic, null, rootEta);

        // first level
        initializeModelFirstLevelNodes(priorTopics, initEtas);
        Queue<Node> queue = new LinkedList<>();
        for (Node child : root.getChildren()) {
            queue.add(child);
        }

        // from 2nd-level downward
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            int level = node.getLevel();
            if (level < L - 1) {
                for (int kk = 0; kk < Ks[level]; kk++) {
                    DirMult topic = new DirMult(V, getBeta(level + 1) * V, uniform);
                    Node child = new Node(iter, kk, level + 1, topic, node,
                            SamplerUtils.getGaussian(mu, getSigma(level)));
                    node.addChild(kk, child);
                    queue.add(child);
                }
            }
        }

        // initialize pi's and theta's
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getLevel() < L - 1) {
                node.initializeGlobalTheta();
                node.initializeGlobalPi();
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        if (verbose) {
            logln("--- --- Initialized model structure.\n" + printGlobalTreeSummary());
        }
    }

    protected void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }
        this.z = new Node[D][];
        for (int dd = 0; dd < D; dd++) {
            this.z[dd] = new Node[words[dd].length];
        }
        this.docMeans = new double[D];
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments. " + initState);
        }
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
    }

    private void initializePresetAssignments() {
//        SLDA slda = new SLDA();
//        slda.setDebug(false);
//        slda.setVerbose(verbose);
//        slda.setLog(false);
//        
//        slda.configure(new File(foldFolder, modelFolder).getAbsolutePath(),
//                trainData.getWordVocab().size(), K,
//                alpha, beta, rho, mu, sigma,
//                initState, paramOpt, hasBias,
//                burn_in, max_iters, sample_lag, report_interval);
//        
//        LDA lda = runLDA(words, Ks[0], V, null);
//        int[][] ldaZs = lda.getZs();
//
//        for (int dd = 0; dd < D; dd++) {
//            for (int nn = 0; nn < words[dd].length; nn++) {
//                Node sampledNode = sampleNode(dd, nn, root.getChild(ldaZs[dd][nn]));
//                addToken(dd, nn, sampledNode, ADD, ADD);
//            }
//        }
    }

    /**
     * Add a token to a node.
     *
     * @param dd
     * @param nn
     * @param node
     * @param addToData
     * @param addToModel
     */
    private void addToken(int dd, int nn, Node node,
            boolean addToData, boolean addToModel) {
        if (addToModel) {
            node.getContent().increment(words[dd][nn]);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.incrementSubtreeWordCount(words[dd][nn]);
                tempNode = tempNode.getParent();
            }
        }
        if (addToData) {
            docMeans[dd] += node.pathEta / this.words[dd].length;
            node.nodeDocCounts.increment(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeDocCounts.increment(dd);
                tempNode = tempNode.getParent();
            }
        }
    }

    /**
     * Remove a token from a node.
     *
     * @param dd
     * @param nn
     * @param node
     * @param removeFromData
     * @param removeFromModel
     */
    private void removeToken(int dd, int nn, Node node,
            boolean removeFromData, boolean removeFromModel) {
        if (removeFromData) {
            docMeans[dd] -= node.pathEta / this.words[dd].length;
            node.nodeDocCounts.decrement(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeDocCounts.decrement(dd);
                tempNode = tempNode.getParent();
            }
        }
        if (removeFromModel) {
            node.getContent().decrement(words[dd][nn]);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.decrementSubtreeWordCount(words[dd][nn]);
                tempNode = tempNode.getParent();
            }
        }
    }

    @Override
    public void iterate() {
        if (isReporting) {
            System.out.println("\n");
            logln("Iteration " + iter + " / " + MAX_ITER);
        }
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);
        updateEtas();
    }

    // ONLY FOR DEBUGGING
    private Node sampleNodeSimple(int dd, int nn, boolean observed) {
        double[] logprobs = new double[Ks[0]];
        for (int kk = 0; kk < Ks[0]; kk++) {
            Node node = root.getChild(kk);
            logprobs[kk] = Math.log(node.subtreeDocCounts.getCount(dd) + getAlpha(0))
                    + Math.log(node.getNodeWordProbability(words[dd][nn]));
            if (observed) {
                double mean = docMeans[dd] + node.pathEta / words[dd].length;
                logprobs[kk] += StatUtils.logNormalProbability(responses[dd], mean, Math.sqrt(rho));
            }
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        return root.getChild(sampledZ);
    }

    /**
     * Gibbs sample node assignment for all tokens.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (isReporting) {
            logln("+++ Gibbs sampling Zs ...");
        }
        numTokensChanged = 0;
        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                Node sampledNode = sampleNode(dd, nn, root);
                if (z[dd][nn] == null || !z[dd][nn].equals(sampledNode)) {
                    numTokensChanged++;
                }
                z[dd][nn] = sampledNode;

                // add
                addToken(dd, nn, z[dd][nn], addToData, addToModel);
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
            );
        }
        return eTime;
    }

    /**
     * MH sample node assignment for all tokens.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observed
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData, boolean observed) {
        if (isReporting) {
            logln("+++ MH-sampling Zs ...");
        }
        numTokensChanged = 0;
        numTokensAccepted = 0;

        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                boolean accept = false;
                Node sampledNode = sampleNode(dd, nn, root);
                if (z[dd][nn] == null) {
                    accept = true;
                    numTokensChanged++;
                    numTokensAccepted++;
                } else if (sampledNode.equals(z[dd][nn])) {
                    accept = true;
                    numTokensAccepted++;
                } else {
                    if (evaluateProposedNode(dd, nn, z[dd][nn], sampledNode, observed)) {
                        accept = true;
                        numTokensAccepted++;
                    }
                }
                if (accept) {
                    if (z[dd][nn] != null && !z[dd][nn].equals(sampledNode)) {
                        numTokensChanged++;
                    }
                    z[dd][nn] = sampledNode;
                }

                // add
                addToken(dd, nn, z[dd][nn], addToData, addToModel);
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # accepted: " + numTokensAccepted
                    + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Recursively sample node level-by-level.
     *
     * @param dd
     * @param nn
     * @param curNode
     */
    private Node sampleNode(int dd, int nn, Node curNode) {
        if (curNode.isLeaf()) {
            return curNode;
        }
        if (curNode.isRoot()) {
            cachedProbabilities = new HashMap<>();
        }

        int level = curNode.getLevel();
        ArrayList<Node> nodeList = new ArrayList<>();
        ArrayList<Double> probList = new ArrayList<>();

        // staying at this node
        double gamma = getGamma(level);
        double stayprob;
        if (curNode.isRoot() && !isRooted) {
            stayprob = 0.0;
        } else {
            stayprob = (curNode.nodeDocCounts.getCount(dd) + gamma * curNode.pi)
                    / (curNode.subtreeDocCounts.getCount(dd) + gamma);
            double wordprob = curNode.getNodeWordProbability(words[dd][nn]);
            double prob = stayprob * wordprob;
            probList.add(prob);
            nodeList.add(curNode);

            if (curNode.isRoot()) {
                cachedProbabilities.put(curNode, prob);
            }
        }

        // moving to one of the children nodes
        double alpha = getAlpha(level);
        double passprob = 1.0 - stayprob;
        int KK = curNode.getNumChildren();
        double norm = curNode.getPassingCount(dd) + alpha * KK;
        for (Node child : curNode.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeDocCounts.getCount(dd)
                    + alpha * KK * curNode.theta[kk]) / norm;
            double wordprob = child.getSubtreeWordProbability(words[dd][nn]);
            double prob = passprob * pathprob * wordprob;
            probList.add(prob);
            nodeList.add(child);

            if (curNode.isRoot()) {
                cachedProbabilities.put(child, prob);
            }
        }

        int sampledIdx = SamplerUtils.scaleSample(probList);
        Node sampledNode = nodeList.get(sampledIdx);

        if (sampledNode.equals(curNode)) {
            return curNode;
        } else {
            return sampleNode(dd, nn, sampledNode);
        }
    }

    /**
     * Evaluate proposed node using Metropolis-Hastings algorithm.
     *
     * @param dd
     * @param nn
     * @param curNode
     * @param newNode
     * @param observed
     */
    private boolean evaluateProposedNode(int dd, int nn,
            Node curNode, Node newNode, boolean observed) {
        double newNodeTrueProb = getTrueLogProbability(dd, nn, newNode, observed);
        double curNodeTrueProb = getTrueLogProbability(dd, nn, curNode, observed);

        double newNodePropProb = getProporalProbability(dd, nn, newNode);
        double curNodePropProb = getProporalProbability(dd, nn, curNode);

        double ratio = (newNodeTrueProb * curNodePropProb) / (curNodeTrueProb * newNodePropProb);
        return rand.nextDouble() < Math.min(1.0, ratio);
    }

    private double getTrueLogProbability(int dd, int nn, Node node, boolean observed) {
        double lp = node.getNodeWordProbability(words[dd][nn]);
        if (observed) {
            lp *= Math.exp(getResponseLogLikelihood(dd, node));
        }
        lp *= getActualPathProbability(dd, nn, node, node);
        Node source = node.getParent();
        Node target = node;
        while (!target.isRoot()) {
            lp *= getActualPathProbability(dd, nn, source, target);
            target = source;
            source = source.getParent();
        }
        return lp;
    }

    private double getActualPathProbability(int dd, int nn, Node source, Node target) {
        int level = source.getLevel();
        if (level == L - 1) { // leaf node
            return 1.0;
        }
        double stayprob;
        if (source.isRoot() && !isRooted) {
            stayprob = 0.0;
        } else {
            stayprob = (source.nodeDocCounts.getCount(dd) + getGamma(level) * source.pi)
                    / (source.subtreeDocCounts.getCount(dd) + getGamma(level));
        }
        if (source.equals(target)) {
            return (stayprob);
        }

        double alpha = getAlpha(level);
        double passprob = 1.0 - stayprob;
        int KK = source.getNumChildren();
        double pathprob = (target.subtreeDocCounts.getCount(dd) + alpha * KK * source.theta[target.getIndex()])
                / (source.getPassingCount(dd) + alpha * KK);
        return (passprob * pathprob);
    }

    private double getProporalProbability(int dd, int nn, Node node) {
        double prob = getProposalPathProbability(dd, nn, node, node);
        Node source = node.getParent();
        Node target = node;
        while (!target.isRoot()) {
            prob *= getProposalPathProbability(dd, nn, source, target);
            target = source;
            source = source.getParent();
        }
        return prob;
    }

    private double getProposalPathProbability(int dd, int nn, Node source, Node target) {
        int level = source.getLevel();
        if (level == 0) { // use cached probabilities
            double totalProb = 0.0;
            for (double prob : cachedProbabilities.values()) {
                totalProb += prob;
            }
            Double prob = cachedProbabilities.get(target);
            if (prob == null) {
                throw new RuntimeException("Null probability");
            }
            return (prob / totalProb);
        } else if (level == L - 1) { // leaf node
            return 1.0;
        }

        double gamma = getGamma(level);
        double num = 0.0;
        double den = 0.0;

        double stayprob;
        if (source.isRoot() && !isRooted) {
            stayprob = 0.0;
        } else {
            stayprob = (source.nodeDocCounts.getCount(dd) + gamma * source.pi)
                    / (source.subtreeDocCounts.getCount(dd) + gamma);
            double wordprob = source.getNodeWordProbability(words[dd][nn]);
            double prob = stayprob * wordprob;
            if (source.equals(target)) {
                num = prob;
            }
            den += prob;
        }

        double alpha = getAlpha(level);
        double passprob = 1.0 - stayprob;
        int KK = source.getNumChildren();
        double norm = source.getPassingCount(dd) + alpha * KK;
        for (Node child : source.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeDocCounts.getCount(dd)
                    + alpha * KK * source.theta[kk]) / norm;
            double wordprob = child.getSubtreeWordProbability(words[dd][nn]);
            double prob = passprob * pathprob * wordprob;
            if (target.equals(child)) {
                num = prob;
            }
            den += prob;
        }
        return (num / den);
    }

    private double getResponseLogLikelihood(int dd, Node node) {
        double aMean = docMeans[dd] + node.pathEta / this.words[dd].length;
        double resLLh;
        if (isBinary) {
            resLLh = getLabelLogLikelihood(labels[dd], aMean);
        } else {
            resLLh = StatUtils.logNormalProbability(responses[dd], aMean, Math.sqrt(rho));
        }
        return resLLh;
    }

    private double getLabelLogLikelihood(int label, double dotProb) {
        double logNorm = Math.log(Math.exp(dotProb) + 1);
        if (label == POSITVE) {
            return dotProb - logNorm;
        } else {
            return -logNorm;
        }
    }

    /**
     * Update regression parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating eta's ...");
        }
        long sTime = System.currentTimeMillis();

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        // design matrix
        SparseVector[] designMatrix = new SparseVector[D];
        for (int aa = 0; aa < D; aa++) {
            designMatrix[aa] = new SparseVector(N);
        }

        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            if (node.isRoot()) {
                for (int dd = 0; dd < D; dd++) {
                    designMatrix[dd].set(kk, 1.0);
                }
            } else {
                for (int dd : node.subtreeDocCounts.getIndices()) {
                    int count = node.subtreeDocCounts.getCount(dd);
                    double val = (double) count / this.words[dd].length;
                    designMatrix[dd].change(kk, val);
                }
            }
        }

        // current params
        double[] etaArray = new double[N];
        double[] sigmaArray = new double[N];
        for (int kk = 0; kk < N; kk++) {
            etaArray[kk] = nodeList.get(kk).eta;
            sigmaArray[kk] = getSigma(nodeList.get(kk).getLevel());
        }

        boolean converged = false;
        if (isBinary) {
            RidgeLogisticRegressionOptimizable optimizable = new RidgeLogisticRegressionOptimizable(
                    labels, etaArray, designMatrix, mu, sigmaArray);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            // update regression parameters
            for (int kk = 0; kk < N; kk++) {
                nodeList.get(kk).eta = optimizable.getParameter(kk);
            }
        } else {
            RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                    responses, etaArray, designMatrix, rho, mu, sigmaArray);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);

            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            // update regression parameters
            for (int kk = 0; kk < N; kk++) {
                nodeList.get(kk).eta = optimizable.getParameter(kk);
            }
        }

        // update document means
        for (int dd = 0; dd < D; dd++) {
            docMeans[dd] = 0.0;
            for (int kk : designMatrix[dd].getIndices()) {
                docMeans[dd] += designMatrix[dd].get(kk) * nodeList.get(kk).eta;
            }
        }

        // update path thetas
        updatePathEtas();

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged
                    + ". " + designMatrix.length + " x " + nodeList.size());
            logln("--- --- time: " + eTime);
            evaluatePerformances();
        }
        return eTime;
    }

    protected void evaluatePerformances() {
        if (isBinary) {
            double[] predVals = new double[D];
            for (int d = 0; d < D; d++) {
                double expDotProd = Math.exp(docMeans[d]);
                double docPred = expDotProd / (expDotProd + 1);
                predVals[d] = docPred;
            }

            ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<RankingItem<Integer>>();
            for (int d = 0; d < D; d++) {
                rankDocs.add(new RankingItem<Integer>(d, predVals[d]));
            }
            Collections.sort(rankDocs);
            int[] preds = new int[D];
            for (int ii = 0; ii < this.positives.size(); ii++) {
                int d = rankDocs.get(ii).getObject();
                preds[d] = POSITVE;
            }

            ClassificationEvaluation eval = new ClassificationEvaluation(labels, preds);
            eval.computePRF1();
            for (Measurement measurement : eval.getMeasurements()) {
                logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
            }
        } else {
            RegressionEvaluation eval = new RegressionEvaluation(responses, docMeans);
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
    }

    /**
     * Update the eta sum for each path, which is represented by a node.
     */
    private void updatePathEtas() {
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.poll();

            node.pathEta = node.eta;
            if (!node.isRoot()) {
                node.pathEta += node.getParent().pathEta;
            }

            for (Node child : node.getChildren()) {
                queue.add(child);
            }
        }
    }

    /**
     * Flatten the tree.
     */
    private ArrayList<Node> getNodeList() {
        ArrayList<Node> nodeList = new ArrayList<>();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            nodeList.add(node);
        }
        return nodeList;
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.validate(msg);
        }
    }

    public void outputState(String filepath, boolean outputModel, boolean outputAssignment) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        StringBuilder modelStr = new StringBuilder();
        if (outputModel) {
            Stack<Node> stack = new Stack<>();
            stack.add(root);
            while (!stack.isEmpty()) {
                Node node = stack.pop();
                modelStr.append(Integer.toString(node.born)).append("\n");
                modelStr.append(node.getPathString()).append("\n");
                modelStr.append(node.eta).append("\n");
                modelStr.append(node.pi).append("\n");
                if (node.theta != null) {
                    modelStr.append(MiscUtils.arrayToString(node.theta));
                }
                modelStr.append("\n");
                modelStr.append(DirMult.output(node.getContent())).append("\n");
                modelStr.append(SparseCount.output(node.subtreeWordCounts)).append("\n");
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
            }
        }

        StringBuilder assignStr = new StringBuilder();
        if (outputAssignment) {
            for (int dd = 0; dd < z.length; dd++) {
                for (int nn = 0; nn < z[dd].length; nn++) {
                    assignStr.append(dd)
                            .append("\t").append(nn)
                            .append("\t").append(z[dd][nn].getPathString()).append("\n");
                }
            }
        }

        try { // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);

            this.outputZipFile(filepath, contentStrs, entryFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        outputState(filepath, true, true);
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
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                String pathStr = reader.readLine();
                double eta = Double.parseDouble(reader.readLine());
                double pi = Double.parseDouble(reader.readLine());
                line = reader.readLine().trim();
                double[] theta = null;
                if (!line.isEmpty()) {
                    theta = MiscUtils.stringToDoubleArray(line);
                }
                DirMult topic = DirMult.input(reader.readLine());
                SparseCount subtreeWordCounts = SparseCount.input(reader.readLine());

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                Node parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }
                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, eta);
                node.pi = pi;
                node.theta = theta;
                node.subtreeWordCounts = subtreeWordCounts;

                if (node.getLevel() == 0) {
                    root = node;
                }
                if (parent != null) {
                    parent.addChild(node.getIndex(), node);
                }
                nodeMap.put(pathStr, node);
            }

            updatePathEtas();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input a set of assignments.
     *
     * @param zipFilepath Compressed learned state file
     */
    public void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }
        try {
            z = new Node[D][];
            for (int d = 0; d < D; d++) {
                z[d] = new Node[words[d].length];
            }

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int dd = 0; dd < z.length; dd++) {
                for (int nn = 0; nn < z[dd].length; nn++) {
                    String[] sline = reader.readLine().split("\t");
                    if (dd != Integer.parseInt(sline[0])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[0]), dd);
                    }
                    if (nn != Integer.parseInt(sline[1])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[1]), nn);
                    }
                    String pathStr = sline[2];
                    z[dd][nn] = getNode(pathStr);
                    addToken(dd, nn, z[dd][nn], ADD, ADD);
                }
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Parse the node path string.
     *
     * @param nodePath The node path string
     * @return
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
    private Node getNode(int[] parsedPath) {
        Node node = root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    /**
     * Get a node in the tree given its path.
     *
     * @param pathStr
     */
    private Node getNode(String pathStr) {
        return getNode(parseNodePath(pathStr));
    }

    /**
     * Summary of the current tree.
     *
     * @return Summary of the current tree
     */
    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        SparseCount nodeCountPerLevel = new SparseCount();
        SparseCount obsCountPerLevel = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        int totalObs = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            if (node.isEmpty()) {
                continue;
            }

            int level = node.getLevel();
            nodeCountPerLevel.increment(level);
            obsCountPerLevel.changeCount(level, node.nodeDocCounts.getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeDocCounts.getCountSum());
            totalObs += node.nodeDocCounts.getCountSum();
        }
        str.append("global tree:\n\t>>> node count per level:\n");
        for (int l : nodeCountPerLevel.getSortedIndices()) {
            int obsCount = obsCountPerLevel.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLevel.getCount(l);
            str.append("\t>>> >>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * obsCount / numTokens)).append("%")
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * subtreeObsCount / numTokens)).append("%")
                    .append("]\n");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLevel.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * The current tree.
     *
     * @return The current tree
     */
    public String printGlobalTree() {
        SparseCount nodeCountPerLvl = new SparseCount();
        SparseCount obsCountPerLvl = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();
        int totalNumObs = 0;
        int numWords = 10;

        StringBuilder str = new StringBuilder();
        str.append("global tree\n");

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            String indentation = node.getIndentation();

            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            // top words according to the distribution
            str.append(indentation);
            str.append(node.getPathString())
                    .append(" (").append(node.born)
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.eta))
                    .append("; ").append(MiscUtils.formatDouble(node.pathEta))
                    .append(")");
            str.append("\n");

            if (!node.isEmpty()) {
                // words with highest probabilities at subtree
                if (node.getLevel() < L - 1) {
                    String[] subtreeTopWords = node.getSubtreeTopWords(numWords);
                    str.append(indentation).append("@ subtree: ");
                    for (String topWord : subtreeTopWords) {
                        str.append(" ").append(topWord);
                    }
                    str.append("\n");
                }

                // words with highest probabilities at node
                String[] nodeTopWords = node.getNodeTopWords(numWords);
                str.append(indentation).append("@ node: ");
                for (String topWord : nodeTopWords) {
                    str.append(" ").append(topWord);
                }
                str.append("\n");

                // top assigned words
                str.append(indentation);
                str.append(node.getTopObservations()).append("\n\n");

                int level = node.getLevel();
                nodeCountPerLvl.increment(level);
                obsCountPerLvl.changeCount(level, node.nodeDocCounts.getCountSum());
                subtreeObsCountPerLvl.changeCount(level, node.subtreeDocCounts.getCountSum());
                totalNumObs += node.getContent().getCountSum();
            }

        }
        str.append("Tree summary").append("\n");
        for (int l : nodeCountPerLvl.getSortedIndices()) {
            int obsCount = obsCountPerLvl.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLvl.getCount(l);
            str.append("\t>>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * obsCount / numTokens)).append("%")
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append(", ").append(MiscUtils.formatDouble((double) 100 * subtreeObsCount / numTokens)).append("%")
                    .append("]\n");
        }
        str.append("\t>>> # observations = ").append(totalNumObs).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLvl.getCountSum()).append("\n");
        return str.toString();
    }

    /**
     * Output top words for each topic in the tree to text file.
     *
     * @param outputFile The output file
     * @param numWords Number of top words
     */
    @Override
    public void outputTopicTopWords(File outputFile, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            String indentation = node.getIndentation();

            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            // top words according to the distribution
            str.append(indentation);
            str.append(node.getPathString())
                    .append(" (")
                    .append(node.born).append("; ")
                    .append(node.getContent().getCountSum()).append("; ")
                    //.append(MiscUtils.formatDouble(node.eta)).append("; eta: ")
                    .append(MiscUtils.formatDouble(node.pathEta))
                    .append(")");
            str.append(" ");

            if (!node.isEmpty()) {
                // words with highest probabilities at subtree
                if (node.getLevel() < L - 1) {
                    String[] subtreeTopWords = node.getSubtreeTopWords(numWords);
//                    str.append(indentation);
                    for (String topWord : subtreeTopWords) {
                        str.append(topWord).append(" ");
                    }
                    str.append("\n");
                } else { // words with highest probabilities at node
                    String[] nodeTopWords = node.getNodeTopWords(numWords);
//                    str.append(indentation);
                    for (String topWord : nodeTopWords) {
                        str.append(topWord).append(" ");
                    }
                    str.append("\n");
                }

                // top assigned words
//                str.append(indentation);
//                str.append(node.getTopObservations()).append("\n\n");
            }
            str.append("\n");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(str.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topics "
                    + outputFile);
        }
    }

    /**
     * Output posterior distribution over non-rooted nodes in the tree of all
     * documents.
     *
     * @param outputFile Output file
     */
    public void outputNodePosteriors(File outputFile) {
        ArrayList<Node> nodeList = getNodeList();
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int dd = 0; dd < D; dd++) {
                double[] nodePos = new double[nodeList.size()];
                for (int kk = 0; kk < nodeList.size(); kk++) {
                    Node node = nodeList.get(kk);
                    nodePos[kk] = (double) node.nodeDocCounts.getCount(dd) / words[dd].length;
                }
                writer.write(Integer.toString(dd));
                for (int kk = 0; kk < nodePos.length; kk++) {
                    writer.write("\t" + nodePos[kk]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while output to " + outputFile);
        }
    }

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected SparseCount subtreeDocCounts;
        protected SparseCount nodeDocCounts;
        protected double[] theta;
        protected double pi;
        protected double eta;
        protected double pathEta;
        protected SparseCount subtreeWordCounts;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.subtreeDocCounts = new SparseCount();
            this.nodeDocCounts = new SparseCount();
            this.eta = eta;
            this.subtreeWordCounts = new SparseCount();
        }

        void incrementSubtreeWordCount(int vv) {
            subtreeWordCounts.increment(vv); // currently only for maximal assumption
        }

        void decrementSubtreeWordCount(int vv) {
            subtreeWordCounts.decrement(vv); // currently only for maximal assumption
        }

        double getNodeWordProbability(int vv) {
            return this.content.getProbability(vv);
        }

        double getSubtreeWordProbability(int vv) {
            return (content.getCount(vv) + subtreeWordCounts.getCount(vv)
                    + content.getConcentration() * content.getCenterElement(vv))
                    / (content.getCountSum() + subtreeWordCounts.getCountSum()
                    + content.getConcentration());
        }

        void setPathEta(double pathEta) {
            this.pathEta = pathEta;
        }

        void initializeGlobalPi() {
            this.pi = getPi(level);
        }

        void initializeGlobalTheta() {
            int KK = getNumChildren();
            this.theta = new double[KK];
            Arrays.fill(this.theta, 1.0 / KK);
        }

        /**
         * Return the number of tokens of a given document which are assigned to
         * any nodes below this node.
         *
         * @param dd Document index
         */
        int getPassingCount(int dd) {
            return subtreeDocCounts.getCount(dd) - nodeDocCounts.getCount(dd);
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
        }

        String[] getNodeTopWords(int numTopWords) {
            double[] phi = new double[V];
            for (int vv = 0; vv < V; vv++) {
                phi[vv] = getNodeWordProbability(vv);
            }
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(phi, wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }

        String[] getSubtreeTopWords(int numTopWords) {
            double[] phi = new double[V];
            for (int vv = 0; vv < V; vv++) {
                phi[vv] = getSubtreeWordProbability(vv);
            }
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(phi, wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }

        String getTopObservations() {
            return getTopObservations(getContent().getSparseCounts());
        }

        String getTopObservations(SparseCount counts) {
            ArrayList<RankingItem<Integer>> rankObs = new ArrayList<RankingItem<Integer>>();
            for (int obs : counts.getIndices()) {
                rankObs.add(new RankingItem<Integer>(obs, counts.getCount(obs)));
            }
            Collections.sort(rankObs);
            StringBuilder str = new StringBuilder();
            for (int ii = 0; ii < Math.min(10, rankObs.size()); ii++) {
                RankingItem<Integer> obs = rankObs.get(ii);
                str.append(wordVocab.get(obs.getObject())).append(":")
                        .append(obs.getPrimaryValue()).append(" ");
            }
            return str.toString();
        }

        void validate(String msg) {
            this.nodeDocCounts.validate(msg);
            this.subtreeDocCounts.validate(msg);
            if (theta != null && theta.length != getNumChildren()) {
                throw new RuntimeException(msg + ". MISMATCH. " + this.toString());
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", c (").append(getChildren().size()).append(")");
            // word types
            str.append(", (").append(getContent().getCountSum()).append(")");
            // token counts
            str.append(", (").append(subtreeDocCounts.getCountSum());
            str.append(", ").append(nodeDocCounts.getCountSum()).append(")");
            str.append(", ").append(MiscUtils.formatDouble(eta));
            str.append(", ").append(MiscUtils.formatDouble(pathEta));
            str.append("]");
            return str.toString();
        }

        String getIndentation() {
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < this.getLevel(); i++) {
                str.append("\t");
            }
            return str.toString();
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + SNLDA.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        String example = new String();
        return example;
    }

    private static void addOpitions() throws Exception {
        parser = new BasicParser();
        options = new Options();

        // data input
        addOption("dataset", "Dataset");
        addOption("word-voc-file", "Word vocabulary file");
        addOption("word-file", "Document word file");
        addOption("info-file", "Document info file");
        addOption("selected-docs-file", "(Optional) Indices of selected documents");
        addOption("prior-topic-file", "File containing prior topics");
        addOption("init-eta-file", "File containing initial etas");
        addOption("num-top-words", "Number of top words per topic");

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("alphas", "Alpha");
        addOption("betas", "Beta");
        addOption("pis", "Mean");
        addOption("gammas", "Scale");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigmas", "Sigmas");
        addOption("Ks", "Number of topics");
        addOption("path", "Path assumption");

        // configurations
        addOption("init", "Initialization");

        options.addOption("train", false, "train");
        options.addOption("test", false, "test");
        options.addOption("parallel", false, "parallel");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
        options.addOption("example", false, "Example command");
        options.addOption("binary", false, "Binary responses");
        options.addOption("root", false, "Is rooted");
    }

    private static void runModel() throws Exception {
        // sampling configurations
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 20);
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 500);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 50);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 25);
        boolean paramOpt = cmd.hasOption("paramOpt");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
        InitialState initState;
        switch (init) {
            case "random":
                initState = InitialState.RANDOM;
                break;
            case "preset":
                initState = InitialState.PRESET;
                break;
            default:
                throw new RuntimeException("Initialization " + init + " not supported");
        }

        // model parameters
        int[] Ks = CLIUtils.getIntArrayArgument(cmd, "Ks", new int[]{15, 4}, ",");
        int L = Ks.length + 1;

        double[] alphas = CLIUtils.getDoubleArrayArgument(cmd, "alphas", new double[]{2.0, 1.0}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas", new double[]{0.5, 0.25, 0.1}, ",");
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis", new double[]{0.2, 0.2}, ",");
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas", new double[]{100, 10}, ",");
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas", new double[]{0.5, 2.5}, ",");
        String path = CLIUtils.getStringArgument(cmd, "path", "none");
        PathAssumption pathAssumption = getPathAssumption(path);

        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");

        // data output
        String outputFolder = cmd.getOptionValue("output-folder");

        double[][] priorTopics = null;
        if (cmd.hasOption("prior-topic-file")) {
            String priorTopicFile = cmd.getOptionValue("prior-topic-file");
            priorTopics = IOUtils.input2DArray(new File(priorTopicFile));
        }

        double[] initEtas = null;
        if (cmd.hasOption("init-eta-file")) {
            String initEtaFile = cmd.getOptionValue("init-eta-file");
            initEtas = IOUtils.inputArray(new File(initEtaFile));
        }

        File docInfoFile = null;
        if (cmd.hasOption("info-file")) {
            docInfoFile = new File(cmd.getOptionValue("info-file"));
        }

        SNLDA sampler = new SNLDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);

        boolean isBinary = cmd.hasOption("binary");
        boolean isRooted = cmd.hasOption("root");
        ResponseTextDataset contData = new ResponseTextDataset(datasetName);
        LabelTextDataset binData = new LabelTextDataset(datasetName);
        int V;
        if (isBinary) {
            binData.loadFormattedData(new File(wordVocFile),
                    new File(docWordFile),
                    docInfoFile,
                    null);
            V = binData.getWordVocab().size();
            sampler.setWordVocab(binData.getWordVocab());
            sampler.configureBinary(outputFolder, V, Ks,
                    alphas, betas, pis, gammas, mu, sigmas,
                    initState, pathAssumption, paramOpt, isRooted,
                    burnIn, maxIters, sampleLag, repInterval);
        } else {
            contData.loadFormattedData(new File(wordVocFile),
                    new File(docWordFile),
                    docInfoFile,
                    null);
            V = contData.getWordVocab().size();
            sampler.setWordVocab(contData.getWordVocab());
            sampler.configureContinuous(outputFolder, V, Ks,
                    alphas, betas, pis, gammas, rho, mu, sigmas,
                    initState, pathAssumption, paramOpt, isRooted,
                    burnIn, maxIters, sampleLag, repInterval);
        }

        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (isTraining()) {
            ArrayList<Integer> trainDocIndices;
            if (isBinary) {
                trainDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());
                sampler.train(binData.getWords(), trainDocIndices, binData.getSingleLabels());
            } else {
                trainDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());
                double[] docResponses = contData.getResponses();
                if (cmd.hasOption("z")) { // z-normalization
                    ZNormalizer zNorm = new ZNormalizer(docResponses);
                    docResponses = zNorm.normalize(docResponses);
                }
                sampler.train(contData.getWords(), trainDocIndices, docResponses);
            }

            sampler.initialize(priorTopics, initEtas);
            sampler.metaIterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            sampler.outputNodePosteriors(new File(samplerFolder, "train-node-posteriors.txt"));
        }

        if (isTesting()) {
            int[][] testWords;
            ArrayList<Integer> testDocIndices;
            if (isBinary) {
                testWords = binData.getWords();
                testDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());

            } else {
                testWords = contData.getWords();
                testDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());
            }

            File testAssignmentFolder = new File(samplerFolder, AbstractSampler.IterAssignmentFolder);
            IOUtils.createFolder(testAssignmentFolder);

            File testPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
            IOUtils.createFolder(testPredFolder);

            double[] predictions;
            if (cmd.hasOption("parallel")) { // using multiple stored models
                predictions = SNLDA.parallelTest(testWords, testDocIndices, testPredFolder, testAssignmentFolder, sampler);
            } else { // using the last model
                File stateFile = sampler.getFinalStateFile();
                File outputPredFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".txt");
                File outputStateFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".zip");
                sampler.test(testWords, testDocIndices);
                predictions = sampler.sampleTest(stateFile, outputStateFile, outputPredFile);
                sampler.outputNodePosteriors(new File(samplerFolder, "test-node-posteriors.txt"));
            }

            File teResultFolder = new File(samplerFolder,
                    AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            if (isBinary) {
                PredictionUtils.outputClassificationPredictions(
                        new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                        binData.getDocIds(), binData.getSingleLabels(), predictions);
                PredictionUtils.outputBinaryClassificationResults(
                        new File(teResultFolder, AbstractExperiment.RESULT_FILE),
                        binData.getSingleLabels(), predictions);
            } else {
                double[] docResponses = contData.getResponses();
                if (cmd.hasOption("z")) { // z-normalization
                    ZNormalizer zNorm = new ZNormalizer(docResponses);
                    docResponses = zNorm.normalize(docResponses);
                }
                PredictionUtils.outputRegressionPredictions(
                        new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                        contData.getDocIds(), docResponses, predictions);
                PredictionUtils.outputRegressionResults(
                        new File(teResultFolder, AbstractExperiment.RESULT_FILE), docResponses,
                        predictions);
            }
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOpitions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            } else if (cmd.hasOption("example")) {
                System.out.println(getExampleCmd());
                return;
            }

            runModel();

            // date and time
            DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
            Date dateobj = new Date();
            long eTime = (System.currentTimeMillis() - sTime) / 1000;
            System.out.println("Elapsed time: " + eTime + "s");
            System.out.println("End time: " + df.format(dateobj));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newDocIndices Indices of test documents
     * @param iterPredFolder Output folder
     * @param iterStateFolder Folder to store assignments
     * @param sampler The configured sampler
     */
    public static double[] parallelTest(int[][] newWords,
            ArrayList<Integer> newDocIndices,
            File iterPredFolder,
            File iterStateFolder,
            SNLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[] avgPredictions = null;
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            ArrayList<File> partPredFiles = new ArrayList<>();
            for (String filename : filenames) { // all learned models
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);

                String stateFilename = IOUtils.removeExtension(filename);
                File iterOutputPredFile = new File(iterPredFolder, stateFilename + ".txt");
                File iterOutputStateFile = new File(iterStateFolder, stateFilename + ".zip");

                SNLDATestRunner runner = new SNLDATestRunner(sampler,
                        newWords, newDocIndices,
                        stateFile.getAbsolutePath(),
                        iterOutputStateFile.getAbsolutePath(),
                        iterOutputPredFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
                partPredFiles.add(iterOutputPredFile);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // average predictions
            avgPredictions = PredictionUtils.computeMultipleAverage(partPredFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
        return avgPredictions;
    }
}

class SNLDATestRunner implements Runnable {

    SNLDA sampler;
    int[][] newWords;
    ArrayList<Integer> newDocIndices;
    String stateFile;
    String outputStateFile;
    String outputPredictionFile;

    public SNLDATestRunner(SNLDA sampler,
            int[][] newWords,
            ArrayList<Integer> newDocIndices,
            String stateFile,
            String outputStateFile,
            String outputPredFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newDocIndices = newDocIndices;
        this.stateFile = stateFile;
        this.outputStateFile = outputStateFile;
        this.outputPredictionFile = outputPredFile;
    }

    @Override
    public void run() {
        SNLDA testSampler = new SNLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        try {
            testSampler.test(newWords, newDocIndices);
            testSampler.sampleTest(
                    new File(stateFile),
                    new File(outputStateFile),
                    new File(outputPredictionFile));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
