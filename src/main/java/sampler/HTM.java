package sampler;

import sampler.unsupervised.LDA;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.util.Randoms;
import core.AbstractExperiment;
import core.AbstractSampler;
import data.LabelTextDataset;
import data.ResponseTextDataset;
import data.TextDataset;
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
import sampling.likelihood.CascadeDirMult.PathAssumption;
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
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;
import util.normalizer.ZNormalizer;

/**
 * Holistic Topic Model (HTM) which provides the following models:
 *
 * <ul>
 * <li>Latent Dirichlet Allocation (LDA)</li>
 * <li>Hierarchical Dirichlet Process (HDP)</li>
 * <li>Nested Hierarchical Dirichlet Process (NHDP)</li>
 * <li>Supervised Latent Dirichlet Allocation (SLDA)</li>
 * <li>Supervised Hierarchical Dirichlet Process (SHDP)</li>
 * <li>Supervised Nested Hierarchical Dirichlet Process (SNHDP)</li>
 * </ul>
 *
 * @author vietan
 */
public class HTM extends AbstractSampler {

    public static enum Mode {

        UNSUPERVISED, SUPERVISED_BINARY, SUPERVISED_CATEGORICAL, SUPERVISED_CONTINUOUS
    }

    public static final int NEW_CHILD_INDEX = -1;
    public static final int PROPOSE_INDEX = 0;
    public static final int ASSIGN_INDEX = 1;
    public static final int POSITVE = 1;
    public static final int NEGATIVE = -1;
    public static Randoms random = new Randoms(1);

    // hyperparameters for fixed-height tree
    protected int L;
    protected double[] globalAlphas;   // [L-1]
    protected double[] localAlphas;    // [L-1]
    protected double[] betas;           // [L]
    protected double[] pis;             // probability of staying 
    protected double[] gammas;          // scale for pi
    protected double rho;
    protected double mu;
    protected double[] sigmas;
    protected double sigma;

    // inputs
    protected int[][] words; // all words
    protected ArrayList<Integer> docIndices; // indices of docs under consideration
    protected double[] responses; // [D] continous responses
    protected int[] labels; // [D] binary responses
    protected int V; // vocabulary size
    protected double[][] priors; // priors for first-level nodes
    protected Mode mode;
    protected boolean isRooted; // whether tokens are assigned to root node
    protected int[] Ks; // initial number of nodes per level
    protected boolean[] unbounded; // whether a level is unbounded

    // derived
    protected int D; // number of documents
    protected Set<Integer> positives;
    protected boolean isExtendable;
    protected SparseVector[] lexicalDesginMatrix;

    // latent
    private Node[][] z;
    private Node root;
    private double[] dotprods;
    private double uniform;

    private double[] tau;
    private boolean isLexical = false;

    // configuration
    protected PathAssumption path;
    // internal
    protected int numTokensAccepted;

    public HTM() {
        this.basename = "HTM";
    }

    public HTM(String bname) {
        this.basename = bname;
    }

    protected double getSigma(int l) {
        return this.sigmas[l];
    }

    protected double getLocalAlpha(int l) {
        return this.localAlphas[l];
    }

    protected double getGlobalAlpha(int l) {
        return this.globalAlphas[l];
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

    /**
     * Whether a non-terminal node is extendable.
     *
     * @param level
     */
    protected boolean isExtendable(int level) {
        return (level < L - 1 && this.Ks[level] == 0);
    }

    protected boolean isUnbounded(int level) {
        return this.unbounded[level];
    }

    /**
     * *
     * Check whether a given level is terminal.
     *
     * @param level
     * @return
     */
    private boolean isTerminal(int level) {
        return level == this.L - 1;
    }

    /**
     * Return whether running supervised version.
     */
    private boolean isSupervised() {
        return this.mode != Mode.UNSUPERVISED;
    }

    public double[] getPredictedValues() {
        if (this.mode == Mode.SUPERVISED_CONTINUOUS) {
            return this.dotprods;
        } else {
            throw new RuntimeException("Mode " + mode + " is not supported.");
        }
    }

    public void configure(HTM sampler) {
        if (sampler.mode == Mode.UNSUPERVISED) {
            this.configure(sampler.folder,
                    sampler.V,
                    sampler.L,
                    sampler.Ks, sampler.unbounded, sampler.priors,
                    sampler.globalAlphas,
                    sampler.localAlphas,
                    sampler.betas,
                    sampler.pis,
                    sampler.gammas,
                    sampler.initState,
                    sampler.path,
                    sampler.isRooted,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        } else if (sampler.mode == Mode.SUPERVISED_BINARY) {
            this.configureBinary(sampler.folder,
                    sampler.V,
                    sampler.L,
                    sampler.Ks, sampler.unbounded, sampler.priors,
                    sampler.globalAlphas,
                    sampler.localAlphas,
                    sampler.betas,
                    sampler.pis,
                    sampler.gammas,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.sigma,
                    sampler.initState,
                    sampler.path,
                    sampler.isRooted,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);

        } else if (sampler.mode == Mode.SUPERVISED_CONTINUOUS) {
            this.configureContinuous(sampler.folder,
                    sampler.V,
                    sampler.L,
                    sampler.Ks, sampler.unbounded, sampler.priors,
                    sampler.globalAlphas,
                    sampler.localAlphas,
                    sampler.betas,
                    sampler.pis,
                    sampler.gammas,
                    sampler.rho,
                    sampler.mu,
                    sampler.sigmas,
                    sampler.sigma,
                    sampler.initState,
                    sampler.path,
                    sampler.isRooted,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        } else {
            throw new RuntimeException("Mode " + sampler.mode + " not supported");
        }
    }

    public void configure(String folder,
            int V, int L,
            int[] Ks, boolean[] unbounded, double[][] priors,
            double[] globalAlphas,
            double[] localAlphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean isRooted,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / V;
        this.L = L;
        this.Ks = Ks;
        this.unbounded = unbounded;
        this.priors = priors;
        this.isExtendable = false;
        for (int K : Ks) {
            if (K == 0) {
                this.isExtendable = true;
            }
        }

        this.globalAlphas = globalAlphas;
        this.localAlphas = localAlphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.path = pathAssumption;
        this.isRooted = isRooted;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.mode = Mode.UNSUPERVISED;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- L = " + L);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- global alphas:\t" + MiscUtils.arrayToString(globalAlphas));
            logln("--- local alphas:\t" + MiscUtils.arrayToString(localAlphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(pis));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gammas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
            logln("--- mode:\t" + this.mode);
            logln("--- is rooted:\t" + this.isRooted);
        }

        validateInputHyperparameters();
    }

    public void configureBinary(String folder,
            int V, int L,
            int[] Ks, boolean[] unbounded, double[][] priors,
            double[] globalAlphas,
            double[] localAlphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            double mu,
            double[] sigmas,
            double sigma,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean isRooted,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / V;
        this.L = L;
        this.Ks = Ks;
        this.unbounded = unbounded;
        this.priors = priors;
        this.isExtendable = false;
        for (int K : Ks) {
            if (K == 0) {
                this.isExtendable = true;
            }
        }

        this.globalAlphas = globalAlphas;
        this.localAlphas = localAlphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.mu = mu;
        this.sigmas = sigmas;
        this.sigma = sigma;
        this.isLexical = this.sigma > 0;
        this.path = pathAssumption;
        this.isRooted = isRooted;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.mode = Mode.SUPERVISED_BINARY;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- L = " + L);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- global alphas:\t" + MiscUtils.arrayToString(globalAlphas));
            logln("--- local alphas:\t" + MiscUtils.arrayToString(localAlphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(pis));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- is lexical:\t" + this.isLexical);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
            logln("--- mode:\t" + this.mode);
            logln("--- is rooted:\t" + this.isRooted);
        }

        validateInputHyperparameters();
    }

    public void configureContinuous(String folder,
            int V, int L,
            int[] Ks, boolean[] unbounded, double[][] priors,
            double[] globalAlphas,
            double[] localAlphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            double rho,
            double mu,
            double[] sigmas,
            double sigma,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean isRooted,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.uniform = 1.0 / V;
        this.L = L;
        this.Ks = Ks;
        this.unbounded = unbounded;
        this.priors = priors;
        this.isExtendable = false;
        for (int K : Ks) {
            if (K == 0) {
                this.isExtendable = true;
            }
        }

        this.globalAlphas = globalAlphas;
        this.localAlphas = localAlphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.rho = rho;
        this.mu = mu;
        this.sigmas = sigmas;
        this.sigma = sigma;
        this.isLexical = this.sigma > 0;
        this.path = pathAssumption;
        this.isRooted = isRooted;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.mode = Mode.SUPERVISED_CONTINUOUS;

        this.setName();

        if (verbose) {
            logln("--- V = " + V);
            logln("--- L = " + L);
            logln("--- Ks = " + MiscUtils.arrayToString(this.Ks));
            logln("--- folder\t" + folder);
            logln("--- global alphas:\t" + MiscUtils.arrayToString(globalAlphas));
            logln("--- local alphas:\t" + MiscUtils.arrayToString(localAlphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(pis));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- is lexical:\t" + this.isLexical);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.path);
            logln("--- mode:\t" + this.mode);
            logln("--- is rooted:\t" + this.isRooted);
        }

        validateInputHyperparameters();
    }

    /**
     * Validate the input hyper-parameters and make sure the dimensions are
     * valid.
     */
    private void validateInputHyperparameters() {
        if (globalAlphas.length != L - 1) {
            throw new MismatchRuntimeException(globalAlphas.length, L - 1);
        }
        if (localAlphas.length != L - 1) {
            throw new MismatchRuntimeException(localAlphas.length, L - 1);
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
        if (sigmas != null && sigmas.length != L) {
            throw new MismatchRuntimeException(sigmas.length, L - 1);
        }
        if (Ks.length != L - 1) {
            throw new MismatchRuntimeException(Ks.length, L - 1);
        }
        if (priors != null && Ks[0] != priors.length) {
            throw new MismatchRuntimeException(this.Ks[0], this.priors.length);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename);
        str.append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_l-").append(L);
        str.append("_Ks");
        for (int K : Ks) {
            str.append("-").append(K);
        }
        str.append("_ga");
        for (double ga : globalAlphas) {
            str.append("-").append(MiscUtils.formatDouble(ga));
        }
        str.append("_la");
        for (double la : localAlphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_p");
        for (double p : pis) {
            str.append("-").append(MiscUtils.formatDouble(p));
        }
        str.append("_g");
        for (double g : gammas) {
            str.append("-").append(MiscUtils.formatDouble(g));
        }
        if (isSupervised()) {
            str.append("_r-").append(MiscUtils.formatDouble(rho));
            str.append("_m-").append(MiscUtils.formatDouble(mu));
            str.append("_s");
            for (double s : sigmas) {
                str.append("-").append(MiscUtils.formatDouble(s));
            }
            str.append(this.sigma > 0 ? ("_s-" + sigma) : "");
        }
        str.append("_opt-").append(this.paramOptimized);
        str.append("_prior-").append(this.priors != null);
        str.append("_root-").append(this.isRooted);
        str.append("_").append(path);
        str.append("_").append(mode);
        this.name = str.toString();
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath() + "\n" + printGlobalTreeSummary() + "\n";
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
     * Set up training data for unsupervised version.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     */
    public void train(int[][] docWords, ArrayList<Integer> docIndices) {
        setupTextData(docWords, docIndices);
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
        if (!isSupervised()) {
            throw new RuntimeException("Should have configured for supervised version");
        }
        setupTextData(docWords, docIndices);
        setContinuousResponses(docResponses);
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
        if (!isSupervised()) {
            throw new RuntimeException("Should have configured for supervised version");
        }
        setupTextData(docWords, docIndices);
        setBinaryResponses(docLabels);
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
     * Set up test data.
     *
     * @param docWords Test documents
     * @param docIndices Indices of test documents
     */
    public void test(int[][] docWords, ArrayList<Integer> docIndices) {
        setupTextData(docWords, docIndices);
    }

    /**
     * Sample during test.
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
        inputModel(stateFile.toString());
        initializeDataStructure();

        // store predictions at different test iterations
        ArrayList<double[]> predResponsesList = new ArrayList<double[]>();

        // sample topic assignments for test document
        for (iter = 0; iter < this.testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + ". current thread: " + Thread.currentThread().getId();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            if (iter == 0) {
//                sampleZs_MH(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, !EXTEND);
                sampleZs_Gibbs(!REMOVE, !ADD, !REMOVE, ADD, !EXTEND);
            } else {
//                sampleZs_MH(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, !EXTEND);
                sampleZs_Gibbs(!REMOVE, !ADD, REMOVE, ADD, !EXTEND);
            }

            // store prediction (on all documents) at a test iteration
            if (isSupervised() && iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double[] predResponses = new double[D];
                System.arraycopy(dotprods, 0, predResponses, 0, D); // now only for continuous
                predResponsesList.add(predResponses);

                if (responses != null) { // debug
                    evaluatePerformances();
                }
            }
        }

        // output state file containing the assignments for test documents
        if (testStateFile != null) {
            outputState(testStateFile);
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
        initialize(priors);
    }

    public void initialize(double[][] priors) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;
        initializeModelStructure(priors);
        initializeDataStructure();
        initializeAssignments();
        if (isSupervised()) {
            if (isLexical) {
                updateEtasTaus();
            } else {
                updateEtas();
            }
        }

        if (verbose) {
            logln("--- Done initializing.\n" + printGlobalTree());
            logln("\n" + printGlobalTreeSummary() + "\n");
            getLogLikelihood();
        }

        if (debug) {
            outputTopicTopWords(new File(getSamplerFolderPath(), "init-" + TopWordFile), 20);
            validate("Initialized");
        }
    }

    protected void initializeModelFirstLevelNodes(double[][] priorTopics, double[] initEtas) {
//        int level = 1;
//        for (int kk = 0; kk < Ks[0]; kk++) {
//            // prior topic
//            double[] prior;
//            if (priorTopics == null) {
//                prior = new double[V];
//                Arrays.fill(prior, uniform);
//            } else {
//                prior = priorTopics[kk];
//            }
//
//            // initial eta
//            double eta;
//            if (initEtas != null) {
//                eta = initEtas[kk];
//            } else {
//                eta = SamplerUtils.getGaussian(mu, getSigma(level));
//            }
//
//            // initialize
//            DirMult topic = new DirMult(V, getBeta(1) * V, prior);
//            Node node = new Node(iter, kk, level, topic, root, eta);
//            this.root.addChild(kk, node);
//        }
    }

    private void initializeModelStructure(double[][] priors) {
        if (Ks[0] == 0) {
            throw new RuntimeException("Number of first-level nodes needs to be defined");
        }

        if (isLexical) {
            this.tau = new double[V];
            for (int vv = 0; vv < V; vv++) {
                this.tau[vv] = SamplerUtils.getGaussian(mu, sigma);
            }
        }

        double[] background = new double[V];
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                background[words[dd][nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            background[vv] /= numTokens;
        }

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, background);
        this.root = new Node(iter, 0, 0, isExtendable(0), rootTopic, null, 0.0);
        for (int kk = 0; kk < Ks[0]; kk++) {
            int level = 1;
            DirMult topic;
            if (priors != null) { // seeded prior
                topic = new DirMult(V, getBeta(level) * V, priors[kk]);
            } else { // uninformed prior
                topic = new DirMult(V, getBeta(level) * V, uniform);
            }
            double eta = 0.0;
            if (isSupervised()) {
                eta = SamplerUtils.getGaussian(mu, getSigma(level));
            }
            Node issueNode = new Node(iter, kk, level, isExtendable(level), topic, root, eta);
            root.addChildUpdate(kk, issueNode);
        }

        // initialize theta and pi
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            node.changeStatus();

            int level = node.getLevel();
            if (level < L - 1) {
                if (!node.isRoot() && !isExtendable(level)) {
                    for (int kk = 0; kk < Ks[level]; kk++) {
                        DirMult childTopic = new DirMult(V, getBeta(level + 1) * V, uniform);
                        double eta = 0.0;
                        if (isSupervised()) {
                            eta = SamplerUtils.getGaussian(mu, getSigma(level + 1));
                        }
                        Node child = new Node(iter, kk, level + 1,
                                isExtendable(level + 1), childTopic, node, eta);
                        node.addChildUpdate(kk, child);
                    }
                }

                for (Node child : node.getChildren()) {
                    stack.add(child);
                }

                node.initializeGlobalTheta();
                node.initializeGlobalPi();
            }
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
        if (isSupervised()) {
            this.dotprods = new double[D];

            if (isLexical) {
                this.lexicalDesginMatrix = new SparseVector[D];
                for (int dd = 0; dd < D; dd++) {
                    lexicalDesginMatrix[dd] = new SparseVector(V);

                    double val = 1.0 / words[dd].length;
                    for (int nn = 0; nn < words[dd].length; nn++) {
                        lexicalDesginMatrix[dd].change(words[dd][nn], val);
                    }

                    for (int vv : lexicalDesginMatrix[dd].getIndices()) {
                        dotprods[dd] += lexicalDesginMatrix[dd].get(vv) * tau[vv];
                    }
                }
            }
        }
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

        // remove empty nodes after initial assignments and update thetas
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();

            if (node.getLevel() < L - 1) {
                ArrayList<Node> emptyChildren = new ArrayList<>();
                for (Node child : node.getChildren()) {
                    if (child.isEmpty()) {
                        emptyChildren.add(child);
                    } else {
                        stack.add(child);
                    }
                }

                for (Node emptyChild : emptyChildren) {
                    if (verbose) {
                        logln("Initializing. Removing empty child: " + emptyChild.toString());
                    }
                    node.removeChildUpdate(emptyChild.getIndex());
                }
            }
        }
    }

    /**
     * Run LDA to initialize the first-level nodes.
     */
    private void initializePresetAssignments() {
        if (Ks[0] > 0) {
            LDA lda = runLDA(words, Ks[0], V, priors);
            for (int dd = 0; dd < D; dd++) {
                for (int nn = 0; nn < words[dd].length; nn++) {
                    int kk = lda.getZs()[dd][nn];
                    Node node = sampleNode(dd, nn, root.getChild(kk), EXTEND, false);
                    z[dd][nn] = node;
                    addToken(dd, nn, z[dd][nn], ADD, ADD);
                }
            }
        } else {
            // TODO: run HDP
            throw new RuntimeException("Currently, the number of first-level "
                    + "nodes needs to be pre-defined.");
        }
    }

    private void initializeRandomAssignments() {
        sampleZs_MH(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, EXTEND);
    }

    protected void evaluatePerformances() {
        if (mode == Mode.SUPERVISED_BINARY) {
            double[] predVals = new double[D];
            for (int d = 0; d < D; d++) {
                double expDotProd = Math.exp(dotprods[d]);
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
        } else if (mode == Mode.SUPERVISED_CONTINUOUS) {
            RegressionEvaluation eval = new RegressionEvaluation(responses, dotprods);
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeMeanAbsoluteError();
            eval.computeRSquared();
            eval.computePredictiveRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
            }
        } else {
            throw new RuntimeException("Mode " + mode + " is not supported");
        }
    }

    @Override
    public void iterate() {
        if (isReporting) {
            System.out.println("\n");
            logln("Iteration " + iter + " / " + MAX_ITER);
        }

        if (isSupervised()) {
            sampleZs_MH(REMOVE, ADD, REMOVE, ADD, OBSERVED, EXTEND);
        } else {
            sampleZs_Gibbs(REMOVE, ADD, REMOVE, ADD, EXTEND);
        }
//        updateTopics();
        if (this.isExtendable) {
            updateGlobalProbabilities();
        }
        if (isSupervised()) {
            if (isLexical) {
                updateEtasTaus(); // this takes time
            } else {
                updateEtas();
            }
        }
    }

    /**
     * Add a token to a node.
     *
     * @param nn
     * @param dd
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
            if (isSupervised()) {
                dotprods[dd] += node.pathEta / words[dd].length;
            }
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
     * @param nn
     * @param dd
     * @param node
     * @param removeFromData
     * @param removeFromModel
     */
    private void removeToken(int dd, int nn, Node node,
            boolean removeFromData, boolean removeFromModel) {
        if (removeFromData) {
            if (isSupervised()) {
                dotprods[dd] -= node.pathEta / words[dd].length;
            }
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

            if (node.subtreeDocCounts.isEmpty()) {
                if (!node.nodeDocCounts.isEmpty()) {
                    throw new RuntimeException("SubtreeTokenCounts is empty"
                            + " but TokenCounts is not.\n" + node.toString());
                }
                tempNode = node;
                while (tempNode.subtreeDocCounts.isEmpty()) {
                    Node parent = tempNode.getParent();
                    parent.removeChildUpdate(tempNode.getIndex());
                    tempNode = parent;
                }
            }
        }
    }

    /**
     * Sample node assignment for all tokens using Gibbs sampling.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param extend
     * @return Elapsed time
     */
    protected long sampleZs_Gibbs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean extend) {
        if (isReporting) {
            logln("+++ Sampling Zs using Gibbs ...");
        }
        numTokensChanged = 0;

        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                // sample
                Node sampledNode = sampleNode(dd, nn, root, extend, false);
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
     * Sample node assignment for all tokens using Metropolis-Hastings.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe
     * @param extend
     * @return Elapsed time
     */
    protected long sampleZs_MH(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData, boolean observe,
            boolean extend) {
        if (isReporting) {
            logln("+++ Sampling Zs ...");
        }
        numTokensChanged = 0;
        numTokensAccepted = 0;

        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                // sample
                Node sampledNode = sampleNode(dd, nn, root, extend, observe);

                boolean accept = false;
                if (z[dd][nn] == null) { // first iteration, accept anything
                    accept = true;
                    numTokensChanged++;
                    numTokensAccepted++;
                } else if (sampledNode.equals(z[dd][nn])) { // stay the same
                    accept = true;
                    numTokensAccepted++;
                } else { // Metropolis-Hastings
//                    double[] curLogprobs = getLogProbabilities(dd, nn, z[dd][nn], observe);
//                    double[] newLogprobs = getLogProbabilities(dd, nn, sampledNode, observe);
//                    double ratio = Math.min(1.0,
//                            Math.exp(newLogprobs[ACTUAL_INDEX] - curLogprobs[ACTUAL_INDEX]
//                                    + curLogprobs[PROPOSAL_INDEX] - newLogprobs[PROPOSAL_INDEX]));
//                    if (rand.nextDouble() < ratio) {
//                        accept = true;
//                        numTokensAccepted++;
//                    }
                    accept = true;
                    numTokensAccepted++;
                }

                if (accept) { // if accept
                    if (z[dd][nn] != null && !z[dd][nn].equals(sampledNode)) {
                        numTokensChanged++;
                    }
                    z[dd][nn] = sampledNode;
                }

                // add
                addToken(dd, nn, z[dd][nn], addToData, addToModel);

                Node parent = z[dd][nn].getParent();
                int zIdx = z[dd][nn].getIndex();
                if (accept) {
                    // if a new node is sampled and accepted, add it, change its status
                    // (not new node anymore) and udpate the global theta of its parent
                    if (z[dd][nn].newNode) {
                        z[dd][nn].changeStatus();
                        parent.addChildUpdate(zIdx, z[dd][nn]);
                    }
                } else {
                    // if reject the proposed node and the current node was removed
                    // from the tree, we need to add it back to the tree
                    if (!z[dd][nn].isRoot() && !parent.hasChild(zIdx)) {
                        parent.addChildUpdate(zIdx, z[dd][nn]);
                    }
                }
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
     * Updating topics.
     */
//    protected long updateTopics() {
//        if (isReporting) {
//            logln("+++ Updating topics ...");
//        }
//        long sTime = System.currentTimeMillis();
//
//        // get all leaves of the tree
//        ArrayList<Node> leaves = new ArrayList<Node>();
//        Stack<Node> stack = new Stack<Node>();
//        stack.add(root);
//        while (!stack.isEmpty()) {
//            Node node = stack.pop();
//            if (node.getChildren().isEmpty()) {
//                leaves.add(node);
//            }
//            for (Node child : node.getChildren()) {
//                stack.add(child);
//            }
//        }
//
//        // bottom-up smoothing to compute pseudo-counts from children
//        Queue<Node> queue = new LinkedList<Node>();
//        for (Node leaf : leaves) {
//            queue.add(leaf);
//        }
//        while (!queue.isEmpty()) {
//            Node node = queue.poll();
//            Node parent = node.getParent();
//            if (!node.isRoot() && !queue.contains(parent)) {
//                queue.add(parent);
//            }
//            if (node.isLeaf()) {
//                continue;
//            }
//
//            if (!node.isRoot()) {
//                node.computePropagatedCountsFromChildren();
//            }
//        }
//
//        // top-down sampling to get topics
//        queue = new LinkedList<Node>();
//        queue.add(root);
//        while (!queue.isEmpty()) {
//            Node node = queue.poll();
//            for (Node child : node.getChildren()) {
//                queue.add(child);
//            }
//            node.updateTopic();
//        }
//
//        long eTime = System.currentTimeMillis() - sTime;
//        if (isReporting) {
//            logln("--- --- time: " + eTime);
//        }
//        return eTime;
//    }
    /**
     * Recursively sample a node from a current node. The sampled node can be
     * either the same node or one of its children. If the current node is a
     * leaf node, return it.
     *
     * @param dd Document index
     * @param nn Token index
     * @param curNode Current node
     * @param extend Whether the tree structure is extendable
     */
    private Node sampleNode(int dd, int nn, Node curNode, boolean extend,
            boolean observed) {
        if (curNode.isLeaf() && !curNode.extensible) {
            return curNode;
        }

        ArrayList<Node> nodeList = new ArrayList<>();
        ArrayList<Double> logprobList = new ArrayList<>();

        int level = curNode.getLevel();
        double gamma = getGamma(level);
        double stayprob;
        if (curNode.isRoot() && !isRooted) {
            stayprob = 0.0;
        } else {
            stayprob = (curNode.nodeDocCounts.getCount(dd) + gamma * curNode.pi)
                    / (curNode.subtreeDocCounts.getCount(dd) + gamma);

            double wordprob = curNode.getNodeWordProbability(words[dd][nn]);
            double lp = Math.log(stayprob * wordprob);
            if (observed) {
                lp += getResponseLogLikelihood(dd, nn, curNode);
            }
            logprobList.add(lp);
            nodeList.add(curNode);
        }
        double passprob = 1.0 - stayprob;

        // for moving to an existing child node
        double lAlpha = getLocalAlpha(level);
        double norm = curNode.getPassingCount(dd) + lAlpha;
        for (Node child : curNode.getChildren()) {
            double pathprob = (child.subtreeDocCounts.getCount(dd)
                    + lAlpha * curNode.theta.get(child.getIndex())) / norm;
            double lp = Math.log(passprob * pathprob * child.getSubtreeWordProbability(words[dd][nn]));
            if (observed) {
                lp += getResponseLogLikelihood(dd, nn, child);
            }
            nodeList.add(child);
            logprobList.add(lp);
        }

        // for moving to a new child node
        double eta = 0.0;
        if (isSupervised()) {
            eta = SamplerUtils.getGaussian(mu, getSigma(curNode.getLevel() + 1));
        }
        if (extend && curNode.extensible) {
            double pathprob = lAlpha * curNode.theta.get(NEW_CHILD_INDEX) / norm;
            double lp = Math.log(passprob * pathprob * uniform);
            if (observed) {
                lp += getResponseLogLikelihood(dd, nn, curNode.pathEta + eta);
            }
            nodeList.add(null);
            logprobList.add(lp);
        }

        // sample
        int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobList);

        Node sampledNode = nodeList.get(sampledIdx);
        if (sampledNode == null) { // new child
            int nodeIdx = curNode.getNextChildIndex();
            int nodeLevel = curNode.getLevel() + 1;
            boolean extendable = isExtendable(nodeLevel);
            DirMult topic = new DirMult(V, getBeta(nodeLevel) * V, uniform);
            Node newNode = new Node(iter, nodeIdx, nodeLevel, extendable, topic, curNode, eta);
            newNode.pathEta = curNode.pathEta + eta;
            if (extendable) {
                newNode.initializeGlobalTheta();
                newNode.initializeGlobalPi();
            }
            return newNode;
        } else if (sampledNode.equals(curNode)) { // stay at current node
            return sampledNode;
        } else { // recursively move to an existing child
            return sampleNode(dd, nn, sampledNode, extend, observed);
        }
    }

    /**
     * Recursively sample a node from a current node. The sampled node can be
     * either the same node or one of its children. If the current node is a
     * leaf node, return it.
     *
     * @param dd Document index
     * @param nn Token index
     * @param curNode Current node
     * @param extend Whether the tree structure is extendable
     */
//    private Node sampleNodeOld(int dd, int nn, Node curNode, boolean extend) {
//        if (curNode.isLeaf() && !curNode.extensible) {
//            return curNode;
//        }
//
//        int level = curNode.getLevel();
//        double gamma = getGamma(level);
//        double stayprob;
//        if (curNode.isRoot() && !isRooted) {
//            stayprob = 0.0;
//        } else {
//            stayprob = (curNode.nodeDocCounts.getCount(dd) + gamma * curNode.pi)
//                    / (curNode.subtreeDocCounts.getCount(dd) + gamma);
//        }
//        double passprob = 1.0 - stayprob;
//
//        ArrayList<Node> nodeList = new ArrayList<>();
//        ArrayList<Double> nodeProbs = new ArrayList<>();
//
//        // for staying at current node
//        nodeList.add(curNode);
//        nodeProbs.add(stayprob * curNode.getPhi(words[dd][nn]));
//
//        // for moving to an existing child node
//        double lAlpha = getLocalAlpha(level);
//        double norm = curNode.getPassingCount(dd) + lAlpha;
//        for (Node child : curNode.getChildren()) {
//            int childIdx = child.getIndex();
//            nodeList.add(child);
//            double pathprob = (child.subtreeDocCounts.getCount(dd)
//                    + lAlpha * curNode.theta.get(childIdx)) / norm;
//            nodeProbs.add(passprob * pathprob * child.getPhi(words[dd][nn]));
//        }
//
//        // for moving to a new child node
//        if (extend && curNode.extensible) {
//            nodeList.add(null);
//            double pathprob = lAlpha * curNode.theta.get(NEW_CHILD_INDEX) / norm;
//            nodeProbs.add(passprob * pathprob * uniform);
//        }
//
//        // sample
//        int sampledIdx = SamplerUtils.scaleSample(nodeProbs);
//        if (sampledIdx == nodeProbs.size()) { // debug
//            for (int ii = 0; ii < nodeProbs.size(); ii++) {
//                logln(ii
//                        + ". " + nodeList.get(ii)
//                        + ". " + nodeProbs.get(ii));
//            }
//        }
//
//        Node sampledNode = nodeList.get(sampledIdx);
//        if (sampledNode == null) { // new child
//            int nodeIdx = curNode.getNextChildIndex();
//            int nodeLevel = curNode.getLevel() + 1;
//            boolean extendable = isExtendable(nodeLevel);
//            DirMult topic = new DirMult(V, getBeta(nodeLevel) * V, uniform);
//
//            double eta = 0.0;
//            if (isSupervised()) {
//                eta = SamplerUtils.getGaussian(mu, getSigma(nodeLevel));
//            }
//            Node newNode = new Node(iter, nodeIdx, nodeLevel, extendable, topic, curNode, eta);
//            if (extendable) {
//                newNode.initializeGlobalTheta();
//                newNode.initializeGlobalPi();
//            }
//            return newNode;
//        } else if (sampledNode.equals(curNode)) { // stay at current node
//            return sampledNode;
//        } else { // recursively move to an existing child
//            return sampleNode(dd, nn, sampledNode, extend);
//        }
//    }
    /**
     * Compute both the proposal log probabilities and the actual log
     * probabilities of assigning a token to a node.
     *
     * @param dd Document index
     * @param nn Token index
     * @param observed
     * @param node The node to be assigned to
     */
//    private double[] getLogProbabilities(int dd, int nn, Node node, boolean observed) {
//        double[] logprobs = getTransLogProbabilities(dd, nn, node, node);
//        logprobs[ACTUAL_INDEX] = Math.log(node.getPhi(words[dd][nn]));
//        if (isSupervised() && observed) {
//            logprobs[ACTUAL_INDEX] += getResponseLogLikelihood(dd, nn, node);
//        }
//        Node source = node.getParent();
//        Node target = node;
//        while (source != null) {
//            double[] lps = getTransLogProbabilities(dd, nn, source, target);
//            logprobs[PROPOSAL_INDEX] += lps[PROPOSAL_INDEX];
//            logprobs[ACTUAL_INDEX] += lps[ACTUAL_INDEX];
//
//            source = source.getParent();
//            target = target.getParent();
//        }
//        return logprobs;
//    }
    /**
     * Get the transition log probability of moving a token from a source node
     * to a target node. The source node can be the same as the target node.
     *
     * @param dd
     * @param nn
     * @param source
     * @param target
     */
//    private double[] getTransLogProbabilities(int dd, int nn, Node source, Node target) {
//        int level = source.getLevel();
//        if (level == L - 1) { // leaf node
//            if (!source.equals(target)) {
//                throw new RuntimeException("At leaf node. " + source.toString()
//                        + ". " + target.toString());
//            }
//            return new double[2]; // stay with probabilities 1
//        }
//
//        double pNum = 0.0;
//        double pDen = 0.0;
//        double aNum = 0.0;
//        double aDen = 0.0;
//
//        double lAlpha = getLocalAlpha(level);
//        double gamma = getGamma(level);
//        double stayprob = (source.nodeDocCounts.getCount(dd) + gamma * source.pi)
//                / (source.subtreeDocCounts.getCount(dd) + gamma);
//        double passprob = 1.0 - stayprob;
//        double norm = source.subtreeDocCounts.getCount(dd)
//                - source.nodeDocCounts.getCount(dd) + lAlpha;
//
//        // existing children
//        boolean foundTarget = false;
//        for (Node child : source.getChildren()) {
//            double pathprob;
//            double wordprob;
//            if (child.newNode) { // newly created child
//                wordprob = 1.0 / V;
//                pathprob = lAlpha * source.theta.get(NEW_CHILD_INDEX) / norm;
//            } else {
//                try {
//                    pathprob = (child.subtreeDocCounts.getCount(dd)
//                            + lAlpha * source.theta.get(child.getIndex())) / norm;
//                } catch (Exception e) {
//                    e.printStackTrace();
//                    System.out.println("source: " + source.toString());
//                    System.out.println("target: " + target.toString());
//                    System.out.println("child: " + child.toString());
//                    System.out.println("subtree: " + child.subtreeDocCounts.getCount(dd));
//                    System.out.println("theta: " + source.theta.get(child.getIndex()));
//                    throw new RuntimeException("Exception");
//                }
//
//                wordprob = child.getPhi(words[dd][nn]);
//            }
//
//            double aVal = passprob * pathprob;
//            aDen += aVal;
//
//            double pVal = passprob * pathprob * wordprob;
//            pDen += pVal;
//
//            if (target.equals(child)) { // including a new child
//                pNum = pVal;
//                aNum = aVal;
//                foundTarget = true;
//            }
//        }
//
//        // staying at the current node
//        double wordprob = source.getPhi(words[dd][nn]);
//        double pVal = stayprob * wordprob;
//        pDen += pVal;
//        aDen += stayprob;
//
//        if (target.equals(source)) {
//            pNum = pVal;
//            aNum = stayprob;
//            foundTarget = true;
//        }
//
//        if (!foundTarget) {
//            if (!target.isEmpty()) {
//                throw new RuntimeException("Target node is not empty and could not be found");
//            }
//
//            double wProb = 1.0 / V;
//            double pProb = lAlpha * source.theta.get(NEW_CHILD_INDEX) / norm;
//            double aVal = passprob * pProb;
//            aDen += aVal;
//            pVal = passprob * pProb * wProb;
//            pDen += pVal;
//
//            pNum = pVal;
//            aNum = aVal;
//        }
//
//        double[] lps = new double[2];
//        lps[PROPOSAL_INDEX] = Math.log(pNum / pDen);
//        lps[ACTUAL_INDEX] = Math.log(aNum / aDen);
//        return lps;
//    }
    /**
     * Compute the log likelihood of an author's response variable given that a
     * token from the author is assigned to a given node.
     *
     * @param dd
     * @param nn
     * @param node The node
     * @return
     */
    private double getResponseLogLikelihood(int dd, int nn, Node node) {
        return getResponseLogLikelihood(dd, nn, node.pathEta);
    }

    /**
     * Compute the log likelihood of an author's response variable given that a
     * token from the author is assigned to a given node.
     *
     * @param dd
     * @param nn
     * @param pathEta
     * @return
     */
    private double getResponseLogLikelihood(int dd, int nn, double pathEta) {
        double aMean = dotprods[dd] + pathEta / this.words[dd].length;
        double resLLh;
        if (mode == Mode.SUPERVISED_BINARY) {
            resLLh = getLabelLogLikelihood(labels[dd], aMean);
        } else if (mode == Mode.SUPERVISED_CONTINUOUS) {
            resLLh = StatUtils.logNormalProbability(responses[dd], aMean, Math.sqrt(rho));
        } else {
            throw new RuntimeException("Mode " + mode + " is not supported");
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
     * Update both eta's and tau's.
     */
    public long updateEtasTaus() {
        if (isReporting) {
            logln("+++ Updating eta's and tau's ...");
        }
        long sTime = System.currentTimeMillis();

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        SparseVector[] designMatrix = new SparseVector[D];
        for (int dd = 0; dd < D; dd++) {
            designMatrix[dd] = new SparseVector(N + V);
        }
        // topic regression
        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            for (int dd : node.subtreeDocCounts.getIndices()) {
                int count = node.subtreeDocCounts.getCount(dd);
                double val = (double) count / this.words[dd].length;
                designMatrix[dd].change(kk, val);
            }
        }
        // lexical regression
        for (int dd = 0; dd < D; dd++) {
            for (int vv : lexicalDesginMatrix[dd].getIndices()) {
                designMatrix[dd].change(N + vv, lexicalDesginMatrix[dd].get(vv));
            }
        }

        // current params
        double[] etaTauArray = new double[N + V];
        double[] sigmaArray = new double[N + V];
        for (int kk = 0; kk < N; kk++) {
            etaTauArray[kk] = nodeList.get(kk).eta;
            sigmaArray[kk] = getSigma(nodeList.get(kk).getLevel());
        }
        for (int vv = 0; vv < V; vv++) {
            etaTauArray[vv + N] = tau[vv];
            sigmaArray[vv + N] = sigma;
        }

        boolean converged = false;
        if (mode == Mode.SUPERVISED_CONTINUOUS) {
            RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                    responses, etaTauArray, designMatrix, rho, mu, sigmaArray);
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
            for (int vv = 0; vv < V; vv++) {
                tau[vv] = optimizable.getParameter(N + vv);
            }

            // update document means
            for (int dd = 0; dd < D; dd++) {
                dotprods[dd] = 0.0;
                for (int ii : designMatrix[dd].getIndices()) {
                    dotprods[dd] += designMatrix[dd].get(ii) * optimizable.getParameter(ii);
                }
            }
        } else {
            throw new RuntimeException("Mode " + mode + " is not supported");
        }

        updatePathEtas();

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
            evaluatePerformances();
        }
        return eTime;
    }

    /**
     * Update regression parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        // design matrix
        SparseVector[] designMatrix = new SparseVector[D];
        for (int dd = 0; dd < D; dd++) {
            designMatrix[dd] = new SparseVector(N);
        }

        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            for (int dd : node.subtreeDocCounts.getIndices()) {
                int count = node.subtreeDocCounts.getCount(dd);
                double val = (double) count / this.words[dd].length;
                designMatrix[dd].change(kk, val);
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

        if (mode == Mode.SUPERVISED_BINARY) {
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
        } else if (mode == Mode.SUPERVISED_CONTINUOUS) {
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
        } else {
            throw new RuntimeException("Mode " + mode + " is not supported");
        }

        // update document means
        for (int dd = 0; dd < D; dd++) {
            dotprods[dd] = 0.0;
            for (int kk : designMatrix[dd].getIndices()) {
                dotprods[dd] += designMatrix[dd].get(kk) * nodeList.get(kk).eta;
            }
        }

        updatePathEtas();

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
            evaluatePerformances();
        }
        return eTime;
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
     * *
     * Get the list of nodes in the tree excluding the root.
     *
     * @return
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

    /**
     * *
     * Update global theta and omega.
     *
     * @return Elapsed time
     */
    public long updateGlobalProbabilities() {
        if (isReporting) {
            logln("+++ Updating global theta's and pi's ...");
        }
        long sTime = System.currentTimeMillis();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            if (isExtendable(node.getLevel())) {
                node.updateGlobalTheta();
                node.updateGlobalPi();
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
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
        validateModel(msg);
        validateData(msg);
    }

    private void validateModel(String msg) {
        logln("--- Validating model ... " + msg);
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

    private void validateData(String msg) {
        logln("--- Validating data ... " + msg);
        int tokenCount = 0;

        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            tokenCount += node.getContent().getCountSum();
        }
        if (numTokens != tokenCount) {
            throw new MismatchRuntimeException(numTokens, tokenCount);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("Outputing current state to " + filepath);
        }

        // model string
        StringBuilder modelStr = new StringBuilder();
        if (isLexical) {
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(tau[vv]).append("\n");
            }
        }

        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            modelStr.append(Integer.toString(node.born)).append("\n");
            modelStr.append(node.getPathString()).append("\n");
            modelStr.append(node.eta).append("\n");
            modelStr.append(node.pi).append("\n");
            if (node.theta != null) {
                modelStr.append(hashMapToString(node.theta));
            }
            modelStr.append("\n");
            modelStr.append(DirMult.output(node.getContent())).append("\n");
            modelStr.append(SparseCount.output(node.subtreeWordCounts)).append("\n");
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // assignment string
        StringBuilder assignStr = new StringBuilder();
        for (int dd = 0; dd < z.length; dd++) {
            for (int nn = 0; nn < z[dd].length; nn++) {
                assignStr.append(dd)
                        .append("\t").append(nn)
                        .append("\t").append(z[dd][nn].getPathString()).append("\n");
            }
        }

        // output to a compressed file
        try {
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
    public void inputState(String filepath) {
        if (verbose) {
            logln("Inputing state from " + filepath);
        }
        try {
            inputModel(filepath);
            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    /**
     * Input a learned model.
     *
     * @param zipFilepath Compressed learned state file
     */
    void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            if (isLexical) {
                this.tau = new double[V];
                for (int vv = 0; vv < V; vv++) {
                    this.tau[vv] = Double.parseDouble(reader.readLine());
                }
            }

            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                String pathStr = reader.readLine();
                double eta = Double.parseDouble(reader.readLine());
                double pi = Double.parseDouble(reader.readLine());
                line = reader.readLine().trim();
                HashMap<Integer, Double> theta = new HashMap<>();
                if (!line.isEmpty()) {
                    theta = stringToHashMap(line);
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

                Node node = new Node(born, nodeIndex, nodeLevel, false, topic, parent, eta);
                node.changeStatus();
                node.pi = pi;
                node.theta = theta;
                node.setContent(topic);
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
        if (verbose && debug) {
            logln(printGlobalTree());
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

            initializeDataStructure();

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
                    addToken(dd, nn, z[dd][nn], ADD, !ADD);
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

    private Node getNode(String pathStr) {
        return getNode(parseNodePath(pathStr));
    }

    public static String hashMapToString(HashMap<Integer, Double> table) {
        if (table.isEmpty()) {
            return "";
        }
        StringBuilder str = new StringBuilder();
        for (int key : table.keySet()) {
            str.append(key).append(":").append(table.get(key)).append("\t");
        }
        return str.toString();
    }

    public static HashMap<Integer, Double> stringToHashMap(String str) {
        HashMap<Integer, Double> table = new HashMap<>();
        String[] sstr = str.split("\t");
        for (String s : sstr) {
            String[] ss = s.split(":");
            int key = Integer.parseInt(ss[0]);
            double val = Double.parseDouble(ss[1]);
            table.put(key, val);
        }
        return table;
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
            int level = node.getLevel();
            nodeCountPerLevel.increment(level);
            obsCountPerLevel.changeCount(level, node.nodeDocCounts.getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeDocCounts.getCountSum());

            totalObs += node.getContent().getCountSum();

            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("Global tree summary:\n\t>>> node count per level:\n");
        for (int l : nodeCountPerLevel.getSortedIndices()) {
            int obsCount = obsCountPerLevel.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLevel.getCount(l);
            str.append("\t>>> >>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / numTokens))
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
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

        StringBuilder str = new StringBuilder();
        str.append("Global tree\n");

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                if (isSupervised()) {
                    rankChildren.add(new RankingItem<Node>(child, child.eta));
                } else {
                    rankChildren.add(new RankingItem<Node>(child, child.subtreeDocCounts.getCountSum()));
                }
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            int level = node.getLevel();
            nodeCountPerLvl.increment(level);
            obsCountPerLvl.changeCount(level, node.nodeDocCounts.getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeDocCounts.getCountSum());

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString());
            if (!node.isRoot()) {
                str.append(" [")
                        .append(MiscUtils.formatDouble(node.getParent().theta.get(node.getIndex())))
                        .append("]");
            }
            str.append("\n");

            // top words according to subtree distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            String[] subtreeTopWords = node.getSubtreeTopWords(10);
            for (String w : subtreeTopWords) {
                str.append(w).append(" ");
            }
            str.append("\n");

            // top assigned words
            if (!node.getContent().isEmpty()) {
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("\t");
                }
                str.append(node.getTopObservations()).append("\n");
            }

            // lexical regression @ root
            if (node.isRoot() && isSupervised() && isLexical) {
                ArrayList<RankingItem<String>> rankLexItems = new ArrayList<>();
                for (int vv = 0; vv < V; vv++) {
                    rankLexItems.add(new RankingItem<String>(wordVocab.get(vv), tau[vv]));
                }
                Collections.sort(rankLexItems);

                // most positive words
                str.append("+++ ");
                for (int ii = 0; ii < 10; ii++) {
                    RankingItem<String> item = rankLexItems.get(ii);
                    str.append(item.getObject()).append(" (")
                            .append(MiscUtils.formatDouble(item.getPrimaryValue()))
                            .append("); ");
                }
                str.append("\n");

                // most negative words
                str.append("--- ");
                for (int ii = 0; ii < 10; ii++) {
                    RankingItem<String> item = rankLexItems.get(V - 1 - ii);
                    str.append(item.getObject()).append(" (")
                            .append(MiscUtils.formatDouble(item.getPrimaryValue()))
                            .append("); ");
                }
                str.append("\n");
            }

            str.append("\n");

            totalNumObs += node.getContent().getCountSum();

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
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
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

            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                if (isSupervised()) {
                    rankChildren.add(new RankingItem<Node>(child, child.eta));
                } else {
                    rankChildren.add(new RankingItem<Node>(child, child.subtreeDocCounts.getCountSum()));
                }
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            String[] topWords = node.getNodeTopWords(numWords);

            // top words according to the distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.born)
                    .append("; ").append(node.getContent().getCountSum())
                    .append(isSupervised() ? ("; " + MiscUtils.formatDouble(node.eta)) : "")
                    .append(isSupervised() ? ("; " + MiscUtils.formatDouble(node.pathEta)) : "")
                    .append(")");
            str.append(" [")
                    .append(node.isRoot() ? "-" : MiscUtils.formatDouble(node.getParent().theta.get(node.getIndex())))
                    .append(node.pi == 0 ? "" : (", " + MiscUtils.formatDouble(node.pi)))
                    .append("]");
            str.append("\n");

            // words with highest probabilities
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            for (String topWord : topWords) {
                str.append(topWord).append(" ");
            }
            str.append("\n");

            // top assigned words
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getTopObservations()).append("\n");

            // lexical regression @ root
            if (node.isRoot() && isSupervised() && isLexical) {
                ArrayList<RankingItem<String>> rankLexItems = new ArrayList<>();
                for (int vv = 0; vv < V; vv++) {
                    rankLexItems.add(new RankingItem<String>(wordVocab.get(vv), tau[vv]));
                }
                Collections.sort(rankLexItems);

                // most positive words
                str.append("+++ ");
                for (int ii = 0; ii < 10; ii++) {
                    RankingItem<String> item = rankLexItems.get(ii);
                    str.append(item.getObject()).append(" (")
                            .append(MiscUtils.formatDouble(item.getPrimaryValue()))
                            .append("); ");
                }
                str.append("\n");

                // most negative words
                str.append("--- ");
                for (int ii = 0; ii < 10; ii++) {
                    RankingItem<String> item = rankLexItems.get(V - 1 - ii);
                    str.append(item.getObject()).append(" (")
                            .append(MiscUtils.formatDouble(item.getPrimaryValue()))
                            .append("); ");
                }
                str.append("\n");
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

    public void outputRankedLexicalItems(File outputFile) {
        try {
            ArrayList<RankingItem<Integer>> rankLexItems = new ArrayList<>();
            for (int vv = 0; vv < V; vv++) {
                rankLexItems.add(new RankingItem<Integer>(vv, tau[vv]));
            }
            Collections.sort(rankLexItems);

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int vv = 0; vv < V; vv++) {
                RankingItem<Integer> item = rankLexItems.get(vv);
                writer.write(vv
                        + "\t" + item.getObject()
                        + "\t" + wordVocab.get(item.getObject())
                        + "\t" + item.getPrimaryValue()
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected final boolean extensible;
        protected SparseCount subtreeDocCounts;
        protected SparseCount nodeDocCounts;
        protected double eta; // regression parameter
        protected double pathEta;
        protected double pi;
        protected HashMap<Integer, Double> theta;
        protected boolean newNode; // whether this node is newly created
        protected SparseCount subtreeWordCounts;

        public Node(int iter, int index, int level, boolean extendable,
                DirMult content, Node parent, double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.extensible = extendable;
            this.eta = eta;
            this.subtreeDocCounts = new SparseCount();
            this.nodeDocCounts = new SparseCount();
            this.theta = new HashMap<>();
            this.newNode = true;
            this.subtreeWordCounts = new SparseCount();
        }

        void incrementSubtreeWordCount(int vv) {
            this.subtreeWordCounts.increment(vv); // MAXIMAL only
        }

        void decrementSubtreeWordCount(int vv) {
            this.subtreeWordCounts.decrement(vv); // MAXIMAL only
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

        public void removeChildUpdate(int childIndex) {
            this.removeChild(childIndex);
            this.updateGlobalTheta();
        }

        public Node addChildUpdate(int childIndex, Node child) {
            Node newChild = this.addChild(childIndex, child);
            this.updateGlobalTheta();
            return newChild;
        }

        public void setPathEta(double pathEta) {
            this.pathEta = pathEta;
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

        /**
         * Return the number of tokens from all documents which are assigned to
         * any descendant node of this node.
         */
        int getPassingCountSum() {
            return subtreeDocCounts.getCountSum() - nodeDocCounts.getCountSum();
        }

        void changeStatus() {
            this.newNode = false;
        }

        boolean isExtensible() {
            return this.extensible;
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
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

        String[] getSubtreeTopWords(int numTopWords) {
            double[] subtreePhi = new double[V];
            for (int vv = 0; vv < V; vv++) {
                subtreePhi[vv] = getSubtreeWordProbability(vv);
            }
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(subtreePhi, wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
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

        void initializeGlobalPi() {
            this.pi = getPi(level);
        }

        void initializeGlobalTheta() {
            int KK = getNumChildren();
            // initialize
            int thetaSize = KK;
            if (extensible) {
                thetaSize = KK + 1;
            }
            this.theta = new HashMap<>();
            double val = 1.0 / thetaSize;
            for (Node child : getChildren()) {
                this.theta.put(child.getIndex(), val);
            }
            if (extensible) {
                this.theta.put(NEW_CHILD_INDEX, val);
            }
        }

        /**
         * Update the global pi based on the current counts.
         */
        void updateGlobalPi() {
//            int totalStay = this.tokenCounts.getCountSum();
//            int totalStayAndPass = this.subtreeTokenCounts.getCountSum();
//            double gamma = getGamma(level);
//            this.pi = (totalStay + gamma * getPi(level)) / (totalStayAndPass + gamma);
        }

        /**
         * Update the global theta distribution based on the current
         * approximated counts.
         */
        void updateGlobalTheta() {
            double gAlpha = getGlobalAlpha(level);

            // update counts
            SparseCount approxThetaCounts = new SparseCount();
            for (Node child : getChildren()) {
                int childIdx = child.getIndex();
                for (int dd : child.subtreeDocCounts.getIndices()) {
                    int rawCount = child.subtreeDocCounts.getCount(dd);
                    Double thetaVal = this.theta.get(childIdx);
                    if (thetaVal == null) { // this child has just been added
                        thetaVal = theta.get(NEW_CHILD_INDEX);
                    }
                    int approxCount = getApproxCount(rawCount, thetaVal);
                    approxThetaCounts.changeCount(childIdx, approxCount);
                }
            }

            // update theta
            this.theta = new HashMap<>();
            double norm = approxThetaCounts.getCountSum() + gAlpha;
            for (int childIdx : approxThetaCounts.getIndices()) {
                this.theta.put(childIdx,
                        (double) approxThetaCounts.getCount(childIdx) / norm);
            }
            this.theta.put(NEW_CHILD_INDEX, gAlpha / norm);
        }

        /**
         * Compute the approximated count, propagated from lower-level
         * restaurant. This can be approximated using (1) Maximal path
         * assumption, (2) Minimal path assumption, and (3) Sampling from
         * Antoniak distribution.
         *
         * @param count Actual count from lower-level restaurant
         * @param curThetaVal Current theta value
         * @return Approximate count
         */
        int getApproxCount(int count, double curThetaVal) {
            if (count > 1) {
                double val = getGlobalAlpha(level) * (getNumChildren() + 1) * curThetaVal;
                return SamplerUtils.randAntoniak(val, count);
            } else {
                return count;
            }
        }

        public void validate(String msg) {
            this.content.validate(msg);
            this.subtreeDocCounts.validate(msg);
            this.nodeDocCounts.validate(msg);
            int childCountSum = 0;
            for (Node child : getChildren()) {
                childCountSum += child.subtreeDocCounts.getCountSum();
            }
            if (childCountSum + nodeDocCounts.getCountSum() != subtreeDocCounts.getCountSum()) {
                throw new MismatchRuntimeException(subtreeDocCounts.getCountSum(),
                        (childCountSum + nodeDocCounts.getCountSum()));
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", c (").append(getChildren().size()).append(")");
            str.append(", (").append(getContent().getCountSum())
                    .append(", ").append(subtreeDocCounts.getCountSum())
                    .append(", ").append(nodeDocCounts.getCountSum())
                    .append(")");
            if (isSupervised()) {
                str.append(", ").append(MiscUtils.formatDouble(eta));
                str.append(", ").append(MiscUtils.formatDouble(pathEta));
            }
            str.append(", ").append(extensible);
            str.append(", ").append(newNode);
            str.append("]");
            return str.toString();
        }
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
        InitialState initState = getInitialState(init);

        String path = CLIUtils.getStringArgument(cmd, "path", "none");
        PathAssumption pathAssumption = getPathAssumption(path);

        // model parameters
        int L = CLIUtils.getIntegerArgument(cmd, "L", 3);
        int[] Ks = CLIUtils.getIntArrayArgument(cmd, "Ks", new int[]{10, 4}, ",");
        double[] globalAlphas = CLIUtils.getDoubleArrayArgument(cmd, "global-alphas",
                new double[]{2.0, 1.0}, ",");
        double[] localAlphas = CLIUtils.getDoubleArrayArgument(cmd, "local-alphas",
                new double[]{1.0, 1.0}, ",");
        double[] betas = CLIUtils.getDoubleArrayArgument(cmd, "betas",
                new double[]{0.5, 0.25, 0.1}, ",");
        double[] pis = CLIUtils.getDoubleArrayArgument(cmd, "pis",
                new double[]{0.2, 0.2}, ",");
        double[] gammas = CLIUtils.getDoubleArrayArgument(cmd, "gammas",
                new double[]{100, 10}, ",");
        double rho = CLIUtils.getDoubleArgument(cmd, "rho", 1.0);
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double[] sigmas = CLIUtils.getDoubleArrayArgument(cmd, "sigmas",
                new double[]{0.5, 2.5}, ",");
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.0);

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

        File docInfoFile = null;
        if (cmd.hasOption("info-file")) {
            docInfoFile = new File(cmd.getOptionValue("info-file"));
        }

        boolean isRooted = cmd.hasOption("root");

        HTM sampler = new HTM();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);

        File samplerFolder;
        String modeStr = CLIUtils.getStringArgument(cmd, "mode", "unsupervised");
        switch (modeStr) {
            case "unsupervised":
                TextDataset textData = new TextDataset(datasetName);
                textData.loadFormattedData(new File(wordVocFile),
                        new File(docWordFile), docInfoFile, null);

                sampler.setBasename("NHDP");
                sampler.setWordVocab(textData.getWordVocab());
                sampler.configure(outputFolder, textData.getWordVocab().size(),
                        L, Ks, null, priorTopics,
                        globalAlphas, localAlphas, betas,
                        pis, gammas,
                        initState, pathAssumption, isRooted, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);

                samplerFolder = new File(sampler.getSamplerFolderPath());
                IOUtils.createFolder(samplerFolder);

                if (isTraining()) {
                    ArrayList<Integer> trainDocIndices = sampler.getSelectedDocIndices(textData.getDocIds());
                    sampler.train(textData.getWords(), trainDocIndices);
                    sampler.initialize(priorTopics);
                    sampler.metaIterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                }
                break;
            case "binary":
                LabelTextDataset binData = new LabelTextDataset(datasetName);
                binData.loadFormattedData(new File(wordVocFile),
                        new File(docWordFile), docInfoFile, null);

                sampler.setWordVocab(binData.getWordVocab());
                sampler.configureBinary(outputFolder, binData.getWordVocab().size(),
                        L, Ks, null, priorTopics,
                        globalAlphas, localAlphas, betas,
                        pis, gammas, mu, sigmas, sigma,
                        initState, pathAssumption, isRooted, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);

                samplerFolder = new File(sampler.getSamplerFolderPath());
                IOUtils.createFolder(samplerFolder);

                if (isTraining()) {
                    ArrayList<Integer> trainDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());
                    sampler.train(binData.getWords(), trainDocIndices, binData.getSingleLabels());
                    sampler.initialize(priorTopics);
                    sampler.metaIterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);

                    File trResultFolder = new File(samplerFolder,
                            AbstractExperiment.TRAIN_PREFIX + AbstractExperiment.RESULT_FOLDER);
                    IOUtils.createFolder(trResultFolder);

                }

                if (isTesting()) {
                    ArrayList<Integer> testDocIndices = sampler.getSelectedDocIndices(binData.getDocIds());

                    File testAssignmentFolder = new File(samplerFolder, AbstractSampler.IterAssignmentFolder);
                    IOUtils.createFolder(testAssignmentFolder);

                    File testPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
                    IOUtils.createFolder(testPredFolder);

                    double[] predictions;
                    if (cmd.hasOption("parallel")) {
                        predictions = HTM.parallelTest(binData.getWords(), testDocIndices,
                                testPredFolder, testAssignmentFolder, sampler);
                    } else {
                        sampler.test(binData.getWords(), testDocIndices);

                        File stateFile = sampler.getFinalStateFile();
                        File outputPredFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".txt");
                        File outputStateFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".zip");
                        predictions = sampler.sampleTest(stateFile, outputStateFile, outputPredFile);
                    }

                    File teResultFolder = new File(samplerFolder,
                            AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
                    IOUtils.createFolder(teResultFolder);

                    PredictionUtils.outputClassificationPredictions(
                            new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                            binData.getDocIds(), binData.getSingleLabels(), predictions);
                    PredictionUtils.outputBinaryClassificationResults(
                            new File(teResultFolder, AbstractExperiment.RESULT_FILE),
                            binData.getSingleLabels(), predictions);
                }
                break;
            case "continuous":
                ResponseTextDataset contData = new ResponseTextDataset(datasetName);
                contData.loadFormattedData(new File(wordVocFile),
                        new File(docWordFile), docInfoFile, null);

                sampler.setWordVocab(contData.getWordVocab());
                sampler.configureContinuous(outputFolder, contData.getWordVocab().size(),
                        L, Ks, null, priorTopics,
                        globalAlphas, localAlphas, betas,
                        pis, gammas, rho, mu, sigmas, sigma,
                        initState, pathAssumption, isRooted, paramOpt,
                        burnIn, maxIters, sampleLag, repInterval);

                samplerFolder = new File(sampler.getSamplerFolderPath());
                IOUtils.createFolder(samplerFolder);

                double[] docResponses = contData.getResponses();
                if (cmd.hasOption("z")) { // z-normalization
                    ZNormalizer zNorm = new ZNormalizer(docResponses);
                    docResponses = zNorm.normalize(docResponses);
                }

                if (isTraining()) {
                    ArrayList<Integer> trainDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());
                    sampler.train(contData.getWords(), trainDocIndices, docResponses);
                    sampler.initialize(priorTopics);
                    sampler.metaIterate();
                    sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
                }

                if (isTesting()) {
                    ArrayList<Integer> testDocIndices = sampler.getSelectedDocIndices(contData.getDocIds());

                    File testAssignmentFolder = new File(samplerFolder, AbstractSampler.IterAssignmentFolder);
                    IOUtils.createFolder(testAssignmentFolder);

                    File testPredFolder = new File(samplerFolder, AbstractSampler.IterPredictionFolder);
                    IOUtils.createFolder(testPredFolder);

                    if (cmd.hasOption("parallel")) {
                        HTM.parallelTest(contData.getWords(), testDocIndices, testPredFolder, testAssignmentFolder, sampler);
                    } else {
                        sampler.test(contData.getWords(), testDocIndices);

                        File stateFile = sampler.getFinalStateFile();
                        File outputPredFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".txt");
                        File outputStateFile = new File(testPredFolder, "iter-" + sampler.MAX_ITER + ".zip");
                        double[] predictions = sampler.sampleTest(stateFile, outputStateFile, outputPredFile);

                        File teResultFolder = new File(samplerFolder,
                                AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
                        IOUtils.createFolder(teResultFolder);

                        PredictionUtils.outputRegressionPredictions(
                                new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                                contData.getDocIds(), docResponses, predictions);
                        PredictionUtils.outputRegressionResults(
                                new File(teResultFolder, AbstractExperiment.RESULT_FILE),
                                docResponses, predictions);
                    }
                }
                break;
            default:
                throw new RuntimeException("Mode " + modeStr + " not supported");
        }
    }

    private static void addOpitions() throws Exception {
        parser = new BasicParser();
        options = new Options();

        // data input
        addDataOptions();

        // data output
        addOption("output-folder", "Output folder");

        addOption("prior-topic-file", "File containing prior topics");
        options.addOption("z", false, "z-normalize");
        options.addOption("root", false, "Does root generate words?");

        // parameters
        addOption("local-alphas", "Local alphas");
        addOption("global-alphas", "Global alphas");
        addOption("betas", "Betas");
        addOption("pis", "Pis");
        addOption("gammas", "Gammas");
        addOption("rho", "Rho");
        addOption("mu", "Mu");
        addOption("sigmas", "Sigmas");
        addOption("sigma", "Sigma");

        addOption("Ks", "Branching factors");
        addOption("L", "Tree height");
        addOption("num-top-words", "Number of top words per topic");
        addOption("mode", "mode");

        // configurations
        addOption("init", "Initialization");
        addOption("path", "Path assumption");

        // sampling & runnning
        addSamplingOptions();
        addRunningOptions();
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

    public static String getHelpString() {
        return "java -cp \"dist/segan.jar\" " + HTM.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        String example = new String();
        return example;
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
            HTM sampler) {
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

                HTMTestRunner runner = new HTMTestRunner(sampler,
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

    public void debugRegression(File outputFile) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int dd = 0; dd < D; dd++) {
                double topVal = 0.0;
                double lexVal = 0.0;
                int topCount = 0;
                int lexCount = 0;
                for (int nn = 0; nn < words[dd].length; nn++) {
                    Node node = z[dd][nn];
                    if (node.isRoot()) {
                        lexCount++;
                        lexVal += tau[words[dd][nn]] / words.length;
                    } else {
                        topCount++;
                        topVal += node.eta / words[dd].length;
                    }
                }
                writer.write(dd
                        + "\t" + responses[dd]
                        + "\t" + topVal + " (" + topCount + ")"
                        + "\t" + lexVal + " (" + lexCount + ")"
                        + "\t" + (topVal + lexVal)
                        + "\t" + Math.abs(responses[dd] - lexVal - topVal)
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}

class HTMTestRunner implements Runnable {

    HTM sampler;
    int[][] newWords;
    ArrayList<Integer> newDocIndices;
    String stateFile;
    String testStateFile;
    String outputFile;

    public HTMTestRunner(HTM sampler,
            int[][] newWords,
            ArrayList<Integer> newDocIndices,
            String stateFile,
            String testStateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newDocIndices = newDocIndices;
        this.stateFile = stateFile;
        this.testStateFile = testStateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        HTM testSampler = new HTM();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.test(newWords, newDocIndices);
            testSampler.sampleTest(new File(stateFile), new File(testStateFile),
                    new File(outputFile));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
