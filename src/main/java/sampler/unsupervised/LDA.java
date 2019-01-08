package sampler.unsupervised;

import core.AbstractSampler;
import data.TextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 * Implementation of Latent Dirichlet Allocation (LDA).
 *
 * @author vietan
 */
public class LDA extends AbstractSampler {

    // hyperparameters
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    // inputs
    protected int[][] words; // original documents
    protected ArrayList<Integer> docIndices; // [D]: indices of considered docs
    protected int K;
    protected int V;
    // derived
    protected int D;
    // latent
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    protected int[][] z;

    public LDA() {
        this.basename = "LDA";
    }

    public LDA(String basename) {
        this.basename = basename;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.K = K;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_" + this.basename
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(hyperparams.get(ALPHA))
                + "_b-" + formatter.format(hyperparams.get(BETA))
                + "_opt-" + this.paramOptimized;
    }

    /**
     * Return the current topic assignments for all tokens.
     *
     * @return Current topic assignments
     */
    public int[][] getZs() {
        return this.z;
    }

    /**
     * Return the learned topics.
     *
     * @return The learned topics
     */
    public DirMult[] getTopicWords() {
        return this.topicWords;
    }

    /**
     * Return the learned distribution over topics for each document.
     *
     * @return
     */
    public DirMult[] getDocTopics() {
        return this.docTopics;
    }

    public double[][] getThetas() {
        double[][] thetas = new double[D][];
        for (int dd = 0; dd < D; dd++) {
            thetas[dd] = this.docTopics[dd].getDistribution();
        }
        return thetas;
    }

    public double[][] getPhis() {
        double[][] phis = new double[K][];
        for (int kk = 0; kk < K; kk++) {
            phis[kk] = this.topicWords[kk].getDistribution();
        }
        return phis;
    }

    /**
     * Set training data.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     */
    public void train(int[][] docWords, ArrayList<Integer> docIndices) {
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

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure(null);

        initializeDataStructure(null);

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            getLogLikelihood();
        }
    }

    /**
     * Initialized with seeded distributions.
     *
     * @param docTopicPrior Topic distribution for each document
     * @param topicWordPrior Word distribution for each topic
     */
    public void initialize(double[][] docTopicPrior, double[][] topicWordPrior) {
        if (verbose) {
            logln("Initializing with pre-defined topics ...");
        }

        initializeModelStructure(topicWordPrior);

        initializeDataStructure(docTopicPrior);

        initializeAssignments();

        if (debug) {
            logln("--- Done initializing. \t" + getCurrentState());
            getLogLikelihood();
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        if (topics != null && topics.length != K) {
            throw new RuntimeException("Mismatch"
                    + ". K = " + K
                    + ". # prior topics = " + topics.length);
        }

        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (topics != null) {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, topics[k]);
            } else {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }
    }

    protected void initializeDataStructure(double[][] docTopicPrior) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        if (docTopicPrior != null && docTopicPrior.length != D) {
            throw new RuntimeException("Mismatch"
                    + ". D = " + D
                    + ". # prior documents = " + docTopicPrior.length);
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            if (docTopicPrior != null) {
                docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, docTopicPrior[d]);
            } else {
                docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
            }
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }
        sampleZs(!REMOVE, ADD, !REMOVE, ADD);
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report) {
                IOUtils.createFolder(reportFolderPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating report folder."
                    + " " + reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            isReporting = isReporting();
            numTokensChanged = 0;
            if (isReporting) {
                // store llh after every iteration
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            long topicTime = sampleZs(REMOVE, ADD, REMOVE, ADD);

            // parameter optimization by slice sampling
            if (paramOptimized && iter % LAG == 0 && iter >= BURN_IN) {
                sliceSample();
                ArrayList<Double> sparams = new ArrayList<Double>();
                for (double param : this.hyperparams) {
                    sparams.add(param);
                }
                this.sampledParams.add(sparams);

                if (verbose) {
                    for (double p : sparams) {
                        System.out.println(p);
                    }
                }
            }

            if (isReporting) {
                logln("--- --- Time. topic: " + topicTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: "
                        + MiscUtils.formatDouble((double) numTokensChanged / numTokens)
                        + "\n\n");
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
                outputTopicTopWords(new File(reportFolderPath,
                        "topwords-" + iter + ".txt"), 20);
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            outputTopicTopWords(new File(reportFolderPath,
                    "topwords-" + iter + ".txt"), 20);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Sample the topic assignments for all tokens
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < z[dd].length; nn++) {
                sampleZ(dd, nn, removeFromModel, addToModel,
                        removeFromData, addToData);
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample the topic assignment for each token
     *
     * @param dd The document index
     * @param nn The token index
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     */
    protected void sampleZ(int dd, int nn,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        if (removeFromData) {
            docTopics[dd].decrement(z[dd][nn]);
        }
        if (removeFromModel) {
            topicWords[z[dd][nn]].decrement(words[dd][nn]);
        }

        double[] probs = new double[K];
        for (int k = 0; k < K; k++) {
            probs[k] = (docTopics[dd].getCount(k)
                    + hyperparams.get(ALPHA) * K * docTopics[dd].getCenterElement(k))
                    * topicWords[k].getProbability(words[dd][nn]);
        }
        int sampledZ = SamplerUtils.scaleSample(probs);
        if (sampledZ != z[dd][nn]) {
            numTokensChanged++;
        }
        z[dd][nn] = sampledZ;

        if (addToData) {
            docTopics[dd].increment(z[dd][nn]);
        }
        if (addToModel) {
            topicWords[z[dd][nn]].increment(words[dd][nn]);
        }
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += docTopics[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += topicWords[k].getLogLikelihood();
        }
        return docTopicLlh + topicWordLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += docTopics[d].getLogLikelihood(newParams.get(ALPHA) * K,
                    docTopics[d].getCenterVector());
        }
        for (int k = 0; k < K; k++) {
            llh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V,
                    topicWords[k].getCenterVector());
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.docTopics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topicWords[k].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg);
        for (int d = 0; d < D; d++) {
            docTopics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topicWords[k].validate(msg);
        }
    }

    /**
     * Output topics with top words.
     *
     * @param file
     * @param numTopWords
     */
    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing topics to file " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int k = 0; k < K; k++) {
                String[] topWords = getTopWords(topicWords[k].getDistribution(), numTopWords);
                writer.write("[Topic " + k + ": " + topicWords[k].getCountSum() + "]");
                for (String tw : topWords) {
                    writer.write(" " + tw);
                }
                writer.write("\n");
                String topObs = MiscUtils.getTopObservations(wordVocab,
                        topicWords[k].getSparseCounts(), numTopWords);
                writer.write(topObs);
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    public void outputPosterior(File my_file) {
        double[][] postTops = new double[K][];
        for (int i = 0; i < K; i++) {
            postTops[i] = topicWords[i].getDistribution();
        }
        IOUtils.output2DArray(my_file, postTops);
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");
                for (int n = 0; n < z[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
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
            throw new RuntimeException("Exception while loading from " + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeModelStructure(null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                topicWords[k] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure(null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                docTopics[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < z[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + LDA.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        return "java -cp \"dist/segan.jar:lib/*\" sampler.unsupervised.LDA "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-unsupervised/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-unsupervised/amazon-data.dat "
                + "--info-file demo/amazon-data/format-unsupervised/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-unsupervised "
                + "--burnIn 100 "
                + "--maxIter 250 "
                + "--sampleLag 30 "
                + "--report 5 "
                + "--report 1 "
                + "--K 25 "
                + "--alpha 0.1 "
                + "--beta 0.1 "
                + "-v -d";
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

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("alpha", "Alpha");
        addOption("beta", "Beta");
        addOption("K", "Number of topics");
        addOption("num-top-words", "Number of top words per topic");

        // configurations
        addOption("init", "Initialization");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "Help");
        options.addOption("example", false, "Example command");
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
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);

        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");
        String docInfoFile = cmd.getOptionValue("info-file");

        // data output
        String outputFolder = cmd.getOptionValue("output-folder");

        TextDataset data = new TextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile),
                new File(docInfoFile),
                null);
        int V = data.getWordVocab().size();

        LDA sampler = new LDA();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(outputFolder, V, K,
                alpha, beta,
                initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        ArrayList<Integer> selectedDocIndices = null;
        if (cmd.hasOption("selected-docs-file")) {
            String selectedDocFile = cmd.getOptionValue("selected-docs-file");
            selectedDocIndices = new ArrayList<>();
            BufferedReader reader = IOUtils.getBufferedReader(selectedDocFile);
            String line;
            while ((line = reader.readLine()) != null) {
                int docIdx = Integer.parseInt(line);
                if (docIdx >= data.getDocIds().length) {
                    throw new RuntimeException("Out of bound. Doc index " + docIdx);
                }
                selectedDocIndices.add(Integer.parseInt(line));
            }
            reader.close();
        }

        double[][] priorTopics = null;
        if (cmd.hasOption("prior-topic-file")) {
            String priorTopicFile = cmd.getOptionValue("prior-topic-file");
            priorTopics = IOUtils.input2DArray(new File(priorTopicFile));
        }

        sampler.train(data.getWords(), selectedDocIndices);
        sampler.initialize(null, priorTopics);
        sampler.iterate();
        sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
        sampler.outputPosterior(new File(samplerFolder, "posterior.csv"));
        IOUtils.output2DArray(new File(samplerFolder, "phis.txt"), sampler.getPhis());
        IOUtils.output2DArray(new File(samplerFolder, "thetas.txt"), sampler.getThetas());
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
}
