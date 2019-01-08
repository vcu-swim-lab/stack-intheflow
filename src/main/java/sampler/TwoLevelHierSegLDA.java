package sampler;

import core.AbstractSampler;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class TwoLevelHierSegLDA extends AbstractSampler {

    public static final int ALPHA_1 = 0;
    public static final int ALPHA_2 = 1;
    public static final int BETA_1 = 2;
    public static final int BETA_2 = 3;
    protected int K; // number of first-level topics
    protected int L; // number of second-level topics per first-level topics
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z;
    protected int[][] y;
    protected DirMult[] doc_first_topics;
    protected DirMult[] first_topic_words;
    protected DirMult[][] doc_second_topics;
    protected DirMult[][] second_topic_words;
    private int numTokens;
    private int numTokensChanged;

    public DirMult[] getFirstLevelTopics() {
        return this.first_topic_words;
    }

    public DirMult[][] getSecondLevelTopics() {
        return this.second_topic_words;
    }

    public int[][] getFirstLevelAssignments() {
        return this.z;
    }

    public int[][] getSecondLevelAssignments() {
        return this.y;
    }

    public void configure(String folder, int[][] words,
            int V, int K, int L,
            double alpha_1, double alpha_2,
            double beta_1, double beta_2,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.words = words;

        this.K = K;
        this.V = V;
        this.L = L;
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_1);
        this.hyperparams.add(alpha_2);
        this.hyperparams.add(beta_1);
        this.hyperparams.add(beta_2);

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

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- # documents:\t" + D);
            logln("--- # first-level topics:\t" + K);
            logln("--- # second-level topics:\t" + L);
            logln("--- # tokens:\t" + numTokens);
            logln("--- vocab size:\t" + V);
            logln("--- alpha_1:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_1)));
            logln("--- alpha_2:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_2)));
            logln("--- beta_1:\t" + MiscUtils.formatDouble(hyperparams.get(BETA_1)));
            logln("--- beta_1:\t" + MiscUtils.formatDouble(hyperparams.get(BETA_2)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_hier-seg-LDA"
                + "_K-" + K
                + "_L-" + L
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a1-" + formatter.format(this.hyperparams.get(ALPHA_1))
                + "_a2-" + formatter.format(this.hyperparams.get(ALPHA_2))
                + "_b1-" + formatter.format(this.hyperparams.get(BETA_1))
                + "_b2-" + formatter.format(this.hyperparams.get(BETA_2))
                + "_opt-" + this.paramOptimized;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure();

        initializeDataStructure();

        if (debug) {
            validate("Initialized");
        }
    }

    public void initialize(double[][] topics) {
        if (verbose) {
            logln("Initializing with priors ...");
        }

        initializeModelStructure(topics);

        initializeDataStructure();

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (verbose) {
            logln("--- Initializing model structure with prior ...");
        }
        first_topic_words = new DirMult[K];
        for (int k = 0; k < K; k++) {
            first_topic_words[k] = new DirMult(V, hyperparams.get(BETA_1) * V, topics[k]);
        }
        second_topic_words = new DirMult[K][L];
        for (int k = 0; k < K; k++) {
            for (int l = 0; l < L; l++) {
                second_topic_words[k][l] = new DirMult(V, hyperparams.get(BETA_2) * V, 1.0 / V);
            }
        }
    }

    protected void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        first_topic_words = new DirMult[K];
        for (int k = 0; k < K; k++) {
            first_topic_words[k] = new DirMult(V, hyperparams.get(BETA_1) * V, 1.0 / V);
        }
        second_topic_words = new DirMult[K][L];
        for (int k = 0; k < K; k++) {
            for (int l = 0; l < L; l++) {
                second_topic_words[k][l] = new DirMult(V, hyperparams.get(BETA_2) * V, 1.0 / V);
            }
        }
    }

    protected void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        doc_first_topics = new DirMult[D];
        doc_second_topics = new DirMult[D][K];
        for (int d = 0; d < D; d++) {
            doc_first_topics[d] = new DirMult(K, hyperparams.get(ALPHA_1) * K, 1.0 / K);
            for (int k = 0; k < K; k++) {
                doc_second_topics[d][k] = new DirMult(L, hyperparams.get(ALPHA_2) * L, 1.0 / L);
            }
        }

        z = new int[D][];
        y = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
            y[d] = new int[words[d].length];
        }
    }

    @Override
    public void iterate() {
        LDA lda = new LDA();
        lda.setVerbose(verbose);
        lda.setDebug(debug);
        lda.setLog(false);
        lda.setReport(false);

        lda.configure(null, words, V, K,
                hyperparams.get(ALPHA_1),
                hyperparams.get(BETA_1),
                initState,
                paramOptimized,
                BURN_IN, MAX_ITER, LAG, REP_INTERVAL);
        double[][] priorTopics = new double[K][];
        for (int k = 0; k < K; k++) {
            priorTopics[k] = first_topic_words[k].getCenterVector();
        }

        lda.initialize(null, priorTopics);
        lda.iterate();

        z = lda.getZ();
        doc_first_topics = lda.doc_topics;
        first_topic_words = lda.topic_words;

        // init
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                int firstTopic = z[d][n];
                y[d][n] = rand.nextInt(L);
                doc_second_topics[d][firstTopic].increment(y[d][n]);
                second_topic_words[firstTopic][y[d][n]].increment(words[d][n]);
            }
        }
        this.logLikelihoods = new ArrayList<Double>();

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        if (report) {
            IOUtils.createFolder(reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        // iterate
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    int firstTopic = z[d][n];

                    // decrement
                    doc_second_topics[d][firstTopic].decrement(y[d][n]);
                    second_topic_words[firstTopic][y[d][n]].decrement(words[d][n]);

                    // sample
                    double[] logprobs = new double[L];
                    for (int l = 0; l < L; l++) {
                        double lp = doc_second_topics[d][firstTopic].getLogLikelihood(l)
                                + second_topic_words[firstTopic][l].getLogLikelihood(words[d][n]);
                        logprobs[l] = lp;
                    }
                    int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);

                    if (y[d][n] != sampledIdx) {
                        numTokensChanged++;
                    }

                    y[d][n] = sampledIdx;

                    // increment
                    doc_second_topics[d][firstTopic].increment(y[d][n]);
                    second_topic_words[firstTopic][y[d][n]].increment(words[d][n]);
                }
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % REP_INTERVAL == 0) {
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio);
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    if (verbose) {
                        logln("--- Slice sampling ...");
                    }

                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        logln(MiscUtils.listToString(sparams));
                    }
                }
            }

            if (debug) {
                validate("iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        if (report && paramOptimized) {
            this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                    "hyperparameters.txt").getAbsolutePath());
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += doc_first_topics[d].getLogLikelihood();
            for (int k = 0; k < K; k++) {
                docTopicLlh += doc_second_topics[d][k].getLogLikelihood();
            }
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += first_topic_words[k].getLogLikelihood();
            for (int l = 0; l < L; l++) {
                topicWordLlh += second_topic_words[k][l].getLogLikelihood();
            }
        }
        return docTopicLlh + topicWordLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            doc_first_topics[d].validate(msg);
            for (int k = 0; k < K; k++) {
                doc_second_topics[d][k].validate(msg);
            }
        }

        for (int k = 0; k < K; k++) {
            first_topic_words[k].validate(msg);
            for (int l = 0; l < L; l++) {
                second_topic_words[k][l].validate(msg);
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
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(first_topic_words[k])).append("\n");
                for (int l = 0; l < L; l++) {
                    modelStr.append(l).append("\n");
                    modelStr.append(DirMult.output(second_topic_words[k][l])).append("\n");
                }
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_first_topics[d])).append("\n");
                for (int k = 0; k < K; k++) {
                    assignStr.append(k).append("\n");
                    assignStr.append(DirMult.output(doc_second_topics[d][k])).append("\n");
                }

                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");

                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(y[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exeption while outputing model to " + filepath);
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

    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int firstTopic = Integer.parseInt(reader.readLine());
                if (firstTopic != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                first_topic_words[k] = DirMult.input(reader.readLine());

                for (int l = 0; l < L; l++) {
                    int secondTopic = Integer.parseInt(reader.readLine());
                    if (secondTopic != l) {
                        throw new RuntimeException("Indices mismatch when loading model");
                    }
                    second_topic_words[k][l] = DirMult.input(reader.readLine());
                }
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
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_first_topics[d] = DirMult.input(reader.readLine());

                for (int k = 0; k < K; k++) {
                    int firstTopic = Integer.parseInt(reader.readLine());
                    if (firstTopic != k) {
                        throw new RuntimeException("Indices mismatch when loading assignments");
                    }
                    doc_second_topics[d][k] = DirMult.input(reader.readLine());
                }

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }

                sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
                    y[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing topics to file " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int k = 0; k < K; k++) {
                double[] firstTopic = first_topic_words[k].getDistribution();
                String[] firstTopWords = getTopWords(firstTopic, numTopWords);
                writer.write("[" + k
                        + ", " + first_topic_words[k].getCountSum()
                        + "]");
                for (String topWord : firstTopWords) {
                    writer.write(" " + topWord);
                }
                writer.write("\n\n");

                for (int l = 0; l < L; l++) {
                    double[] secondTopic = second_topic_words[k][l].getDistribution();
                    String[] secondTopWords = getTopWords(secondTopic, numTopWords);
                    writer.write("\t[" + l
                            + ", " + second_topic_words[k][l].getCountSum()
                            + "]");
                    for (String topWord : secondTopWords) {
                        writer.write(" " + topWord);
                    }
                    writer.write("\n\n");
                }
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + TwoLevelHierSegLDA.class.getName() + " -help";
    }

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static void run(String[] args) throws Exception {
        // create the command line parser
        parser = new BasicParser();

        // create the Options
        options = new Options();

        // directories
        addOption("dataset", "Dataset");
        addOption("output", "Output folder");
        addOption("data-folder", "Processed data folder");
        addOption("format-folder", "Folder holding formatted data");
        addOption("format-file", "Format file name");

        // sampling configurations
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");

        // model parameters
        addOption("K", "Number of first-level topics");
        addOption("L", "Number of second-level topics");
        addOption("numTopwords", "Number of top words per topic");

        // model hyperparameters
        addOption("alpha-1", "Hyperparameter of the symmetric Dirichlet prior "
                + "for first-level topic distributions");
        addOption("alpha-2", "Hyperparameter of the symmetric Dirichlet prior "
                + "for second-level topic distributions");
        addOption("beta-1", "Hyperparameter of the symmetric Dirichlet prior "
                + "for first-level word distributions");
        addOption("beta-2", "Hyperparameter of the symmetric Dirichlet prior "
                + "for second-level word distributions");

        options.addOption("paramOpt", false, "Whether hyperparameter "
                + "optimization using slice sampling is performed");
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "whether standardize (z-score normalization)");
        options.addOption("help", false, "Help");

        cmd = parser.parse(options, args);
        if (cmd.hasOption("help")) {
            CLIUtils.printHelp(getHelpString(), options);
            return;
        }

        runModel();
    }

    public static void runModel() throws Exception {
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "112");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo/govtrack");
        String resultFolder = CLIUtils.getStringArgument(cmd, "output", "demo/govtrack/112/model");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-teaparty");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", "teaparty");

        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 50);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
        int L = CLIUtils.getIntegerArgument(cmd, "L", 5);

        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");
        InitialState initState = InitialState.RANDOM;

        double alpha_1 = CLIUtils.getDoubleArgument(cmd, "alpha-1", 0.1);
        double alpha_2 = CLIUtils.getDoubleArgument(cmd, "alpha-2", 0.01);
        double beta_1 = CLIUtils.getDoubleArgument(cmd, "bete-1", 0.1);
        double beta_2 = CLIUtils.getDoubleArgument(cmd, "bete-2", 0.01);

        verbose = true;
        debug = true;

        ResponseTextDataset data = new ResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        data.prepareTopicCoherence(numTopWords);

        TwoLevelHierSegLDA sampler = new TwoLevelHierSegLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(
                resultFolder,
                data.getWords(), data.getWordVocab().size(), K, L,
                alpha_1, alpha_2,
                beta_1, beta_2,
                initState,
                paramOpt, burnIn, maxIters, sampleLag, repInterval);
        File ldaFolder = new File(resultFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(ldaFolder);
        sampler.initialize();
        sampler.iterate();
        sampler.outputTopicTopWords(new File(ldaFolder, TopWordFile), numTopWords);
    }
}
