package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import main.GlobalConstants;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import sampler.unsupervised.RecursiveLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public abstract class AbstractSampler implements Serializable {

    private static final long serialVersionUID = GlobalConstants.SerialVersionUID;
    public static final int MAX_NUM_PARALLEL_THREADS = 5;
    public static final String IterPredictionFolder = "iter-predictions";
    public static final String IterAssignmentFolder = "iter-assignments";
    public static final String TopWordFile = AbstractExperiment.TopWordFile;
    public static final String TopicCoherenceFile = AbstractExperiment.TopicCoherenceFile;
//    public static final String IterPerplexityFolder = "iter-perplexities";
//    public static final String PerplexityFile = "perplexity.txt";
    public static final String IterPerplexityFolder = "iter-perps";
    public static final String PerplexityFile = "perp.txt";
    public static final String AveragingPerplexityFile = "avg-perplexity.txt";
    public static final String ModelFile = "model.zip";
    public static final String ReportFolder = "report/";
    public static final String AssignmentFileExt = ".assignment";
    public static final String ModelFileExt = ".model";
    public static final String LikelihoodFile = "likelihoods.txt";
    public static final String HyperparameterFile = "hyperparameters.txt";
    public static final int INIT = -1;
    public static final boolean REMOVE = true;
    public static final boolean ADD = true;
    public static final boolean OBSERVED = true;
    public static final boolean EXTEND = true;
    public static final int UNOBSERVED = AbstractExperiment.UNOBSERVED;
    public static final int PROPOSAL_INDEX = 0;
    public static final int ACTUAL_INDEX = 1;

    public int numTokens;
    public int numTokensChanged;

    public static enum InitialState {

        RANDOM, SEEDED, FORWARD, PRESET, PRIOR
    }

    public static enum SamplingType {

        GIBBS, MH
    }
    protected static final long RAND_SEED = 1123581321;
    protected static final double MAX_LOG = Math.log(Double.MAX_VALUE);
    protected static final NumberFormat formatter = new DecimalFormat("###.###");
    protected static Random rand = new Random(RAND_SEED);
    protected static long startTime;
    // sampling configurations
    protected int BURN_IN = 5;          // burn-in
    protected int MAX_ITER = 100;       // maximum number of iterations
    protected int LAG = 1;              // for outputing log-likelihood
    protected int REP_INTERVAL = 10;    // report interval
    // test configuration
    protected int testBurnIn = 50;
    protected int testMaxIter = 100;
    protected int testSampleLag = 5;
    protected int testRepInterval = 5;
    protected String folder;
    protected String name;
    protected String basename;
    protected ArrayList<Double> hyperparams; // should have used a HashMap instead of ArrayList
    protected boolean paramOptimized = false;
    protected String prefix = "";// to store description of predefined configurations (e.g., initialization)
    protected InitialState initState;
    protected double stepSize = 0.1;
    protected int numSliceSamples = 10;
    protected ArrayList<Double> logLikelihoods;
    protected ArrayList<ArrayList<Double>> sampledParams;
    protected ArrayList<String> wordVocab;
    protected int iter;
    protected boolean debug = false;
    protected boolean verbose = true;
    protected boolean log = true;
    protected boolean report = false;
    protected boolean isReporting;
    protected BufferedWriter logger;
    protected static CommandLineParser parser;
    protected static Options options;
    protected static CommandLine cmd;

    protected static void addOption(String optName, String optDesc) {
        options.addOption(OptionBuilder.withLongOpt(optName)
                .withDescription(optDesc)
                .hasArg()
                .withArgName(optName)
                .create());
    }

    public static void addSamplingOptions() {
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");
    }

    public static void addRunningOptions() {
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "help");
        options.addOption("example", false, "example");

        options.addOption("train", false, "train");
        options.addOption("test", false, "test");
        options.addOption("parallel", false, "parallel");
    }

    public static void addDataOptions() {
        addOption("dataset", "Dataset");
        addOption("word-voc-file", "Word vocabulary file");
        addOption("word-file", "Document word file");
        addOption("info-file", "Document info file");
        addOption("selected-doc-indices-file", "File containing selected document indices");
        addOption("selected-doc-ids-file", "File containing selected document IDs");
    }

    public void setFolder(String folder) {
        this.folder = folder;
    }

    public String getFolder() {
        return this.folder;
    }

    public String getName() {
        return this.name;
    }

    public void setBasename(String basename) {
        this.basename = basename;
    }

    public String getBasename() {
        return this.basename;
    }

    public void setTestConfigurations(int tBurnIn, int tMaxIter, int tSampleLag) {
        setTestConfigurations(tBurnIn, tMaxIter, tSampleLag, tSampleLag);
    }

    public void setTestConfigurations(int tBurnIn, int tMaxIter, int tSampleLag,
            int tRepInt) {
        this.testBurnIn = tBurnIn;
        this.testMaxIter = tMaxIter;
        this.testSampleLag = tSampleLag;
        this.testRepInterval = tRepInt;
    }

    public void setSamplerConfiguration(int burn_in, int max_iter, int lag, int repInt) {
        BURN_IN = burn_in;
        MAX_ITER = max_iter;
        LAG = lag;
        REP_INTERVAL = repInt;
    }

    public boolean isReporting() {
        return verbose && iter % REP_INTERVAL == 0;
    }

    public int getBurnIn() {
        return this.BURN_IN;
    }

    public int getMaxIters() {
        return this.MAX_ITER;
    }

    public int getSampleLag() {
        return this.LAG;
    }

    public int getReportInterval() {
        return this.REP_INTERVAL;
    }

    public void setReportInterval(int repInt) {
        REP_INTERVAL = repInt;
    }

    protected String getIteratedStateFile() {
        return "iter-" + iter + ".zip";
    }

    protected String getIteratedTopicFile() {
        return "topwords-" + iter + ".txt";
    }

    public abstract void initialize();

    public abstract void iterate();

    public abstract double getLogLikelihood();

    public abstract double getLogLikelihood(ArrayList<Double> testHyperparameters);

    public abstract void updateHyperparameters(ArrayList<Double> newParams);

    public abstract void validate(String msg);

    public abstract void outputState(String filepath);

    public abstract void inputState(String filepath);

    public LDA runLDA(int[][] words, int K, int V, double[][] priorTopics) {
        return runLDA(words, K, V, null, priorTopics, 0.1, 0.1, 250, 500, 25);
    }

    /**
     * Run LDA.
     *
     * @param words Document
     * @param K Number of topics
     * @param V Vocabulary size
     * @param topicWordPriors
     */
    public LDA runLDA(int[][] words, int K, int V,
            double[][] docTopicPriors, double[][] topicWordPriors,
            double ldaAlpha, double ldaBeta,
            int ldaBurnIn, int ldaMaxIter, int ldaLag) {
        LDA lda = new LDA();
        lda.setDebug(false);
        lda.setVerbose(verbose);
        lda.setLog(false);

        lda.configure(folder, V, K, ldaAlpha, ldaBeta, InitialState.RANDOM, false,
                ldaBurnIn, ldaMaxIter, ldaLag, ldaLag);

        try {
            File ldaFile = new File(lda.getSamplerFolderPath(), basename + ".zip");
            lda.train(words, null);
            if (ldaFile.exists()) {
                if (verbose) {
                    logln("--- --- LDA file exists. Loading from " + ldaFile);
                }
                lda.inputState(ldaFile);
            } else {
                if (verbose) {
                    logln("--- --- LDA not found. Running LDA ...");
                }
                lda.initialize(docTopicPriors, topicWordPriors);
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);
        return lda;
    }

    /**
     * Run LDA recursively.
     *
     * @param words
     * @param Ks
     * @param V
     * @param rldaAlphas
     * @param rldaBetas
     * @param priorTopics
     */
    public RecursiveLDA runRecursiveLDA(int[][] words, int[] Ks,
            double[] rldaAlphas, double[] rldaBetas,
            int V,
            double[][] priorTopics) {
        int rlda_burnin = 10;
        int rlda_maxiter = 100;
        int rlda_samplelag = 10;
        RecursiveLDA rlda = new RecursiveLDA();
        rlda.setDebug(false);
        rlda.setVerbose(verbose);
        rlda.setLog(false);

        rlda.configure(folder, V, Ks, rldaAlphas, rldaBetas,
                InitialState.RANDOM, false,
                rlda_burnin, rlda_maxiter, rlda_samplelag, rlda_samplelag);
        try {
            File ldaZFile = new File(rlda.getSamplerFolderPath(), basename + ".zip");
            rlda.train(words, null); // words are already filtered using docIndices
            if (ldaZFile.exists()) {
                rlda.inputState(ldaZFile);
            } else {
                rlda.setPriorTopics(priorTopics);
                rlda.initialize();
                rlda.iterate();
                IOUtils.createFolder(rlda.getSamplerFolderPath());
                rlda.outputState(ldaZFile);
                rlda.setWordVocab(wordVocab);
                rlda.outputTopicTopWords(new File(rlda.getSamplerFolderPath(), TopWordFile), 20);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running Recursive LDA for initialization");
        }
        setLog(log);
        return rlda;
    }

    public void metaIterate() {
        if (verbose) {
            logln("Iterating ...");
        }
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
            if (isReporting) {
                logln(getCurrentState());
            }

            iterate();

            // parameter optimization
            if (iter % LAG == 0 && iter > BURN_IN) {
                if (paramOptimized) { // slice sampling
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
            }

            if (debug && isReporting) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
                outputTopicTopWords(new File(reportFolderPath,
                        "iter-" + iter + "-" + TopWordFile), 15);
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            outputTopicTopWords(new File(reportFolderPath,
                    "iter-" + iter + "-" + TopWordFile), 15);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Read a list of indices of selected documents.
     *
     * @param docIds Original document IDs
     */
    public ArrayList<Integer> getSelectedDocIndices(String[] docIds) {
        ArrayList<Integer> selectedDocIndices = null;
        if (!cmd.hasOption("selected-doc-indices-file")
                && !cmd.hasOption("selected-doc-ids-file")) {
            return selectedDocIndices;
        }

        if (cmd.hasOption("selected-doc-indices-file")) {
            return loadFromSelectedDocIndices(docIds, cmd.getOptionValue("selected-doc-indices-file"));
        } else {
            assert cmd.hasOption("selected-doc-ids-file");
            return loadFromSelectedDocIds(docIds, cmd.getOptionValue("selected-doc-ids-file"));
        }
    }

    /**
     * Load the list of selected documents where each line is the document
     * index.
     *
     * @param docIds
     * @param filename
     */
    private ArrayList<Integer> loadFromSelectedDocIndices(String[] docIds, String filename) {
        if (verbose) {
            logln("Loading selected document indices from " + filename);
        }
        ArrayList<Integer> selectedDocIndices = new ArrayList<>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(filename);
            String line;
            while ((line = reader.readLine()) != null) {
                int docIdx = Integer.parseInt(line);
                if (docIdx >= docIds.length) {
                    throw new RuntimeException("Out of bound. Doc index " + docIdx);
                }
                selectedDocIndices.add(docIdx);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + filename);
        }
        return selectedDocIndices;
    }

    /**
     * Load the list of selected documents where each line is the document id.
     *
     * @param docIds
     * @param filename
     */
    private ArrayList<Integer> loadFromSelectedDocIds(String[] docIds, String filename) {
        if (verbose) {
            logln("Loading selected document indices from " + filename);
        }
        ArrayList<Integer> selectedDocIndices = new ArrayList<>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(filename);
            String line;
            while ((line = reader.readLine()) != null) {
                int docIdx = findIndex(docIds, line);
                if (docIdx < 0) {
                    throw new RuntimeException("Doc " + line + " not found");
                }
                selectedDocIndices.add(docIdx);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + filename);
        }
        return selectedDocIndices;
    }

    private static int findIndex(String[] set, String q) {
        for (int i = 0; i < set.length; i++) {
            if (set[i].equals(q)) {
                return i;
            }
        }
        System.out.println(q);
        return -1;
    }

    /**
     * Output learned topics.
     *
     * @param file Output file
     * @param numTopWords Number of top words
     */
    public void outputTopicTopWords(File file, int numTopWords) {
        // TOD: make this abstract
    }

    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        return str.toString();
    }

    public File getFinalStateFile() {
        return new File(getReportFolderPath(), "iter-" + MAX_ITER + ".zip");
    }

    public void inputFinalState() {
        this.inputState(new File(getReportFolderPath(), "iter-" + MAX_ITER + ".zip"));
    }

    public void outputState(File file) {
        this.outputState(file.getAbsolutePath());
    }

    public void inputState(File file) {
        this.inputState(file.getAbsolutePath());
    }

    protected void outputZipFile(
            String filepath,
            String modelStr,
            String assignStr) throws Exception {
        String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
        ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

        if (modelStr != null) {
            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();
        }

        if (assignStr != null) {
            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            byte[] data = assignStr.getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();
        }

        writer.close();
    }

    protected void outputZipFile(String filepath,
            ArrayList<String> contentStrs,
            ArrayList<String> entryFiles) throws Exception {
        if (contentStrs.size() != entryFiles.size()) {
            throw new RuntimeException("Mismatch");
        }

        ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);
        for (int ii = 0; ii < contentStrs.size(); ii++) {
            ZipEntry modelEntry = new ZipEntry(entryFiles.get(ii));
            writer.putNextEntry(modelEntry);
            byte[] data = contentStrs.get(ii).getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();
        }
        writer.close();
    }

    public ArrayList<String> getWordVocab() {
        return this.wordVocab;
    }

    public void setWordVocab(ArrayList<String> vocab) {
        this.wordVocab = vocab;
    }

    public String[] getTopWords(double[] distribution, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("Word vocab empty");
        }
        ArrayList<RankingItem<String>> topicSortedVocab = IOUtils.getSortedVocab(distribution, this.wordVocab);
        String[] topWords = new String[numWords];
        for (int i = 0; i < numWords; i++) {
            topWords[i] = topicSortedVocab.get(i).getObject();
        }
        return topWords;
    }

    public String getSamplerName() {
        return this.name;
    }

    public String getSamplerFolder() {
        return this.getSamplerName() + "/";
    }

    public String getSamplerFolderPath() {
        return new File(folder, name).getAbsolutePath();
    }

    public String getReportFolderPath() {
        return new File(getSamplerFolderPath(), ReportFolder).getAbsolutePath();
    }

    public String getFormatNumberString(double value) {
        if (value > 0.001) {
            return formatter.format(value);
        } else {
            return Double.toString(value);
        }
    }

    public void sample() {
        if (log) {
            openLogger();
        }

        logln(getClass().toString() + "\t" + getSamplerName());
        startTime = System.currentTimeMillis();

        initialize();

        iterate();

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                IOUtils.createFolder(getSamplerFolderPath());
                this.outputSampledHyperparameters(
                        new File(getSamplerFolderPath(), HyperparameterFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling");
        }
    }

    public void concatPrefix(String p) {
        if (this.prefix == null) {
            this.prefix = p;
        } else {
            this.prefix += "_" + p;
        }
    }

    public void setPrefix(String p) {
        this.prefix = p;
    }

    public String getPrefix() {
        if (this.prefix == null) {
            return "";
        } else {
            return this.prefix;
        }
    }

    public void openLogger() {
        try {
            this.logger = IOUtils.getBufferedWriter(new File(getSamplerFolderPath(), "log.txt"));
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void closeLogger() {
        try {
            this.logger.close();
            this.logger = null;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setReport(boolean report) {
        this.report = report;
    }

    public void setLog(boolean l) {
        this.log = l;
    }

    public void setDebug(boolean d) {
        this.debug = d;
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    protected void logln(String msg) {
        System.out.println("[LOG] " + msg);
        try {
            if (logger != null) {
                this.logger.write(msg + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public boolean isLogging() {
        return this.logger != null;
    }

    public boolean areParamsOptimized() {
        return this.paramOptimized;
    }

    public void setParamsOptimized(boolean po) {
        this.paramOptimized = po;
    }

    public void outputLogLikelihoods(File file) throws Exception {
        IOUtils.outputLogLikelihoods(logLikelihoods, file.getAbsolutePath());
    }

    public void outputSampledHyperparameters(File file) throws Exception {
        this.outputSampledHyperparameters(file.getAbsolutePath());
    }

    public void outputSampledHyperparameters(String filepath) {
        System.out.println("Outputing sampled hyperparameters to file " + filepath);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (int i = 0; i < this.sampledParams.size(); i++) {
                writer.write(Integer.toString(i));
                for (double p : this.sampledParams.get(i)) {
                    writer.write("\t" + p);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing sampled hyperparameters");
        }
    }

    protected ArrayList<Double> cloneHyperparameters() {
        ArrayList<Double> newParams = new ArrayList<Double>();
        for (Double hyperparam : this.hyperparams) {
            newParams.add(hyperparam);
        }
        return newParams;
    }

    protected void updateHyperparameters() {
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

    /**
     * Slice sampling for hyper-parameter optimization.
     */
    protected void sliceSample() {
        if (hyperparams == null) { // no hyperparameter to optimize
            return;
        }
        int dim = hyperparams.size();
        double[] lefts = new double[dim];
        double[] rights = new double[dim];
        ArrayList<Double> tempParams = hyperparams;

        if (debug) {
            logln("ori params: " + MiscUtils.listToString(hyperparams));
        }

        for (int s = 0; s < numSliceSamples; s++) {
            if (debug) {
                logln("");
                logln("Slice sampling. Iter = " + s);
            }

            double cur_llh = getLogLikelihood(tempParams);
            double log_u_prime = Math.log(rand.nextDouble()) + cur_llh;
            for (int i = 0; i < dim; i++) {
                double r = rand.nextDouble();
                lefts[i] = tempParams.get(i) - r * stepSize;
                rights[i] = lefts[i] + stepSize;
                if (lefts[i] < 0) {
                    lefts[i] = 0;
                }
            }

            if (debug) {
                logln("cur_llh = " + cur_llh + ", log_u' = " + log_u_prime);
                logln("lefts: " + MiscUtils.arrayToString(lefts));
                logln("rights: " + MiscUtils.arrayToString(rights));
            }

            ArrayList<Double> newParams = null;
            while (true) {
                newParams = new ArrayList<Double>();
                for (int i = 0; i < dim; i++) {
                    newParams.add(rand.nextDouble() * (rights[i] - lefts[i]) + lefts[i]);
                }
                double new_llh = getLogLikelihood(newParams);

                if (debug) {
                    logln("new params: " + MiscUtils.listToString(newParams) + "; new llh = " + new_llh);
                }

                if (new_llh > log_u_prime) {
                    break;
                } else {
                    for (int i = 0; i < dim; i++) {
                        if (newParams.get(i) < tempParams.get(i)) {
                            lefts[i] = newParams.get(i);
                        } else {
                            rights[i] = newParams.get(i);
                        }
                    }
                }
            }

            tempParams = newParams;
        }

        updateHyperparameters(tempParams);

        if (debug) {
            logln("sampled params: " + MiscUtils.listToString(hyperparams)
                    + "; final llh = " + getLogLikelihood(hyperparams));
        }
    }

    public static boolean isTraining() {
        return cmd.hasOption("train");
    }

    public static boolean isDeveloping() {
        return cmd.hasOption("dev");
    }

    public static boolean isTesting() {
        return cmd.hasOption("test");
    }

    /**
     * Run multiple threads in parallel.
     *
     * @param threads
     * @throws java.lang.Exception
     */
    public static void runThreads(ArrayList<Thread> threads) throws Exception {
        int c = 0;
        for (int ii = 0; ii < threads.size() / MAX_NUM_PARALLEL_THREADS; ii++) {
            for (int jj = 0; jj < MAX_NUM_PARALLEL_THREADS; jj++) {
                threads.get(ii * MAX_NUM_PARALLEL_THREADS + jj).start();
            }
            for (int jj = 0; jj < MAX_NUM_PARALLEL_THREADS; jj++) {
                threads.get(ii * MAX_NUM_PARALLEL_THREADS + jj).join();
                c++;
            }
        }
        for (int jj = c; jj < threads.size(); jj++) {
            threads.get(jj).start();
        }
        for (int jj = c; jj < threads.size(); jj++) {
            threads.get(jj).join();
        }
    }

    public static PathAssumption getPathAssumption(String path) {
        PathAssumption pathAssumption;
        switch (path) {
            case "max":
                pathAssumption = PathAssumption.MAXIMAL;
                break;
            case "min":
                pathAssumption = PathAssumption.MINIMAL;
                break;
            case "uniproc":
                pathAssumption = PathAssumption.UNIPROC;
                break;
            case "antoniak":
                pathAssumption = PathAssumption.ANTONIAK;
                break;
            case "none": // no cascade
                pathAssumption = PathAssumption.NONE;
                break;
            default:
                throw new RuntimeException("Path assumption " + path + " not supported");
        }
        return pathAssumption;
    }

    public static InitialState getInitialState(String init) {
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
        return initState;
    }
}
