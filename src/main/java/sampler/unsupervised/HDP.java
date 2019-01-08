package sampler.unsupervised;

import cc.mallet.types.Dirichlet;
import core.AbstractSampler;
import data.TextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class HDP extends AbstractSampler {

    public static final int NEW_COMPONENT_INDEX = -1;
    public static final int ALPHA_GLOBAL = 0;
    public static final int ALPHA_LOCAL = 1;
    public static final int BETA = 2;
    // inputs
    protected int[][] words; // original documents
    protected ArrayList<Integer> docIndices; // [D]: indices of considered docs
    protected int K; // initial number of topics
    protected int V;
    // derived
    protected int D; // number of documents
    // latent
    private SparseVector globalWeights;
    private SparseCount[] docTopics;
    private Topics topicWords;
    private int[][] z;
    // internal
    private double uniform;

    public HDP() {
        this.basename = "HDP";
    }

    public HDP(String basename) {
        this.basename = basename;
    }

    public void configure(String folder, int V,
            double alpha_global, double alpha_local, double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.V = V;
        this.uniform = 1.0 / V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_global);
        this.hyperparams.add(alpha_local);
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

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- vocab size:\t" + V);
            logln("--- alpha-global:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_GLOBAL)));
            logln("--- alpha-local:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA_LOCAL)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    /**
     * Set the initial number of topics.
     *
     * @param K Initial number of topics
     */
    public void setK(int K) {
        this.K = K;
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(this.basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_ag-").append(formatter.format(hyperparams.get(ALPHA_GLOBAL)))
                .append("_al-").append(formatter.format(hyperparams.get(ALPHA_LOCAL)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    /**
     * Set training data.
     *
     * @param docWords All documents
     * @param docIndices Indices of selected documents. If this is null, all
     * documents are considered.
     */
    public void train(int[][] docWords, ArrayList<Integer> docIndices) {
        this.words = docWords;
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();

        this.numTokens = 0;
        for (int dd : this.docIndices) {
            this.numTokens += this.words[dd].length;
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

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            getLogLikelihood();
        }
    }

    protected void initializeModelStructure() {
        this.topicWords = new Topics();

        this.globalWeights = new SparseVector();
        this.globalWeights.set(NEW_COMPONENT_INDEX, 1.0);
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        docTopics = new SparseCount[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            z[ii] = new int[words[dd].length];
            docTopics[ii] = new SparseCount();
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializePresetAssignments();
                break;
            case RANDOM:
                this.initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        sampleZs(!REMOVE, ADD, !REMOVE, ADD);
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        if (K == 0) {// this is not set
            K = 50;
        }
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(folder, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaFile = new File(lda.getSamplerFolderPath(), basename + ".zip");
            if (ldaFile.exists()) {
                logln("--- Loading LDA from " + ldaFile);
                lda.inputState(ldaFile);
            } else {
                logln("--- LDA file not found " + ldaFile + ". Sampling LDA ...");
                lda.train(words, docIndices);
                lda.initialize();
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
            ldaZ = lda.getZs();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(true);

        // initialize assignments
        if (verbose) {
            logln("--- LDA loaded. Start initializing assingments ...");
        }

        for (int kk = 0; kk < K; kk++) {
            DirMult topicWord = new DirMult(V, hyperparams.get(BETA) * V, uniform);
            topicWords.createNewComponent(kk, new Topic(iter, topicWord));
        }

        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords.getComponent(z[dd][nn]).phi.increment(words[dd][nn]);
            }
        }

        // initialize tau
        double mean = 1.0 / (K + hyperparams.get(ALPHA_GLOBAL));
        for (int kk = 0; kk < K; kk++) {
            this.globalWeights.set(kk, mean);
        }
        this.globalWeights.set(NEW_COMPONENT_INDEX,
                hyperparams.get(ALPHA_GLOBAL) / (K + hyperparams.get(ALPHA_GLOBAL)));

        sampleGlobalWeights();
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
            numTokensChanged = 0;

            sampleZs(REMOVE, ADD, REMOVE, ADD); // sample topic assignments

            sampleGlobalWeights(); // sample global stick breaking weights

            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);

                String str = "Iter " + iter + "\t llh = " + loglikelihood
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n" + getCurrentState();
                if (iter <= BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

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

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            }

            if (debug) {
                validate("iter " + iter);
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

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(new File(getSamplerFolderPath(),
                        HyperparameterFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception iter = " + iter);
        }
    }

    private void sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        double totalBeta = hyperparams.get(BETA) * V;
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            for (int nn = 0; nn < words[dd].length; nn++) {
                int curZ = z[ii][nn];
                if (removeFromData) {
                    this.docTopics[ii].decrement(curZ);
                }

                if (removeFromModel) {
                    this.topicWords.getComponent(curZ).phi.decrement(words[dd][nn]);
                    if (this.topicWords.getComponent(curZ).phi.isEmpty()) {
                        topicWords.removeComponent(curZ);
                        globalWeights.remove(curZ);
                    }
                }

                ArrayList<Integer> indices = new ArrayList<Integer>();
                ArrayList<Double> probs = new ArrayList<Double>();
                for (int k : topicWords.getIndices()) {
                    indices.add(k);
                    double docTopicProb = docTopics[ii].getCount(k)
                            + hyperparams.get(ALPHA_LOCAL) * globalWeights.get(k);
                    double topicWordProb = topicWords.getComponent(k).getPhi(words[dd][nn]);
                    probs.add(docTopicProb * topicWordProb);
                }

                if (addToModel) {
                    indices.add(NEW_COMPONENT_INDEX);
                    double docTopicProb = hyperparams.get(ALPHA_LOCAL)
                            * globalWeights.get(NEW_COMPONENT_INDEX);
                    double topicWordProb = uniform;
                    probs.add(docTopicProb * topicWordProb);
                }

                int sampledIdx = SamplerUtils.scaleSample(probs);
                int newZ = indices.get(sampledIdx);

                if (curZ != newZ) {
                    numTokensChanged++;
                }

                boolean newTopic = false;
                if (newZ == NEW_COMPONENT_INDEX) {
                    newZ = topicWords.getNextIndex();
                    DirMult topicWord = new DirMult(V, totalBeta, uniform);
                    topicWords.createNewComponent(newZ, new Topic(iter, topicWord));
                    globalWeights.set(newZ, 0.0); // temporarily assigned
                    newTopic = true;
                }
                z[ii][nn] = newZ;

                if (addToData) {
                    this.docTopics[ii].increment(newZ);
                }

                if (addToModel) {
                    this.topicWords.getComponent(newZ).phi.increment(words[dd][nn]);
                }

                if (newTopic) { // if new topic is created, update SBP weights
                    sampleGlobalWeights();
                }
            }
        }
    }

    /**
     * Sample global distribution over topics.
     */
    private void sampleGlobalWeights() {
        if (globalWeights.size() != topicWords.getNumComponents() + 1) {
            throw new RuntimeException("Mismatch: " + globalWeights.size()
                    + " vs. " + topicWords.getNumComponents());
        }

        SparseCount counts = new SparseCount();
        for (int k : topicWords.getIndices()) {
            for (int ii = 0; ii < D; ii++) {
                int count = docTopics[ii].getCount(k);
                if (count > 1) {
                    int c = SamplerUtils.randAntoniak(
                            hyperparams.get(ALPHA_LOCAL) * globalWeights.get(k),
                            count);
                    counts.changeCount(k, c);
                } else {
                    counts.changeCount(k, count);
                }
            }
        }
        double[] dirPrior = new double[topicWords.getNumComponents() + 1];
        ArrayList<Integer> indices = new ArrayList<Integer>();

        int idx = 0;
        for (int kk : topicWords.getIndices()) {
            indices.add(kk);
            dirPrior[idx++] = counts.getCount(kk);
        }

        indices.add(NEW_COMPONENT_INDEX);
        dirPrior[idx] = hyperparams.get(ALPHA_GLOBAL);

        Dirichlet dir = new Dirichlet(dirPrior);
        double[] wts = dir.nextDistribution();
        this.globalWeights = new SparseVector();
        for (int ii = 0; ii < wts.length; ii++) {
            this.globalWeights.set(indices.get(ii), wts[ii]);
        }
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
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
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            docTopics[ii].validate(msg);
            if (docTopics[ii].getCountSum() != words[dd].length) {
                throw new RuntimeException(msg + ". Num tokens mismatch");
            }
        }
        for (int k : topicWords.getIndices()) {
            topicWords.getComponent(k).phi.validate(msg);
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append("# topics: ").append(topicWords.getNumComponents()).append("\n");
        return str.toString();
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int k : topicWords.getSortedIndices()) {
                Topic topic = topicWords.getComponent(k);
                modelStr.append(k).append("\n");
                modelStr.append(globalWeights.get(k)).append("\n");
                modelStr.append(topic.born).append("\n");
                modelStr.append(DirMult.output(topic.phi)).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int ii = 0; ii < D; ii++) {
                assignStr.append(ii).append("\n");
                for (int n = 0; n < z[ii].length; n++) {
                    assignStr.append(z[ii][n]).append("\t");
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
            System.exit(1);
        }

        validate("Done reading state from " + filepath);
    }

    protected void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            this.topicWords = new Topics();
            this.globalWeights = new SparseVector();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            String line;
            while ((line = reader.readLine()) != null) {
                int k = Integer.parseInt(line);
                double weight = Double.parseDouble(reader.readLine());
                this.globalWeights.set(k, weight);

                int born = Integer.parseInt(reader.readLine());
                DirMult topicWord = DirMult.input(reader.readLine());
                topicWords.createNewComponent(k, new Topic(born, topicWord));
            }
            this.topicWords.fillInactives();
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    protected void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }
        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int ii = 0; ii < D; ii++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != ii) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < z[ii].length; n++) {
                    z[ii][n] = Integer.parseInt(sline[n]);
                    docTopics[ii].increment(z[ii][n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k : topicWords.getIndices()) {
            DirMult topic = topicWords.getComponent(k).phi;
            sortedTopics.add(new RankingItem<Integer>(k, topic.getCountSum()));
        }
        Collections.sort(sortedTopics);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (RankingItem<Integer> sortedTopic : sortedTopics) {
                int k = sortedTopic.getObject();
                Topic topic = topicWords.getComponent(k);
                double[] distrs = topic.phi.getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + k + ", " + topic.born
                        + ", " + topic.phi.getCountSum() + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    class Topic {

        private final int born;
        private DirMult phi;

        public Topic(int born, DirMult phi) {
            this.born = born;
            this.phi = phi;
        }

        public double getPhi(int w) {
            return (phi.getCount(w) + hyperparams.get(BETA))
                    / (phi.getCountSum() + hyperparams.get(BETA) * V);
        }
    }

    class Topics {

        private final HashMap<Integer, Topic> actives;
        private SortedSet<Integer> inactives;

        public Topics() {
            this.actives = new HashMap<Integer, Topic>();
            this.inactives = new TreeSet<Integer>();
        }

        public ArrayList<Integer> getSortedIndices() {
            ArrayList<Integer> sortedIndices = new ArrayList<Integer>();
            for (int ii : getIndices()) {
                sortedIndices.add(ii);
            }
            Collections.sort(sortedIndices);
            return sortedIndices;
        }

        public int getNumComponents() {
            return this.actives.size();
        }

        public Set<Integer> getIndices() {
            return this.actives.keySet();
        }

        public boolean isEmpty() {
            return actives.isEmpty();
        }

        public boolean isActive(int idx) {
            return this.actives.containsKey(idx);
        }

        public Topic getComponent(int idx) {
            return this.actives.get(idx);
        }

        public void removeComponent(int idx) {
            this.inactives.add(idx);
            this.actives.remove(idx);
        }

        public void createNewComponent(int idx, Topic c) {
            if (isActive(idx)) {
                throw new RuntimeException("Component " + idx + " exists");
            }
            if (inactives.contains(idx)) {
                inactives.remove(idx);
            }
            this.actives.put(idx, c);
        }

        public void fillInactives() {
            int maxTableIndex = -1;
            for (int idx : actives.keySet()) {
                if (idx > maxTableIndex) {
                    maxTableIndex = idx;
                }
            }
            this.inactives = new TreeSet<Integer>();
            for (int ii = 0; ii < maxTableIndex; ii++) {
                if (!isActive(ii)) {
                    inactives.add(ii);
                }
            }
        }

        public int getNextIndex() {
            int newIdx;
            if (this.inactives.isEmpty()) {
                newIdx = this.actives.size();
            } else {
                newIdx = this.inactives.first();
            }
            return newIdx;
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + HDP.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        return "java -cp \"dist/segan.jar:lib/*\" sampler.unsupervised.HDP "
                + "--dataset amazon-data "
                + "--word-voc-file demo/amazon-data/format-unsupervised/amazon-data.wvoc "
                + "--word-file demo/amazon-data/format-unsupervised/amazon-data.dat "
                + "--info-file demo/amazon-data/format-unsupervised/amazon-data.docinfo "
                + "--output-folder demo/amazon-data/model-unsupervised "
                + "--burnIn 100 "
                + "--maxIter 250 "
                + "--sampleLag 30 "
                + "--report 5 "
                + "--K 25 "
                + "--global-alpha 1 "
                + "--local-alpha 1 "
                + "--beta 0.1 "
                + "--init preset "
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

        // data output
        addOption("output-folder", "Output folder");

        // sampling
        addSamplingOptions();

        // parameters
        addOption("global-alpha", "Global alpha");
        addOption("local-alpha", "Local alpha");
        addOption("beta", "Beta");
        addOption("K", "Initial number of topics");
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
        double globalAlpha = CLIUtils.getDoubleArgument(cmd, "global-alpha", 1);
        double localAlpha = CLIUtils.getDoubleArgument(cmd, "local-alpha", 0.5);
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

        HDP sampler = new HDP();
        sampler.setVerbose(cmd.hasOption("v"));
        sampler.setDebug(cmd.hasOption("d"));
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        if (initState == InitialState.PRESET) { // for initialization
            sampler.setK(K);
        }
        sampler.configure(outputFolder, V,
                globalAlpha, localAlpha, beta,
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

        sampler.train(data.getWords(), selectedDocIndices);
        sampler.initialize();
        sampler.iterate();
        sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
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
