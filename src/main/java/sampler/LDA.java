package sampler;

import core.AbstractSampler;
import data.TextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 * Implementation of a Gibbs sampler for LDA.
 *
 * This is obsolete and retained for backward compatibility. For new
 * implementations which needs LDA, use the one in sampler/unsupervised/LDA.java
 *
 * @author vietan
 */
public class LDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int K;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z;
    protected DirMult[] doc_topics;
    protected DirMult[] topic_words;

    public void configure(LDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    /**
     * TODO: separate configure with train.
     */
    public void configure(String folder, int[][] words,
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
        this.words = words;

        this.K = K;
        this.V = V;
        if (words != null) {
            this.D = this.words.length;
        }

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

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- # documents:\t" + D);
            logln("--- # topics:\t" + K);
            logln("--- # tokens:\t" + numTokens);
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

    public void configure(String folder,
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
            logln("--- # topics:\t" + K);
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

    public void train(int[][] ws) {
        this.words = ws;
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

    protected void setName() {
        this.name = this.prefix
                + "_LDA"
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_b-" + formatter.format(this.hyperparams.get(BETA))
                + "_opt-" + this.paramOptimized;
    }

    public int[][] getZ() {
        return this.z;
    }

    public DirMult[] getDocTopics() {
        return this.doc_topics;
    }

    public DirMult[] getTopicWords() {
        return this.topic_words;
    }

    /**
     * Configure for new documents.
     *
     * @param ws New document words
     */
    public void configure(int[][] ws) {
        this.words = ws;
        this.D = this.words.length;

        this.initializeDataStructure(null);

        // initialize assignments for new documents
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
            }
        }
    }

    /**
     * Sample assignments for new documents given a learned model.
     *
     * @param ws New document words
     */
    public void sample(int[][] ws) {
        if (verbose) {
            logln("Sampling for new documents ...");
        }

        configure(ws);

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

        // sample
        startTime = System.currentTimeMillis();
        logLikelihoods = new ArrayList<Double>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;
            sampleZs(!REMOVE, !ADD, REMOVE, ADD);
            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter + "/" + MAX_ITER
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio)
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }
            if (report && iter > BURN_IN && iter % LAG == 0) {
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
    }

    public void initialize(double[][] docTopicPrior, double[][] topicWordPrior) {
        if (verbose) {
            logln("Initializing with pre-defined topics ...");
        }

        initializeModelStructure(topicWordPrior);

        initializeDataStructure(docTopicPrior);

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeModelStructure(double[][] topics) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        topic_words = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (topics != null) {
                topic_words[k] = new DirMult(V, hyperparams.get(BETA) * V, topics[k]);
            } else {
                topic_words[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }
    }

    protected void initializeDataStructure(double[][] docTopicPrior) {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        doc_topics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            if (docTopicPrior != null) {
                doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, docTopicPrior[d]);
            } else {
                doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
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

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
                topic_words[z[d][n]].increment(words[d][n]);
            }
        }
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

        startTime = System.currentTimeMillis();
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            long eTime = sampleZs(REMOVE, ADD, REMOVE, ADD);

            if (debug) {
                validate("Iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter + "/" + MAX_ITER
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio)
                        + ". time = " + eTime
                        + "\n" + getSamplerFolderPath();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n");
                } else {
                    logln("--- Sampling. " + str + "\n");
                }
            }
            if (report && iter > BURN_IN && iter % LAG == 0) {
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
        double totalBeta = V * hyperparams.get(BETA);
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromData) {
                    doc_topics[d].decrement(z[d][n]);
                }
                if (removeFromModel) {
                    topic_words[z[d][n]].decrement(words[d][n]);
                }

                double[] probs = new double[K];
                for (int k = 0; k < K; k++) {
                    probs[k] = (doc_topics[d].getCount(k) + hyperparams.get(ALPHA))
                            * (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                            / (topic_words[k].getCountSum() + totalBeta);
                }
                int sampledZ = SamplerUtils.scaleSample(probs);
                if (sampledZ != z[d][n]) {
                    numTokensChanged++;
                }
                z[d][n] = sampledZ;

                if (addToData) {
                    doc_topics[d].increment(z[d][n]);
                }
                if (addToModel) {
                    topic_words[z[d][n]].increment(words[d][n]);
                }
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Sample the topic assignment for each token
     *
     * @param d The document index
     * @param n The token index
     * @param remove Whether this token should be removed from the current
     * assigned topic
     * @param add Whether this token should be added to the sampled topic
     */
    protected void sampleZ(int d, int n, boolean remove, boolean add) {
        double totalBeta = V * hyperparams.get(BETA);
        doc_topics[d].decrement(z[d][n]);
        if (remove) {
            topic_words[z[d][n]].decrement(words[d][n]);
        }

        double[] probs = new double[K];
        for (int k = 0; k < K; k++) {
            probs[k] = (doc_topics[d].getCount(k) + hyperparams.get(ALPHA))
                    * (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                    / (topic_words[k].getCountSum() + totalBeta);
        }
        int sampledZ = SamplerUtils.scaleSample(probs);
        if (sampledZ != z[d][n]) {
            numTokensChanged++;
        }
        z[d][n] = sampledZ;

        doc_topics[d].increment(z[d][n]);
        if (add) {
            topic_words[z[d][n]].increment(words[d][n]);
        }
    }

    protected void sampleZ(int d, int ii, int n,
            boolean removeFromData, boolean addToData) {
        if (removeFromData) {
            doc_topics[d].decrement(z[d][ii]);
        }

        double[] probs = new double[K];
        for (int k = 0; k < K; k++) {
            probs[k] = (doc_topics[d].getCount(k) + hyperparams.get(ALPHA))
                    * (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                    / (topic_words[k].getCountSum() + V * hyperparams.get(BETA));
        }
        int sampledZ = SamplerUtils.scaleSample(probs);
        if (sampledZ != z[d][ii]) {
            numTokensChanged++;
        }
        z[d][ii] = sampledZ;

        if (addToData) {
            doc_topics[d].increment(z[d][ii]);
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
            docTopicLlh += doc_topics[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += topic_words[k].getLogLikelihood();
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
            llh += doc_topics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }
        for (int k = 0; k < K; k++) {
            llh += topic_words[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.doc_topics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topic_words[k].setConcentration(this.hyperparams.get(BETA) * V);
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
                String[] topWords = getTopWords(topic_words[k].getDistribution(), numTopWords);
                // output top words
                writer.write("[Topic " + k + ": " + topic_words[k].getCountSum() + "]");
                for (String tw : topWords) {
                    writer.write(" " + tw);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    /**
     * Output topics with top words and top documents associated.
     *
     * @param file Output file
     * @param numTopWords Number of top words
     * @param numTopDocs Number of top documents
     * @param docIds List of document IDs
     */
    public void outputTopicTopWordsWithDocs(File file,
            int numTopWords,
            int numTopDocs,
            String[] docIds) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words and additional info to " + file);
        }

        ArrayList<RankingItem<Integer>>[] topicRankDocs = new ArrayList[K];
        for (int k = 0; k < K; k++) {
            topicRankDocs[k] = new ArrayList<RankingItem<Integer>>();
        }
        for (int dd = 0; dd < D; dd++) {
            double[] topicDist = doc_topics[dd].getDistribution();
            for (int k = 0; k < K; k++) {
                topicRankDocs[k].add(new RankingItem<Integer>(dd, topicDist[k]));
            }
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int k = 0; k < K; k++) {
                String[] topWords = getTopWords(topic_words[k].getDistribution(), numTopWords);
                // output top words
                writer.write("[Topic " + k + ": " + topic_words[k].getCountSum() + "]");
                for (String tw : topWords) {
                    writer.write(" " + tw);
                }
                writer.write("\n");

                // documents
                Collections.sort(topicRankDocs[k]);
                for (int ii = 0; ii < numTopDocs; ii++) {
                    RankingItem<Integer> rankDoc = topicRankDocs[k].get(ii);
                    writer.write("\t" + rankDoc.getObject()
                            + "\t" + docIds[rankDoc.getObject()]
                            + "\t" + rankDoc.getPrimaryValue()
                            + "\n");
                }

                writer.write("\n\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }

    public void outputTopicTopWordsFormatted(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing topics to file " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            // headers
            for (int k = 0; k < K - 1; k++) {
                writer.write("Topic_" + k + "\t");
            }
            writer.write("Topic_" + (K - 1) + "\n");

            // content
            String[][] topWords = new String[K][numTopWords];
            for (int k = 0; k < K; k++) {
                topWords[k] = getTopWords(topic_words[k].getDistribution(), numTopWords);
            }
            for (int ii = 0; ii < numTopWords; ii++) {
                for (int k = 0; k < K - 1; k++) {
                    writer.write(topWords[k][ii] + "\t");
                }
                writer.write(topWords[K - 1][ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + file);
        }
    }

    public void outputTopicTopWordsCummProbs(String filepath, int numTopWords) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        double[][] distrs = new double[K][];
        for (int k = 0; k < K; k++) {
            distrs[k] = topic_words[k].getDistribution();
        }
        IOUtils.outputTopWordsWithProbs(distrs, wordVocab, numTopWords, filepath);
    }

    public void outputTopicWordDistribution(String outputFile) throws Exception {
        double[][] pi = new double[K][];
        for (int k = 0; k < K; k++) {
            pi[k] = this.topic_words[k].getDistribution();
        }
        IOUtils.outputDistributions(pi, outputFile);
    }

    public double[][] inputTopicWordDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    public void outputDocumentTopicDistribution(String outputFile) throws Exception {
        double[][] theta = new double[D][];
        for (int d = 0; d < D; d++) {
            theta[d] = this.doc_topics[d].getDistribution();
        }
        IOUtils.outputDistributions(theta, outputFile);
    }

    public double[][] inputDocumentTopicDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            doc_topics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topic_words[k].validate(msg);
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
                modelStr.append(DirMult.output(topic_words[k])).append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(doc_topics[d])).append("\n");
                for (int n = 0; n < words[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing state to " + filepath);
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
            this.initializeModelStructure(null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                topic_words[k] = DirMult.input(reader.readLine());
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
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine());
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                doc_topics[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                for (int n = 0; n < words[d].length; n++) {
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

    /**
     * Output topic coherence
     *
     * @param file Output file
     * @param topicCoherence Topic coherence
     * @throws java.lang.Exception
     */
    public void outputTopicCoherence(
            File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + file);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            double[] distribution = this.topic_words[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + topic_words[k].getCountSum()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public DirMult[] getTopics() {
        return this.topic_words;
    }

    public double[][] getDocumentEmpiricalDistributions() {
        double[][] docEmpDists = new double[D][K];
        for (int d = 0; d < D; d++) {
            docEmpDists[d] = doc_topics[d].getEmpiricalDistribution();
        }
        return docEmpDists;
    }

    public void outputDocTopicDistributions(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-document topic distribution to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int d = 0; d < D; d++) {
            writer.write(Integer.toString(d));
            double[] docTopicDist = this.doc_topics[d].getDistribution();
            for (int k = 0; k < K; k++) {
                writer.write("\t" + docTopicDist[k]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    /**
     * Output the empirical distributions over words for each topic from a set
     * of documents.
     *
     * @param file The output file
     * @param docIndices The list of document indices
     * @param numTopWords
     */
    public void outputDocTopicDistributions(File file,
            ArrayList<Integer> docIndices,
            int numTopWords) {
        if (verbose) {
            logln("Outputing empirical topic distributions to " + file);
        }
        SparseCount[] empWordCounts = new SparseCount[K];
        for (int k = 0; k < K; k++) {
            empWordCounts[k] = new SparseCount();
        }
        int totalTokenCount = 0;
        if (docIndices == null) {
            for (int d = 0; d < D; d++) {
                totalTokenCount += words[d].length;
                for (int n = 0; n < words[d].length; n++) {
                    empWordCounts[z[d][n]].increment(words[d][n]);
                }
            }
        } else {
            for (int d : docIndices) {
                totalTokenCount += words[d].length;
                for (int n = 0; n < words[d].length; n++) {
                    empWordCounts[z[d][n]].increment(words[d][n]);
                }
            }
        }

        String[][] topWords = new String[K][numTopWords];
        for (int k = 0; k < K; k++) {
            ArrayList<RankingItem<Integer>> rankWords = MiscUtils.getRankingList(empWordCounts[k]);
            for (int ii = 0; ii < numTopWords; ii++) {
                topWords[k][ii] = wordVocab.get(rankWords.get(ii).getObject());
            }
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            // header
            for (int k = 0; k < K - 1; k++) {
                writer.write("Topic" + k + "\t");
            }
            writer.write("Topic" + (K - 1) + "\n");

            // words
            for (int ii = 0; ii < numTopWords; ii++) {
                for (int k = 0; k < K - 1; k++) {
                    writer.write(topWords[k][ii] + "\t");
                }
                writer.write(topWords[K - 1][ii] + "\n");
            }
            writer.close();

            writer = IOUtils.getBufferedWriter(file.getAbsolutePath() + ".prob");
            // header
            for (int k = 0; k < K - 1; k++) {
                writer.write("Topic" + k + "\t");
            }
            writer.write("Topic" + (K - 1) + "\n");

            // empirical probabilities
            for (int k = 0; k < K - 1; k++) {
                writer.write((double) empWordCounts[k].getCountSum() / totalTokenCount + "\t");
            }
            writer.write((double) empWordCounts[K - 1].getCountSum() / totalTokenCount + "\n");
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    /**
     * Get the empirical word counts per topic for a set of documents, given the
     * current assignments.
     *
     * @param docIndices The list of document indices
     * @return Empirical word counts
     */
    public SparseCount[] getEmpiricalWordCounts(ArrayList<Integer> docIndices) {
        SparseCount[] empWordCounts = new SparseCount[K];
        for (int k = 0; k < K; k++) {
            empWordCounts[k] = new SparseCount();
        }
        if (docIndices == null) {
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    empWordCounts[z[d][n]].increment(words[d][n]);
                }
            }
        } else {
            for (int d : docIndices) {
                for (int n = 0; n < words[d].length; n++) {
                    empWordCounts[z[d][n]].increment(words[d][n]);
                }
            }
        }

        return empWordCounts;
    }

    public void outputTopicWordDistributions(File file) throws Exception {
        if (verbose) {
            logln("Outputing per-topic word distribution to " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < K; k++) {
            writer.write(Integer.toString(k));
            double[] topicWordDist = this.topic_words[k].getDistribution();
            for (int v = 0; v < V; v++) {
                writer.write("\t" + topicWordDist[v]);
            }
            writer.write("\n");
        }
        writer.close();
    }

    public ArrayList<double[][]> computeAveragingPerplexities(String stateFile,
            int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices) {
        if (verbose) {
            System.out.println();
            logln("Computing perplexity using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        inputModel(stateFile);

        words = newWords;
        D = words.length;

        numTokens = 0;
        int numTrainTokens = 0;
        int numTestTokens = 0;

        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
            numTrainTokens += trainIndices[d].size();
            numTestTokens += testIndices[d].size();
        }

        if (verbose) {
            logln("Test data:");
            logln("--- D = " + D);
            logln("--- # tokens = " + numTokens);
            logln("--- # train tokens = " + numTrainTokens);
            logln("--- # test tokens = " + numTestTokens);
        }

        // initialize structure
        doc_topics = new DirMult[D];
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
            z[d] = new int[trainIndices[d].size()];
        }

        if (verbose) {
            logln("--- Sampling on test data ...");
        }

        double totalBeta = hyperparams.get(BETA) * V;
        ArrayList<double[][]> tokenProbsList = new ArrayList<double[][]>();
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId()
                        + " " + getSamplerFolderPath());
            }

            for (int d = 0; d < D; d++) {
                for (int ii = 0; ii < trainIndices[d].size(); ii++) {
                    int n = trainIndices[d].get(ii);
                    if (iter == 0) {
                        sampleZ(d, ii, n, !REMOVE, ADD);
                    } else {
                        sampleZ(d, ii, n, REMOVE, ADD);
                    }
                }
            }

            // compute perplexity
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double[][] tokenProbs = new double[D][];
                for (int d = 0; d < D; d++) {
                    tokenProbs[d] = new double[testIndices[d].size()];
                    double[] theta = new double[K];
                    for (int kk = 0; kk < K; kk++) {
                        theta[kk] = (doc_topics[d].getCount(kk) + hyperparams.get(ALPHA))
                                / (doc_topics[d].getCountSum() + hyperparams.get(ALPHA) * K);
                    }

                    for (int i = 0; i < testIndices[d].size(); i++) {
                        int n = testIndices[d].get(i);
                        double val = 0.0;
                        for (int k = 0; k < K; k++) {
                            double phi = (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (topic_words[k].getCountSum() + totalBeta);
                            val += theta[k] * phi;
                        }
                        tokenProbs[d][i] += val;
                    }
                }
                tokenProbsList.add(tokenProbs);
            }
        }
        return tokenProbsList;
    }

    public double computePerplexity(String stateFile,
            int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Computing perplexity using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        inputModel(stateFile);

        words = newWords;
        D = words.length;

        numTokens = 0;
        int numTrainTokens = 0;
        int numTestTokens = 0;

        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
            numTrainTokens += trainIndices[d].size();
            numTestTokens += testIndices[d].size();
        }

        if (verbose) {
            logln("Test data:");
            logln("--- D = " + D);
            logln("--- # tokens = " + numTokens);
            logln("--- # train tokens = " + numTrainTokens);
            logln("--- # test tokens = " + numTestTokens);
        }

        doc_topics = new DirMult[D];
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
            z[d] = new int[trainIndices[d].size()];
        }

        ArrayList<Double> perplexities = new ArrayList<Double>();
        if (verbose) {
            logln("--- Sampling on test data ...");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(stateFile + ".perp");
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId()
                        + " " + getSamplerFolderPath());
            }

            for (int d = 0; d < D; d++) {
                for (int ii = 0; ii < trainIndices[d].size(); ii++) {
                    int n = trainIndices[d].get(ii);
                    if (iter == 0) {
                        sampleZ(d, ii, n, !REMOVE, ADD);
                    } else {
                        sampleZ(d, ii, n, REMOVE, ADD);
                    }
                }
            }

            // compute perplexity
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                perplexities.add(computePerplexity(testIndices, stateFile + ".perp"));
            }
        }
        writer.close();

        double avgPerplexity = StatUtils.mean(perplexities);
        return avgPerplexity;
    }

    private double computePerplexity(ArrayList<Integer>[] testIndices, String outFile) {
        double totalBeta = hyperparams.get(BETA) * V;
        double totalLogprob = 0.0;
        int numTestTokens = 0;
        for (int d = 0; d < D; d++) {
            numTestTokens += testIndices[d].size();
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outFile);
            for (int d = 0; d < D; d++) {
                double[] docTheta = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    docTheta[kk] = (doc_topics[d].getCount(kk) + hyperparams.get(ALPHA))
                            / (doc_topics[d].getCountSum() + hyperparams.get(ALPHA) * K);
                }

                double docLogProb = 0.0;
                for (int n : testIndices[d]) {
                    double val = 0.0;
                    for (int k = 0; k < K; k++) {
                        double phi = (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                / (topic_words[k].getCountSum() + totalBeta);
                        val += docTheta[k] * phi;
                    }
                    docLogProb += Math.log(val);
                }
                totalLogprob += docLogProb;
                writer.write(d
                        + "\t" + words[d].length
                        + "\t" + testIndices[d].size()
                        + "\t" + docLogProb + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
        double perplexity = Math.exp(-totalLogprob / numTestTokens);
        return perplexity;
    }

    public double computePerplexity(String stateFile, int[][] newWords) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Computing perplexity using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        words = newWords;
        D = words.length;
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
        }

        doc_topics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            doc_topics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        ArrayList<Double> perplexities = new ArrayList<Double>();
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(stateFile + ".perp1");
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId());
            }

            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD);
            }

            // compute perplexity
            double totalBeta = hyperparams.get(BETA) * V;
            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                double totalLogprob = 0.0;
                for (int d = 0; d < D; d++) {
                    double docLogProb = 0.0;
                    for (int n = 0; n < words[d].length; n++) {
                        double val = 0.0;
                        for (int k = 0; k < K; k++) {
                            double theta = (doc_topics[d].getCount(k) + hyperparams.get(ALPHA))
                                    / (doc_topics[d].getCountSum() + hyperparams.get(ALPHA) * K);
                            double phi = (topic_words[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (topic_words[k].getCountSum() + totalBeta);
                            val += theta * phi;
                        }
                        docLogProb += Math.log(val);
                    }
                    totalLogprob += docLogProb;
                    writer.write(iter
                            + "\t" + d
                            + "\t" + words[d].length
                            + "\t" + docLogProb + "\n");
                }
                double perplexity = Math.exp(-totalLogprob / numTokens);
                perplexities.add(perplexity);
            }
        }
        writer.close();
        double avgPerplexity = StatUtils.mean(perplexities);
        return avgPerplexity;
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

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + LDA.class.getName() + " -help";
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
        addOption("K", "Number of topics");
        addOption("numTopwords", "Number of top words per topic");

        // model hyperparameters
        addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                + "for topic distributions");
        addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                + "for word distributions");

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

        // data 
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "demo/"
                + datasetName + "/" + formatFolder + "-model");
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        // sampler
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 25);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 50);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        if (verbose) {
            System.out.println("Loading data ...");
        }
        TextDataset dataset = new TextDataset(datasetName, datasetFolder);
        dataset.setFormatFilename(formatFile);
        dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder));
        dataset.prepareTopicCoherence(numTopWords);

        int V = dataset.getWordVocab().size();
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("Running LDA ...");
        }
        LDA sampler = new LDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(dataset.getWordVocab());
        sampler.setPrefix("prior_");

        sampler.configure(outputFolder, dataset.getWords(),
                V, K, alpha, beta, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File ldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(ldaFolder);

        double ratio = 100;
        double prob = 1.0 / (K - 1 + ratio);
        double[][] docTopicPrior = new double[dataset.getWords().length][K];
        for (int d = 0; d < dataset.getWords().length; d++) {
            docTopicPrior[d][0] = ratio * prob;
            for (int k = 1; k < K; k++) {
                docTopicPrior[d][k] = prob;
            }
        }

        sampler.initialize(docTopicPrior, null);
        sampler.iterate();
        sampler.outputTopicTopWords(new File(ldaFolder, TopWordFile), numTopWords);
        sampler.outputTopicCoherence(new File(ldaFolder, TopicCoherenceFile), dataset.getTopicCoherence());
        sampler.outputDocTopicDistributions(new File(ldaFolder, "doc-topic.txt"));
        sampler.outputTopicWordDistributions(new File(ldaFolder, "topic-word.txt"));
    }

    public static void parallelPerplexity(int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            File iterPerplexityFolder,
            File resultFolder,
            LDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPerplexityFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (String filename : filenames) {
                if (!filename.endsWith("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPerplexityFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                LDAPerplexityRunner runner = new LDAPerplexityRunner(sampler,
                        newWords, trainIndices, testIndices,
                        stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // summarize multiple perplexities
            String[] ppxFiles = iterPerplexityFolder.list();
            ArrayList<Double> ppxs = new ArrayList<Double>();
            for (String ppxFile : ppxFiles) {
                double ppx = IOUtils.inputPerplexity(new File(iterPerplexityFolder, ppxFile));
                ppxs.add(ppx);
            }

            // averaging
            File ppxResultFile = new File(resultFolder, PerplexityFile);
            IOUtils.outputPerplexities(ppxResultFile, ppxs);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing perplexity parallel test.");
        }
    }

    public static void parallelAveragingPerplexity(
            int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            File iterPerplexityFolder,
            File resultFolder,
            LDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPerplexityFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPerplexityFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                LDAAveragingPerplexityRunner runner = new LDAAveragingPerplexityRunner(
                        sampler,
                        newWords, trainIndices, testIndices,
                        stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // summarize multiple perplexities
            String[] ppxFiles = iterPerplexityFolder.list();
            ArrayList<ArrayList<double[][]>> allTokenProbs = new ArrayList<ArrayList<double[][]>>();
            for (String ppxFile : ppxFiles) {
                ArrayList<double[][]> tokenProbsList = LDA.inputTokenProbabilities(
                        new File(iterPerplexityFolder, ppxFile), testIndices);
                allTokenProbs.add(tokenProbsList);
            }

            int I = allTokenProbs.size();
            int J = allTokenProbs.get(0).size();

            // single final
            double[][] singleFinalProbs = new double[newWords.length][];
            for (int dd = 0; dd < newWords.length; dd++) {
                singleFinalProbs[dd] = new double[newWords[dd].length];
                System.arraycopy(allTokenProbs.get(I - 1).get(J - 1)[dd], 0,
                        singleFinalProbs[dd], 0, newWords[dd].length);
            }
            double singleFinal = LDA.computePerplexity(singleFinalProbs);
            System.out.println("Single final = " + singleFinal);

            // single average
            double[][] singleAvgProbs = new double[newWords.length][];
            for (int dd = 0; dd < newWords.length; dd++) {
                singleAvgProbs[dd] = new double[newWords[dd].length];
                for (int nn = 0; nn < newWords[dd].length; nn++) {
                    double sum = 0.0;
                    for (int jj = 0; jj < J; jj++) {
                        sum += allTokenProbs.get(I - 1).get(jj)[dd][nn];
                    }
                    singleAvgProbs[dd][nn] = sum / J;
                }
            }
            double singleAvg = LDA.computePerplexity(singleAvgProbs);
            System.out.println("Single avg = " + singleAvg);

            // multiple final©
            double[][] multipleFinalProbs = new double[newWords.length][];
            for (int dd = 0; dd < newWords.length; dd++) {
                multipleFinalProbs[dd] = new double[newWords[dd].length];
                for (int nn = 0; nn < newWords[dd].length; nn++) {
                    double sum = 0.0;
                    for (int ii = 0; ii < I; ii++) {
                        sum += allTokenProbs.get(ii).get(J - 1)[dd][nn];
                    }
                    multipleFinalProbs[dd][nn] = sum / I;
                }
            }
            double multipleFinal = LDA.computePerplexity(multipleFinalProbs);
            System.out.println("Multiple final = " + multipleFinal);

            // multiple average
            double[][] multipleAvgProbs = new double[newWords.length][];
            for (int dd = 0; dd < newWords.length; dd++) {
                multipleAvgProbs[dd] = new double[newWords[dd].length];
                for (int nn = 0; nn < newWords[dd].length; nn++) {
                    double sum = 0.0;
                    for (int ii = 0; ii < I; ii++) {
                        for (int jj = 0; jj < J; jj++) {
                            sum += allTokenProbs.get(ii).get(jj)[dd][nn];
                        }
                    }
                    multipleAvgProbs[dd][nn] = sum / (I * J);
                }
            }
            double multipleAvg = LDA.computePerplexity(multipleAvgProbs);
            System.out.println("Mutiple avg = " + multipleAvg);

            // averaging
            File ppxResultFile = new File(resultFolder, AveragingPerplexityFile);
            BufferedWriter writer = IOUtils.getBufferedWriter(ppxResultFile);
            writer.write("SingleFinal\t" + singleFinal + "\n");
            writer.write("SingleAvg\t" + singleAvg + "\n");
            writer.write("MultipleFinal\t" + multipleFinal + "\n");
            writer.write("MultipleAvg\t" + multipleAvg + "\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing perplexity parallel test.");
        }
    }

    public static double computePerplexity(double[][] tokenProbs) {
        double val = 0.0;
        int num = 0;
        for (double[] tokenProb : tokenProbs) {
            num += tokenProb.length;
            for (int nn = 0; nn < tokenProb.length; nn++) {
                val += Math.log(tokenProb[nn]);
            }
        }
        return Math.exp(-val / num);
    }

    public double[][] computeAvgTopicCoherence(File file,
            MimnoTopicCoherence topicCoherence) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing averaged topic coherence to file " + file);

        }

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[][] avgTopics = new double[K][V];
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file.getAbsolutePath() + ".iter");
            writer.write("Iteration");
            for (int k = 0; k < K; k++) {
                writer.write("\tTopic_" + k);
            }
            writer.write("\n");

            // partial score
            ArrayList<double[][]> aggTopics = new ArrayList<double[][]>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }
                inputModel(new File(reportFolder, filename).getAbsolutePath());
                double[][] pointTopics = new double[K][V];

                writer.write(filename);
                for (int k = 0; k < K; k++) {
                    pointTopics[k] = topic_words[k].getDistribution();
                    int[] topic = SamplerUtils.getSortedTopic(pointTopics[k]);
                    double score = topicCoherence.getCoherenceScore(topic);

                    writer.write("\t" + score);
                }
                writer.write("\n");
                aggTopics.add(pointTopics);
            }

            // averaging
            writer.write("Average");
            ArrayList<Double> scores = new ArrayList<Double>();
            for (int k = 0; k < K; k++) {
                double[] avgTopic = new double[V];
                for (int v = 0; v < V; v++) {
                    for (double[][] aggTopic : aggTopics) {
                        avgTopic[v] += aggTopic[k][v] / aggTopics.size();
                    }
                }
                int[] topic = SamplerUtils.getSortedTopic(avgTopic);
                double score = topicCoherence.getCoherenceScore(topic);
                writer.write("\t" + score);
                scores.add(score);
                avgTopics[k] = avgTopic;
            }
            writer.write("\n");
            writer.close();

            // output aggregated topic coherence scores
            IOUtils.outputTopicCoherences(file, scores);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
        return avgTopics;
    }

    public static void outputTokenProbabilities(File outputFile,
            ArrayList<double[][]> tokenProbsList,
            ArrayList<Integer>[] testIndices) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(tokenProbsList.size() + "\n");
            for (double[][] tokenProbs : tokenProbsList) {
                for (int dd = 0; dd < testIndices.length; dd++) {
                    writer.write(dd + "\t" + testIndices[dd].size());
                    for (int jj = 0; jj < testIndices[dd].size(); jj++) {
                        writer.write("\t" + tokenProbs[dd][jj]);
                    }
                    writer.write("\n");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public static ArrayList<double[][]> inputTokenProbabilities(File intputFile,
            ArrayList<Integer>[] testIndices) {
        ArrayList<double[][]> tokenProbsList = new ArrayList<double[][]>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(intputFile);
            String[] sline;
            int num = Integer.parseInt(reader.readLine());
            for (int ii = 0; ii < num; ii++) {
                double[][] tokenProbs = new double[testIndices.length][];
                for (int dd = 0; dd < testIndices.length; dd++) {
                    sline = reader.readLine().split(" ");
                    if (Integer.parseInt(sline[0]) != dd) {
                        throw new RuntimeException("Mismatch");
                    }
                    if (Integer.parseInt(sline[1]) != testIndices[dd].size()) {
                        throw new RuntimeException("Mismatch");
                    }
                    if (testIndices[dd].size() != sline.length - 2) {
                        throw new RuntimeException("Mismatch");
                    }

                    tokenProbs[dd] = new double[testIndices[dd].size()];
                    for (int nn = 0; nn < testIndices[dd].size(); nn++) {
                        tokenProbs[dd][nn] = Double.parseDouble(sline[nn + 2]);
                    }
                }
                tokenProbsList.add(tokenProbs);
                reader.readLine();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + intputFile);
        }
        return tokenProbsList;
    }
}

class LDAPerplexityRunner implements Runnable {

    LDA sampler;
    int[][] newWords;
    ArrayList<Integer>[] trainIndices;
    ArrayList<Integer>[] testIndices;
    String stateFile;
    String outputFile;

    public LDAPerplexityRunner(LDA sampler,
            int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.trainIndices = trainIndices;
        this.testIndices = testIndices;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        LDA testSampler = new LDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            double perplexity = testSampler.computePerplexity(stateFile, newWords, trainIndices, testIndices);
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(perplexity + "\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}

class LDAAveragingPerplexityRunner implements Runnable {

    LDA sampler;
    int[][] newWords;
    ArrayList<Integer>[] trainIndices;
    ArrayList<Integer>[] testIndices;
    String stateFile;
    String outputFile;

    public LDAAveragingPerplexityRunner(LDA sampler,
            int[][] newWords,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.trainIndices = trainIndices;
        this.testIndices = testIndices;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        LDA testSampler = new LDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            ArrayList<double[][]> tokenProbsList = testSampler.computeAveragingPerplexities(
                    stateFile, newWords, trainIndices, testIndices);
            LDA.outputTokenProbabilities(new File(outputFile), tokenProbsList, testIndices);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
