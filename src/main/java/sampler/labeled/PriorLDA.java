package sampler.labeled;

import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class PriorLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1; // 0.1
    public static final int ETA = 2;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] observed topics; for some doc, this can be partially or totally unobserved
    protected int K; // number of topics;
    protected int V; // vocab size
    protected int D; // number of documents
    protected double[][] docLabelPriors;
    protected double[] docLabelPriorSums;
    // latent
    private DirMult[] labelWords; // K multinomials over V words
    private DirMult[] docLabels; // D multinomials over K topics
    protected int[][] z;
    // internal
    private int numTokens;      // number of token assignments to be sampled
    private int numTokensChange;
    private ArrayList<String> labelVocab;

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
    }

    public void configure(PriorLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(ETA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V, int K,
            double alpha,
            double beta,
            double eta,
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
        this.hyperparams.add(eta);

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
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- eta:\t" + MiscUtils.formatDouble(hyperparams.get(ETA)));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_prior-LDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_e-").append(formatter.format(hyperparams.get(ETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void train(int[][] ws, int[][] ls) {
        this.words = ws;
        this.labels = ls;
        this.D = this.words.length;

        this.numTokens = 0;
        int numLabels = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
            numLabels += labels[d].length;
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- # label instances:\t" + numLabels);
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }
//
//    private boolean hasLabel(int d) {
//        if (labels != null && labels[d].length > 0) {
//            return true;
//        }
//        return false;
//    }
//

    protected void initializeModelStructure() {
        this.labelWords = new DirMult[K];
        for (int kk = 0; kk < K; kk++) {
            this.labelWords[kk] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    protected void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        docLabels = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docLabels[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
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
                docLabels[d].increment(z[d][n]);
                labelWords[z[d][n]].increment(words[d][n]);
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
            numTokensChange = 0;

            long eTime = sampleZs(REMOVE, ADD, REMOVE, ADD);

            if (debug) {
                validate("Iter " + iter);
            }

            if (verbose && iter % REP_INTERVAL == 0) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                double changeRatio = (double) numTokensChange / numTokens;
                String str = "Iter " + iter + "/" + MAX_ITER
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChange
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio)
                        + ". time = " + eTime
                        + "\n" + getSamplerFolderPath()
                        + "\n";
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
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        long sTime = System.currentTimeMillis();
        double totalBeta = V * hyperparams.get(BETA);
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromData) {
                    docLabels[d].decrement(z[d][n]);
                }
                if (removeFromModel) {
                    labelWords[z[d][n]].decrement(words[d][n]);
                }

                double[] probs = new double[K];
                for (int k = 0; k < K; k++) {
                    probs[k] = (docLabels[d].getCount(k) + hyperparams.get(ALPHA))
                            * (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                            / (labelWords[k].getCountSum() + totalBeta);
                }
                int sampledZ = SamplerUtils.scaleSample(probs);
                if (sampledZ != z[d][n]) {
                    numTokensChange++;
                }
                z[d][n] = sampledZ;

                if (addToData) {
                    docLabels[d].increment(z[d][n]);
                }
                if (addToModel) {
                    labelWords[z[d][n]].increment(words[d][n]);
                }
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += docLabels[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += labelWords[k].getLogLikelihood();
        }
        return docTopicLlh + topicWordLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        throw new RuntimeException("Not supported yet");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        throw new RuntimeException("Not supported yet");
    }

    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (this.labelVocab == null) {
            throw new RuntimeException("The topic vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int k = 0; k < K; k++) {
                double[] distrs = labelWords[k].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + k
                        + ", " + labelVocab.get(k)
                        + ", " + labelWords[k].getCountSum()
                        + "]");
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

    @Override
    public void validate(String msg) {
        for (int d = 0; d < D; d++) {
            docLabels[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            labelWords[k].validate(msg);
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
                modelStr.append(DirMult.output(labelWords[k])).append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docLabels[d])).append("\n");
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
            throw new RuntimeException("Exception while inputing state from " + filepath);
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
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                labelWords[k] = DirMult.input(reader.readLine());
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
                docLabels[d] = DirMult.input(reader.readLine());

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

    public double computePerplexity(String stateFile, int[][] newWords,
            int[][] newLabels) {
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
        labels = newLabels;
        D = words.length;
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
        }

        // initialize structure
        initializeDataStructure();

        ArrayList<Double> perplexities = new ArrayList<Double>();
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
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
                    for (int n = 0; n < words[d].length; n++) {
                        double val = 0.0;
                        for (int k = 0; k < K; k++) {
                            double theta = (docLabels[d].getCount(k) + hyperparams.get(ALPHA))
                                    / (docLabels[d].getCountSum() + hyperparams.get(ALPHA) * K);
                            double phi = (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (labelWords[k].getCountSum() + totalBeta);
                            val += theta * phi;
                        }
                        totalLogprob += Math.log(val);
                    }
                }
                double perplexity = Math.exp(-totalLogprob / numTokens);
                perplexities.add(perplexity);
            }
        }
        double avgPerplexity = StatUtils.mean(perplexities);
        return avgPerplexity;
    }

//    private void initializeDataStructure() {
//        this.docLabels = new DirMult[D];
//        this.docLabelPriors = new double[D][];
//        this.docLabelPriorSums = new double[D];
//        for (int dd = 0; dd < D; dd++) {
//            docLabelPriors[dd] = new double[K];
//            Arrays.fill(docLabelPriors[dd], hyperparams.get(ALPHA));
//            if (hasLabel(dd)) {
//                for (int ll : labels[dd]) {
//                    docLabelPriors[dd][ll] += hyperparams.get(ETA);
//                }
//            }
//            docLabelPriorSums[dd] = StatisticsUtils.sum(docLabelPriors[dd]);
//            docLabels[dd] = new DirMult(docLabelPriors[dd]);
//        }
//
//        this.z = new int[D][];
//        for (int d = 0; d < D; d++) {
//            this.z[d] = new int[words[d].length];
//        }
//    }
//
//    protected void initializeAssignments() {
//        switch (initState) {
//            case RANDOM:
//                this.initializeRandomAssignments();
//                break;
//            default:
//                throw new RuntimeException("Initialization not supported");
//        }
//    }
//
//    private void initializeRandomAssignments() {
//        for (int d = 0; d < D; d++) {
//            for (int n = 0; n < words[d].length; n++) {
//                this.z[d][n] = rand.nextInt(K);
//                this.docLabels[d].increment(z[d][n]);
//                this.labelWords[z[d][n]].increment(words[d][n]);
//            }
//        }
//    }
//
//    @Override
//    public void iterate() {
//        if (verbose) {
//            logln("Iterating ...");
//        }
//        logLikelihoods = new ArrayList<Double>();
//
//        try {
//            if (report) {
//                IOUtils.createFolder(new File(this.getSamplerFolderPath(), ReportFolder));
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while creating report folder "
//                    + new File(getSamplerFolderPath(), ReportFolder));
//        }
//
//        logln(getClass().toString());
//        startTime = System.currentTimeMillis();
//
//        for (iter = 0; iter < MAX_ITER; iter++) {
//            numTokensChange = 0;
//
//            // sample topic assignments
//            long eTime = sampleZs(REMOVE, ADD, REMOVE, ADD);
//
//            if (verbose && iter % REP_INTERVAL == 0) {
//                double loglikelihood = this.getLogLikelihood();
//                logLikelihoods.add(loglikelihood);
//                if (iter < BURN_IN) {
//                    logln("--- Burning in. Iter " + iter
//                            + "\t llh = " + loglikelihood
//                            + "\t time = " + eTime
//                            + "\n" + getCurrentState());
//                } else {
//                    logln("--- Sampling. Iter " + iter
//                            + "\t llh = " + loglikelihood
//                            + "\t time = " + eTime
//                            + "\n" + getCurrentState());
//                }
//            }
//
//            // parameter optimization
//            if (iter % LAG == 0 && iter >= BURN_IN) {
//                if (paramOptimized) { // slice sampling
//                    sliceSample();
//                    ArrayList<Double> sparams = new ArrayList<Double>();
//                    for (double param : this.hyperparams) {
//                        sparams.add(param);
//                    }
//                    this.sampledParams.add(sparams);
//
//                    if (verbose) {
//                        for (double p : sparams) {
//                            System.out.println(p);
//                        }
//                    }
//                }
//            }
//
//            if (verbose && iter % REP_INTERVAL == 0) {
//                logln("--- --- # tokens: " + numTokens
//                        + ". # token changed: " + numTokensChange
//                        + ". change ratio: " + (double) numTokensChange / numTokens);
//            }
//
//            if (debug) {
//                validate("iter " + iter);
//            }
//
//            if (verbose && iter % REP_INTERVAL == 0) {
//                System.out.println();
//            }
//
//            // store model
//            if (report && iter > BURN_IN && iter % LAG == 0) {
//                outputState(new File(this.getReportFolderPath(),
//                        "iter-" + iter + ".zip").getAbsolutePath());
//            }
//        }
//
//        if (report) { // output final model
//            outputState(new File(this.getReportFolderPath(),
//                    "iter-" + iter + ".zip").getAbsolutePath());
//        }
//
//        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
//        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");
//
//        if (log && isLogging()) {
//            closeLogger();
//        }
//    }
//
//    @Override
//    public String getCurrentState() {
//        return this.getSamplerFolderPath();
//    }
//
//    /**
//     * Sample topic assignments for all tokens.
//     *
//     * @param removeFromModel
//     * @param addToModel
//     * @param removeFromData
//     * @param addToData
//     */
//    private long sampleZs(boolean removeFromModel, boolean addToModel,
//            boolean removeFromData, boolean addToData) {
//        long sTime = System.currentTimeMillis();
//        double totalBeta = hyperparams.get(BETA) * V;
//        for (int d = 0; d < D; d++) {
//            for (int n = 0; n < words[d].length; n++) {
//                if (removeFromData) {
//                    docLabels[d].decrement(z[d][n]);
//                }
//                if (removeFromModel) {
//                    labelWords[z[d][n]].decrement(words[d][n]);
//                }
//
//                double[] probs = new double[K];
//                for (int k = 0; k < K; k++) {
//                    double theta = docLabels[d].getCount(k) + docLabelPriors[d][k];
////                    double theta = docLabels[d].getCount(k) + hyperparams.get(ALPHA);
//                    double phi = (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
//                            / (labelWords[k].getCountSum() + totalBeta);
//                    probs[k] = theta * phi;
//                }
//                int sampledZ = SamplerUtils.scaleSample(probs);
//                if (z[d][n] != sampledZ) {
//                    numTokensChange++;
//                }
//
//                z[d][n] = sampledZ;
//
//                if (addToData) {
//                    docLabels[d].increment(z[d][n]);
//                }
//                if (addToModel) {
//                    labelWords[z[d][n]].increment(words[d][n]);
//                }
//            }
//        }
//        return System.currentTimeMillis() - sTime;
//    }
//
//    @Override
//    public double getLogLikelihood() {
//        double doc_topic = 0.0;
//        for (int d = 0; d < D; d++) {
//            doc_topic += this.docLabels[d].getLogLikelihood();
//        }
//        double topic_word = 0.0;
//        for (int k = 0; k < K; k++) {
//            topic_word += this.labelWords[k].getLogLikelihood();
//        }
//
//        double llh = doc_topic + topic_word;
//        if (verbose && iter % REP_INTERVAL == 0) {
//            logln(">>> topic-word: " + MiscUtils.formatDouble(topic_word)
//                    + "\tdoc-topic: " + MiscUtils.formatDouble(doc_topic)
//                    + "\tllh: " + MiscUtils.formatDouble(llh));
//        }
//
//        return llh;
//    }
//
//    @Override
//    public double getLogLikelihood(ArrayList<Double> newParams) {
//        double val = 0.0;
//        return val;
//    }
//
//    @Override
//    public void updateHyperparameters(ArrayList<Double> newParams) {
//    }
//
//    @Override
//    public void validate(String msg) {
//        for (int kk = 0; kk < K; kk++) {
//            this.labelWords[kk].validate(msg);
//        }
//        for (int dd = 0; dd < D; dd++) {
//            this.docLabels[dd].validate(msg);
//        }
//    }
//
//    @Override
//    public void outputState(String filepath) {
//        if (verbose) {
//            logln("--- Outputing current state to " + filepath);
//        }
//
//        try {
//            // model
//            StringBuilder modelStr = new StringBuilder();
//            for (int k = 0; k < K; k++) {
//                modelStr.append(k).append("\n");
//                modelStr.append(DirMult.output(labelWords[k])).append("\n");
//            }
//
//            // assignments
//            StringBuilder assignStr = new StringBuilder();
//            for (int d = 0; d < D; d++) {
//                assignStr.append(d).append("\n");
//                assignStr.append(DirMult.output(docLabels[d])).append("\n");
//
//                for (int n = 0; n < words[d].length; n++) {
//                    assignStr.append(z[d][n]).append("\t");
//                }
//                assignStr.append("\n");
//            }
//
//            // output to a compressed file
//            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while outputing state to " + filepath);
//        }
//    }
//
//    @Override
//    public void inputState(String filepath) {
//        if (verbose) {
//            logln("--- Reading state from " + filepath);
//        }
//
//        try {
//            inputModel(filepath);
//
//            inputAssignments(filepath);
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while inputing state from " + filepath);
//        }
//
//        validate("Done reading state from " + filepath);
//    }
//
//    public void inputFinalModel() {
//        try {
//            inputModel(new File(this.getReportFolderPath(), "iter-" + MAX_ITER + ".zip").getAbsolutePath());
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while loading final predictor model");
//        }
//    }
//
//    public void inputModel(String zipFilepath) {
//        if (verbose) {
//            logln("--- Loading model from " + zipFilepath);
//        }
//        try {
//            this.initializeModelStructure();
//
//            String filename = IOUtils.removeExtension(new File(zipFilepath).getName());
//            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
//            for (int k = 0; k < K; k++) {
//                int topicIdx = Integer.parseInt(reader.readLine());
//                if (topicIdx != k) {
//                    throw new RuntimeException("Topic indices mismatch when loading model");
//                }
//                labelWords[k] = DirMult.input(reader.readLine());
//            }
//            reader.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while loading model from "
//                    + zipFilepath);
//        }
//    }
//
//    private void inputAssignments(String zipFilepath) throws Exception {
//        if (verbose) {
//            logln("--- --- Loading assignments from " + zipFilepath);
//        }
//
//        try {
//            // initialize
//            this.initializeDataStructure();
//
//            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
//            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
//            for (int d = 0; d < D; d++) {
//                int docIdx = Integer.parseInt(reader.readLine());
//                if (docIdx != d) {
//                    throw new RuntimeException("Indices mismatch when loading assignments");
//                }
//                docLabels[d] = DirMult.input(reader.readLine());
//
//                String[] sline = reader.readLine().split("\t");
//                for (int n = 0; n < words[d].length; n++) {
//                    z[d][n] = Integer.parseInt(sline[n]);
//                }
//            }
//            reader.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while inputing assignments from "
//                    + zipFilepath);
//        }
//    }
//
//    public void outputTopicTopWords(File file, int numTopWords) throws Exception {
//        if (this.wordVocab == null) {
//            throw new RuntimeException("The word vocab has not been assigned yet");
//        }
//
//        if (this.labelVocab == null) {
//            throw new RuntimeException("The topic vocab has not been assigned yet");
//        }
//
//        if (verbose) {
//            logln("Outputing per-topic top words to " + file);
//        }
//
//        BufferedWriter writer = IOUtils.getBufferedWriter(file);
//        for (int k = 0; k < K; k++) {
//            double[] distrs = labelWords[k].getDistribution();
//            String[] topWords = getTopWords(distrs, numTopWords);
//            writer.write("[" + k
//                    + ", " + labelVocab.get(k)
//                    + ", " + labelWords[k].getCountSum()
//                    + "]");
//            for (String topWord : topWords) {
//                writer.write("\t" + topWord);
//            }
//            writer.write("\n\n");
//        }
//        writer.close();
//    }
//
//    public void sampleNewDocuments(String stateFile,
//            int[][] newWords,
//            String outputResultFile) throws Exception {
//        if (verbose) {
//            System.out.println();
//            logln("Perform prediction using model from " + stateFile);
//            logln("--- Test burn-in: " + this.testBurnIn);
//            logln("--- Test max-iter: " + this.testMaxIter);
//            logln("--- Test sample-lag: " + this.testSampleLag);
//        }
//
//        // input model
//        inputModel(stateFile);
//
//        words = newWords;
//        labels = null; // for evaluation
//        D = words.length;
//
//        // initialize structure
//        initializeDataStructure();
//
//        if (verbose) {
//            logln("test data");
//            logln("--- V = " + V);
//            logln("--- D = " + D);
//            int docTopicCount = 0;
//            for (int d = 0; d < D; d++) {
//                docTopicCount += docLabels[d].getCountSum();
//            }
//            int topicWordCount = 0;
//            for (int k = 0; k < labelWords.length; k++) {
//                topicWordCount += labelWords[k].getCountSum();
//            }
//            logln("--- docTopics: " + docLabels.length + ". " + docTopicCount);
//            logln("--- topicWords: " + labelWords.length + ". " + topicWordCount);
//        }
//
//        // initialize assignments
//        sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
//
//        // sample an store predictions
//        double[][] predictedScores = new double[D][K];
//        int count = 0;
//        for (iter = 0; iter < testMaxIter; iter++) {
//            if (iter == 0) {
//                sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
//            } else {
//                sampleZs(!REMOVE, !ADD, REMOVE, ADD);
//            }
//
//            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
//                if (verbose) {
//                    logln("--- iter = " + iter + " / " + this.testMaxIter);
//                }
//                for (int dd = 0; dd < D; dd++) {
//                    for (int ll = 0; ll < K; ll++) {
//                        double predProb = (docLabels[dd].getCount(ll) + docLabelPriors[dd][ll])
//                                / (docLabels[dd].getCountSum() + docLabelPriorSums[dd]);
//                        predictedScores[dd][ll] += predProb;
//                    }
//                }
//                count++;
//            }
//        }
//
//        // output result during test time
//        if (verbose) {
//            logln("--- Outputing result to " + outputResultFile);
//        }
//        for (int dd = 0; dd < D; dd++) {
//            for (int ll = 0; ll < K; ll++) {
//                predictedScores[dd][ll] /= count;
//            }
//        }
//        PredictionUtils.outputSingleModelClassifications(
//                new File(outputResultFile), predictedScores);
//    }
//
//    public double computePerplexity(String stateFile,
//            int[][] newWords, int[][] newLabels) {
//        if (verbose) {
//            System.out.println();
//            logln("Computing perplexity using model from " + stateFile);
//            logln("--- Test burn-in: " + this.testBurnIn);
//            logln("--- Test max-iter: " + this.testMaxIter);
//            logln("--- Test sample-lag: " + this.testSampleLag);
//        }
//
//        // input model
//        inputModel(stateFile);
//
//        words = newWords;
//        labels = newLabels;
//        D = words.length;
//        numTokens = 0;
//        for (int d = 0; d < D; d++) {
//            numTokens += words[d].length;
//        }
//
//        // initialize structure
//        initializeDataStructure();
//
//        ArrayList<Double> perplexities = new ArrayList<Double>();
//        if (verbose) {
//            logln("--- Sampling on test data ...");
//        }
//        for (iter = 0; iter < testMaxIter; iter++) {
//            if (iter % testSampleLag == 0) {
//                logln("--- --- iter " + iter + "/" + testMaxIter
//                        + " @ thread " + Thread.currentThread().getId()
//                        + " " + getSamplerFolderPath());
//            }
//
//            if (iter == 0) {
//                sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
//            } else {
//                sampleZs(!REMOVE, !ADD, REMOVE, ADD);
//            }
//
//            // compute perplexity
//            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
//                perplexities.add(computePerplexity());
//            }
//        }
//        double avgPerplexity = StatisticsUtils.mean(perplexities);
//        return avgPerplexity;
//    }
//
//    /**
//     * Compute perplexity.
//     */
//    private double computePerplexity() {
//        double totalBeta = hyperparams.get(BETA) * V;
//        double totalLogprob = 0.0;
//        for (int d = 0; d < D; d++) {
//            for (int n = 0; n < words[d].length; n++) {
//                double val = 0.0;
//                for (int ll = 0; ll < K; ll++) {
//                    double theta = (docLabels[d].getCount(ll) + docLabelPriors[d][ll])
//                            / (docLabels[d].getCountSum() + docLabelPriorSums[d]);
//                    double phi = (labelWords[ll].getCount(words[d][n]) + hyperparams.get(BETA))
//                            / (labelWords[ll].getCountSum() + totalBeta);
//                    val += theta * phi;
//                }
//                totalLogprob += Math.log(val);
//            }
//        }
//        double perplexity = Math.exp(-totalLogprob / numTokens);
//        return perplexity;
//    }
//
    public static void parallelPerplexity(int[][] newWords,
            int[][] newLabels,
            File iterPerplexityFolder,
            File resultFolder,
            PriorLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPerplexityFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPerplexityFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                PriorLDAPerplexityRunner runner = new PriorLDAPerplexityRunner(sampler,
                        newWords, newLabels,
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

    public double[] predictNewDocument(int[] newWords) {
        throw new RuntimeException("To be removed");
    }
//}
//
//class PriorLDATestRunner implements Runnable {
//
//    PriorLDA sampler;
//    int[][] newWords;
//    String stateFile;
//    String outputFile;
//
//    public PriorLDATestRunner(PriorLDA sampler,
//            int[][] newWords,
//            String stateFile,
//            String outputFile) {
//        this.sampler = sampler;
//        this.newWords = newWords;
//        this.stateFile = stateFile;
//        this.outputFile = outputFile;
//    }
//
//    @Override
//    public void run() {
//        PriorLDA testSampler = new PriorLDA();
//        testSampler.setVerbose(true);
//        testSampler.setDebug(false);
//        testSampler.setLog(false);
//        testSampler.setReport(false);
//        testSampler.configure(sampler);
//        testSampler.setTestConfigurations(sampler.getBurnIn(),
//                sampler.getMaxIters(), sampler.getSampleLag());
//
//        try {
//            testSampler.sampleNewDocuments(stateFile, newWords, outputFile);
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException();
//        }
//    }
}

class PriorLDAPerplexityRunner implements Runnable {

    PriorLDA sampler;
    int[][] newWords;
    int[][] newLabels;
    String stateFile;
    String outputFile;

    public PriorLDAPerplexityRunner(PriorLDA sampler,
            int[][] newWords,
            int[][] newLabels,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newLabels = newLabels;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        PriorLDA testSampler = new PriorLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            double perplexity = testSampler.computePerplexity(stateFile, newWords, newLabels);
            IOUtils.outputPerplexity(outputFile, perplexity);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
