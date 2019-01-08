package sampler.labeled;

import core.AbstractSampler;
import data.LabelTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;
import util.PredictionUtils;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.MimnoTopicCoherence;

/**
 * This is an implementation of a Gibbs sampler for Labeled LDA (Ramage et. al.
 * EMNLP 09).
 *
 * Each document is associated with a set of labels.
 *
 * @author vietan
 */
    public class LabeledLDA extends AbstractSampler implements Serializable {

    private static final long serialVersionUID = 1123581321L;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected ArrayList<Integer> docIndices;
    protected int[][] words; // [D] x [N_d]
    protected int[][] labels; // [D] x [T_d] 
    protected int L;
    protected int V;
    protected int D;
    private DirMult[] docLabels;
    private DirMult[] labelWords;
    private int[][] z;
    private ArrayList<String> labelVocab;
    private int numTokensChange;

    public LabeledLDA() {
        this.basename = "L-LDA";
    }

    public LabeledLDA(String basename) {
        this.basename = basename;
    }

    public void setLabelVocab(ArrayList<String> labelVocab) {
        this.labelVocab = labelVocab;
    }

    public int[][] getZ() {
        return this.z;
    }

    public void configure(LabeledLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.L,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL,
                sampler.report);
    }
    
    public void configure(String folder,
            int V, int L,
            double alpha,
            double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval) {
        configure(folder,
                V, L,
                alpha,
                beta,
                initState,
                paramOpt,
                burnin, maxiter, samplelag, repInterval, false);
    }

    public void configure(String folder,
            int V, int L,
            double alpha,
            double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInterval, boolean reportState) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.L = L;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInterval;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();
        this.report = reportState;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- label vocab:\t" + L);
            logln("--- word vocab:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(alpha));
            logln("--- beta:\t" + MiscUtils.formatDouble(beta));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- report model state:\t" + report);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_K-").append(L)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_a-").append(MiscUtils.formatDouble(hyperparams.get(ALPHA)))
                .append("_b-").append(MiscUtils.formatDouble(hyperparams.get(BETA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public DirMult[] getTopicWordDistributions() {
        return this.labelWords;
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath();
    }

    /**
     * Set training data.
     *
     * @param docIndices Indices of selected documents
     * @param words Document words
     * @param labels Document labels
     */
    public void train(ArrayList<Integer> docIndices, int[][] words, int[][] labels) {
        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.labels = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = words[dd];
            this.labels[ii] = labels[dd];
        }

        this.numTokens = 0;
        int numLabels = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
            numLabels += labels[d].length;
        }

        if (verbose) {
            logln("--- # all documents:\t" + words.length);
            logln("--- # selected documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            logln("--- # label instances:\t" + numLabels);
        }
    }

    public void test(int[][] ws) {
        this.words = ws;
        this.labels = null;
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
    }

    private void initializeModelStructure() {
        if (verbose) {
            logln("--- Initializing model structure ...");
        }

        labelWords = new DirMult[L];
        for (int ll = 0; ll < L; ll++) {
            labelWords[ll] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    private void initializeDataStructure() {
        if (verbose) {
            logln("--- Initializing data structure ...");
        }

        docLabels = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docLabels[d] = new DirMult(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }
    }

    private void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                int[] dls = labels[d];
                if (dls.length > 0) {
                    z[d][n] = dls[rand.nextInt(dls.length)];
                } else {
                    z[d][n] = rand.nextInt(L);
                }
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

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        if (report) {
            if (this.wordVocab == null) {
                throw new RuntimeException("The word vocab has not been assigned yet");
            }

            if (this.labelVocab == null) {
                throw new RuntimeException("The label vocab has not been assigned yet");
            }
            IOUtils.createFolder(reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChange = 0;

            sampleZs(REMOVE, ADD, REMOVE, ADD);

            if (debug) {
                validate("iter " + iter);
            }

            if (isReporting()) {
                double loglikelihood = this.getLogLikelihood();
                String str = "Iter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\t tokens changed: " + numTokensChange
                        + " (" + MiscUtils.formatDouble((double) numTokensChange / numTokens) + ")"
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str + "\n");
                } else {
                    logln("--- Sampling. " + str + "\n");
                }
                System.out.println();
            }

            if (paramOptimized && iter % LAG == 0 && iter >= BURN_IN) {
                this.updateHyperparameters();
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, getIteratedStateFile()), true, true);
                outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 20);
            }
        }

        if (report) {
            // outputState(new File(reportFolderPath, getIteratedStateFile()), true, false);
            outputState(new File(reportFolderPath, getIteratedStateFile()), true, false);
            outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 20);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Sample topic assignments for all tokens. This is a little bit faster than
     * calling sampleZ repeatedly.
     *
     * @param removeFromModel Whether the current assignment should be removed
     * from the model (i.e., label-word distributions)
     * @param addToModel Whether the new assignment should be added to the model
     * @param removeFromData Whether the current assignment should be removed
     * from the data (i.e., doc-label distributions)
     * @param addToData Whether the new assignment should be added to the data
     */
    public void sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData) {
        double totalBeta = V * hyperparams.get(BETA);
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    labelWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docLabels[d].decrement(z[d][n]);
                }

                int sampledZ;
                if (labels != null && labels[d].length > 0) {
                    double[] probs = new double[labels[d].length];
                    for (int ii = 0; ii < labels[d].length; ii++) {
                        int k = labels[d][ii];
                        probs[ii] = (docLabels[d].getCount(k) + hyperparams.get(ALPHA))
                                * (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                / (labelWords[k].getCountSum() + totalBeta);
                    }
                    sampledZ = labels[d][SamplerUtils.scaleSample(probs)];
                } else { // for documents without labels and for test documents
                    double[] probs = new double[L];
                    for (int ll = 0; ll < L; ll++) {
                        probs[ll] = (docLabels[d].getCount(ll) + hyperparams.get(ALPHA))
                                * (labelWords[ll].getCount(words[d][n]) + hyperparams.get(BETA))
                                / (labelWords[ll].getCountSum() + totalBeta);
                    }
                    sampledZ = SamplerUtils.scaleSample(probs);
                }

                if (sampledZ != z[d][n]) {
                    numTokensChange++;
                }
                z[d][n] = sampledZ;

                if (addToModel) {
                    labelWords[z[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docLabels[d].increment(z[d][n]);
                }
            }
        }
    }

    public double[] predictNewDocument(int[] newDoc) throws Exception {
        // initialize assignments
        DirMult docTopic = new DirMult(L, hyperparams.get(ALPHA) * L, 1.0 / L);
        int[] newZ = new int[newDoc.length];
        for (int n = 0; n < newZ.length; n++) {
            newZ[n] = rand.nextInt(L);
            docTopic.increment(newZ[n]);
        }
        // sample
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int n = 0; n < newZ.length; n++) {
                // decrement
                docTopic.decrement(newZ[n]);

                // sample
                double[] logprobs = new double[L];
                for (int l = 0; l < L; l++) {
                    logprobs[l] = docTopic.getLogLikelihood(l)
                            + labelWords[l].getLogLikelihood(newDoc[n]);
                }
                newZ[n] = SamplerUtils.logMaxRescaleSample(logprobs);

                // increment
                docTopic.increment(newZ[n]);
            }
        }
        return docTopic.getDistribution();
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += docLabels[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int l = 0; l < L; l++) {
            topicWordLlh += labelWords[l].getLogLikelihood();
        }

        double llh = docTopicLlh + topicWordLlh;
        if (verbose) {
            logln(">>> doc-topic: " + MiscUtils.formatDouble(docTopicLlh)
                    + "\ttopic-word: " + MiscUtils.formatDouble(topicWordLlh)
                    + "\tllh: " + MiscUtils.formatDouble(llh));
        }
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += docLabels[d].getLogLikelihood(newParams.get(ALPHA) * L, 1.0 / L);
        }
        for (int l = 0; l < L; l++) {
            llh += labelWords[l].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.docLabels[d].setConcentration(this.hyperparams.get(ALPHA) * L);
        }
        for (int l = 0; l < L; l++) {
            this.labelWords[l].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    @Override
    public void validate(String msg) {
        validateData(msg);
        validateModel(msg);
    }

    private void validateModel(String msg) {
        for (int l = 0; l < L; l++) {
            this.labelWords[l].validate(msg);
        }
    }

    private void validateData(String msg) {
        for (int d = 0; d < D; d++) {
            this.docLabels[d].validate(msg);
        }

        int total = 0;
        for (int d = 0; d < D; d++) {
            total += docLabels[d].getCountSum();
        }
        if (total != numTokens) {
            throw new RuntimeException("Token counts mismatch. "
                    + total + " vs. " + numTokens);
        }
    }

    /**
     * Output current state including the learned model and the current
     * assignments.
     *
     * @param filepath Output file
     */
    @Override
    public void outputState(String filepath) {
        outputState(filepath, true, true);
    }

    /**
     * Output current state.
     *
     * @param filepath Output file
     * @param outputModel Whether to output the model
     * @param outputData Whether to output the assignments
     */
    public void outputState(File filepath, boolean outputModel, boolean outputData) {
        this.outputState(filepath.getAbsolutePath(), outputModel, outputData);
    }

    /**
     * Output current state.
     *
     * @param filepath Output file
     * @param outputModel Whether to output the model
     * @param outputData Whether to output the assignments
     */
    public void outputState(String filepath, boolean outputModel, boolean outputData) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
            logln("--- --- Outputing model? " + outputModel);
            logln("--- --- Outputing assignments? " + outputData);
        }

        try {
            // model
            String modelStr = null;
            if (outputModel) {
                StringBuilder modelStrBuilder = new StringBuilder();
                for (int k = 0; k < L; k++) {
                    modelStrBuilder.append(k).append("\n");
                    modelStrBuilder.append(DirMult.output(labelWords[k])).append("\n");
                }
                modelStr = modelStrBuilder.toString();
            }

            // data
            String assignStr = null;
            if (outputData) {
                StringBuilder assignStrBuilder = new StringBuilder();
                for (int d = 0; d < D; d++) {
                    assignStrBuilder.append(d).append("\n");
                    assignStrBuilder.append(DirMult.output(docLabels[d])).append("\n");

                    for (int n = 0; n < words[d].length; n++) {
                        assignStrBuilder.append(z[d][n]).append("\t");
                    }
                    assignStrBuilder.append("\n");
                }
                assignStr = assignStrBuilder.toString();
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr, assignStr);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing state to "
                    + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        inputState(filepath, true, true);
    }

    /**
     * Input model state.
     *
     * @param filepath Output file
     * @param inputModel Whether to input the model
     * @param inputData Whether to input the assignments
     */
    public void inputState(File filepath, boolean inputModel, boolean inputData) {
        this.inputState(filepath.getAbsolutePath(), inputModel, inputData);
    }

    /**
     * Input model state.
     *
     * @param filepath Output file
     * @param inputModel Whether to input the model
     * @param inputData Whether to input the assignments
     */
    public void inputState(String filepath, boolean inputModel, boolean inputData) {
        if (verbose) {
            logln("--- Inputing state to " + filepath);
            logln("--- --- Inputing model? " + inputModel);
            logln("--- --- Inputing assignments? " + inputData);
        }
        try {
            if (inputModel) {
                inputModel(filepath);
            }
            if (inputData) {
                inputAssignments(filepath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing from " + filepath);
        }
    }

    /**
     * Input learned model.
     *
     * @param zipFilepath Input file
     */
    private void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            // initialize
            this.initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < L; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                labelWords[k] = DirMult.input(reader.readLine());
            }
            reader.close();
            validateModel("Loaded from " + zipFilepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    /**
     * Input assignments.
     *
     * @param zipFilepath Input file
     */
    private void inputAssignments(String zipFilepath) {
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

            validateData("Loaded from " + zipFilepath);
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

        if (this.labelVocab == null) {
            throw new RuntimeException("The label vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        try {
            // get label frequencies
            SparseCount labelFreqs = new SparseCount();
            for (int[] label : labels) {
                for (int ll : label) {
                    labelFreqs.increment(ll);
                }
            }

            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int kk = 0; kk < L; kk++) {
                double[] distrs = labelWords[kk].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + kk
                        + ", " + labelVocab.get(kk)
                        + ", " + labelFreqs.getCount(kk)
                        + ", " + labelWords[kk].getCountSum()
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    public void outputTopicCoherence(File file,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing topic coherence to file " + file);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(file);
        for (int k = 0; k < L; k++) {
            double[] distribution = this.labelWords[k].getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(k
                    + "\t" + labelWords[k].getCountSum()
                    + "\t" + MiscUtils.formatDouble(score));
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    /**
     * Return the feature vector extracted from training data.
     *
     * Indices start at 1.
     *
     * @return
     */
    public SparseVector[] getTrainingFeatureVectors() {
        SparseVector[] featVecs = new SparseVector[D];
        for (int d = 0; d < D; d++) {
            featVecs[d] = new SparseVector();
        }
        double[][] sumDists = new double[D][L];

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            int numModels = 0;
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                inputState(new File(reportFolder, filename).getAbsolutePath());
                for (int d = 0; d < D; d++) {
                    double[] docDist = docLabels[d].getDistribution();
                    for (int ll = 0; ll < L; ll++) {
                        sumDists[d][ll] += docDist[ll];
                    }
                }
                numModels++;
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int ll = 0; ll < L; ll++) {
                    double val = sumDists[d][ll] / numModels;
                    featVecs[d].set(ll + 1, val); // index start at 1
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting training feature vectors.");
        }
        return featVecs;
    }

    public SparseVector[] getTestFeatureVectors(File iterPredFolder) {
        SparseVector[] featVecs = new SparseVector[D];
        for (int d = 0; d < D; d++) {
            featVecs[d] = new SparseVector();
        }
        double[][] sumDists = new double[D][L];

        String[] filenames = iterPredFolder.list();
        try {
            for (String filename : filenames) {
                double[][] singlePreds = PredictionUtils.inputSingleModelClassifications(
                        new File(iterPredFolder, filename));
                for (int d = 0; d < D; d++) {
                    for (int ll = 0; ll < L; ll++) {
                        sumDists[d][ll] += singlePreds[d][ll];
                    }
                }
            }

            // average
            for (int d = 0; d < D; d++) {
                for (int ll = 0; ll < L; ll++) {
                    double val = sumDists[d][ll] / filenames.length;
                    featVecs[d].set(ll + 1, val);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while getting test feature vectors.");
        }

        return featVecs;
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
        double[][] avgTopics = new double[L][V];
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file.getAbsolutePath() + ".iter");
            writer.write("Iteration");
            for (int k = 0; k < L; k++) {
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
                double[][] pointTopics = new double[L][V];

                writer.write(filename);
                for (int k = 0; k < L; k++) {
                    pointTopics[k] = labelWords[k].getDistribution();
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
            for (int k = 0; k < L; k++) {
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
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
        return avgTopics;
    }

    public double[][] hack(int[][] newWords) {
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        test(newWords);
        double[][] finalPredictions = new double[D][L];
        int count = 0;
        try {
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                inputModel(new File(reportFolder, filename).getAbsolutePath());
                SparseVector[] topics = new SparseVector[L];
                for (int ll = 0; ll < L; ll++) {
                    topics[ll] = new SparseVector();
                    for (int v : labelWords[ll].getSparseCounts().getIndices()) {
                        double val = (double) labelWords[ll].getCount(v) / labelWords[ll].getCountSum();
                        topics[ll].set(v, val);
                    }
                }

                int ss = MiscUtils.getRoundStepSize(D, 10);
                for (int d = 0; d < D; d++) {
                    if (d % ss == 0) {
                        logln("--- Predicting d = " + d + " / " + D);
                    }
                    SparseCount docTokenCount = new SparseCount();
                    for (int n = 0; n < words[d].length; n++) {
                        docTokenCount.increment(words[d][n]);
                    }

                    SparseVector doc = new SparseVector();
                    for (int v : docTokenCount.getIndices()) {
                        double val = (double) docTokenCount.getCount(v) / words[d].length;
                        doc.set(v, val);
                    }

                    double[] docScores = new double[L];
                    for (int ll = 0; ll < L; ll++) {
                        docScores[ll] = doc.cosineSimilarity(topics[ll]);
                    }

                    for (int ll = 0; ll < L; ll++) {
                        finalPredictions[d][ll] += docScores[ll];
                    }
                }

                count++;
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }

        for (int d = 0; d < D; d++) {
            for (int ll = 0; ll < L; ll++) {
                finalPredictions[d][ll] /= count;
            }
        }
        return finalPredictions;
    }

    public void test(int[][] newWords, File iterPredFolder) {
        if (verbose) {
            logln("Test sampling ...");
        }
        this.setTestConfigurations(BURN_IN, MAX_ITER, LAG);
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        try {
            IOUtils.createFolder(iterPredFolder);
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                sampleNewDocuments(
                        new File(reportFolder, filename).getAbsolutePath(),
                        newWords,
                        partialResultFile.getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    public void computePerplexities(int[][] newWords, int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            File outputFile) {
        if (verbose) {
            logln("Computing perplexities & outputing to " + outputFile);
        }
        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Iteration\tPerplexity\n");
            ArrayList<Double> pps = new ArrayList<Double>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                double pp = computePerplexity(new File(reportFolder, filename).getAbsolutePath(),
                        newWords, newLabels, trainIndices, testIndices);
                pps.add(pp);
                writer.write(filename + "\t" + pp + "\n");
            }
            writer.write("Average\t" + StatUtils.mean(pps) + "\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
    }

    public void sampleZ(int d, int i, int n,
            boolean removeFromData, boolean addToData) {
        double totalBeta = V * hyperparams.get(BETA);
        if (removeFromData) {
            docLabels[d].decrement(z[d][i]);
        }

        int sampledZ;
        if (labels != null && labels[d].length > 0) {
            double[] probs = new double[labels[d].length];
            for (int ii = 0; ii < labels[d].length; ii++) {
                int k = labels[d][ii];
                probs[ii] = (docLabels[d].getCount(k) + hyperparams.get(ALPHA) * labels[d].length / L)
                        * (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                        / (labelWords[k].getCountSum() + totalBeta);
            }
            sampledZ = labels[d][SamplerUtils.scaleSample(probs)];
        } else { // for documents without labels and for test documents
            double[] probs = new double[L];
            for (int ll = 0; ll < L; ll++) {
                probs[ll] = (docLabels[d].getCount(ll) + hyperparams.get(ALPHA))
                        * (labelWords[ll].getCount(words[d][n]) + hyperparams.get(BETA))
                        / (labelWords[ll].getCountSum() + totalBeta);
            }
            sampledZ = SamplerUtils.scaleSample(probs);
        }

        if (sampledZ != z[d][i]) {
            numTokensChange++;
        }
        z[d][i] = sampledZ;

        if (addToData) {
            docLabels[d].increment(z[d][i]);
        }
    }

    /**
     * Sampling to compute perplexity using a learned model stored in a state
     * file.
     *
     * TODO: this could be merged with sampleNewDocuments.
     *
     * @param stateFile The state file storing the learned model
     * @param newWords Words of test documents
     * @param newLabels Labels of test documents
     * @param trainIndices
     * @param testIndices
     * @return
     */
    public double computePerplexity(String stateFile,
            int[][] newWords, int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices) {
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

        docLabels = new DirMult[D];
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            docLabels[d] = new DirMult(L, hyperparams.get(ALPHA) * L, 1.0 / L);
            z[d] = new int[trainIndices[d].size()];
        }

        ArrayList<Double> perplexities = new ArrayList<Double>();
        if (verbose) {
            logln("--- Sampling on test data ...");
        }
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter % testSampleLag == 0) {
                logln("--- --- iter " + iter + "/" + testMaxIter
                        + " @ thread " + Thread.currentThread().getId()
                        + "\n" + getSamplerFolderPath());
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
                double docLogProb = 0.0;
                for (int n : testIndices[d]) {
                    double val = 0.0;
                    if (labels[d].length > 0) {
                        for (int ii = 0; ii < labels[d].length; ii++) {
                            int k = labels[d][ii];
                            double theta = (docLabels[d].getCount(k) + hyperparams.get(ALPHA))
                                    / (docLabels[d].getCountSum() + hyperparams.get(ALPHA) * labels[d].length);
                            double phi = (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (labelWords[k].getCountSum() + totalBeta);
                            val += theta * phi;
                        }
                    } else { // for documents without labels and for test documents
                        for (int k = 0; k < L; k++) {
                            double theta = (docLabels[d].getCount(k) + hyperparams.get(ALPHA))
                                    / (docLabels[d].getCountSum() + hyperparams.get(ALPHA) * L);
                            double phi = (labelWords[k].getCount(words[d][n]) + hyperparams.get(BETA))
                                    / (labelWords[k].getCountSum() + totalBeta);
                            val += theta * phi;
                        }
                    }
                    docLogProb += Math.log(val);
                }
                totalLogprob += docLogProb;
                writer.write(d
                        + "\t" + words[d].length
                        + "\t" + labels[d].length
                        + "\t" + testIndices[d].size()
                        + "\t" + docLogProb + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException();
        }

        double perplexity = Math.exp(-totalLogprob / numTestTokens);
        return perplexity;
    }

    public void sampleNewDocuments(String stateFile,
            int[][] newWords,
            String outputResultFile) throws Exception {
        if (verbose) {
            System.out.println();
            logln("Perform prediction using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile);

        // test data
        test(newWords);

        // initialize structure
        initializeDataStructure();

        if (verbose) {
            logln("test data");
            logln("--- V = " + V);
            int docTopicCount = 0;
            for (int d = 0; d < D; d++) {
                docTopicCount += docLabels[d].getCountSum();
            }
            int topicWordCount = 0;
            for (DirMult label_word : labelWords) {
                topicWordCount += label_word.getCountSum();
            }
            logln("--- docTopics: " + docLabels.length + ". " + docTopicCount);
            logln("--- topicWords: " + labelWords.length + ". " + topicWordCount);
        }

        // initialize assignments
        sampleZs(!REMOVE, !ADD, !REMOVE, ADD);

        // sample an store predictions
        double[][] predictedScores = new double[D][L];
        int count = 0;
        for (iter = 0; iter < testMaxIter; iter++) {
            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD);
            }

            if (iter >= this.testBurnIn && iter % this.testSampleLag == 0) {
                if (verbose) {
                    logln("--- iter = " + iter + " / " + this.testMaxIter);
                }
                for (int dd = 0; dd < D; dd++) {
                    double[] predProbs = docLabels[dd].getDistribution();
                    for (int ll = 0; ll < L; ll++) {
                        predictedScores[dd][ll] += predProbs[ll];
                    }
                }
                count++;
            }
        }

        // output result during test time
        if (verbose) {
            logln("--- Outputing result to " + outputResultFile);
        }
        for (int dd = 0; dd < D; dd++) {
            for (int ll = 0; ll < L; ll++) {
                predictedScores[dd][ll] /= count;
            }
        }
        PredictionUtils.outputSingleModelClassifications(
                new File(outputResultFile), predictedScores);
    }

    public static void parallelPerplexity(int[][] newWords,
            int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            File iterPerplexityFolder,
            File resultFolder,
            LabeledLDA sampler) {
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
                LabeledLDAPerplexityRunner runner = new LabeledLDAPerplexityRunner(sampler,
                        newWords, newLabels, trainIndices, testIndices,
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

    public static void parallelTest(int[][] newWords, File iterPredFolder, LabeledLDA sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                LabeledLDATestRunner runner = new LabeledLDATestRunner(sampler,
                        newWords, stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }

    public static void main(String[] args) {
        run(args);
    }

    public static void run(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("dataset", "Dataset");
            addOption("data-folder", "Processed data folder");
            addOption("format-folder", "Folder holding formatted data");
            addOption("format-file", "Formatted file name");
            addOption("output", "Output folder");

            // sampling configurations
            addSamplingOptions();

            // model parameters
            addOption("K", "Number of topics");
            addOption("numTopwords", "Number of top words per topic");
            addOption("min-label-freq", "Minimum label frequency");

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
            options.addOption(OptionBuilder.withLongOpt("report-state")
                    .withDescription("Report model state during iteration")
                    .hasArg(false)
                    .create());

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            runModel();
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + LabeledLDA.class.getName() + " -help";
    }

    private static void runModel() throws Exception {
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "112");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "L:/Dropbox/github/data");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "L:/Dropbox/github/data/112/format-label/model");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format-label");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        int minLabelFreq = CLIUtils.getIntegerArgument(cmd, "min-label-freq", 300);

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);

        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");
        boolean reportState = cmd.hasOption("report-state");

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        LabelTextDataset data = new LabelTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());
        data.filterLabelsByFrequency(minLabelFreq);
        data.prepareTopicCoherence(numTopWords);

        int V = data.getWordVocab().size();
        int K = data.getLabelVocab().size();
        boolean paramOpt = cmd.hasOption("paramOpt");
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("\tRunning Labeled-LDA sampler ...");
        }
        LabeledLDA sampler = new LabeledLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(data.getWordVocab());
        sampler.setLabelVocab(data.getLabelVocab());

        sampler.configure(outputFolder,
                V, K, alpha, beta, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval, reportState);
        sampler.train(null, data.getWords(), data.getLabels());
        File lldaFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(lldaFolder);
        sampler.sample();
        sampler.outputTopicTopWords(
                new File(lldaFolder, TopWordFile),
                numTopWords);
        sampler.outputTopicCoherence(
                new File(lldaFolder, TopicCoherenceFile),
                data.getTopicCoherence());
    }
}

class LabeledLDAPerplexityRunner implements Runnable {

    LabeledLDA sampler;
    int[][] newWords;
    int[][] newLabels;
    ArrayList<Integer>[] trainIndices;
    ArrayList<Integer>[] testIndices;
    String stateFile;
    String outputFile;

    public LabeledLDAPerplexityRunner(LabeledLDA sampler,
            int[][] newWords,
            int[][] newLabels,
            ArrayList<Integer>[] trainIndices,
            ArrayList<Integer>[] testIndices,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.newLabels = newLabels;
        this.trainIndices = trainIndices;
        this.testIndices = testIndices;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        LabeledLDA testSampler = new LabeledLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            double perplexity = testSampler.computePerplexity(stateFile, newWords,
                    newLabels, trainIndices, testIndices);
            IOUtils.outputPerplexity(outputFile, perplexity);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}

class LabeledLDATestRunner implements Runnable {

    LabeledLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public LabeledLDATestRunner(LabeledLDA sampler,
            int[][] newWords,
            String stateFile,
            String outputFile) {
        this.sampler = sampler;
        this.newWords = newWords;
        this.stateFile = stateFile;
        this.outputFile = outputFile;
    }

    @Override
    public void run() {
        LabeledLDA testSampler = new LabeledLDA();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
