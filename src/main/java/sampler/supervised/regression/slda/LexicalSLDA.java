package sampler.supervised.regression.slda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL2Norm;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class LexicalSLDA extends SLDA {

    public void configure(LexicalSLDA sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.hyperparams.get(RHO),
                sampler.hyperparams.get(MU),
                sampler.hyperparams.get(SIGMA),
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    @Override
    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_lex-sLDA")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_s-").append(formatter.format(hyperparams.get(SIGMA)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    protected void initializeModelStructure() {
        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
        regParams = new double[K + V];
    }

    @Override
    protected void initializeDataStructure() {
        super.initializeDataStructure();
        designMatrix = new double[D][K + V];
        for (int d = 0; d < D; d++) {
            double denom = 1.0 / words[d].length;
            for (int n = 0; n < words[d].length; n++) {
                designMatrix[d][K + words[d][n]] += denom;
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

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            // store llh after every iteration
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                String str = "Iter " + iter + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter <= BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    sampleZ(d, n, REMOVE, ADD, REMOVE, ADD, OBSERVED);
                }
            }

            // update the regression parameters
            int step = (int) Math.log(iter + 1) + 1;
            if (iter % step == 0) {
                updateTopicRegressionParameters();
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

            if (verbose && iter % REP_INTERVAL == 0) {
                evaluateRegressPrediction(responses, docRegressMeans);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n\n");
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
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
    protected void sampleZ(int d, int n,
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (removeFromModel) {
            topicWords[z[d][n]].decrement(words[d][n]);
        }
        if (removeFromData) {
            docTopics[d].decrement(z[d][n]);
            docRegressMeans[d] -= regParams[z[d][n]] * K / (words[d].length * V);
        }

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] = docTopics[d].getLogLikelihood(k)
                    + topicWords[k].getLogLikelihood(words[d][n]);
            if (observe) {
                double mean = docRegressMeans[d] + regParams[k] * K / (words[d].length * V);
                logprobs[k] += StatUtils.logNormalProbability(responses[d],
                        mean, Math.sqrt(hyperparams.get(RHO)));
            }
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);

        if (z[d][n] != sampledZ) {
            numTokensChanged++; // for debugging
        }
        // update
        z[d][n] = sampledZ;

        if (addToModel) {
            topicWords[z[d][n]].increment(words[d][n]);
        }
        if (addToData) {
            docTopics[d].increment(z[d][n]);
            docRegressMeans[d] += regParams[z[d][n]] * K / (words[d].length * V);
        }
    }

    @Override
    protected void updateTopicRegressionParameters() {
        for (int d = 0; d < D; d++) {
            double[] empDist = docTopics[d].getEmpiricalDistribution();
            for (int k = 0; k < K; k++) {
                designMatrix[d][k] = empDist[k] * K / V;
            }
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
        mlr.setRho(hyperparams.get(RHO));
        double[] means = new double[V + K];
        double[] sigmas = new double[V + K];
        for (int v = 0; v < V + K; v++) {
            means[v] = hyperparams.get(MU);
            sigmas[v] = hyperparams.get(SIGMA);
        }
        mlr.setMeans(means);
        mlr.setSigmas(sigmas);
        double[] params = mlr.solve();
        System.arraycopy(params, 0, regParams, 0, V + K);

        // update current predictions
        updatePredictionValues();
    }

    @Override
    protected void updatePredictionValues() {
        this.docRegressMeans = new double[D];
        for (int d = 0; d < D; d++) {
            for (int ii = 0; ii < K + V; ii++) {
                docRegressMeans[d] += regParams[ii] * designMatrix[d][ii];
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood();
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood();
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            responseLlh += StatUtils.logNormalProbability(
                    responses[d],
                    docRegressMeans[d],
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int ii = 0; ii < K + V; ii++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[ii],
                    hyperparams.get(MU),
                    Math.sqrt(hyperparams.get(SIGMA)));
        }

        if (verbose && iter % REP_INTERVAL == 0) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". response: " + MiscUtils.formatDouble(responseLlh)
                    + ". regParam: " + MiscUtils.formatDouble(regParamLlh));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }

        double responseLlh = 0.0;
        for (int d = 0; d < D; d++) {
            responseLlh += StatUtils.logNormalProbability(
                    responses[d],
                    docRegressMeans[d],
                    Math.sqrt(hyperparams.get(RHO)));
        }

        double regParamLlh = 0.0;
        for (int ii = 0; ii < K + V; ii++) {
            regParamLlh += StatUtils.logNormalProbability(
                    regParams[ii],
                    hyperparams.get(MU),
                    Math.sqrt(hyperparams.get(SIGMA)));
        }

        double llh = wordLlh
                + topicLlh
                + responseLlh
                + regParamLlh;
        return llh;
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int k = 0; k < K; k++) {
                modelStr.append(k).append("\n");
                modelStr.append(DirMult.output(topicWords[k])).append("\n");
            }
            for (int ii = 0; ii < V + K; ii++) {
                modelStr.append(ii).append("\n");
                modelStr.append(regParams[ii]).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");

                for (int n = 0; n < words[d].length; n++) {
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
    protected void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            // initialize
            initializeModelStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            for (int k = 0; k < K; k++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != k) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                topicWords[k] = DirMult.input(reader.readLine());
            }
            for (int ii = 0; ii < V + K; ii++) {
                int lexIdx = Integer.parseInt(reader.readLine());
                if (lexIdx != ii) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                regParams[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void outputLexicalParameters(File filepath) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing lexical weights to " + filepath);
        }

        ArrayList<RankingItem<Integer>> sortedWeights = new ArrayList<RankingItem<Integer>>();
        for (int v = 0; v < V; v++) {
            sortedWeights.add(new RankingItem<Integer>(v, regParams[K + v]));
        }
        Collections.sort(sortedWeights);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (int v = 0; v < V; v++) {
                RankingItem<Integer> rankItem = sortedWeights.get(v);
                int lexIdx = rankItem.getObject();
                writer.write(lexIdx
                        + "\t" + this.wordVocab.get(lexIdx)
                        + "\t" + this.regParams[K + lexIdx]
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing lexical weights to "
                    + filepath);
        }
    }

    public static void parallelTest(int[][] newWords, File iterPredFolder, LexicalSLDA sampler) {
        // debug
        System.out.println("Parallel test in LexicalSLDA");

        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < filenames.length; i++) {
                String filename = filenames[i];
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                LexicalSLDATestRunner runner = new LexicalSLDATestRunner(
                        sampler, newWords,
                        stateFile.getAbsolutePath(),
                        partialResultFile.getAbsolutePath());
                Thread thread = new Thread(runner);
                threads.add(thread);
            }

            runThreads(threads);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
    }
}

class LexicalSLDATestRunner implements Runnable {

    LexicalSLDA sampler;
    int[][] newWords;
    String stateFile;
    String outputFile;

    public LexicalSLDATestRunner(LexicalSLDA sampler,
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
        LexicalSLDA testSampler = new LexicalSLDA();
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