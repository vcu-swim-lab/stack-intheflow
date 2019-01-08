package core;

import core.AbstractSampler.InitialState;
import core.crossvalidation.Fold;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import regression.AbstractRegressor;
import util.CLIUtils;
import util.IOUtils;
import util.RankingItem;
import util.StatUtils;
import util.evaluation.Measurement;

/**
 * @param <D> dataset
 * @author vietan
 */
public abstract class AbstractExperiment<D extends AbstractDataset> extends AbstractRunner {

    public static final String PREDICTION_FILE = AbstractRegressor.PREDICTION_FILE;
    public static final String RESULT_FILE = AbstractRegressor.RESULT_FILE;
    public static final String SUMMARY_FILE = "summary.txt";
    public static final String SUPERVISED_FOLDER = "supervised/";
    public static final String UNSUPERVISED_FOLDER = "unsupervised/";
    public static final String SAMPLER_FOLDER = "data/sampler/";
    public static final String TRAIN_PREFIX = "tr_";
    public static final String DEV_PREFIX = "de_";
    public static final String TEST_PREFIX = "te_";
    public static final String TRAIN_POSFIX = "_tr";
    public static final String DEV_POSFIX = "_de";
    public static final String TEST_POSFIX = "_te";
    public static final String PARALLEL_POSFIX = "_p";
    public static final String RESULT_FOLDER = "result";
    public static final String RANKING_FOLDER = "ranking";
    public static final String SURVEY_FOLDER = "survey";
    public static final String MODEL_FILE = "model";
    public static final String PERPLEXITY_FILE = "perplexity.txt";
    public static final int UNOBSERVED = -1;

    public static int burn_in = 100;
    public static int max_iters = 1000;
    public static int sample_lag = 50;
    public static int report_interval = 5;
    protected InitialState initState;
    protected boolean paramOpt;
    public static String experimentPath;
    protected D data;

    public abstract void preprocess() throws Exception;

    public abstract void setup() throws Exception;

    public abstract void run() throws Exception;

    public abstract void evaluate() throws Exception;

    public String getDatasetFolder() {
        return data.getFolder();
    }

    public boolean isTraining() {
        return cmd.hasOption("train");
    }

    public boolean isDeveloping() {
        return cmd.hasOption("dev");
    }

    public boolean isTesting() {
        return cmd.hasOption("test");
    }

    public void setupSampling() {
        burn_in = CLIUtils.getIntegerArgument(cmd, "burnIn", 5);
        max_iters = CLIUtils.getIntegerArgument(cmd, "maxIter", 10);
        sample_lag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        report_interval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        paramOpt = cmd.hasOption("paramOpt");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
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
    }

    public static void addSamplingOptions() {
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");
        addOption("init", "Report interval");

        addOption("test-burnIn", "Burn-in");
        addOption("test-maxIter", "Maximum number of iterations");
        addOption("test-sampleLag", "Sample lag");
    }

    public static void addCrossValidationOptions() {
        addOption("num-folds", "Number of folds. Default 5.");
        addOption("tr2dev-ratio", "Training-to-development ratio. Default 0.8.");
        addOption("cv-folder", "Folder to store cross validation folds");
        addOption("fold", "The cross-validation fold to run");
        addOption("num-classes", "Number of classes when performing stratified sampling");
        options.addOption("train", false, "Train");
        options.addOption("dev", false, "Develop");
        options.addOption("test", false, "Test");
        options.addOption("parallel", false, "Parallel sampling");
    }

    public static void addCorpusProcessorOptions() {
        addOption("u", "The minimum count of raw unigrams");
        addOption("b", "The minimum count of raw bigrams");
        addOption("bs", "The minimum score of bigrams");
        addOption("V", "Maximum vocab size");
        addOption("min-tf", "Term frequency minimum cutoff");
        addOption("max-tf", "Term frequency maximum cutoff");
        addOption("min-df", "Document frequency minimum cutoff");
        addOption("max-df", "Document frequency maximum cutoff");
        addOption("min-doc-length", "Document minimum length");
        options.addOption("s", false, "Whether stopwords are filtered");
        options.addOption("l", false, "Whether lemmatization is performed");
        options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
    }

    public static void addGreekParametersOptions() {
        addOption("alpha", "alpha");
        addOption("beta", "beta");
        addOption("gamma", "gamma");
        addOption("delta", "delta");
        addOption("lambda", "lambda");
        addOption("sigma", "sigma");
        addOption("mu", "mu");
        addOption("rho", "rho");
        addOption("epsilon", "epsilon");
        addOption("eta", "eta");
        addOption("pi", "pi");
    }

    /**
     * Summarize evaluation measurements over multiple folds.
     *
     * @param numFolds Number of folds
     * @param resultFolder The folder containing results
     * @param modelFolder The sub-folder in each foldFolder which stores the
     * outputs of different models
     * @param phase Whether it is training/development/test
     * @param resultFile Result file
     * @throws java.lang.Exception
     */
    protected void evaluate(
            String resultFolder,
            String modelFolder,
            int numFolds,
            String phase,
            String resultFile) throws Exception {
        ArrayList<String> modelNames = new ArrayList<String>();
        HashMap<String, ArrayList<Measurement>>[] results = new HashMap[numFolds];
        for (int f = 0; f < numFolds; f++) {
            Fold fold = new Fold(f, null);
            String foldName = fold.getFoldName();
            File foldFolder = new File(resultFolder, foldName);
            if (!foldFolder.exists()) {
                continue;
            }
            File foldModelFolder = foldFolder;
            if (modelFolder != null) {
                foldModelFolder = new File(foldFolder, modelFolder);
            }
            if (!foldModelFolder.exists()) {
                continue;
            }
            if (verbose) {
                logln("--- Reading results from " + foldModelFolder);
            }

            String[] modelFolders = foldModelFolder.list();
            ArrayList<String> modelFolderList = new ArrayList<String>();
            modelFolderList.addAll(Arrays.asList(modelFolders));
            Collections.sort(modelFolderList);

            results[f] = new HashMap<String, ArrayList<Measurement>>();

            File foldSummary = new File(foldModelFolder, phase + SUMMARY_FILE);
            BufferedWriter writer = IOUtils.getBufferedWriter(foldSummary);
            if (verbose) {
                logln("--- Summarizing fold " + f + ". Writing to " + foldSummary);
            }

            int count = 0;
            for (String mFolder : modelFolderList) {
                File teResultFolder = new File(new File(foldModelFolder, mFolder),
                        phase + RESULT_FOLDER);
                if (!teResultFolder.exists()) {
                    continue;
                }
                File teResultFile = new File(teResultFolder, resultFile);
                if (!teResultFile.exists()) {
                    continue;
                }

                // read measurements
                BufferedReader reader = IOUtils.getBufferedReader(teResultFile);
                String line;
                ArrayList<Measurement> measurements = new ArrayList<Measurement>();
                while ((line = reader.readLine()) != null) {
                    Measurement m = new Measurement(line.split("\t")[0],
                            Double.parseDouble(line.split("\t")[1]));
                    measurements.add(m);
                }
                reader.close();

                if (!modelNames.contains(mFolder)) {
                    modelNames.add(mFolder);
                }
                results[f].put(mFolder, measurements);

                if (count == 0) {
                    writer.write("Model");
                    for (Measurement m : measurements) {
                        writer.write("\t" + m.getName());
                    }
                    writer.write("\n");
                }

                writer.write(mFolder);
                for (Measurement m : measurements) {
                    writer.write("\t" + m.getValue());
                }
                writer.write("\n");

                count++;
            }
            writer.close();
        }
        Collections.sort(modelNames);

        // summarize across folds
        File mergeFile = new File(resultFolder, phase + "merged-" + SUMMARY_FILE);
        File metaSumFile = new File(resultFolder, phase + "meta-" + SUMMARY_FILE);
        if (verbose) {
            System.out.println("--- Meta summarizing " + metaSumFile);
            System.out.println("--- Merge summarizing " + mergeFile);
        }
        ArrayList<String> measureNames = null;

        BufferedWriter writer = IOUtils.getBufferedWriter(metaSumFile);
        BufferedWriter mergeWriter = IOUtils.getBufferedWriter(mergeFile);
        mergeWriter.write("Model\tMetric\tValue\tFold\n");
        for (int f = 0; f < results.length; f++) {
            if (results[f] == null) {
                continue;
            }
            writer.write("Fold " + f + "\n");
            for (String modelName : modelNames) {
                ArrayList<Measurement> modelFoldMeasurements = results[f].get(modelName);
                if (modelFoldMeasurements != null) {
                    writer.write(modelName);
                    for (Measurement m : modelFoldMeasurements) {
                        writer.write("\t" + m.getValue());
                        mergeWriter.write(modelName
                                + "\t" + m.getName()
                                + "\t" + m.getValue()
                                + "\t" + f
                                + "\n");
                    }

                    if (measureNames == null) {
                        measureNames = new ArrayList<String>();
                        for (Measurement m : modelFoldMeasurements) {
                            measureNames.add(m.getName());
                        }
                    }
                    writer.write("\n");
                }
            }
            writer.write("\n\n");
        }

        // average
        if (measureNames != null) {
            for (String measure : measureNames) {
                ArrayList<RankingItem<String>> rankModels = new ArrayList<>();
                HashMap<String, ArrayList<Double>> modelVals = new HashMap<>();
                for (String model : modelNames) {
                    ArrayList<Double> vals = new ArrayList<Double>();
                    for (HashMap<String, ArrayList<Measurement>> result : results) {
                        if (result == null) {
                            continue;
                        }
                        ArrayList<Measurement> modelFoldMeasurements = result.get(model);
                        if (modelFoldMeasurements != null) {
                            for (Measurement m : modelFoldMeasurements) {
                                if (m.getName().equals(measure)) {
                                    vals.add(m.getValue());
                                }
                            }
                        }
                    }

                    double avg = StatUtils.mean(vals);
                    rankModels.add(new RankingItem<String>(model, avg));
                    modelVals.put(model, vals);

                }
                Collections.sort(rankModels);

                writer.write(measure + "\n");
                writer.write("Model\tNum-folds\tAverage\tStdv\n");
                for (RankingItem<String> rankModel : rankModels) {
                    String model = rankModel.getObject();
                    ArrayList<Double> vals = modelVals.get(model);
                    double std = StatUtils.standardDeviation(vals);
                    writer.write(model
                            + "\t" + vals.size()
                            + "\t" + rankModel.getPrimaryValue()
                            + "\t" + std
                            + "\n");
                }
                writer.write("\n\n");
            }
        }
        writer.close();
        mergeWriter.close();
    }
}
