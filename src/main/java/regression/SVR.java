package regression;

import core.crossvalidation.Fold;
import data.ResponseTextDataset;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import svm.SVMLight;
import svm.SVMUtils;
import util.CLIUtils;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public class SVR<D extends ResponseTextDataset> extends AbstractRegressor implements Regressor<D> {

    public static final double DEFAULT_C = -1; // use default value from SVMLight
    protected SVMLight svm;
    // option C in SVM Light: trade-off between training error and margin
    // (default: [avg. x*x]^{-1})
    private double c;

    public SVR(String folder) {
        super(folder);
        this.svm = new SVMLight();
        this.c = DEFAULT_C;
    }

    public SVR(String folder, double c) {
        super(folder);
        this.svm = new SVMLight();
        this.c = c;
    }

    @Override
    public String getName() {
        if (name == null) {
            if (this.c == DEFAULT_C) {
                return "SVR";
            } else {
                return "SVR-" + MiscUtils.formatDouble(this.c);
            }
        }
        return name;
    }

    public SVMLight getSVM() {
        return svm;
    }

    @Override
    public void input(File inputFile) {
    }

    @Override
    public void output(File outputFile) {
    }

    public void train(
            int[][] trWords,
            double[] trResponses,
            int V,
            File trainFile,
            File modelFile) {
        int D = trWords.length;
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < trWords[d].length; n++) {
                designMatrix[d][trWords[d][n]]++;
            }
        }
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] /= trWords[d].length;
            }
        }
        SVMUtils.outputSVMLightFormat(trainFile, designMatrix, trResponses);

        String[] opts = {"-z r"};
        if (this.c != DEFAULT_C) {
            opts = new String[2];
            opts[0] = "-z r";
            opts[1] = "-c " + this.c;
        }
        try {
            svm.learn(opts, trainFile, modelFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while training");
        }
    }

    public void train(int[][] trWords,
            double[] trResponses,
            int V,
            ArrayList<Double>[] addFeatures,
            File trainFile,
            File modelFile) {
        int D = trWords.length;
        int F = addFeatures[0].size();

        double[][] allFeatures = new double[D][V + F];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < trWords[d].length; n++) {
                allFeatures[d][trWords[d][n]]++;
            }
            for (int f = 0; f < F; f++) {
                allFeatures[d][V + f] = addFeatures[d].get(f);
            }
        }
        SVMUtils.outputSVMLightFormat(trainFile, allFeatures, trResponses);

        String[] opts = {"-z r"};
        if (this.c != DEFAULT_C) {
            opts = new String[2];
            opts[0] = "-z r";
            opts[1] = "-c " + this.c;
        }
        try {
            svm.learn(opts, trainFile, modelFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while training");
        }
    }

    @Override
    public void train(D trainData) {
        if (verbose) {
            System.out.println("Training ...");
        }
        int[][] trWords = trainData.getWords();
        double[] trResponses = trainData.getResponses();
        int V = trainData.getWordVocab().size();
        File trainFile = new File(getRegressorFolder(), DATA_FILE + Fold.TrainingExt);
        File modelFile = new File(getRegressorFolder(), MODEL_FILE);
        train(trWords, trResponses, V, trainFile, modelFile);
    }

    public void test(File testFile, File modelFile, File resultFile) {
        String[] opts = null;
        try {
            svm.classify(opts, testFile, modelFile, resultFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while training");
        }
    }

    public void test(int[][] teWords, double[] teResponses, int V,
            File testFile, File modelFile, File resultFile) {
        int D = teWords.length;
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < teWords[d].length; n++) {
                designMatrix[d][teWords[d][n]]++;
            }
        }
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] /= teWords[d].length;
            }
        }
        SVMUtils.outputSVMLightFormat(testFile, designMatrix, teResponses);

        test(testFile, modelFile, resultFile);
    }

    public void test(int[][] teWords, double[] teResponses, int V,
            ArrayList<Double>[] addFeatures,
            File testFile, File modelFile, File resultFile) {
        int D = teWords.length;
        int F = addFeatures[0].size();

        double[][] allFeatures = new double[D][V + F];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < teWords[d].length; n++) {
                allFeatures[d][teWords[d][n]]++;
            }
            for (int f = 0; f < F; f++) {
                allFeatures[d][V + f] = addFeatures[d].get(f);
            }
        }
        SVMUtils.outputSVMLightFormat(testFile, allFeatures, teResponses);

        test(testFile, modelFile, resultFile);
    }

    @Override
    public void test(D testData) {
        if (verbose) {
            System.out.println("Testing ...");
        }
        String[] teDocIds = testData.getDocIds();
        int[][] teWords = testData.getWords();
        double[] teResponses = testData.getResponses();

        int V = testData.getWordVocab().size();
        File testFile = new File(getRegressorFolder(), DATA_FILE + Fold.TestExt);
        File modelFile = new File(getRegressorFolder(), MODEL_FILE);
        File resultFile = new File(getRegressorFolder(), "svm-" + PREDICTION_FILE + Fold.TestExt);
        test(teWords, teResponses, V, testFile, modelFile, resultFile);

        File predFile = new File(getRegressorFolder(), PREDICTION_FILE + Fold.TestExt);
        double[] predictions = svm.getPredictedValues(resultFile);
        outputPredictions(predFile, teDocIds, teResponses, predictions);

        File regFile = new File(getRegressorFolder(), RESULT_FILE + Fold.TestExt);
        outputRegressionResults(regFile, teResponses, predictions);
    }

    public static String getHelpString() {
        return "java -cp \"dist/segan.jar:dist/lib/*\" " + SVR.class.getName() + " -help";
    }

    public static void main(String[] args) {
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

            // running configurations
            addOption("cv-folder", "Cross validation folder");
            addOption("num-folds", "Number of folds");
            addOption("fold", "The cross-validation fold to run");
            addOption("run-mode", "Running mode");

            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("z", false, "standardize (z-score normalization)");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            if (cmd.hasOption("cv-folder")) {
                runCrossValidation();
            } else {
                runModel();
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    private static void runModel() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String outputFolder = cmd.getOptionValue("output");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        if (verbose) {
            System.out.println("\nLoading formatted data ...");
        }
        ResponseTextDataset data = new ResponseTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder).getAbsolutePath());

        if (cmd.hasOption("z")) {
            data.zNormalize();
        }

        if (verbose) {
            System.out.println("--- Loaded. " + data.toString());
        }

        SVR svr = new SVR(outputFolder);
        svr.train(data);
    }

    private static void runCrossValidation() throws Exception {
        String cvFolder = cmd.getOptionValue("cv-folder");
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
        String resultFolder = cmd.getOptionValue("output");

        int foldIndex = -1;
        if (cmd.hasOption("fold")) {
            foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
        }

        for (int ii = 0; ii < numFolds; ii++) {
            if (foldIndex != -1 && ii != foldIndex) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + foldIndex);
            }

            Fold fold = new Fold(ii, cvFolder);
            File foldFolder = new File(resultFolder, fold.getFoldName());
            ResponseTextDataset[] foldData = ResponseTextDataset.loadCrossValidationFold(fold);
            ResponseTextDataset trainData = foldData[Fold.TRAIN];
            ResponseTextDataset devData = foldData[Fold.DEV];
            ResponseTextDataset testData = foldData[Fold.TEST];

            if (cmd.hasOption("z")) {
                ResponseTextDataset.zNormalize(trainData, devData, testData);
            }

            if (verbose) {
                System.out.println("Fold " + fold.getFoldName());
                System.out.println("--- training: " + trainData.toString());
                System.out.println("--- development: " + devData.toString());
                System.out.println("--- test: " + testData.toString());
                System.out.println();
            }

            SVR svr = new SVR(foldFolder.getAbsolutePath());
            svr.train(trainData);
            svr.test(testData);
        }
    }
}
