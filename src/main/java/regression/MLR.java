package regression;

import core.crossvalidation.Fold;
import data.ResponseTextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import optimization.GurobiMLRL1Norm;
import optimization.GurobiMLRL2Norm;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 * @param <D> Any dataset where each document is associated with a response.
 */
public class MLR<D extends ResponseTextDataset> extends AbstractRegressor implements Regressor<D> {

    public static enum Regularizer {

        L1, L2
    }
    protected Regularizer regularizer;
    protected double[] weights;
    protected double param;

    public MLR(String folder, Regularizer reg, double t) {
        super(folder);
        this.regularizer = reg;
        this.param = t;
    }

    @Override
    public String getName() {
        if (name == null) {
            name = "MLR";
        }
        return name + "-" + regularizer + "-" + param;
    }

    public void train(double[][] designMatrix, double[] responses) {
        if (regularizer == Regularizer.L1) {
            GurobiMLRL1Norm mlr = new GurobiMLRL1Norm(designMatrix, responses, param);
            this.weights = mlr.solve();
        } else if (regularizer == Regularizer.L2) {
            GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responses);
            mlr.setSigma(param);
            this.weights = mlr.solve();
        } else {
            throw new RuntimeException(regularizer + " regularization is not supported");
        }
        IOUtils.createFolder(getRegressorFolder());
        output(new File(getRegressorFolder(), MODEL_FILE));
    }

    public void train(int[][] trWords, double[] trResponses, int V) {
        int D = trWords.length;
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < trWords[d].length; n++) {
                designMatrix[d][trWords[d][n]]++;
            }
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] /= trWords[d].length;
            }
        }
        train(designMatrix, trResponses);
    }

    @Override
    public void train(D trainData) {
        if (verbose) {
            System.out.println("Training ...");
        }
        int[][] trWords = trainData.getWords();
        double[] trResponses = trainData.getResponses();
        int V = trainData.getWordVocab().size();
        train(trWords, trResponses, V);
    }

    public double[] test(double[][] designMatrix) {
        int D = designMatrix.length;
        double[] predictions = new double[D];
        for (int d = 0; d < D; d++) {
            double predVal = 0.0;
            for (int v = 0; v < designMatrix[d].length; v++) {
                predVal += designMatrix[d][v] * this.weights[v];
            }

            predictions[d] = predVal;
        }
        return predictions;
    }

    public double[] test(int[][] teWords, int V) {
        input(new File(getRegressorFolder(), MODEL_FILE));

        int D = teWords.length;
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < teWords[d].length; n++) {
                designMatrix[d][teWords[d][n]]++;
            }
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] /= teWords[d].length;
            }
        }
        return test(designMatrix);
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

        double[] predictions = test(teWords, V);
        File predFile = new File(getRegressorFolder(), PREDICTION_FILE + Fold.TestExt);
        outputPredictions(predFile, teDocIds, teResponses, predictions);

        File regFile = new File(getRegressorFolder(), RESULT_FILE + Fold.TestExt);
        outputRegressionResults(regFile, teResponses, predictions);
    }

    @Override
    public void output(File file) {
        if (verbose) {
            System.out.println("Outputing model to " + file);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                writer.write(weights[ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    @Override
    public void input(File file) {
        if (verbose) {
            System.out.println("Inputing model from " + file);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int V = Integer.parseInt(reader.readLine());
            this.weights = new double[V];
            for (int ii = 0; ii < V; ii++) {
                this.weights[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + file);
        }
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double[] getWeights() {
        return this.weights;
    }

    public static double[] inputWeights(File inputFile) {
        double[] ws = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            int dim = Integer.parseInt(reader.readLine());
            ws = new double[dim];
            for (int ii = 0; ii < dim; ii++) {
                ws[ii] = Double.parseDouble(reader.readLine().split("\t")[0]);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing weights from "
                    + inputFile);
        }
        return ws;
    }

    public static void outputWeights(File outputFile, double[] weights) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                writer.write(weights[ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing weights to "
                    + outputFile);
        }
    }

    public static double[] inputWeights(File weightFile,
            ArrayList<String> weightNames) {
        double[] weights = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(weightFile);
            int numWeights = Integer.parseInt(reader.readLine());
            if (numWeights != weightNames.size()) {
                throw new RuntimeException("MISMATCH");
            }

            weights = new double[numWeights];
            String[] sline;
            for (int ii = 0; ii < numWeights; ii++) {
                sline = reader.readLine().split("\t");
                String weightName = sline[0];
                double weight = Double.parseDouble(sline[1]);
                int idx = weightNames.indexOf(weightName);
                weights[idx] = weight;
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing weights from "
                    + weightFile);
        }
        return weights;
    }

    public static void outputWeights(File outputFile,
            double[] weights,
            ArrayList<String> weightNames) {
        if (weights.length != weightNames.size()) {
            throw new RuntimeException("Dimensions mismatch. " + weights.length
                    + " vs. " + weightNames.size());
        }
        try {
            ArrayList<RankingItem<String>> rankFeatures = new ArrayList<RankingItem<String>>();
            for (int ii = 0; ii < weightNames.size(); ii++) {
                rankFeatures.add(new RankingItem<String>(weightNames.get(ii), weights[ii]));
            }
            Collections.sort(rankFeatures);

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                RankingItem<String> item = rankFeatures.get(ii);
                writer.write(item.getObject() + "\t" + item.getPrimaryValue() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing weights to "
                    + outputFile);
        }
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar:dist/lib/*' "
                + MLR.class.getName() + " -help";
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

            addOption("regularizer", "Regularizer (L1, L2)");
            addOption("param", "Parameter");

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

        String regularizer = cmd.getOptionValue("regularizer");
        double param = Double.parseDouble(cmd.getOptionValue("param"));

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

        MLR mlr;
        if (regularizer.equals("L1")) {
            mlr = new MLR(outputFolder, Regularizer.L1, param);
        } else if (regularizer.equals("L2")) {
            mlr = new MLR(outputFolder, Regularizer.L2, param);
        } else {
            throw new RuntimeException(regularizer + " regularization is not supported");
        }
        File mlrFolder = new File(outputFolder, mlr.getName());
        IOUtils.createFolder(mlrFolder);
        mlr.train(data);
    }

    private static void runCrossValidation() throws Exception {
        String cvFolder = cmd.getOptionValue("cv-folder");
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));
        String resultFolder = cmd.getOptionValue("output");

        String regularizer = cmd.getOptionValue("regularizer");
        double param = Double.parseDouble(cmd.getOptionValue("param"));
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

            MLR mlr;
            if (regularizer.equals("L1")) {
                mlr = new MLR(foldFolder.getAbsolutePath(), Regularizer.L1, param);
            } else if (regularizer.equals("L2")) {
                mlr = new MLR(foldFolder.getAbsolutePath(), Regularizer.L2, param);
            } else {
                throw new RuntimeException(regularizer + " regularization is not supported");
            }
            File mlrFolder = new File(foldFolder, mlr.getName());
            IOUtils.createFolder(mlrFolder);
            mlr.train(trainData);
            mlr.test(testData);
        }
    }
}
