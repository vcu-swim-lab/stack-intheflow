package optimization;

import core.AbstractExperiment;
import core.AbstractLinearModel;
import data.ResponseTextDataset;
import edu.stanford.nlp.optimization.DiffFunction;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;
import util.PredictionUtils;
import util.SparseVector;
import util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class OWLQNLinearRegression extends AbstractLinearModel {

    private final double l1;
    private final double l2;
    private int maxIters;

    public OWLQNLinearRegression(String basename) {
        this(basename, 1.0, 1.0);
    }

    public OWLQNLinearRegression(String basename, double l1, double l2) {
        this(basename, l1, l2, 1000);
    }

    public OWLQNLinearRegression(String basename, double l1, double l2,
            int maxIters) {
        super(basename);
        this.l1 = l1;
        this.l2 = l2;
        this.maxIters = maxIters;
    }

    @Override
    public String getName() {
        return this.name + "_l1-" + l1 + "_l2-" + l2 + "_m-" + maxIters;
    }

    public double getL1() {
        return this.l1;
    }

    public double getL2() {
        return this.l2;
    }

    public void train(int[][] docWords,
            ArrayList<Integer> docIndices,
            double[] docResponses,
            int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        double[] responses = new double[D];
        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            responses[ii] = docResponses[dd];
            designMatrix[ii] = new SparseVector(V);
            double val = 1.0 / docWords[dd].length;
            for (int nn = 0; nn < docWords[dd].length; nn++) {
                designMatrix[ii].change(docWords[dd][nn], val);
            }
        }
        train(designMatrix, responses, V);
    }

    public void train(SparseVector[] designMatrix, double[] responses, int K) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        OWLQN minimizer = new OWLQN();
        minimizer.setQuiet(quiet);
        minimizer.setMaxIters(maxIters);
        DiffFunc diffFunction = new DiffFunc(designMatrix, responses, l2);
        double[] initParams = new double[K];
        this.weights = minimizer.minimize(diffFunction, initParams, l1);
    }

    public void train(SparseVector[] designMatrix, double[] responses, double[] initParams) {
        if (verbose) {
            System.out.println("Training ...");
            System.out.println("--- # instances: " + designMatrix.length + ". " + responses.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        OWLQN minimizer = new OWLQN();
        minimizer.setQuiet(quiet);
        minimizer.setMaxIters(maxIters);
        DiffFunc diffFunction = new DiffFunc(designMatrix, responses, l2);
        this.weights = minimizer.minimize(diffFunction, initParams, l1);
    }

    public double[] test(int[][] docWords, ArrayList<Integer> docIndices, int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        SparseVector[] designMatrix = new SparseVector[D];
        for (int ii = 0; ii < D; ii++) {
            int dd = docIndices.get(ii);
            designMatrix[ii] = new SparseVector(V);
            double val = 1.0 / docWords[dd].length;
            for (int nn = 0; nn < docWords[dd].length; nn++) {
                designMatrix[ii].change(docWords[dd][nn], val);
            }
        }

        return test(designMatrix);
    }

    public double[] test(SparseVector[] designMatrix) {
        if (verbose) {
            System.out.println("Testing ...");
            System.out.println("--- # instances: " + designMatrix.length);
            System.out.println("--- # features: " + designMatrix[0].getDimension());
        }
        double[] predictions = new double[designMatrix.length];
        for (int d = 0; d < predictions.length; d++) {
            predictions[d] = designMatrix[d].dotProduct(weights);
        }
        return predictions;
    }

    class DiffFunc implements DiffFunction {

        // inputs
        private final double[] values; // [N]-dim vector
        private final SparseVector[] designMatrix; // [N]x[K] sparse matrix
        // derived
        private final int N;
        private final int K;
        private final double l2;

        public DiffFunc(SparseVector[] designMatrix, double[] values, double l2) {
            this.designMatrix = designMatrix;
            this.values = values;
            this.l2 = l2;
            // derived statistics
            this.N = this.designMatrix.length;
            this.K = this.designMatrix[0].getDimension();
            if (this.K <= 0) {
                throw new RuntimeException("Number of features = " + this.K);
            }
        }

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int n = 0; n < N; n++) {
                double dotprod = dotprod(designMatrix[n], w);
                for (int k : this.designMatrix[n].getIndices()) {
                    grads[k] -= 2 * (values[n] - dotprod) * designMatrix[n].get(k);
                }
            }
            if (l2 > 0) {
                for (int k = 0; k < w.length; k++) {
                    grads[k] += 2 * l2 * w[k];
                }
            }
            return grads;
        }

        @Override
        public double valueAt(double[] w) {
            double loss = 0.0;
            for (int n = 0; n < N; n++) {
                double dotprod = dotprod(designMatrix[n], w);
                double diff = values[n] - dotprod;
                loss += diff * diff;
            }
            double val = loss;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        private double dotprod(SparseVector designVec, double[] w) {
            double dotprod = 0.0;
            for (int k : designVec.getIndices()) {
                dotprod += w[k] * designVec.get(k);
            }
            return dotprod;
        }
    }

    private static void addOpitions() throws Exception {
        parser = new BasicParser();
        options = new Options();

        // data input
        addOption("dataset", "Dataset");
        addOption("word-voc-file", "Word vocabulary file");
        addOption("word-file", "Document word file");
        addOption("info-file", "Document info file");

        // data output
        addOption("output-folder", "Output folder");

        // parameters
        addOption("l1", "L1");
        addOption("l2", "L2");
        addOption("maxIter", "Maximum number of iterations");

        // running
        options.addOption("train", false, "Train");
        options.addOption("test", false, "Test");

        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("z", false, "z-normalize");
        options.addOption("help", false, "Help");
        options.addOption("example", false, "Example command");
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

    private static void runModel() throws Exception {
        // data input
        String datasetName = cmd.getOptionValue("dataset");
        String wordVocFile = cmd.getOptionValue("word-voc-file");
        String docWordFile = cmd.getOptionValue("word-file");
        String docInfoFile = cmd.getOptionValue("info-file");

        String outputFolder = cmd.getOptionValue("output-folder");
        ResponseTextDataset data = new ResponseTextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile),
                new File(docInfoFile),
                null);
        int V = data.getWordVocab().size();

        double[] docResponses = data.getResponses();
        if (cmd.hasOption("z")) { // z-normalization
            ZNormalizer zNorm = new ZNormalizer(docResponses);
            docResponses = zNorm.normalize(docResponses);
        }

        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 1.0);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);

        OWLQNLinearRegression mlr = new OWLQNLinearRegression("MLR-OWLQN", l1, l2, maxIters);
        File mlrFolder = new File(outputFolder, mlr.getName());
        IOUtils.createFolder(mlrFolder);

        if (cmd.hasOption("train")) { // train
            mlr.train(data.getWords(), null, docResponses, V);
            mlr.output(new File(mlrFolder, MODEL_FILE));
            
            File trResultFolder = new File(mlrFolder,
                    AbstractExperiment.TRAIN_PREFIX + AbstractExperiment.RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);

            double[] trPredictions = mlr.test(data.getWords(), null, V);
            PredictionUtils.outputRegressionPredictions(
                    new File(trResultFolder, AbstractExperiment.PREDICTION_FILE),
                    data.getDocIds(), docResponses, trPredictions);
            PredictionUtils.outputRegressionResults(
                    new File(trResultFolder, AbstractExperiment.RESULT_FILE),
                    docResponses, trPredictions);
        }

        if (cmd.hasOption("test")) {
            mlr.input(new File(mlrFolder, MODEL_FILE));
            double[] tePredictions = mlr.test(data.getWords(), null, V);
            
            File teResultFolder = new File(mlrFolder,
                    AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            PredictionUtils.outputRegressionPredictions(
                    new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                    data.getDocIds(), docResponses, tePredictions);
            PredictionUtils.outputRegressionResults(
                    new File(teResultFolder, AbstractExperiment.RESULT_FILE),
                    docResponses, tePredictions);
        }
    }

    public static String getHelpString() {
        return "java -cp \"dist/segan.jar\" " + OWLQNLinearRegression.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        String example = new String();
        return example;
    }
}
