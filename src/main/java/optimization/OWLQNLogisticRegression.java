package optimization;

import core.AbstractExperiment;
import core.AbstractLinearModel;
import data.LabelTextDataset;
import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedWriter;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class OWLQNLogisticRegression extends AbstractLinearModel {

    private final double l1;
    private final double l2;
    private int maxIters;

    public OWLQNLogisticRegression(String basename) {
        this(basename, 1.0, 1.0);
    }

    public OWLQNLogisticRegression(String basename, double l1, double l2) {
        this(basename, l1, l2, 1000);
    }

    public OWLQNLogisticRegression(String basename, double l1, double l2,
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
            int[] docResponses,
            int V) {
        if (docIndices == null) {
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < docWords.length; dd++) {
                docIndices.add(dd);
            }
        }

        int D = docIndices.size();
        int[] responses = new int[D];
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

    public void train(SparseVector[] designMatrix, int[] responses, int K) {
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
            double expdotprod = Math.exp(designMatrix[d].dotProduct(weights));
            predictions[d] = expdotprod / (1.0 + expdotprod);
        }
        return predictions;
    }

    public void outputRankedWeights(File outputFile, ArrayList<String> featureNames) {
        try {
            ArrayList<RankingItem<Integer>> rankItems = new ArrayList<>();
            for (int vv = 0; vv < this.weights.length; vv++) {
                rankItems.add(new RankingItem<Integer>(vv, weights[vv]));
            }
            Collections.sort(rankItems);

            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (RankingItem<Integer> rankItem : rankItems) {
                int idx = rankItem.getObject();
                double val = rankItem.getPrimaryValue();
                writer.write(idx
                        + "\t" + featureNames.get(idx)
                        + "\t" + val
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    class DiffFunc implements DiffFunction {

        // inputs
        private final int[] values; // [N]-dim binary vector {0, 1}
        private final SparseVector[] designMatrix; // [N]x[K] sparse matrix
        // derived
        private final int N;
        private final int K;
        private final double l2;

        public DiffFunc(SparseVector[] designMatrix, int[] values, double l2) {
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
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int nn = 0; nn < N; nn++) {
                double dotProb = designMatrix[nn].dotProduct(w);
                llh -= values[nn] * dotProb - Math.log(Math.exp(dotProb) + 1);
            }

            double val = llh;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int nn = 0; nn < N; nn++) {
                double dotprod = designMatrix[nn].dotProduct(w);
                double expDotprod = Math.exp(dotprod);
                double pred = expDotprod / (expDotprod + 1);
                for (int kk = 0; kk < K; kk++) {
                    grads[kk] -= (values[nn] - pred) * designMatrix[nn].get(kk);
                }
            }
            if (l2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += 2 * l2 * w[kk];
                }
            }
            return grads;
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
        LabelTextDataset data = new LabelTextDataset(datasetName);
        data.loadFormattedData(new File(wordVocFile),
                new File(docWordFile), new File(docInfoFile), null);
        int V = data.getWordVocab().size();

        double l1 = CLIUtils.getDoubleArgument(cmd, "l1", 0.0);
        double l2 = CLIUtils.getDoubleArgument(cmd, "l2", 1.0);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 1000);

        OWLQNLogisticRegression mlr = new OWLQNLogisticRegression("MLR-OWLQN", l1, l2, maxIters);
        File mlrFolder = new File(outputFolder, mlr.getName());
        IOUtils.createFolder(mlrFolder);

        // TODO: allow training and test for a subset of the data by using the list of indices
        if (cmd.hasOption("train")) { // train
            mlr.train(data.getWords(), null, data.getSingleLabels(), V);
            mlr.output(new File(mlrFolder, MODEL_FILE));

            File trResultFolder = new File(mlrFolder,
                    AbstractExperiment.TRAIN_PREFIX + AbstractExperiment.RESULT_FOLDER);
            IOUtils.createFolder(trResultFolder);

            double[] trPredictions = mlr.test(data.getWords(), null, V);
            PredictionUtils.outputClassificationPredictions(
                    new File(trResultFolder, AbstractExperiment.PREDICTION_FILE),
                    data.getDocIds(), data.getSingleLabels(), trPredictions);
            PredictionUtils.outputBinaryClassificationResults(
                    new File(trResultFolder, AbstractExperiment.RESULT_FILE),
                    data.getSingleLabels(), trPredictions);
        }

        if (cmd.hasOption("test")) {
            mlr.input(new File(mlrFolder, MODEL_FILE));
            double[] tePredictions = mlr.test(data.getWords(), null, V);

            File teResultFolder = new File(mlrFolder,
                    AbstractExperiment.TEST_PREFIX + AbstractExperiment.RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);

            PredictionUtils.outputClassificationPredictions(
                    new File(teResultFolder, AbstractExperiment.PREDICTION_FILE),
                    data.getDocIds(), data.getSingleLabels(), tePredictions);
            PredictionUtils.outputBinaryClassificationResults(
                    new File(teResultFolder, AbstractExperiment.RESULT_FILE),
                    data.getSingleLabels(), tePredictions);
        }
    }

    public static String getHelpString() {
        return "java -cp \"dist/segan.jar\" " + OWLQNLogisticRegression.class.getName() + " -help";
    }

    public static String getExampleCmd() {
        String example = new String();
        return example;
    }
}
