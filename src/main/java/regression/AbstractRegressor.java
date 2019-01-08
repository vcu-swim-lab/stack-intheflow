package regression;

import core.AbstractRunner;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.IOUtils;
import util.PredictionUtils;
import util.RankingItem;
import util.RankingItemList;
import util.evaluation.ClassificationEvaluation;
import util.evaluation.Measurement;
import util.evaluation.RankingPerformance;

/**
 *
 * @author vietan
 */
public abstract class AbstractRegressor extends AbstractRunner {
    
    public static final String DATA_FILE = "data";
    public static final String MODEL_FILE = "model";
    public static final String PREDICTION_FILE = "predictions";
    public static final String RESULT_FILE = "result";
    protected String folder;
    protected String name;
    
    public AbstractRegressor(String folder) {
        this.folder = folder;
    }
    
    public abstract String getName();
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getFolder() {
        return this.folder;
    }
    
    public String getRegressorFolder() {
        return new File(folder, getName()).getAbsolutePath();
    }
    
    public double[] inputPredictions(File inputFile) {
        if (verbose) {
            logln(">>> Input predictions to " + inputFile);
        }
        return PredictionUtils.inputPredictions(inputFile);
    }

    /**
     * Output predictions.
     *
     * @param outputFile The output file
     * @param instanceIds List of instance IDs
     * @param trueValues List of true values
     * @param predValues List of predicted values
     *
     */
    public void outputPredictions(
            File outputFile,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues) {
        if (verbose) {
            logln(">>> Output predictions to " + outputFile);
        }
        PredictionUtils.outputRegressionPredictions(outputFile, instanceIds, trueValues, predValues);
    }

    /**
     * Output regression results.
     *
     * @param outputFile The output file
     * @param trueValues List of true values
     * @param predValues List of predicted values
     * @return 
     */
    public ArrayList<Measurement> outputRegressionResults(
            File outputFile,
            double[] trueValues,
            double[] predValues) {
        // output different measurements
        if (verbose) {
            logln(">>> Output regression results to " + outputFile);
        }
        return PredictionUtils.outputRegressionResults(outputFile, trueValues, predValues);
    }
    
    public ArrayList<Measurement> outputClassificationResults(
            File outputFile,
            int[] trueClasses,
            int[] predClasses) throws Exception {
        // output different measurements
        if (verbose) {
            logln(">>> Output classification results to " + outputFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        ClassificationEvaluation eval = new ClassificationEvaluation(trueClasses, predClasses);
        eval.computePRF1();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement m : measurements) {
            writer.write(m.getName() + "\t" + m.getValue() + "\n");
        }
        writer.close();
        return measurements;
    }
    
    public void outputRankingPerformance(
            File rankFolder,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues) {
        if (verbose) {
            logln(">>> Output ranking performance to " + rankFolder);
        }
        IOUtils.createFolder(rankFolder);

        // predictions
        RankingItemList<String> preds = new RankingItemList<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            preds.addRankingItem(new RankingItem<String>(instanceIds[ii], predValues[ii]));
        }
        preds.sortDescending();

        // groundtruth
        RankingItemList<String> truths = new RankingItemList<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            truths.addRankingItem(new RankingItem<String>(instanceIds[ii], trueValues[ii]));
        }
        truths.sortDescending();
        
        RankingPerformance<String> rankPerf = new RankingPerformance<String>(preds,
                rankFolder.getAbsolutePath());
        rankPerf.computeAndOutputNDCGs(truths);
    }
    
    public void outputRankingPerformance(
            File rankFolder,
            String[] instanceIds,
            double[] trueValues,
            double[] predValues,
            double threshold) {
        IOUtils.createFolder(rankFolder);
        RankingItemList<String> preds = new RankingItemList<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            preds.addRankingItem(new RankingItem<String>(instanceIds[ii], predValues[ii]));
        }
        preds.sortDescending();
        
        Set<String> groundtruth = new HashSet<String>();
        for (int ii = 0; ii < instanceIds.length; ii++) {
            if (trueValues[ii] >= threshold) {
                groundtruth.add(instanceIds[ii]);
            }
        }
        
        RankingPerformance<String> rankPerf = new RankingPerformance<String>(preds,
                groundtruth, rankFolder.getAbsolutePath());
        rankPerf.computePrecisionsAndRecalls();
        rankPerf.outputPrecisionRecallF1();
        
        rankPerf.outputAUCListFile();
        rankPerf.outputRankingResultsWithGroundtruth();
        
        rankPerf.computeAUC();
        rankPerf.outputAUC();
    }
}
