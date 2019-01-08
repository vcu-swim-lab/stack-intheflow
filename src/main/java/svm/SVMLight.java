package svm;

/**
 *
 * @author vanguyen
 */
import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.IOUtils;
import util.RankingItem;
import util.RankingItemList;
import util.evaluation.RankingPerformance;

public class SVMLight {

    public static final String Svm2WeightFile = "lib/svm2weight.pl";
    // Inputs
    private String svmLightLearn;
    private String svmLightClassify;
    // Outputs
    private RankingPerformance<Integer> performance;

    public SVMLight() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            this.svmLightLearn = "lib/svm_light_windows/svm_learn.exe";
            this.svmLightClassify = "lib/svm_light_windows/svm_classify.exe";
        } else if (os.contains("mac")) {
            this.svmLightLearn = "lib/svm_light_osx/svm_learn";
            this.svmLightClassify = "lib/svm_light_osx/svm_classify";
        } else if (os.contains("nix") || os.contains("nux")) {
            this.svmLightLearn = "lib/svm_light_linux/svm_learn";
            this.svmLightClassify = "lib/svm_light_linux/svm_classify";
        } else {
            throw new RuntimeException("OS " + os + " not supported.");
        }
    }

    public SVMLight(String svmLightLearn, String svmLightClassify) {
        this.svmLightLearn = svmLightLearn;
        this.svmLightClassify = svmLightClassify;
    }

    /**
     * Train an SVM using SVM Light.
     *
     * @param options SVM Light options
     * @param trainingFile Training file in SVM Light format
     * @param modelFile File to store the model learned
     */
    public void learn(
            String[] options,
            File trainingFile,
            File modelFile) {
        System.out.println("\nStart learning ...");
        String cmd = svmLightLearn;
        if (options != null) {
            for (String option : options) {
                cmd += " " + option;
            }
        }

        try {
            cmd += " " + trainingFile.getAbsolutePath() + " " + modelFile.getAbsolutePath();
            System.out.println("Learn cmd: " + cmd);

            Process proc = Runtime.getRuntime().exec(cmd);
            InputStream istr = proc.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(istr));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while learning SVM-Light");
        }
    }

    /**
     * Run a learned SVM model on unseen data.
     *
     * @param options SVM Light options
     * @param testingFile Test data file in SVM Light format
     * @param modelFile File storing the learned model
     * @param resultFile File to store the result
     */
    public void classify(
            String[] options,
            File testingFile,
            File modelFile,
            File resultFile) {
        System.out.println("Start classifying ...");

        String cmd = svmLightClassify;
        if (options != null) {
            for (String option : options) {
                cmd += " " + option;
            }
        }

        try {
            cmd += " " + testingFile.getAbsolutePath()
                    + " " + modelFile.getAbsolutePath()
                    + " " + resultFile.getAbsolutePath();
            System.out.println("Classify cmd: " + cmd);

            Process proc = Runtime.getRuntime().exec(cmd);

            InputStream istr = proc.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(istr));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while classifying SVM-Light");
        }
    }

    /**
     * Load the predicted values from a result file.
     *
     * @param predFile The result file
     */
    public double[] getPredictedValues(File predFile) {
        ArrayList<Double> list = new ArrayList<Double>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(predFile);
            String line;
            while ((line = reader.readLine()) != null) {
                list.add(Double.parseDouble(line));
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + predFile);
        }

        double[] predVals = new double[list.size()];
        for (int ii = 0; ii < predVals.length; ii++) {
            predVals[ii] = list.get(ii);
        }
        return predVals;
    }

    /**
     * Evaluate SVM-based ranker (using SVMRank)
     */
    public void evaluateRanker(File testingFilePath, File resultFile, File evaluateFolder) throws Exception {
        // load ranked result
        RankingItemList<Integer> resultRankingItemList = new RankingItemList<Integer>();
        BufferedReader reader = IOUtils.getBufferedReader(resultFile);
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            double score = Double.parseDouble(line);
            resultRankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        reader.close();
        resultRankingItemList.sortDescending();

        // load groundtruth from testingFilepath
        RankingItemList<Integer> groundtruthRankingItemList = new RankingItemList<Integer>();
        reader = IOUtils.getBufferedReader(testingFilePath);
        count = 0;
        while ((line = reader.readLine()) != null) {
            double score = Double.parseDouble(line.split(" ")[0]);
            groundtruthRankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        reader.close();

        performance = new RankingPerformance<Integer>(resultRankingItemList, evaluateFolder.getAbsolutePath());
        performance.computeAndOutputNDCGs(groundtruthRankingItemList);
    }

    /**
     * Evaluate SVM-based classifier (using SVMLight)
     */
    public void evaluateClassifier(File testingFilePath, File resultFile, File evaluateFolder) throws Exception {
        // Load actual classes
        Set<Integer> positiveSet = new HashSet<Integer>();
        BufferedReader testIn = IOUtils.getBufferedReader(testingFilePath);
        String line;
        int count = 0;
        while ((line = testIn.readLine()) != null) {
            String[] sline = line.split(" ");
            int cls = Integer.parseInt(sline[0]);
            if (cls == 1) {
                positiveSet.add(count);
            }
            count++;
        }
        testIn.close();
        System.out.println("# positive set: " + positiveSet.size());

        // Load predicted scores
        RankingItemList<Integer> rankingItemList = new RankingItemList<Integer>();

        BufferedReader resultIn = IOUtils.getBufferedReader(resultFile);
        count = 0;
        while ((line = resultIn.readLine()) != null) {
            double score = Double.parseDouble(line);
            rankingItemList.addRankingItem(new RankingItem<Integer>(count, score));
            count++;
        }
        resultIn.close();
        rankingItemList.sortDescending();
        System.out.println("# total data points: " + rankingItemList.size());

        performance = new RankingPerformance<Integer>(rankingItemList, positiveSet,
                evaluateFolder.getAbsolutePath());
        performance.outputRankingResultsWithGroundtruth();
        performance.computePrecisionsAndRecalls();
        performance.outputPrecisionRecallF1();
        performance.outputAUCListFile();
        performance.computeAUC();
        performance.outputAUC();
    }

    //String testFilePath, String resultFilePath, String outputFilePath
    public int[][] confusionMatrix(File testingFilePath, File resultFile,
            File confusionMatrixFile) throws Exception {
        int[][] conMatrix = new int[2][2];
        BufferedReader testIn = IOUtils.getBufferedReader(testingFilePath);
        BufferedReader resultIn = IOUtils.getBufferedReader(resultFile);

        ArrayList<Integer> test = new ArrayList<Integer>();
        ArrayList<Integer> result = new ArrayList<Integer>();

        String line;
        while ((line = testIn.readLine()) != null) {
            String[] sline = line.split(" ");
            test.add(Integer.valueOf(sline[0]));
        }

        while ((line = resultIn.readLine()) != null) {
            double res = Double.parseDouble(line);
            if (res > 0) {
                result.add(1);
            } else {
                result.add(-1);
            }
        }

        for (int i = 0; i < test.size(); i++) {
            int actual = test.get(i);
            int predict = result.get(i);
            if (actual == 1 && predict == 1) {
                conMatrix[0][0]++;
            } else if (actual == 1 && predict == -1) {
                conMatrix[1][0]++;
            } else if (actual == -1 && predict == 1) {
                conMatrix[0][1]++;
            } else if (actual == -1 && predict == -1) {
                conMatrix[1][1]++;
            } else {
                System.out.println("Error");
            }
        }

        testIn.close();
        resultIn.close();

        for (int[] row : conMatrix) {
            for (int j = 0; j < conMatrix[0].length; j++) {
                System.out.print(row[j] + "\t");
            }
            System.out.println();
        }

        // output
        BufferedWriter writer = IOUtils.getBufferedWriter(confusionMatrixFile);
        writer.write(conMatrix[0][0] + "\t" + conMatrix[0][1] + "\n");
        writer.write(conMatrix[1][0] + "\t" + conMatrix[1][1] + "\n");
        writer.close();

        return conMatrix;
    }

    public double[] getFeatureWeights(File modelFile, File perlSvm2Weight,
            File featureWeightsFile) throws Exception {

        String cmd = "perl " + perlSvm2Weight.getAbsolutePath()
                + " " + modelFile.getAbsolutePath();
        System.out.println("Processing command " + cmd);
        Process proc = Runtime.getRuntime().exec(cmd);
        InputStream istr = proc.getInputStream();
        BufferedReader in = new BufferedReader(new InputStreamReader(istr));
        String line;
        ArrayList<String> w = new ArrayList<String>();
        while ((line = in.readLine()) != null) {
            String[] sline = line.split(":");
            if (sline.length > 1) {
                w.add(sline[1]);
            } else {
                w.add("0.0");
            }
        }
        double[] weights = new double[w.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Double.parseDouble(w.get(i));
        }

        //output
        BufferedWriter writer = IOUtils.getBufferedWriter(featureWeightsFile);
        for (int i = 0; i < weights.length; i++) {
            writer.write(weights[i] + "\n");
        }
        writer.close();

        return weights;
    }

    public double[] loadFeatureWeights(File featureWeightsFile) throws Exception {
        ArrayList<Double> weights = new ArrayList<Double>();
        BufferedReader reader = IOUtils.getBufferedReader(featureWeightsFile);
        String line;
        while ((line = reader.readLine()) != null) {
            weights.add(Double.parseDouble(line));
        }
        reader.close();
        double[] featureWeights = new double[weights.size()];
        for (int i = 0; i < featureWeights.length; i++) {
            featureWeights[i] = weights.get(i);
        }
        return featureWeights;
    }

    public RankingPerformance<Integer> getPerformanceMeasure() {
        return this.performance;
    }
}
