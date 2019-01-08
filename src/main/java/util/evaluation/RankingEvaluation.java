package util.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import util.IOUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public class RankingEvaluation {

    public static final String AUCCalculatorPath = "lib/auc.jar";
    private final double[] scores;
    private final Set<Integer> relevants;
    // internal
    private final ArrayList<Measurement> measurements;
    private final ArrayList<RankingItem<Integer>> ranking;
    private String aucListFile;

    public RankingEvaluation(double[] scores, Set<Integer> relevants) {
        this.scores = scores;
        this.relevants = relevants;
        this.measurements = new ArrayList<Measurement>();
        this.ranking = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < this.scores.length; ii++) {
            this.ranking.add(new RankingItem<Integer>(ii, scores[ii]));
        }
        Collections.sort(this.ranking);
    }

    public void setAUCListFile(String auc) {
        this.aucListFile = auc;
    }

    public ArrayList<Measurement> getMeasurements() {
        return this.measurements;
    }

    public void computePRF() {
        double[] precisions = new double[scores.length];
        double[] recalls = new double[scores.length];
        double[] f1s = new double[scores.length];
        int correctCount = 0;
        for (int i = 0; i < this.ranking.size(); i++) {
            int item = this.ranking.get(i).getObject();
            if (this.relevants.contains(item)) {
                correctCount++;
            }

            precisions[i] = (double) correctCount / (i + 1);
            recalls[i] = (double) correctCount / this.relevants.size();
            if (precisions[i] == 0 && recalls[i] == 0) {
                f1s[i] = 0;
            } else {
                f1s[i] = (2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]);
            }
        }

        this.measurements.add(new Measurement("N", ranking.size()));
        this.measurements.add(new Measurement("#positives", relevants.size()));

        this.measurements.add(new Measurement("PRF1", f1s[this.relevants.size() - 1]));
        this.measurements.add(new Measurement("P@1", precisions[0]));
        this.measurements.add(new Measurement("R@1", recalls[0]));
        this.measurements.add(new Measurement("F1@1", f1s[0]));

        this.measurements.add(new Measurement("P@5", precisions[4]));
        this.measurements.add(new Measurement("R@5", recalls[4]));
        this.measurements.add(new Measurement("F1@5", f1s[4]));

        this.measurements.add(new Measurement("P@10", precisions[9]));
        this.measurements.add(new Measurement("R@10", recalls[9]));
        this.measurements.add(new Measurement("F1@10", f1s[9]));
    }

    public void computeAUCs() {
        try {
            File aucFile = new File(AUCCalculatorPath);
            if (!aucFile.exists()) {
                throw new RuntimeException(AUCCalculatorPath + " not found.");
            }

            // create a temp folder
            File tempFolder = new File("temp");
            if (aucListFile == null) {
                IOUtils.createFolder(tempFolder);
                this.aucListFile = new File(tempFolder, "auc-list").getAbsolutePath();
            }
            // output temporary results
            BufferedWriter writer = IOUtils.getBufferedWriter(aucListFile);
            for (RankingItem<Integer> rankingItem : this.ranking) {
                int item = rankingItem.getObject();
                double value = rankingItem.getPrimaryValue();
                int cls;
                if (this.relevants.contains(item)) {
                    cls = 1;
                } else {
                    cls = 0;
                }
                writer.write(value + " " + cls + "\n");
            }
            writer.close();

            // compute 
            String cmd = "java -jar " + AUCCalculatorPath + " " + aucListFile + " list";

            Process proc = Runtime.getRuntime().exec(cmd);
            InputStream istr = proc.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(istr));

            String line;
            while ((line = in.readLine()) != null) {
                if (line.contains("Area Under the Curve for Precision - Recall is")) {
                    String[] sline = line.split(" ");
                    double aucPR = Double.parseDouble(sline[sline.length - 1]);
                    this.measurements.add(new Measurement("AUC-PRC", aucPR));
                } else if (line.contains("Area Under the Curve for ROC is")) {
                    String[] sline = line.split(" ");
                    double aucROC = Double.parseDouble(sline[sline.length - 1]);
                    this.measurements.add(new Measurement("AUC-ROC", aucROC));
                }
            }
            in.close();

            // delete the temp folder
            if (tempFolder.exists()) {
                IOUtils.deleteFolderContent(tempFolder.getAbsolutePath());
                tempFolder.delete();
            }
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing AUCs");
        }
    }
}
