package util.evaluation;

/**
 *
 * @author vanguyen
 */
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Set;
import util.IOUtils;
import util.RankingItem;
import util.RankingItemList;

/**
 * This is to replace the class Evaluation
 * @param <A>
 */
public class RankingPerformance<A> {

    public static final String AUCCalculatorPath = "lib/auc.jar";
    public static final String AUCListFile = "AUCList.txt";
    public static final String AUCFile = "AUC.txt";
    public static final String PRF1File = "PRF1.txt";
    public static final String NDCGFile = "NDCG.txt";
    public static final String RankedResultsWithGroundtruthFile = "RankingResultWithGroundtruth.txt";
    // Inputs
    protected RankingItemList<A> rankingItemLists;
    protected Set<A> groundtruthSet;
    protected String performanceFolder;
    // Outputs
    protected double[] precisions;
    protected double[] recalls;
    protected double[] f1s;
    protected double[] ndcgs;
    protected double aucROC;
    protected double aucPR;
    protected double f1;
    
    public RankingPerformance(RankingItemList<A> rankedItems, Set<A> groundtruthSet, String folder) {
        this.rankingItemLists = rankedItems;
        this.groundtruthSet = groundtruthSet;
        this.performanceFolder = folder;

        this.precisions = new double[this.rankingItemLists.size()];
        this.recalls = new double[this.rankingItemLists.size()];
        this.f1s = new double[this.rankingItemLists.size()];
    }

    public RankingPerformance(RankingItemList<A> rankedItems, String folder) {
        this.rankingItemLists = rankedItems;
        this.performanceFolder = folder;

        this.ndcgs = new double[this.rankingItemLists.size()];
    }

    public RankingPerformance(String folder) {
        this.performanceFolder = folder;
    }

    public void loadAUCAndF1() {
        try {
            BufferedReader reader = IOUtils.getBufferedReader(new File(performanceFolder, AUCFile));

            //AUC_ROC
            String line = reader.readLine();
            String[] sline = line.split("\t");
            this.aucROC = Double.parseDouble(sline[1]);

            // AUC_PR
            line = reader.readLine();
            sline = line.split("\t");
            this.aucPR = Double.parseDouble(sline[1]);

            // F1
            line = reader.readLine();
            sline = line.split("\t");
            this.f1 = Double.parseDouble(sline[1]);

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public double getAUCROC() {
        return this.aucROC;
    }

    public double getAUCPR() {
        return this.aucPR;
    }

    public double getF1() {
        return this.f1;
    }

    public double[] getNDCGs() {
        return this.ndcgs;
    }

    public static double[] inputNDCG(File ndcgFile) {
        double[] ndcg = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(ndcgFile);
            String line;
            String[] sline;
            ArrayList<String> ndcgList = new ArrayList<String>();
            reader.readLine();
            while ((line = reader.readLine()) != null) {
                sline = line.split("\t");
                ndcgList.add(sline[sline.length - 1]);
            }
            ndcg = new double[ndcgList.size()];
            for (int i = 0; i < ndcg.length; i++) {
                ndcg[i] = Double.parseDouble(ndcgList.get(i));
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing NDCG from " + ndcgFile);
        }
        return ndcg;
    }

    public void inputNDCGs() {
        this.ndcgs = inputNDCG(new File(performanceFolder, NDCGFile));
    }

    public void computeAndOutputNDCGsNormalize(RankingItemList<A> groundtruthRakingItems) {
        // groundtruth
        double minTrueVal = Double.MAX_VALUE;
        for (int ii = 0; ii < groundtruthRakingItems.size(); ii++) {
            double val = groundtruthRakingItems.getRankingItem(ii).getPrimaryValue();
            if (minTrueVal > val) {
                minTrueVal = val;
            }
        }

        RankingItemList<A> normTrueList = new RankingItemList<A>();
        for (int i = 0; i < this.rankingItemLists.size(); i++) {
            A rec_item = this.rankingItemLists.getRankingItem(i).getObject();
            RankingItem<A> groundtruth_item = groundtruthRakingItems.get(rec_item);
            if (groundtruth_item == null) {
                normTrueList.addRankingItem(new RankingItem<A>(rec_item, 0));
            } else {
                double val = groundtruth_item.getPrimaryValue();
                normTrueList.addRankingItem(new RankingItem<A>(rec_item, val - minTrueVal));
            }
        }
        normTrueList.sortDescending();

        // pred list
        double minPredVal = Double.MAX_VALUE;
        for (int ii = 0; ii < rankingItemLists.size(); ii++) {
            double val = rankingItemLists.getRankingItem(ii).getPrimaryValue();
            if (minPredVal > val) {
                minPredVal = val;
            }
        }

        RankingItemList<A> normPredList = new RankingItemList<A>();
        for (int ii = 0; ii < rankingItemLists.size(); ii++) {
            RankingItem<A> item = rankingItemLists.getRankingItem(ii);
            RankingItem<A> normItem = new RankingItem<A>(item.getObject(),
                    item.getPrimaryValue() - minPredVal);
            normPredList.addRankingItem(normItem);
        }
        normPredList.sortDescending();

        // create hashtable for easy locating items
        Hashtable<A, RankingItem<A>> groundtruth_table = new Hashtable<A, RankingItem<A>>();
        for (RankingItem<A> item : normTrueList.getRankingItems()) {
            groundtruth_table.put(item.getObject(), item);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder, NDCGFile));
            writer.write("order\tpredicted_value\ttrue_value\taccumulated_gain"
                    + "\tnormalization_value\tndcg\n");

            double predicted_value, true_value, value;
            double accumulated_gain = 0;
            double normalization_value = 0;

            for (int i = 0; i < this.rankingItemLists.size(); i++) {
                RankingItem<A> rec_item = normPredList.getRankingItem(i);
                predicted_value = groundtruth_table.get(rec_item.getObject()).getPrimaryValue();
                //value = (Math.pow(2.0, Math.log(predicted_value + 1)) - 1) / (Math.log(i + 2));
                value = (Math.pow(2.0, predicted_value) - 1) / ((Math.log(i + 2) / Math.log(2)));
                accumulated_gain += value;

                true_value = groundtruth_table.get(normTrueList.getRankingItem(i).getObject()).getPrimaryValue();
                value = (Math.pow(2.0, true_value) - 1) / ((Math.log(i + 2) / Math.log(2)));
                normalization_value += value;

                ndcgs[i] = accumulated_gain / normalization_value;
                writer.write((i + 1) + "\t" + predicted_value + "\t" + true_value
                        + "\t" + accumulated_gain + "\t" + normalization_value + "\t" + ndcgs[i] + "\n");
            }

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing NDCG");
        }
    }

    /**
     * Compute and output NDCG scores of a rank list. Note that the ranking
     * score should be positive.
     *
     * @param groundtruthRakingItems The true ranking list with ground truth
     * scores
     */
    public void computeAndOutputNDCGs(RankingItemList<A> groundtruthRakingItems) {
        RankingItemList<A> pseudo_groundtruth_list = new RankingItemList<A>();
        for (int i = 0; i < this.rankingItemLists.size(); i++) {
            A rec_item = this.rankingItemLists.getRankingItem(i).getObject();
            RankingItem<A> groundtruth_item = groundtruthRakingItems.get(rec_item);
            if (groundtruth_item == null) {
                pseudo_groundtruth_list.addRankingItem(new RankingItem<A>(rec_item, 0));
            } else {
                pseudo_groundtruth_list.addRankingItem(groundtruth_item);
            }
        }
        pseudo_groundtruth_list.sortDescending();

        // create hashtable for easy locating items
        Hashtable<A, RankingItem<A>> groundtruth_table = new Hashtable<A, RankingItem<A>>();
        for (RankingItem<A> item : pseudo_groundtruth_list.getRankingItems()) {
            groundtruth_table.put(item.getObject(), item);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder, NDCGFile));
            writer.write("order\tpredicted_value\ttrue_value\taccumulated_gain"
                    + "\tnormalization_value\tndcg\n");

            double predicted_value, true_value, value;
            double accumulated_gain = 0;
            double normalization_value = 0;

            for (int i = 0; i < this.rankingItemLists.size(); i++) {
                RankingItem<A> rec_item = this.rankingItemLists.getRankingItem(i);
                predicted_value = groundtruth_table.get(rec_item.getObject()).getPrimaryValue();
                //value = (Math.pow(2.0, Math.log(predicted_value + 1)) - 1) / (Math.log(i + 2));
                value = (Math.pow(2.0, predicted_value) - 1) / ((Math.log(i + 2) / Math.log(2)));
                accumulated_gain += value;

                true_value = groundtruth_table.get(pseudo_groundtruth_list.getRankingItem(i).getObject()).getPrimaryValue();
                value = (Math.pow(2.0, true_value) - 1) / ((Math.log(i + 2) / Math.log(2)));
                normalization_value += value;

                ndcgs[i] = accumulated_gain / normalization_value;
                writer.write((i + 1) + "\t" + predicted_value + "\t" + true_value
                        + "\t" + accumulated_gain + "\t" + normalization_value + "\t" + ndcgs[i] + "\n");
            }

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while computing NDCG");
        }
    }

    public void outputRankingResultsWithGroundtruth() {
        System.out.println("Outputing ranking results with groundtruth ... "
                + this.performanceFolder
                + RankedResultsWithGroundtruthFile);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder,
                    RankedResultsWithGroundtruthFile));
            for (RankingItem<A> rankingItem : this.rankingItemLists.getRankingItems()) {
                A obj = rankingItem.getObject();
                int rankingOrder = rankingItem.getRankingOrder();
                double rankingScore = rankingItem.getPrimaryValue();
                int cls = 0;
                if (this.groundtruthSet.contains(obj)) {
                    cls = 1;
                }
                writer.write(rankingOrder + "\t" + rankingScore + "\t" + obj.toString() + "\t" + cls + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void computePrecisionsAndRecalls() {
        System.out.println("Computing precision and recall ...");

        int correctCount = 0;
        for (int i = 0; i < this.rankingItemLists.size(); i++) {
            A item = this.rankingItemLists.getRankingItem(i).getObject();
            if (groundtruthSet.contains(item)) {
                correctCount++;
            }

            this.precisions[i] = (double) correctCount / (i + 1);
            this.recalls[i] = (double) correctCount / this.groundtruthSet.size();
            if (this.precisions[i] == 0 && this.recalls[i] == 0) {
                this.f1s[i] = 0;
            } else {
                this.f1s[i] = (2 * this.precisions[i] * this.recalls[i]) / (this.precisions[i] + this.recalls[i]);
            }
        }

        this.f1 = this.f1s[this.groundtruthSet.size() - 1];
    }

    public void outputPrecisionRecallF1() {
        System.out.println("Outputing precision and recall ... " + this.performanceFolder + PRF1File);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder, PRF1File));
            writer.write("k\tPrecision\tRecall\tF1\n");
            for (int i = 0; i < this.rankingItemLists.size(); i++) {
                writer.write(Integer.toString(i + 1));
                writer.write("\t" + this.precisions[i]);
                writer.write("\t" + this.recalls[i]);
                writer.write("\t" + this.f1s[i]);
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void outputAUCListFile() {
        System.out.println("Outputing AUC List file ... " + this.performanceFolder + AUCListFile);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder, AUCListFile));
            for (int i = 0; i < this.rankingItemLists.size(); i++) {
                A item = this.rankingItemLists.getRankingItem(i).getObject();
                double value = this.rankingItemLists.getRankingItem(i).getPrimaryValue();
                int cls;
                if (this.groundtruthSet.contains(item)) {
                    cls = 1;
                } else {
                    cls = 0;
                }
                writer.write(value + " " + cls + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void computeAUC() {
        try {
            File aucFile = new File(AUCCalculatorPath);
            if (!aucFile.exists()) {
                throw new RuntimeException(AUCCalculatorPath + " not found.");
            }

            String cmd = "java -jar " + AUCCalculatorPath + " " + new File(performanceFolder, AUCListFile + " list");
            Process proc = Runtime.getRuntime().exec(cmd);
            InputStream istr = proc.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(istr));

            String line;
            while ((line = in.readLine()) != null) {
                if (line.contains("Area Under the Curve for Precision - Recall is")) {
                    String[] sline = line.split(" ");
                    this.aucPR = Double.parseDouble(sline[sline.length - 1]);
                } else if (line.contains("Area Under the Curve for ROC is")) {
                    String[] sline = line.split(" ");
                    this.aucROC = Double.parseDouble(sline[sline.length - 1]);
                }
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void outputAUC() {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(performanceFolder, AUCFile));
            writer.write("AUC_ROC\t" + this.aucROC + "\n");
            writer.write("AUC_PR\t" + this.aucPR + "\n");
            writer.write("F1\t" + this.f1s[this.groundtruthSet.size() - 1] + "\n");

            writer.write("#Positive: " + this.groundtruthSet.size() + "\n");
            writer.write("#Total: " + this.rankingItemLists.size());

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public RankingItemList<A> getRankingResult() {
        return this.rankingItemLists;
    }

    public Set<A> getGroundtruthSet() {
        return this.groundtruthSet;
    }

    public RankingItemList<A> getBestRanking() throws Exception {
        // result ranking
        RankingItemList<A> resultRank = new RankingItemList<A>();
        for (RankingItem<A> rankingItem : this.rankingItemLists.getRankingItems()) {
            resultRank.addRankingItem(rankingItem);
        }
        resultRank.sortDescending();

        // put the result ranking in a hashtable
        Hashtable<A, Double> resultRankedTable = new Hashtable<A, Double>();
        for (int i = 0; i < resultRank.size(); i++) {
            RankingItem<A> rankingItem = resultRank.getRankingItem(i);
            resultRankedTable.put(rankingItem.getObject(), rankingItem.getPrimaryValue());
        }

        // group the ties by their order
        HashMap<Integer, Set<A>> resultGroupByOrder = resultRank.getGroupByOrder();
        ArrayList<Integer> orders = new ArrayList<Integer>();
        for (Integer order : resultGroupByOrder.keySet()) {
            orders.add(order);
        }
        Collections.sort(orders); // get the ranking order

        // Best ranking result
        //ArrayList<RankingItem<A>> bestResultRanking = new ArrayList<RankingItem<A>>();
        RankingItemList<A> bestResultRanking = new RankingItemList<A>();
        for (Integer order : orders) {
            Set<A> ties = resultGroupByOrder.get(order);
            ArrayList<RankingItem<A>> tiesRankedByActualValue = new ArrayList<RankingItem<A>>();
            for (A tie : ties) {
                double groundtruthValue = 0;
                if (this.groundtruthSet.contains(tie)) {
                    groundtruthValue = 1;
                }
                RankingItem<A> rankingItem = new RankingItem<A>(tie, groundtruthValue);
                tiesRankedByActualValue.add(rankingItem);
            }
            Collections.sort(tiesRankedByActualValue);

            for (RankingItem<A> rankingItem : tiesRankedByActualValue) {
                A item = rankingItem.getObject();
                double resultValue = resultRankedTable.get(item);
                RankingItem<A> resultItem = new RankingItem<A>(item, resultValue);
                resultItem.setRankingOrder(order);
                //bestResultRanking.add(resultItem);
                bestResultRanking.addRankingItem(resultItem);
            }
        }

        return bestResultRanking;
    }

    public RankingItemList<A> getWorstRanking() throws Exception {
        // result ranking
        RankingItemList<A> resultRank = new RankingItemList<A>();
        ArrayList<RankingItem<A>> rankingItems = this.getRankingResult().getRankingItems();
        for (RankingItem<A> rankingItem : rankingItems) {
            resultRank.addRankingItem(rankingItem);
        }
        resultRank.sortDescending();

        // put the result ranking in a hashtable
        Hashtable<A, Double> resultRankedTable = new Hashtable<A, Double>();
        for (int i = 0; i < resultRank.size(); i++) {
            RankingItem<A> rankingItem = resultRank.getRankingItem(i);
            resultRankedTable.put(rankingItem.getObject(), rankingItem.getPrimaryValue());
        }

        // group the ties by their order
        HashMap<Integer, Set<A>> resultGroupByOrder = resultRank.getGroupByOrder();
        ArrayList<Integer> orders = new ArrayList<Integer>();
        for (Integer order : resultGroupByOrder.keySet()) {
            orders.add(order);
        }
        Collections.sort(orders); // get the ranking order

        // Worst ranking result
        RankingItemList<A> worstResultRanking = new RankingItemList<A>();
        for (Integer order : orders) {
            Set<A> ties = resultGroupByOrder.get(order);
            ArrayList<RankingItem<A>> tiesRankedByActualValue = new ArrayList<RankingItem<A>>();
            for (A tie : ties) {
                double groundtruthValue = 0;
                if (this.getGroundtruthSet().contains(tie)) {
                    groundtruthValue = 1;
                }
                RankingItem<A> rankingItem = new RankingItem<A>(tie, groundtruthValue);
                tiesRankedByActualValue.add(rankingItem);
            }
            Collections.sort(tiesRankedByActualValue);
            Collections.reverse(tiesRankedByActualValue);

            for (RankingItem<A> rankingItem : tiesRankedByActualValue) {
                A item = rankingItem.getObject();
                double resultValue = resultRankedTable.get(item);
                RankingItem<A> resultItem = new RankingItem<A>(item, resultValue);
                resultItem.setRankingOrder(order);
                worstResultRanking.addRankingItem(resultItem);
            }
        }
        return worstResultRanking;
    }

    public boolean exists() {
        File aFile = new File(performanceFolder, AUCFile);
        return aFile.exists();
    }
}
