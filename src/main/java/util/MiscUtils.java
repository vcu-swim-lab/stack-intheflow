package util;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import sampling.util.SparseCount;

/**
 *
 * @author vietan
 */
public class MiscUtils {

    protected static final NumberFormat formatter = new DecimalFormat("###.###");
    
    public static ArrayList<ArrayList<String>> getPowerSet(ArrayList<String> list) {
        ArrayList<ArrayList<String>> powerSet = new ArrayList<>();
        powerSet.add(new ArrayList<String>());
        for (String item : list) {
            ArrayList<ArrayList<String>> newPs = new ArrayList<>();
            for (ArrayList<String> subset : powerSet) {
                newPs.add(subset);

                ArrayList<String> newSubset = new ArrayList<String>(subset);
                newSubset.add(item);
                newPs.add(newSubset);
            }
            powerSet = newPs;
        }
        return powerSet;
    }

    public static String[] getTopWords(ArrayList<String> wordVocab,
            double[] probs, int numTopObs) {
        ArrayList<RankingItem<Integer>> rankObs = new ArrayList<RankingItem<Integer>>();
        for (int vv = 0; vv < probs.length; vv++) {
            rankObs.add(new RankingItem<Integer>(vv, probs[vv]));
        }
        Collections.sort(rankObs);
        String[] topWords = new String[numTopObs];
        for (int ii = 0; ii < numTopObs; ii++) {
            topWords[ii] = wordVocab.get(rankObs.get(ii).getObject());
        }
        return topWords;
    }

    public static String getTopObservations(ArrayList<String> wordVocab,
            double[] probs, int numTopObs) {
        ArrayList<RankingItem<Integer>> rankObs = new ArrayList<RankingItem<Integer>>();
        for (int vv = 0; vv < probs.length; vv++) {
            rankObs.add(new RankingItem<Integer>(vv, probs[vv]));
        }
        Collections.sort(rankObs);
        StringBuilder str = new StringBuilder();
        for (int ii = 0; ii < Math.min(numTopObs, rankObs.size()); ii++) {
            RankingItem<Integer> obs = rankObs.get(ii);
            str.append(wordVocab.get(obs.getObject())).append(":")
                    .append(MiscUtils.formatDouble(obs.getPrimaryValue())).append(" ");
        }
        return str.toString();
    }

    public static String getTopObservations(ArrayList<String> wordVocab,
            SparseCount counts, int numTopObs) {
        ArrayList<RankingItem<Integer>> rankObs = new ArrayList<RankingItem<Integer>>();
        for (int obs : counts.getIndices()) {
            rankObs.add(new RankingItem<Integer>(obs, counts.getCount(obs)));
        }
        Collections.sort(rankObs);
        StringBuilder str = new StringBuilder();
        for (int ii = 0; ii < Math.min(numTopObs, rankObs.size()); ii++) {
            RankingItem<Integer> obs = rankObs.get(ii);
            str.append(wordVocab.get(obs.getObject())).append(":")
                    .append(MiscUtils.formatDouble(obs.getPrimaryValue())).append(" ");
        }
        return str.toString();
    }

    public static double[] getIDFs(int[][] words, ArrayList<Integer> docIndices, int V) {
        int D = docIndices.size();
        int[] dfs = new int[V];
        for (int dd : docIndices) {
            Set<Integer> uniqueWords = new HashSet<Integer>();
            for (int nn = 0; nn < words[dd].length; nn++) {
                uniqueWords.add(words[dd][nn]);
            }
            for (int uw : uniqueWords) {
                dfs[uw]++;
            }
        }

        double[] idfs = new double[V];
        for (int vv = 0; vv < V; vv++) {
            idfs[vv] = Math.log(D) - Math.log(dfs[vv] + 1);
        }
        return idfs;
    }

    public static double[] getIDFs(int[][] words, int V) {
        int D = words.length;
        int[] dfs = new int[V];
        for (int d = 0; d < D; d++) {
            Set<Integer> uniqueWords = new HashSet<Integer>();
            for (int n = 0; n < words[d].length; n++) {
                uniqueWords.add(words[d][n]);
            }
            for (int uw : uniqueWords) {
                dfs[uw]++;
            }
        }

        double[] idfs = new double[V];
        for (int v = 0; v < V; v++) {
            idfs[v] = Math.log(D) - Math.log(dfs[v] + 1);
        }
        return idfs;
    }

    public static ArrayList<RankingItem<Integer>> getRankingList(SparseVector vector) {
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int idx : vector.getIndices()) {
            rankItems.add(new RankingItem<Integer>(idx, vector.get(idx)));
        }
        Collections.sort(rankItems);
        return rankItems;
    }

    public static ArrayList<RankingItem<Integer>> getRankingList(SparseCount counts) {
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int idx : counts.getIndices()) {
            rankItems.add(new RankingItem<Integer>(idx, counts.getCount(idx)));
        }
        Collections.sort(rankItems);
        return rankItems;
    }

    public static ArrayList<RankingItem<Integer>> getRankingList(double[] scores) {
        ArrayList<RankingItem<Integer>> rankItems = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < scores.length; ii++) {
            rankItems.add(new RankingItem<Integer>(ii, scores[ii]));
        }
        Collections.sort(rankItems);
        return rankItems;
    }

    public static void incrementMap(HashMap<Integer, Integer> map, Integer key) {
        Integer count = map.get(key);
        if (count == null) {
            map.put(key, 1);
        } else {
            map.put(key, count + 1);
        }
    }

    public static void incrementMap(HashMap<String, Integer> map, String key) {
        Integer count = map.get(key);
        if (count == null) {
            map.put(key, 1);
        } else {
            map.put(key, count + 1);
        }
    }

    public static int getRoundStepSize(int total, int numSteps) {
        int stepSize = (int) Math.pow(10, (int) Math.log10(total / numSteps));
        if (stepSize == 0) {
            return 1;
        }
        return stepSize;
    }

    public static double[] flatten2DArray(double[][] array) {
        int length = 0;
        for (double[] arr : array) {
            length += arr.length;
        }
        double[] flattenArray = new double[length];
        int count = 0;
        for (double[] arr : array) {
            for (int j = 0; j < arr.length; j++) {
                flattenArray[count++] = arr[j];
            }
        }
        return flattenArray;
    }

    public static ArrayList<Double> stringToList(String str) {
        str = str.substring(1, str.length() - 1);
        String[] sline = str.split(",");
        ArrayList<Double> list = new ArrayList<>();
        for (String s : sline) {
            list.add(Double.parseDouble(s));
        }
        return list;
    }

    public static String listToString(List<Double> list) {
        if (list.isEmpty()) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(list.get(0)));
        for (int i = 1; i < list.size(); i++) {
            str.append(", ").append(formatDouble(list.get(i)));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(String[] array) {
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(array[0]);
        for (int i = 1; i < array.length; i++) {
            str.append(" ").append(array[i]);
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(double[] array) {
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static double[] stringToDoubleArray(String str) {
        str = str.substring(1, str.length() - 1);
        String[] sline = str.split(",");
        double[] arr = new double[sline.length];
        for (int ii = 0; ii < arr.length; ii++) {
            arr[ii] = Double.parseDouble(sline[ii]);
        }
        return arr;
    }

    public static String arrayToString(float[] array) {
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToString(int[] array) {
        if (array.length == 0) {
            return "[]";
        }
        StringBuilder str = new StringBuilder();
        str.append("[").append(formatDouble(array[0]));
        for (int i = 1; i < array.length; i++) {
            str.append(", ").append(formatDouble(array[i]));
        }
        str.append("]");
        return str.toString();
    }

    public static String arrayToSVMLightString(int[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String arrayToSVMLightString(float[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String arrayToSVMLightString(double[] array) {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                str.append(i).append(":").append(array[i]).append(" ");
            }
        }
        return str.toString();
    }

    public static String formatDouble(double value) {
        return formatter.format(value);
    }

    public static String formatDouble(double value, int n) {
        StringBuilder str = new StringBuilder("###.");
        for (int nn = 0; nn < n; nn++) {
            str.append("#");
        }
        NumberFormat newFormatter = new DecimalFormat(str.toString());
        return newFormatter.format(value);
    }
}
