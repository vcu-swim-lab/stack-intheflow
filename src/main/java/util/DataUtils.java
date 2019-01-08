package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.*;

/**
 *
 * @author vietan
 */
public class DataUtils {

    public static Set<String> loadStopwords(String stopwordFilepath) {
        Set<String> stopwords = new HashSet<String>();
        try {
            BufferedReader reader = IOUtils.getBufferedReader(stopwordFilepath);
            String line;
            while ((line = reader.readLine()) != null) {
                stopwords.add(line);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return stopwords;
    }

    public static Hashtable<String, Integer> changeFrequency(
            Hashtable<String, Integer> freqTable, String term, int delta) {
        Integer freq = freqTable.get(term);
        if (freq == null) {
            freqTable.put(term, delta);
        } else {
            freqTable.put(term, freq + delta);
        }
        return freqTable;
    }

    public static Hashtable<String, Integer> incrementFrequency(
            Hashtable<String, Integer> freqTable, String term) {
        Integer freq = freqTable.get(term);
        if (freq == null) {
            freqTable.put(term, 1);
        } else {
            freqTable.put(term, freq + 1);
        }
        return freqTable;
    }

    public static HashMap<String, Integer> incrementFrequency(
            HashMap<String, Integer> freqTable, String term) {
        Integer freq = freqTable.get(term);
        if (freq == null) {
            freqTable.put(term, 1);
        } else {
            freqTable.put(term, freq + 1);
        }
        return freqTable;
    }

    public static HashMap<String, Integer> changeFrequency(
            HashMap<String, Integer> freqTable, String term, int delta) {
        Integer freq = freqTable.get(term);
        if (freq == null) {
            freqTable.put(term, delta);
        } else {
            freqTable.put(term, freq + delta);
        }
        return freqTable;
    }

    public static ArrayList<RankingItem<String>> createWordVocab(
            Hashtable<String, Integer> termFreq,
            Hashtable<String, Integer> docFreq,
            Set<String> stopwords) {
        ArrayList<RankingItem<String>> rankTokens = new ArrayList<RankingItem<String>>();
        for (String token : termFreq.keySet()) {
            if (stopwords.contains(token)
                    || token.length() < 3
                    || (token.length() == 3 && token.contains("'"))
                    || token.contains("...")
                    || token.contains("\"")
                    || token.matches("[^A-Za-z]+")
                    || token.matches("([A-Z]+)")
                    || token.matches("[^A-Za-z]+.*")) {
                continue;
            }

            double tfidf = (double) termFreq.get(token) / docFreq.get(token);
            rankTokens.add(new RankingItem<String>(token, tfidf, termFreq.get(token)));
        }
        Collections.sort(rankTokens);

        return rankTokens;
    }

    public static void outputRawWordVocab(String filepath,
            ArrayList<RankingItem<String>> rankTokens,
            Hashtable<String, Integer> termFreq,
            Hashtable<String, Integer> docFreq) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < rankTokens.size(); i++) {
            String token = rankTokens.get(i).getObject();
            writer.write(i
                    + "\t" + rankTokens.get(i).getObject()
                    + "\t" + rankTokens.get(i).getPrimaryValue()
                    + "\t" + termFreq.get(token)
                    + "\t" + docFreq.get(token)
                    + "\n");
        }
        writer.close();
    }
    
    public static void outputVocab(File filepath, ArrayList<String> vocab) throws Exception {
        outputVocab(filepath.getAbsolutePath(), vocab);
    }

    public static void outputVocab(String filepath, ArrayList<String> vocab) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (String element : vocab) {
            writer.write(element + "\n");
        }
        writer.close();
    }

    public static ArrayList<String> inputWordVocab(String filepath) throws Exception {
        ArrayList<String> wordVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        while ((line = reader.readLine()) != null) {
            wordVocab.add(line);
        }
        reader.close();
        return wordVocab;
    }

    public static int getIndex(ArrayList<String> list, String value) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i).equals(value)) {
                return i;
            }
        }
        return -1;
    }
}
