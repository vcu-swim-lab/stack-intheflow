/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package core.crossvalidation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Each instance corresponds to a document which has a bag of words and a
 * response variable.
 *
 * @author vietan
 */
public class RegressionDocumentInstance extends Instance<String> {

    private int[] tokens;
    private double response;

    public RegressionDocumentInstance(String id, int[] tokens, double response) {
        super(id);
        this.tokens = tokens;
        this.response = response;
    }

    public int[] getTokens() {
        return tokens;
    }

    public double getResponse() {
        return response;
    }

    private HashMap<Integer, Double> getFullVocabFeatures() {
        HashMap<Integer, Double> unigrams = new HashMap<Integer, Double>();
        for (int i = 0; i < tokens.length; i++) {
            Double count = unigrams.get(tokens[i]);
            if (count == null) {
                unigrams.put(tokens[i], 1.0);
            } else {
                unigrams.put(tokens[i], count + 1.0);
            }
        }
        return unigrams;
    }

    public String getFullVocabSVMLigthString() {
        StringBuilder str = new StringBuilder();
        str.append(response);
        HashMap<Integer, Double> features = getFullVocabFeatures();

        ArrayList<Integer> sortedFeatures = new ArrayList<Integer>();
        for (int f : features.keySet()) {
            sortedFeatures.add(f);
        }
        Collections.sort(sortedFeatures);

        for (int f : sortedFeatures) {
            str.append(" ").append(f + 1).append(":").append(features.get(f));
        }

        return str.toString();
    }
}
