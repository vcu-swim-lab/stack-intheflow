package io.github.vcuswimlab.stackintheflow.controller.info;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by chase on 2/1/17.
 */
public class TfIdf {

    public static Map<String, Integer> getTermFrequencies(String document, Pattern pattern) {

        Map<String, Integer> termFrequencies = new HashMap<>();
        Matcher matcher = pattern.matcher(document);

        while(matcher.find()) {
            String term = matcher.group(1);
            termFrequencies.put(term, termFrequencies.getOrDefault(term, 0) + 1);
        }

        return termFrequencies;
    }

    public static double getInverseDocumentFrequency(String term, String[] documents) {

        int termFrequency = 0;

        Pattern pattern = Pattern.compile("\\b"+term+"\\b");

        for(String doc : documents) {
            Matcher matcher = pattern.matcher(doc);
            while (matcher.find()) {
                termFrequency++;
            }
        }

        return Math.log(documents.length / termFrequency);
    }
}
