package io.github.vcuswimlab.stackintheflow.controller;

import io.github.vcuswimlab.stackintheflow.controller.info.TfIdf;
import io.github.vcuswimlab.stackintheflow.controller.info.match.StringMatchUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryGenerator {

    private static final int MAX_QUERY_TERMS = 5;


    public static String generateQuery(String editorText) {

        String cleanedText = StringMatchUtils.removeComments(editorText);

        //Collects the MAX_QUERY_TERMS most frequent elements in the list
        List<String> top = TfIdf.getTermFrequencies(cleanedText, StringMatchUtils.TERM_PATTERN)
                .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).limit(MAX_QUERY_TERMS)
                .map(Map.Entry::getKey).collect(Collectors.toList());

        return top.stream().collect(Collectors.joining(" "));
    }
}
