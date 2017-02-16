package io.github.vcuswimlab.stackintheflow.controller;

import io.github.vcuswimlab.stackintheflow.controller.info.match.StringMatchUtils;
import io.github.vcuswimlab.stackintheflow.model.score.combiner.Combiner;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryGenerator {

    private static final int MAX_QUERY_TERMS = 3;


    public static String generateQuery(String editorText, Combiner combiner) {

        Set<String> imports = StringMatchUtils.extractImports(editorText);

        Set<String> terms = new HashSet<>();
        imports.forEach(i -> terms.addAll(Arrays.asList(i.toLowerCase().split("\\."))));
        Map<String, Double> scores = terms.stream().collect(Collectors.toMap(s -> s, combiner::generateCumulativeScore));

        //Collects the MAX_QUERY_TERMS most frequent elements in the list
        List<String> top = scores
                .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).limit(MAX_QUERY_TERMS)
                .map(Map.Entry::getKey).collect(Collectors.toList());

        return top.stream().collect(Collectors.joining(" "));
    }
}
