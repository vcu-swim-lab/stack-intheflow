package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.editor.CaretModel;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.editor.LogicalPosition;
import io.github.vcuswimlab.stackintheflow.controller.info.match.StringMatchUtils;
import io.github.vcuswimlab.stackintheflow.model.score.combiner.Combiner;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Chase on 1/7/2017.
 */
public class AutoQueryGenerator {

    private static final int MAX_QUERY_TERMS = 3;


    public static String generateQuery(Editor editor, Combiner combiner) {

        CaretModel caretModel = editor.getCaretModel();
        LogicalPosition logicalPosition = caretModel.getLogicalPosition();

        final Document document = editor.getDocument();
        String editorText = document.getText();

        Set<String> imports = StringMatchUtils.extractImports(editorText);

        Map<String, Integer> termsFreqMap = new HashMap<>();
        imports.forEach(i -> Arrays.stream(i.toLowerCase().split("\\."))
                .forEach(t -> termsFreqMap.put(t, 1 + termsFreqMap.getOrDefault(t, 0))));

        String[] lines = editorText.split("\\n");
        String currentLine = lines[logicalPosition.line];

        Arrays.stream(currentLine.toLowerCase().split("\\b"))
                .forEach(t -> termsFreqMap.put(t, 2 + termsFreqMap.getOrDefault(t, 0)));

        Map<String, Double> scores =
                termsFreqMap.entrySet().stream()
                        .collect(Collectors.toMap(Map.Entry::getKey, e -> combiner.generateCumulativeScore(e.getKey())));

        //Collects the MAX_QUERY_TERMS most frequent elements in the list
        List<String> top = scores
                .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).limit(MAX_QUERY_TERMS)
                .map(Map.Entry::getKey).collect(Collectors.toList());

        return top.stream().collect(Collectors.joining(" "));
    }
}
