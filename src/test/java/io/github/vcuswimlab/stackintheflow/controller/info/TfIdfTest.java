package io.github.vcuswimlab.stackintheflow.controller.info;

import org.junit.Test;

import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TfIdfTest {

    @Test
    public void getTermFrequencies() {
        String document = "asdf test asdf test\n rare";
        Pattern pattern = Pattern.compile("rare");
        Map<String, Integer> termFrequencies = TfIdf.getTermFrequencies(document, pattern);
        termFrequencies.forEach((term, count) -> System.out.println(term + ":\t" + count));
        assertEquals(java.util.Optional.of(1).get(), termFrequencies.get("rare"));
    }

    @Test
    public void getInverseDocumentFrequency() {
        String term = "rare";
        String[] documents = {"asdf asdf asdf fdsa test", "rare", "another test document", "test2", "asdf test"};
        double idf = TfIdf.getInverseDocumentFrequency(term, documents);
        System.out.println(idf);
        assertTrue(idf > 0);
    }
}