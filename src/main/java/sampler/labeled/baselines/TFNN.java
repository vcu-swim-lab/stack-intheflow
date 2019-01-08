package sampler.labeled.baselines;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class TFNN {

    protected int[][] words;
    protected int[][] labels;
    protected int L;
    protected int V;
    protected int D;
    protected SparseVector[] labelVectors; // L x V;
    protected int minWordTypeCount = 0;
    protected double[] labelL2Norms;
    
    public TFNN(
            int[][] docWords,
            int[] labels,
            int L,
            int V,
            int minWordTypeCount) {
        this.words = docWords;
        this.L = L;
        this.V = V;
        this.D = this.words.length;
        this.minWordTypeCount = minWordTypeCount;
        this.labels = new int[labels.length][1];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii][0] = labels[ii];
        }
    }

    public TFNN(
            int[][] docWords,
            int[][] labels,
            int L,
            int V,
            int minWordTypeCount) {
        this.words = docWords;
        this.labels = labels;
        this.L = L;
        this.V = V;
        this.D = this.words.length;
        this.minWordTypeCount = minWordTypeCount;
    }

    public String getName() {
        return "tf-nn-" + minWordTypeCount;
    }

    public SparseVector[] getLabelVectors() {
        return this.labelVectors;
    }

    public void setMinWordTypeCount(int minTypeCount) {
        this.minWordTypeCount = minTypeCount;
    }
    
    public void outputVWFormat(File outputFile, int[][] words, int[][] labels,
            ArrayList<String> vocab) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int d = 0; d < words.length; d++) {
            if (labels[d].length > 0) {
                for (int ll : labels[d]) {
                    writer.write(ll + ":1 ");
                }
                writer.write("|w");

                SparseCount count = new SparseCount();
                for (int n = 0; n < words[d].length; n++) {
                    count.increment(words[d][n]);
                }

                for (int idx : count.getSortedIndices()) {
                    String word = vocab.get(idx);
                    writer.write(" " + word + ":" + count.getCount(idx));
                }
                writer.write("\n");
            }
        }
        writer.close();
    }

    public SparseVector getFeatureVector(int[] newWords) {
        SparseCount typeCount = new SparseCount();
        for (int n = 0; n < newWords.length; n++) {
            typeCount.increment(newWords[n]);
        }

        SparseVector docVector = new SparseVector();
        for (int idx : typeCount.getIndices()) {
            double score = (double) typeCount.getCount(idx) / newWords.length;
            docVector.set(idx, score); // index used to start at 1
        }

        return docVector;
    }

    public void learn() {
        this.labelVectors = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            this.labelVectors[ll] = new SparseVector();
        }
        int[] labelDocCounts = new int[L];
        System.out.println("Aggregate label vectors ...");
        for (int d = 0; d < D; d++) {
            int[] docTopics = this.labels[d];
            // skip unlabeled document or very short (after filtered) documents
            if (docTopics == null
                    || docTopics.length == 0
                    || words[d].length < minWordTypeCount) {
                continue;
            }
            SparseVector docVector = getFeatureVector(words[d]);
            for (int ll : docTopics) {
                labelDocCounts[ll]++;
                this.labelVectors[ll].add(docVector);
            }
        }

        // average
        System.out.println("Averaging ...");
        for (int ll = 0; ll < L; ll++) {
            int docCount = labelDocCounts[ll];
            if (docCount > 0) {
                this.labelVectors[ll].divide(docCount);
            }
        }

        computeLabelL2Norms();
    }

    protected void computeLabelL2Norms() {
        labelL2Norms = new double[L];
        for (int ll = 0; ll < L; ll++) {
            labelL2Norms[ll] = labelVectors[ll].getL2Norm();
        }
    }

    public double[][] predict(int[][] newWords) {
        double[][] predictions = new double[newWords.length][];
        int stepSize = MiscUtils.getRoundStepSize(newWords.length, 10);
        for (int dd = 0; dd < predictions.length; dd++) {
            if (dd % stepSize == 0) {
                System.out.println("--- Predicting doc = " + dd + " / " + newWords.length);
            }
            predictions[dd] = this.predict(newWords[dd]);
        }
        return predictions;
    }

    public double[] predict(int[] newWords) {
        double[] scores = new double[L];
        if (newWords.length == 0) {
            return scores;
        }
        SparseVector docVector = getFeatureVector(newWords);
        double newDocL2Norm = docVector.getL2Norm();
        for (int l = 0; l < L; l++) {
            if (labelVectors[l].size() > 0) { // skip topics that didn't have enough training data for
                scores[l] = labelVectors[l].dotProduct(docVector)
                        / (labelL2Norms[l] * newDocL2Norm);
            }
        }
        return scores;
    }

    public ArrayList<Integer> predictLabel(int[] newWords, int topK) {
        double[] scores = predict(newWords);
        ArrayList<RankingItem<Integer>> rank = new ArrayList<RankingItem<Integer>>();
        for (int ii = 0; ii < scores.length; ii++) {
            rank.add(new RankingItem<Integer>(ii, scores[ii]));
        }
        Collections.sort(rank);
        ArrayList<Integer> rankLabels = new ArrayList<Integer>();
        for (int ii = 0; ii < topK; ii++) {
            RankingItem<Integer> item = rank.get(ii);
            int label = item.getObject();
            double score = item.getPrimaryValue();
            if (score == 0.0) {
                break;
            }
            rankLabels.add(label);
        }
        return rankLabels;
    }

    public void outputPredictor(File predictorFile) {
        System.out.println("Outputing learned model to " + predictorFile);
        try {
            StringBuilder labelVecStr = new StringBuilder();
            labelVecStr.append("num-labels\t").append(L).append("\n");
            labelVecStr.append("num-dimensions\t").append(V).append("\n");
            for (SparseVector labelVector : this.labelVectors) {
                labelVecStr.append(labelVector.toString()).append("\n");
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(predictorFile.getName());
            ZipOutputStream writer = IOUtils.getZipOutputStream(predictorFile.getAbsolutePath());

            ZipEntry modelEntry = new ZipEntry(filename + ".label");
            writer.putNextEntry(modelEntry);
            byte[] data = labelVecStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to "
                    + predictorFile);
        }
    }

    public void inputPredictor(File predictorFile) {
        System.out.println("Inputing learned model from " + predictorFile);
        try {
            String filename = IOUtils.removeExtension(predictorFile.getName());

            ZipFile zipFile = new ZipFile(predictorFile);
            BufferedReader reader = IOUtils.getBufferedReader(zipFile,
                    zipFile.getEntry(filename + ".label"));
            L = Integer.parseInt(reader.readLine().split("\t")[1]);
            V = Integer.parseInt(reader.readLine().split("\t")[1]);
            this.labelVectors = new SparseVector[L];
            for (int l = 0; l < L; l++) {
                labelVectors[l] = SparseVector.parseString(reader.readLine());
            }
            reader.close();

            labelL2Norms = new double[L];
            for (int ll = 0; ll < L; ll++) {
                labelL2Norms[ll] = labelVectors[ll].getL2Norm();
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictor from "
                    + predictorFile);
        }
    }

    public void outputTopWords(File outputFile,
            ArrayList<String> labelVocab,
            ArrayList<String> wordVocab,
            int numTopWords) {
        System.out.println("Outputing top words to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int ll = 0; ll < L; ll++) {
                ArrayList<RankingItem<Integer>> rankWords = new ArrayList<RankingItem<Integer>>();
                for (int v : labelVectors[ll].getIndices()) {
                    rankWords.add(new RankingItem<Integer>(v, labelVectors[ll].get(v)));
                }
                Collections.sort(rankWords);

                String topicStr = "Label-" + ll;
                if (labelVocab != null) {
                    topicStr = labelVocab.get(ll);
                }
                writer.write(topicStr);

                for (int ii = 0; ii < numTopWords; ii++) {
                    RankingItem<Integer> item = rankWords.get(ii);
                    writer.write("\t" + wordVocab.get(item.getObject()));
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing top words to "
                    + outputFile);
        }
    }
}
