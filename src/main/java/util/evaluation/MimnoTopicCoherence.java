package util.evaluation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public class MimnoTopicCoherence {

    private int[][] corpus;
    private int vocabSize;
    private int numTokens;
    private int[][] coDocFreq;

    public MimnoTopicCoherence(int[][] corpus, int vocSize, int numTokens) {
        this.corpus = corpus;
        this.vocabSize = vocSize;
        this.numTokens = numTokens;
    }

    public int getNumTokens() {
        return this.numTokens;
    }

    public void prepare() {
        this.coDocFreq = new int[vocabSize][vocabSize];
        for (int d = 0; d < corpus.length; d++) {
            Set<Integer> uniqueTokens = new HashSet<Integer>();
            for (int n = 0; n < corpus[d].length; n++) {
                uniqueTokens.add(corpus[d][n]);
            }

            for (int token : uniqueTokens) {
                for (int otherToken : uniqueTokens) {
                    // System.out.println("token: " + token + " otherToken: " + otherToken);
                    coDocFreq[token][otherToken]++;
                }
            }
        }
    }

    public double getCoherenceScore(int[] topic) {
        double score = 0.0;
        for (int m = 1; m < numTokens; m++) {
            int tokenM = topic[m];
            for (int l = 0; l < m; l++) {
                int tokenL = topic[l];
                score += Math.log(coDocFreq[tokenM][tokenL] + 1) - Math.log(coDocFreq[tokenL][tokenL]);
            }
        }
        return score;
    }

    public double[] getCoherenceScores(int[][] topics) {
        double[] scores = new double[topics.length];
        for (int k = 0; k < topics.length; k++) {
            scores[k] = getCoherenceScore(topics[k]);
        }
        return scores;
    }

    public double[] getCoherenceScores(ArrayList<int[]> topics) {
        double[] scores = new double[topics.size()];
        for (int k = 0; k < topics.size(); k++) {
            scores[k] = getCoherenceScore(topics.get(k));
        }
        return scores;
    }

    public static void main(String[] args) {
        Random rand = new Random(1);

        int D = 5;
        int V = 5;
        int N = 10;
        int[][] obs = new int[D][N];
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < N; n++) {
                obs[d][n] = rand.nextInt(V);
            }
        }

        for (int d = 0; d < D; d++) {
            System.out.println(MiscUtils.arrayToString(obs[d]));
        }

        MimnoTopicCoherence tc = new MimnoTopicCoherence(obs, V, 5);
        tc.prepare();
        for (int i = 0; i < tc.coDocFreq.length; i++) {
            System.out.println(MiscUtils.arrayToString(tc.coDocFreq[i]));
        }
    }
}
