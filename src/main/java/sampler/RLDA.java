package sampler;

import java.util.ArrayList;
import java.util.Arrays;
import util.MiscUtils;

/**
 * Implementation of a node in a LDA-tree.
 *
 * @author vietan
 */
public class RLDA extends LDA {

    public static final int INVALID = -1;
    private final boolean[][] valid;
    private final int index;
    private final int level;
    private final RLDA parent;
    private final ArrayList<RLDA> children;

    public RLDA(int index, int level, boolean[][] v, RLDA parent) {
        this.index = index;
        this.level = level;
        this.valid = v;
        this.parent = parent;
        this.children = new ArrayList<RLDA>();
    }

    public boolean[][] getValid() {
        return this.valid;
    }

    public void updateStatistics() {
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (this.valid[d][n]) {
                    numTokens++;
                }
            }
        }
    }

    public int getIndex() {
        return this.index;
    }

    public int getLevel() {
        return this.level;
    }

    public RLDA getParent() {
        return this.parent;
    }

    public ArrayList<RLDA> getChildren() {
        return this.children;
    }

    public void addChild(RLDA child) {
        this.children.add(child);
    }

    public int getNumTokens() {
        return this.numTokens;
    }

    /**
     * Return the unique path string for each node in the tree
     * @return Path of this node
     */
    public String getPathString() {
        if (parent == null) {
            return Integer.toString(this.index);
        } else {
            return this.parent.getPathString() + ":" + this.index;
        }
    }

    protected void initializeAssignments(int[][] seededZs) {
        if (verbose) {
            logln("--- Initializing assignments with seeded assignments ...");
        }

        for (int d = 0; d < D; d++) {
            Arrays.fill(z[d], INVALID);
            for (int n = 0; n < words[d].length; n++) {
                if (valid[d][n]) {
                    z[d][n] = seededZs[d][n];
                    doc_topics[d].increment(z[d][n]);
                    topic_words[z[d][n]].increment(words[d][n]);
                }
            }
        }
    }

    @Override
    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        for (int d = 0; d < D; d++) {
            Arrays.fill(z[d], INVALID);
            for (int n = 0; n < words[d].length; n++) {
                if (valid[d][n]) {
                    z[d][n] = rand.nextInt(K);
                    doc_topics[d].increment(z[d][n]);
                    topic_words[z[d][n]].increment(words[d][n]);
                }
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        updateStatistics();
        logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    if (valid[d][n]) {
                        sampleZ(d, n, REMOVE, ADD);
                    }
                }
            }

            if (debug) {
                validate("Iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose && iter % REP_INTERVAL == 0) {
                double changeRatio = (double) numTokensChanged / numTokens;
                String str = "Iter " + iter + "/" + MAX_ITER
                        + ". llh = " + MiscUtils.formatDouble(loglikelihood)
                        + ". numTokensChanged = " + numTokensChanged
                        + ". change ratio = " + MiscUtils.formatDouble(changeRatio);
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }
        }
    }

    @Override
    public void validate(String msg) {
        super.validate(msg);
        int totalValid = 0;
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < valid[d].length; n++) {
                if (valid[d][n]) {
                    totalValid++;
                }
            }
        }

        int totalDocTopicCount = 0;
        for (int d = 0; d < D; d++) {
            totalDocTopicCount += doc_topics[d].getCountSum();
        }

        if (totalValid != totalDocTopicCount) {
            throw new RuntimeException(msg + ". Total count mismatch. "
                    + totalValid + " vs. " + totalDocTopicCount);
        }

        int totalTopicWordCount = 0;
        for (int k = 0; k < K; k++) {
            totalTopicWordCount += topic_words[k].getCountSum();
        }

        if (totalValid != totalTopicWordCount) {
            throw new RuntimeException(msg + ". Total count mismatch. "
                    + totalValid + " vs. " + totalTopicWordCount);
        }
    }
}
