package sampling;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import sampling.util.SparseCount;

/**
 * An abstract likelihood model of generating countable finite observations.
 * Observations are indexed by integers from the vocabulary.
 *
 * Each likelihood model needs to store the hyper-parameters of the
 * corresponding prior.
 * 
* @author vietan
 */
public abstract class AbstractDiscreteFiniteLikelihoodModel
        implements Cloneable, Serializable {

    private static final long serialVersionUID = 1123581321L;
    public static final int RANDOM_SEED = 1123581321;
    // this is currently used for likelihood models that does not have/use
    // conjugate prior and we need to sample from the prior
    protected static Random rand = new Random(RANDOM_SEED);
    // observations
    protected int dimension;
    protected SparseCount observations;

    public AbstractDiscreteFiniteLikelihoodModel(int dim) {
        this.dimension = dim;
        this.observations = new SparseCount();
    }

    public abstract String getModelName();

    public abstract double getLogLikelihood(int observation);

    public abstract double getLogLikelihood();

    public abstract double[] getDistribution();

    /**
     * Sample the parameter(s) for a new component using the prior. This is
     * mainly used for non-conjugate prior where computing the likelihood of the
     * new table/component given observed data is difficult.
     */
    public abstract void sampleFromPrior();

    @Override
    public AbstractDiscreteFiniteLikelihoodModel clone() throws CloneNotSupportedException {
        AbstractDiscreteFiniteLikelihoodModel m = (AbstractDiscreteFiniteLikelihoodModel) super.clone();
        m.observations = this.observations.clone();
        return m;
    }

    public void clear() {
        this.observations = new SparseCount();
    }

    public boolean isEmpty() {
        return this.observations.isEmpty();
    }

    public int getCount(int observation) {
        return this.observations.getCount(observation);
    }

    public HashMap<Integer, Integer> getObservations() {
        return this.observations.getObservations();
    }

    public Set<Integer> getUniqueObservations() {
        return this.observations.getIndices();
    }

    public int[] getCounts() {
        int[] counts = new int[this.dimension];
        for (int obs : this.observations.getIndices()) {
            counts[obs] = this.observations.getCount(obs);
        }
        return counts;
    }

    public SparseCount getSparseCounts() {
        return this.observations;
    }

    public void setCounts(int[] c) {
        this.observations = new SparseCount();
        for (int i = 0; i < c.length; i++) {
            if (c[i] > 0) {
                this.observations.setCount(i, c[i]);
            }
        }
    }

    public int getCountSum() {
        return this.observations.getCountSum();
    }

    public int getDimension() {
        return this.dimension;
    }

    /**
     * Change the count of a given observation
     *
     * @param observation The observation in [0, dim)
     * @param delta Change in the count
     */
    public void changeCount(int observation, int delta) {
        int count = this.getCount(observation);
        this.observations.setCount(observation, count + delta);
    }

    /**
     * Decrement the count of a given observation
     *
     * @param observation The observation whose count is decremented
     */
    public void decrement(int observation) {
        this.observations.decrement(observation);
    }

    /**
     * Increment the count of a given observation
     *
     * @param observation The observation whose count is incremented
     */
    public void increment(int observation) {
        this.observations.increment(observation);
    }

    public void validate(String msg) {
        this.observations.validate(msg);
    }

    public String getDebugString() {
        StringBuilder str = new StringBuilder();
        str.append("Dimension = ").append(this.dimension).append("\n");
        str.append("Count sum = ").append(this.getCountSum()).append("\n");
        str.append("Counts = ").append(java.util.Arrays.toString(this.getCounts())).append("\n");
        return str.toString();
    }
}
