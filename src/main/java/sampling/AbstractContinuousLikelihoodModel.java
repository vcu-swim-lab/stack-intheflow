package sampling;

import java.util.Random;

/**
 *
 * @author vietan
 */
public abstract class AbstractContinuousLikelihoodModel {

    public static final int RANDOM_SEED = 1123581321;
    protected static Random rand = new Random(RANDOM_SEED);

    public abstract double sampleFromPrior();

    public abstract double getLogLikelihood(double observation);
}
