package util.normalizer;

/**
 *
 * @author vietan
 */
public abstract class AbstractNormalizer {

    public abstract double normalize(double originalValue);

    public abstract double denormalize(double normalizedValue);
}
