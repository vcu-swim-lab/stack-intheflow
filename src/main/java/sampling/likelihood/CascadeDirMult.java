package sampling.likelihood;

import java.io.Serializable;
import sampling.AbstractDiscreteFiniteLikelihoodModel;
import sampling.util.SparseCount;

/**
 *
 * @author vietan
 */
public class CascadeDirMult extends AbstractDiscreteFiniteLikelihoodModel
        implements Serializable {

    private static final long serialVersionUID = 1123581321L;

    public static enum PathAssumption {

        MINIMAL, MAXIMAL, ANTONIAK, UNIPROC, NONE
    }
    private SparseCount pseudoObservations;
    private double[] distribution;
    private double[] logDistribution;

    public CascadeDirMult(int dim) {
        super(dim);
        this.observations = new SparseCount();
        this.pseudoObservations = null;
        this.distribution = null;
        this.logDistribution = null;
    }

    public boolean containsObservation(int obs) {
        return this.observations.containsIndex(obs);
    }

    /**
     * Get the count to be cascaded upward when an observation is added.
     *
     * @param pathAssumption The path assumption, can be either MAXIMAL or
     * MINIMAL
     * @param obs The observation to be added
     * @param count The number of observations to be added
     */
    public int getPassingCountIncrease(PathAssumption pathAssumption, int obs,
            int count) {
        if (pathAssumption == PathAssumption.MAXIMAL) {
            return count;
        } else if (pathAssumption == PathAssumption.MINIMAL) {
            if (!this.containsObservation(obs)) {
                return 1;
            }
            return 0;
        } else {
            throw new InvalidPathAssumptionException();
        }
    }

    /**
     * Get the count to be cascaded upward when an observation is removed.
     *
     * @param pathAssumption The path assumption, can be either MAXIMAL or
     * MINIMAL
     * @param obs The observation to be added
     * @param count The number of observations to be removed
     */
    public int getPassingCountDecrease(PathAssumption pathAssumption, int obs,
            int count) {
        if (pathAssumption == PathAssumption.MAXIMAL) {
            return count;
        } else if (pathAssumption == PathAssumption.MINIMAL) {
            if (this.observations.getCount(obs) == count) {
                return 1;
            }
            return 0;
        } else {
            throw new InvalidPathAssumptionException();
        }
    }

    public void setDistribution(double[] t) {
        this.distribution = t;
        this.logDistribution = new double[distribution.length];
        for (int ii = 0; ii < logDistribution.length; ii++) {
            logDistribution[ii] = Math.log(distribution[ii]);
        }
    }

    @Override
    public void sampleFromPrior() {
        throw new RuntimeException("Not supported");
    }

    @Override
    public double[] getDistribution() {
        return this.distribution;
    }

    public double[] getLogDistribution() {
        return this.logDistribution;
    }

    public SparseCount getSparseCount() {
        return this.observations;
    }

    public SparseCount getPseudoObservations() {
        return this.pseudoObservations;
    }

    public void setPseudoObservations(SparseCount sp) {
        this.pseudoObservations = sp;
    }

    public void changeCountObservation(int obs, int delta) {
        this.observations.changeCount(obs, delta);
    }

    public void incrementObservation(int obs) {
        this.observations.increment(obs);
    }

    public void decrementObservation(int obs) {
        this.observations.decrement(obs);
    }

    @Override
    public String getModelName() {
        return "Cascaded-Dirichlet-Multinomial";
    }

    @Override
    public double getLogLikelihood(int observation) {
        return this.logDistribution[observation];
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;
        for (int obs : this.observations.getIndices()) {
            llh += this.observations.getCount(obs) * getLogLikelihood(obs);
        }
        return llh;
    }

    class InvalidPathAssumptionException extends RuntimeException {

        public InvalidPathAssumptionException() {
            super("Invalid path assumption");
        }
    }
}
