package sampling.likelihood;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import sampling.AbstractDiscreteFiniteLikelihoodModel;
import sampling.util.SparseCount;
import util.SamplerUtils;
import weka.core.SerializedObject;

/**
 * Implementation of a multinomial likelihood model in which the multinomial
 * distribution is integrated out.
 *
 * @author vietan
 */
public class DirMult extends AbstractDiscreteFiniteLikelihoodModel implements Serializable {

    private static final long serialVersionUID = 1123581321L;
    private double concentration; // concentration parameter
    private double[] center; // the mean vector for asymmetric distribution
    private double centerElement; // an element in the mean vector for symmetric distribution
    private double[] distribution;

    public DirMult(int dim, double concentration, double centerElement) {
        super(dim);
        this.centerElement = centerElement;
        this.concentration = concentration;
    }

    /*TODO: dim can be inferred from the dimension of centerVector. remove the
     * argument "dim"! */
    public DirMult(int dim, double concentration, double[] centerVector) {
        super(dim);
        this.center = centerVector;
        this.concentration = concentration;
    }

    public DirMult(double[] p) {
        super(p.length);
        this.concentration = 0.0;
        for (double v : p) {
            this.concentration += v;
        }
        this.center = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            this.center[i] = p[i] / this.concentration;
        }
    }

    public void setSamplingDistribution(double[] dist) {
        this.distribution = dist;
    }

    public double[] getSamplingDistribution() {
        return this.distribution;
    }

    public void setHyperparameters(double[] p) {
        this.concentration = 0.0;
        for (double v : p) {
            this.concentration += v;
        }
        this.center = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            this.center[i] = p[i] / this.concentration;
        }
    }

    public void setConcentration(double conc) {
        this.concentration = conc;
    }

    public void getCenterVector(double[] ce) {
        this.center = ce;
    }

    public double getConcentration() {
        return concentration;
    }

    public double[] getCenterVector() {
        if (center == null) { // if this is null, this has a symmetric Dirichlet prior
            center = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                center[i] = 1.0 / dimension;
            }
        }
        return center;
    }

    public double getCenterElement(int index) {
        if (center == null) {
            return centerElement;
        }
        return this.center[index];
    }

    @Override
    public String getModelName() {
        return "Dirichlet-Multinomial";
    }

    @Override
    public void sampleFromPrior() {
        // Do nothing here since in this case, we integrate over all possible 
        // multinomials due to the conjugacy betwen Dirichlet and multinomial
        // distributions.
//        throw new RuntimeException(DirichletMultinomialModel.class 
//                + " is not currently supporting sampling from prior since the "
//                + "multinomial is integrated out.");
    }

    @Override
    public DirMult clone() {
        DirMult newMult = null;
        try {
            newMult = (DirMult) super.clone();
            if (!isShortRepresented()) {
                newMult.center = (double[]) this.center.clone();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while cloning.");
        }
        return newMult;
    }

    public double getLogLikelihood(ArrayList<Integer> observations) {
        HashMap<Integer, Integer> observationMap = new HashMap<Integer, Integer>();
        for (int obs : observations) {
            Integer count = observationMap.get(obs);
            if (count == null) {
                observationMap.put(obs, 1);
            } else {
                observationMap.put(obs, count + 1);
            }
        }
        return getLogLikelihood(observationMap);
    }

    public double getLogLikelihood(HashMap<Integer, Integer> observations) {
        double llh = 0.0;
        int j = 0;
        for (int observation : observations.keySet()) {
            for (int i = 0; i < observations.get(observation); i++) {
                llh += Math.log(concentration * getCenterElement(observation)
                        + getCount(observation) + i)
                        - Math.log(concentration + getCountSum() + j);
                j++;
            }
        }
        return llh;
    }

    public double getLogLikelihood(SparseCount observations) {
        return getLogLikelihood(observations.getObservations());
    }

    @Override
    public double getLogLikelihood(int observation) {
        double prior;
        if (isShortRepresented()) {
            prior = this.centerElement * this.concentration;
        } else {
            prior = this.center[observation] * this.concentration;
        }
        return Math.log(this.getCount(observation) + prior)
                - Math.log(this.getCountSum() + this.concentration);
    }

    @Override
    public double getLogLikelihood() {
        if (isShortRepresented()) {
            return SamplerUtils.computeLogLhood(getCounts(), getCountSum(), centerElement * concentration);
        } else {
            double[] params = new double[this.getDimension()];
            for (int i = 0; i < this.getDimension(); i++) {
                params[i] = center[i] * concentration;
            }
            return SamplerUtils.computeLogLhood(getCounts(), getCountSum(), params);
        }
    }

    public double getLogLikelihood(double[] params) {
        return SamplerUtils.computeLogLhood(getCounts(), getCountSum(), params);
    }

    public double getLogLikelihood(double concentr, double centerE) {
        return SamplerUtils.computeLogLhood(getCounts(), getCountSum(), centerE * concentr);
    }

    public double getLogLikelihood(double concentr, double[] centerV) {
        double[] params = new double[this.getDimension()];
        for (int i = 0; i < this.getDimension(); i++) {
            params[i] = centerV[i] * concentr;
        }
        return SamplerUtils.computeLogLhood(getCounts(), getCountSum(), params);
    }

    @Override
    public double[] getDistribution() {
        double[] distr = new double[getDimension()];
        for (int k = 0; k < distr.length; k++) {
            if (isShortRepresented()) {
                distr[k] = (getCount(k) + concentration * centerElement) / (getCountSum() + concentration);
            } else {
                distr[k] = (getCount(k) + concentration * center[k]) / (getCountSum() + concentration);
            }
        }
        return distr;
    }

    public double getProbability(int w) {
        return (getCount(w) + this.concentration * getCenterElement(w))
                / (getCountSum() + this.concentration);
    }

    public double[] getEmpiricalDistribution() {
        double[] empDist = new double[getDimension()];
        for (int k = 0; k < empDist.length; k++) {
            empDist[k] = (double) getCount(k) / getCountSum();
        }
        return empDist;
    }

    /**
     * Return true if the Dirichlet is short-represented by a scalar, indicating
     * a symmetric Dirichlet distribution.
     * @return 
     */
    public boolean isShortRepresented() {
        return this.center == null;
    }

    @Override
    public String getDebugString() {
        StringBuilder str = new StringBuilder();
        str.append("Dimension = ").append(this.dimension).append("\n");
        str.append("Count sum = ").append(this.getCountSum()).append("\n");
        str.append("Counts = ").append(java.util.Arrays.toString(this.getCounts())).append("\n");
        str.append("Concentration = ").append(this.concentration).append("\n");
        str.append("Short-represented = ").append(isShortRepresented()).append("\n");
        if (isShortRepresented()) {
            str.append("Mean element = ").append(this.centerElement).append("\n");
        } else {
            str.append("Mean vector = ").append(java.util.Arrays.toString(this.center)).append("\n");
        }
        return str.toString();
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("Dimension = ").append(this.dimension).append("\n");
        str.append("Count sum = ").append(this.getCountSum()).append("\n");
        return str.toString();
    }

    public static String outputDistribution(double[] dist) {
        StringBuilder str = new StringBuilder();
        for (int ii = 0; ii < dist.length; ii++) {
            str.append(dist[ii]).append("\t");
        }
        return str.toString().trim();
    }

    public static double[] inputDistribution(String str) {
        String[] sstr = str.split("\t");
        double[] dist = new double[sstr.length];
        for (int ii = 0; ii < dist.length; ii++) {
            dist[ii] = Double.parseDouble(sstr[ii]);
        }
        return dist;
    }

    public static String output(DirMult model) {
        StringBuilder str = new StringBuilder();
        str.append(model.dimension)
                .append("\t").append(model.concentration);
        for (int v = 0; v < model.dimension; v++) {
            str.append("\t").append(model.getCenterElement(v));
        }
        for (int v = 0; v < model.dimension; v++) {
            str.append("\t").append(model.getCount(v));
        }
        return str.toString();
    }

    public static DirMult input(String str) {
        String[] sline = str.split("\t");
        int dim = Integer.parseInt(sline[0]);
        double concentration = Double.parseDouble(sline[1]);
        double[] mean = new double[dim];
        int idx = 2;
        for (int v = 0; v < dim; v++) {
            mean[v] = Double.parseDouble(sline[idx++]);
        }
        DirMult model = new DirMult(dim, concentration, mean);
        for (int v = 0; v < dim; v++) {
            model.changeCount(v, Integer.parseInt(sline[idx++]));
        }
        return model;
    }

    public static void main(String[] args) {
        try {
//            testClone();

//            testPrior();
//            testLlh();
            testSerializable();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testSerializable() throws Exception {
        double[] mean = {0.5, 0.5};
        double scale = 2;
        DirMult dmm = new DirMult(mean.length, scale, mean);
        dmm.increment(0);
        System.out.println(dmm);
        System.out.println(DirMult.output(dmm));

        DirMult copy = (DirMult) new SerializedObject(dmm).getObject();
        System.out.println(copy);
        System.out.println(DirMult.output(copy));
    }

    private static void testLlh() throws Exception {
        double[] mean = {0.5, 0.5};
        double scale = 2;
        DirMult dmm = new DirMult(mean.length, scale, mean);
        System.out.println(dmm.getLogLikelihood(0));
        dmm.increment(0);
        System.out.println(dmm.getLogLikelihood());

        System.out.println(dmm.getDebugString());
        System.out.println(dmm.getLogLikelihood(1));
        dmm.increment(1);
        System.out.println(dmm.getDebugString());
        System.out.println(dmm.getLogLikelihood());
    }

    private static void testPrior() throws Exception {
        double[] mean = {0.7, 0.2, 0.1};
        double conc = 10000;
        DirMult mm = new DirMult(mean.length, conc, mean);

        mm.increment(0);
        System.out.println(java.util.Arrays.toString(mm.getDistribution()));
    }

    private static void testClone() throws Exception {
        int dim = 10;
        double con = 0.5;
        double ce = 0.1;
        DirMult mm = new DirMult(dim, con, ce);
        mm.increment(1);

        DirMult newMM = (DirMult) mm.clone();
        System.out.println(mm.getDebugString());
        System.out.println(newMM.getDebugString());

        mm.increment(0);
        newMM.concentration = 10;

        System.out.println(mm.getDebugString());
        System.out.println(newMM.getDebugString());
    }
}
