package util.evaluation;

import java.util.ArrayList;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class RegressionEvaluation {

    private double[] trueValues;
    private double[] predValues;
    private ArrayList<Measurement> measurements;

    public RegressionEvaluation(double[] tv, double[] pv) {
        this.trueValues = tv;
        this.predValues = pv;
        this.measurements = new ArrayList<Measurement>();
    }

    public ArrayList<Measurement> getMeasurements() {
        return this.measurements;
    }

    public void computePredictiveRSquared() {
        this.measurements.add(new Measurement("pR-squared", StatUtils.computePredictedRSquared(trueValues, predValues)));
    }

    public void computeMeanSquareError() {
        this.measurements.add(new Measurement("MSE", StatUtils.computeMeanSquaredError(trueValues, predValues)));
    }

    public void computeMeanAbsoluteError() {
        this.measurements.add(new Measurement("MAE", StatUtils.computeMeanAbsoluteError(trueValues, predValues)));
    }

    public void computeCorrelationCoefficient() {
        this.measurements.add(new Measurement("Correlation-coefficient", StatUtils.computeCorrelationCoefficient(trueValues, predValues)));
    }

    public void computeRSquared() {
        this.measurements.add(new Measurement("R-squared", Math.pow(StatUtils.computeCorrelationCoefficient(trueValues, predValues), 2)));
    }
}
