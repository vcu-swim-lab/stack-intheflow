package optimization;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.optimization.DiffFunction;
import gnu.trove.THashSet;
import java.util.*;

/**
 * Class implementing the Orthant-Wise Limited-memory Quasi-Newton algorithm
 * (OWL-QN). OWN-QN is a numerical optimization procedure for finding the
 * optimum of an objective of the form smooth function plus L1-norm of the
 * parameters. It has been used for training log-linear models (such as logistic
 * regression) with L1-regularization. The algorithm is described in "Scalable
 * training of L1-regularized log-linear models" by Galen Andrew and Jianfeng
 * Gao. This implementation includes built-in capacity to train logistic
 * regression or least-squares models with L1 regularization. It is also
 * possible to use OWL-QN to optimize any arbitrary smooth convex loss plus L1
 * regularization by defining the function and its gradient using the supplied
 * "DifferentiableFunction" class, and passing an instance of the function to
 * the OWLQN object. For more information, please read the included file
 * README.txt. Also included in the distribution are the ICML paper and slide
 * presentation.
 *
 * Significant portions of this code are taken from
 * <a
 * href="http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/default.aspx">Galen
 * Andew's implementation</a>
 *
 * @author Michel Galley
 *
 * modified by Michael Heilman (mheilman@cmu.edu) -allow for bias/intercept
 * parameters that shouldn't be penalized -make outside API calls easier
 * (11/9/2010) -removed lots of extraneous stuff from the stanford API
 *
 */
public class OWLQN {

    private int maxIters = Integer.MAX_VALUE;

    interface TerminationCriterion {

        double getValue(OptimizerState state, StringBuilder out);
    }

    static class RelativeMeanImprovementCriterion implements TerminationCriterion {

        int numItersToAvg;
        Queue<Double> prevVals;

        RelativeMeanImprovementCriterion() {
            this(2);
        }

        RelativeMeanImprovementCriterion(int numItersToAvg) {
            this.numItersToAvg = numItersToAvg;
            this.prevVals = new LinkedList<Double>();
        }

        @Override
        public double getValue(OptimizerState state, StringBuilder out) {

            double retVal = Double.POSITIVE_INFINITY;

            if (prevVals.size() > 2) {
                double prevVal = prevVals.peek();
                if (prevVals.size() == 10) {
                    prevVals.poll();
                }
                double averageImprovement = (prevVal - state.getValue()) / prevVals.size();
                double relAvgImpr = averageImprovement / Math.abs(state.getValue());
                String relAvgImprStr = String.format("%.4e", relAvgImpr);
                out.append("  (").append(relAvgImprStr).append(") ");
                retVal = relAvgImpr;
            } else {
                out.append("  (wait for two iters) ");
            }

            prevVals.offer(state.getValue());
            return retVal;
        }
    } // end static class RelativeMeanImprovementCriterion

    boolean quiet;
    boolean responsibleForTermCrit;

    public static Set<Integer> biasParameters = new THashSet<Integer>();

    TerminationCriterion termCrit;

    public OWLQN(boolean quiet) {
        this.quiet = quiet;
        this.termCrit = new RelativeMeanImprovementCriterion(5);
        this.responsibleForTermCrit = true;
    }

    public OWLQN() {
        this(false);
    }

    public OWLQN(TerminationCriterion termCrit, boolean quiet) {
        this.quiet = quiet;
        this.termCrit = termCrit;
        this.responsibleForTermCrit = false;
    }

    public void setQuiet(boolean q) {
        quiet = q;
    }

    public double[] minimize(DiffFunction function, double[] initial) {
        return minimize(function, initial, 1.0);
    }

    public double[] minimize(DiffFunction function, double[] initial, double l1weight) {
        return minimize(function, initial, l1weight, 1e-5);
    }

    public double[] minimize(DiffFunction function, double[] initial, double l1weight, double tol) {
        return minimize(function, initial, l1weight, tol, 10);
    }

    public double[] minimize(DiffFunction function, double[] initial, double l1weight, double tol, int m) {

        OptimizerState state = new OptimizerState(function, initial, m, l1weight, quiet);

        if (!quiet) {
            System.err.printf("Optimizing function of %d variables with OWL-QN parameters:\n", state.dim);
            System.err.printf("   l1 regularization weight: %f.\n", l1weight);
            System.err.printf("   L-BFGS memory parameter (m): %d\n", m);
            System.err.printf("   Convergence tolerance: %f\n\n", tol);
            System.err.printf("Iter    n:\tnew_value\tdf\t(conv_crit)\tline_search\n");
            System.err.printf("Iter    0:\t%.4e\t\t(***********)\t", state.value);
        }

        StringBuilder buf = new StringBuilder();
        termCrit.getValue(state, buf);

        for (int i = 0; i < maxIters; i++) {
            buf.setLength(0);
            state.updateDir();
            state.backTrackingLineSearch();

            double termCritVal = termCrit.getValue(state, buf);
            if (!quiet) {
                int numnonzero = ArrayMath.countNonZero(state.newX);
                System.err.printf("Iter %4d:\t%.4e\t%d", state.iter, state.value, numnonzero);
                System.err.print("\t" + buf.toString());
            }

            if (termCritVal < tol) {
                break;
            }

            state.shift();
        }

        if (!quiet) {
            System.err.println();
            System.err.printf("Finished with optimization.  %d/%d non-zero weights.\n",
                    ArrayMath.countNonZero(state.newX), state.newX.length);
            //System.err.println(Arrays.toString(state.newX));
        }

        return state.newX;
    }

    public void setMaxIters(int maxIters) {
        this.maxIters = maxIters;
    }

    public int getMaxIters() {
        return maxIters;
    }

} // end static class OWLQN

class OptimizerState {

    double[] x, grad, newX, newGrad, dir;
    double[] steepestDescDir;
    LinkedList<double[]> sList = new LinkedList<double[]>();
    LinkedList<double[]> yList = new LinkedList<double[]>();
    LinkedList<Double> roList = new LinkedList<Double>();
    double[] alphas;
    double value;
    int iter, m;
    int dim;
    DiffFunction func;
    double l1weight;

    boolean quiet;

    void mapDirByInverseHessian() {
        int count = sList.size();

        if (count != 0) {
				//check that the ro values are all nonzero.
            //if they aren't, then don't use information about the hessian
            //to change the descent direction.
            for (int i = count - 1; i >= 0; i--) {
                if (roList.get(i) == 0.0) {
                    return;
                }
            }

            for (int i = count - 1; i >= 0; i--) {
                alphas[i] = -ArrayMath.innerProduct(sList.get(i), dir) / roList.get(i);
                ArrayMath.addMultInPlace(dir, yList.get(i), alphas[i]);
            }

            double[] lastY = yList.get(count - 1);
            double yDotY = ArrayMath.innerProduct(lastY, lastY);
            double scalar = roList.get(count - 1) / yDotY;
            ArrayMath.multiplyInPlace(dir, scalar);

            for (int i = 0; i < count; i++) {
                double beta = ArrayMath.innerProduct(yList.get(i), dir) / roList.get(i);
                ArrayMath.addMultInPlace(dir, sList.get(i), -alphas[i] - beta);
            }
        }
    }

    void makeSteepestDescDir() {
        if (l1weight == 0) {
            ArrayMath.multiplyInto(dir, grad, -1);
        } else {

            for (int i = 0; i < dim; i++) {
                if (OWLQN.biasParameters.contains(i)) {
                    dir[i] = -grad[i];
                    continue;
                }
                if (x[i] < 0) {
                    dir[i] = -grad[i] + l1weight;
                } else if (x[i] > 0) {
                    dir[i] = -grad[i] - l1weight;
                } else {
                    if (grad[i] < -l1weight) {
                        dir[i] = -grad[i] - l1weight;
                    } else if (grad[i] > l1weight) {
                        dir[i] = -grad[i] + l1weight;
                    } else {
                        dir[i] = 0;
                    }
                }
            }
        }
        steepestDescDir = dir.clone(); // deep copy needed
    }

    void fixDirSigns() {
        if (l1weight > 0) {
            for (int i = 0; i < dim; i++) {
                if (OWLQN.biasParameters.contains(i)) {
                    continue;
                }
                if (dir[i] * steepestDescDir[i] <= 0) {
                    dir[i] = 0;
                }
            }
        }
    }

    void updateDir() {
        makeSteepestDescDir();
        mapDirByInverseHessian();
        fixDirSigns();
        //if(!quiet) testDirDeriv();
    }

    /*
     void testDirDeriv() {
     double dirNorm = Math.sqrt(ArrayMath.innerProduct(dir, dir));
     double eps = 1.05e-8 / dirNorm;
     getNextPoint(eps);
     //double val2 = evalL1();
     //double numDeriv = (val2 - value) / eps;
     //double deriv = dirDeriv();
     //if (!quiet) System.err.print("  Grad check: " + numDeriv + " vs. " + deriv + "  ");
     }
     */
    double dirDeriv() {
        if (l1weight == 0) {
            return ArrayMath.innerProduct(dir, grad);
        } else {
            double val = 0.0;
            for (int i = 0; i < dim; i++) {
                if (OWLQN.biasParameters.contains(i)) {
                    val += dir[i] * grad[i];
                    continue;
                }
                if (dir[i] != 0) {
                    if (x[i] < 0) {
                        val += dir[i] * (grad[i] - l1weight);
                    } else if (x[i] > 0) {
                        val += dir[i] * (grad[i] + l1weight);
                    } else if (dir[i] < 0) {
                        val += dir[i] * (grad[i] - l1weight);
                    } else if (dir[i] > 0) {
                        val += dir[i] * (grad[i] + l1weight);
                    }
                }
            }
            return val;
        }
    }

    private boolean getNextPoint(double alpha) {
        ArrayMath.addMultInto(newX, x, dir, alpha);
        if (l1weight > 0) {
            for (int i = 0; i < dim; i++) {
                if (OWLQN.biasParameters.contains(i)) {
                    continue;
                }
                if (x[i] * newX[i] < 0.0) {
                    newX[i] = 0.0;
                }
            }
        }
        return true;
    }

    double evalL1() {

        double val = func.valueAt(newX);
			// Don't remove clone(), otherwise newGrad and grad may end up referencing the same vector
        // (that's the case with LogisticObjectiveFunction)
        newGrad = func.derivativeAt(newX).clone();
        if (l1weight > 0) {
            for (int i = 0; i < dim; i++) {
                if (OWLQN.biasParameters.contains(i)) {
                    continue;
                }
                val += Math.abs(newX[i]) * l1weight;
            }
        }

        return val;
    }

    void backTrackingLineSearch() {

        double origDirDeriv = dirDeriv();
			// if a non-descent direction is chosen, the line search will break anyway, so throw here
        // The most likely reason for this is a bug in your function's gradient computation
        if (origDirDeriv >= 0) {
            throw new RuntimeException("+");
            //throw new RuntimeException("L-BFGS chose a non-descent direction: check your gradient!");
        }

        double alpha = 1.0;
        double backoff = 0.5;
        if (iter == 1) {
            double normDir = Math.sqrt(ArrayMath.innerProduct(dir, dir));
            alpha = (1 / normDir);
            backoff = 0.1;
        }

        double c1 = 1e-4;
        double oldValue = value;
        while (true) {
            getNextPoint(alpha);
            value = evalL1();

            if (value <= oldValue + c1 * origDirDeriv * alpha) {
                break;
            }
            if (alpha < 1e-30) {
                value = oldValue;
                break;
            }

            if (!quiet) {
                System.err.print(".");
            }

            alpha *= backoff;
        }

        if (!quiet) {
            System.err.println();
        }
    }

    void shift() {
        double[] nextS = null, nextY = null;

        int listSize = sList.size();

        if (listSize < m) {
            try {
                nextS = new double[dim];
                nextY = new double[dim];
            } catch (OutOfMemoryError e) {
                m = listSize;
                nextS = null;
            }
        }

        if (nextS == null) {
            nextS = sList.poll();
            nextY = yList.poll();
            roList.poll();
        }

        ArrayMath.addMultInto(nextS, newX, x, -1);
        ArrayMath.addMultInto(nextY, newGrad, grad, -1);

        double ro = ArrayMath.innerProduct(nextS, nextY);
        assert (ro != 0.0);

        sList.offer(nextS);
        yList.offer(nextY);
        roList.offer(ro);

        double[] tmpX = newX;
        newX = x;
        x = tmpX;

        double[] tmpGrad = newGrad;
        newGrad = grad;
        grad = tmpGrad;

        ++iter;
    }

    double getValue() {
        return value;
    }

    OptimizerState(DiffFunction f, double[] init, int m, double l1weight, boolean quiet) {
        this.x = init;
        this.grad = new double[init.length];
        this.newX = init.clone();
        this.newGrad = new double[init.length];
        this.dir = new double[init.length];
        this.steepestDescDir = newGrad.clone();
        this.alphas = new double[m];
        this.iter = 1;
        this.m = m;
        this.dim = init.length;
        this.func = f;
        this.l1weight = l1weight;
        this.quiet = quiet;

        if (m <= 0) {
            throw new RuntimeException("m must be an integer greater than zero.");
        }

        value = evalL1();
        grad = newGrad.clone();
    }

}
