package optimization;

//import gurobi.GRB;
//import gurobi.GRBEnv;
//import gurobi.GRBLinExpr;
//import gurobi.GRBModel;
//import gurobi.GRBQuadExpr;
//import gurobi.GRBVar;
import java.util.Random;
import util.MiscUtils;
import util.SamplerUtils;

/**
 * Obsolete. To be removed. L2-norm multiple linear regression.
 *
 * @author vietan
 */
public class GurobiMLRL2Norm {

    private double[][] designMatrix;
    private double[] responseVector;
    private double rho; // variance of observations
    private double sigma; // variance of variables
    private double mean; // mean of variables
    private double[] means; // one mean for each variable
    private double[] sigmas; // one variance for each variable

    public GurobiMLRL2Norm(double[][] X, double[] y, double rho, double mean, double sigma) {
        this.designMatrix = X;
        this.responseVector = y;
        this.rho = rho;
        this.sigma = sigma;
        this.mean = mean;
    }

    public GurobiMLRL2Norm(double[][] X, double[] y, double rho) {
        this(X, y, rho, 0.0, 1.0);
    }

    public GurobiMLRL2Norm(double[][] X, double[] y) {
        this(X, y, 1.0, 0.0, 1.0);
    }

    public GurobiMLRL2Norm() {
    }

    public double[] solveExact() {
        double[] solution = new double[getNumVariables()];
        int D = getNumObservations();
        int V = getNumVariables();

//        System.out.println("Solving MLR L2 ...");
//        System.out.println("# observations: " + D);
//        System.out.println("# variables: " + V);
//        
//        try {
//            GRBEnv env = new GRBEnv();
//            GRBModel model = new GRBModel(env);
//
//            // variables
//            GRBVar[] variables = new GRBVar[getNumVariables()];
//            for (int v = 0; v < getNumVariables(); v++) {
//                variables[v] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 1.0, GRB.CONTINUOUS, "x-" + v);
//            }
//            model.update();
//
//            // objective function
//            GRBQuadExpr obj = new GRBQuadExpr();
//
//            // likelihood
//            double constTerm = 0.0;
//            double[] firstOrderTerms = new double[V];
//            double[] secondOrderTerms = new double[V];
//            double[][] crossTerms = new double[V][V];
//            for(int d=0; d<D; d++) {
//                constTerm += responseVector[d] * responseVector[d] / rho;
//                for(int v=0; v<V; v++) {
//                    firstOrderTerms[v] += -2 * responseVector[d] * designMatrix[d][v] / rho;
//                    secondOrderTerms[v] += designMatrix[d][v] * designMatrix[d][v] / rho;
//                }
//                for(int ii=0; ii<V; ii++) {
//                    for(int jj=0; jj<ii; jj++) {
//                        crossTerms[ii][jj] += 2 * designMatrix[d][ii] * designMatrix[d][jj] / rho;
//                    }
//                }
//            }
//            obj.addConstant(constTerm);
//            for(int v=0; v<V; v++) {
//                obj.addTerm(firstOrderTerms[v], variables[v]);
//                obj.addTerm(secondOrderTerms[v], variables[v], variables[v]);
//                for(int u=0; u<v; u++) {
//                    obj.addTerm(crossTerms[v][u], variables[v], variables[u]);
//                }
//            }
//
//            // prior
//            for (int v = 0; v < V; v++) {
//                obj.addConstant(getMean(v) * getMean(v) / getSigma(v));
//                obj.addTerm(-2 * getMean(v) / getSigma(v), variables[v]);
//                obj.addTerm(1.0 / getSigma(v), variables[v], variables[v]);
//            }
//
//            model.setObjective(obj, GRB.MINIMIZE);
//
//            // optimize
//            model.optimize();
//
//            // get solution
//            for (int v = 0; v < getNumVariables(); v++) {
//                solution[v] = variables[v].get(GRB.DoubleAttr.X);
//            }
//
//            // dispose of model and environment
//            model.dispose();
//            env.dispose();
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Exception while solving " + GurobiMLRL2Norm.class.getName());
//        }
        return solution;
    }

    public double[] solve() {
        double[] solution = new double[getNumVariables()];
//        try {
//            GRBEnv env = new GRBEnv();
//            GRBModel model = new GRBModel(env);
//
//            // add variables
//            GRBVar[] regParams = new GRBVar[getNumVariables()];
//            for (int v = 0; v < getNumVariables(); v++) {
//                regParams[v] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "var-" + v);
//            }
//
//            GRBVar[] docAuxParams = new GRBVar[getNumObservations()];
//            for (int d = 0; d < getNumObservations(); d++) {
//                docAuxParams[d] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "dvar-" + d);
//            }
//
//            model.update();
//
//            // objective function
//            GRBQuadExpr obj = new GRBQuadExpr();
//            for (GRBVar docAuxParam : docAuxParams) {
//                obj.addTerm(1.0 / getRho(), docAuxParam, docAuxParam);
//            }
//            for (int v = 0; v < getNumVariables(); v++) {
//                obj.addTerm(1.0 / getSigma(v), regParams[v], regParams[v]);
//            }
//            model.setObjective(obj, GRB.MINIMIZE);
//
//            // constraints
//            for (int d = 0; d < getNumObservations(); d++) {
//                GRBLinExpr expr = new GRBLinExpr();
//                expr.addTerm(1.0, docAuxParams[d]);
//                for (int v = 0; v < getNumVariables(); v++) {
//                    expr.addTerm(designMatrix[d][v], regParams[v]);
//                }
//                model.addConstr(expr, GRB.EQUAL, responseVector[d], "c-" + d);
//            }
//
//            // optimize
//            model.optimize();
//
//            // get solution
//            for (int v = 0; v < getNumVariables(); v++) {
//                solution[v] = regParams[v].get(GRB.DoubleAttr.X);
//            }
//
//            // dispose of model and environment
//            model.dispose();
//            env.dispose();
//        } catch (Exception e) {
//            e.printStackTrace();
//            System.exit(1);
//        }
        return solution;
    }

    public double getRho() {
        return rho;
    }

    public void setRho(double rho) {
        this.rho = rho;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    public double getMean() {
        return mean;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public double[] getSigmas() {
        return sigmas;
    }

    public void setSigmas(double[] sigmas) {
        this.sigmas = sigmas;
    }

    public void setMeans(double[] means) {
        this.means = means;
    }

    public double getMean(int v) {
        if (this.means == null) {
            return mean;
        } else {
            return this.means[v];
        }
    }

    public double getSigma(int v) {
        if (this.sigmas == null) {
            return this.sigma;
        } else {
            return this.sigmas[v];
        }
    }

    public void setDesignMatrix(double[][] d) {
        this.designMatrix = d;
    }

    public void setResponseVector(double[] r) {
        this.responseVector = r;
    }

    public int getNumObservations() {
        return designMatrix.length;
    }

    public int getNumVariables() {
        return designMatrix[0].length;
    }

    public static void main(String[] args) {
        test();
    }

    private static void test() {
        Random rand = new Random(1);
        double sigma = 1.0;
        double rho = 100;
        double mean = 0.0;

        int D = 10000;
        int V = 10;

        double[] trueParams = new double[V];
        for (int v = 0; v < V; v++) {
            trueParams[v] = SamplerUtils.getGaussian(mean, sigma);
        }

        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] = rand.nextFloat();
            }
        }

        // generate response
        double[] responseVector = new double[D];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                responseVector[d] += designMatrix[d][v] * trueParams[v];
            }
        }
        for (int d = 0; d < D; d++) {
            responseVector[d] = SamplerUtils.getGaussian(responseVector[d], rho);
        }

        GurobiMLRL2Norm mlr = new GurobiMLRL2Norm(designMatrix, responseVector, rho, mean, sigma);
        double[] solution = mlr.solveExact();
        System.out.println("solution:\t" + MiscUtils.arrayToString(solution));
        System.out.println("groundtruth:\t" + MiscUtils.arrayToString(trueParams));

//        for(int d=0; d<D; d++) {
//            double predVal = 0.0;
//            double trueVal = 0.0;
//            for(int v=0; v<V; v++) {
//                predVal += designMatrix[d][v] * solution[v];
//                trueVal += designMatrix[d][v] * trueParams[v];
//            }
//            System.out.println(d + "\t" + responseVector[d]
//                    + "\t" + predVal + "\t" + trueVal);
//        }
    }
}
