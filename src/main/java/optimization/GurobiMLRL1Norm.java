package optimization;

//import gurobi.GRB;
//import gurobi.GRBEnv;
//import gurobi.GRBLinExpr;
//import gurobi.GRBModel;
//import gurobi.GRBQuadExpr;
//import gurobi.GRBVar;
import java.util.Random;
import util.MiscUtils;

/**
 * Obsolete. To be removed.
 *
 * @author vietan
 */
public class GurobiMLRL1Norm {

    private double[][] designMatrix;
    private double[] responseVector;
    private double t;

    public GurobiMLRL1Norm(double t) {
        this.t = t;
    }

    public GurobiMLRL1Norm(double[][] X, double[] y, double t) {
        this.designMatrix = X;
        this.responseVector = y;
        this.t = t;
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

    public double[] solve() {
        int D = getNumObservations();
        int V = getNumVariables();

        double[] solution = new double[getNumVariables()];
//        try {
//            GRBEnv env = new GRBEnv();
//            GRBModel model = new GRBModel(env);
//
//            // add variables
//            GRBVar[] pRegParams = new GRBVar[V];
//            for (int v = 0; v < V; v++) {
//                pRegParams[v] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "pvar-" + v);
//            }
//            GRBVar[] nRegParams = new GRBVar[V];
//            for (int v = 0; v < V; v++) {
//                nRegParams[v] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "nvar-" + v);
//            }
//            GRBVar[] docAuxParams = new GRBVar[D];
//            for (int d = 0; d < D; d++) {
//                docAuxParams[d] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "dvar-" + d);
//            }
//            model.update();
//
//            // objective function
//            GRBQuadExpr obj = new GRBQuadExpr();
//            for (int d = 0; d < D; d++) {
//                obj.addTerm(1.0, docAuxParams[d], docAuxParams[d]);
//            }
//            model.setObjective(obj, GRB.MINIMIZE);
//
//            // constraints
//            for (int d = 0; d < D; d++) {
//                GRBLinExpr expr = new GRBLinExpr();
//                expr.addTerm(-1.0, docAuxParams[d]);
//                for (int v = 0; v < V; v++) {
//                    expr.addTerm(designMatrix[d][v], pRegParams[v]);
//                }
//                for (int v = 0; v < V; v++) {
//                    expr.addTerm(-designMatrix[d][v], nRegParams[v]);
//                }
//                model.addConstr(expr, GRB.EQUAL, responseVector[d], "c-" + d);
//            }
//
//            for(int v=0; v<V; v++){
//                GRBLinExpr expr = new GRBLinExpr();
//                expr.addTerm(1.0, pRegParams[v]);
//                model.addConstr(expr, GRB.GREATER_EQUAL, 0.0, "pc" + v);
//            }
//            
//            for(int v=0; v<V; v++) {
//                GRBLinExpr expr = new GRBLinExpr();
//                expr.addTerm(1.0, nRegParams[v]);
//                model.addConstr(expr, GRB.GREATER_EQUAL, 0.0, "nc" + v);
//            }
//            
//            GRBLinExpr expr = new GRBLinExpr();
//            for(int v=0; v<V; v++){
//                expr.addTerm(1.0, pRegParams[v]);
//                expr.addTerm(1.0, nRegParams[v]);
//            }
//            model.addConstr(expr, GRB.LESS_EQUAL, this.t, "total");
//            
//            // optimize
//            model.optimize();
//
//            // get solution
//            for (int v = 0; v < V; v++) {
//                solution[v] = pRegParams[v].get(GRB.DoubleAttr.X)
//                        - nRegParams[v].get(GRB.DoubleAttr.X);
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

    public static void main(String[] args) {
        test();
    }

    private static void test() {
        Random rand = new Random(1);

        int D = 100;
        int V = 10;
        double[][] designMatrix = new double[D][V];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                designMatrix[d][v] = rand.nextFloat();
            }
        }

        double[] trueParams = new double[V];
        for (int i = 0; i < 3; i++) {
            trueParams[i] = i + 1;
            trueParams[V - 1 - i] = -i - 1;
        }
        System.out.println("true params: " + MiscUtils.arrayToString(trueParams));

        // generate response
        double[] responseVector = new double[D];
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                responseVector[d] += designMatrix[d][v] * trueParams[v];
            }
        }

        GurobiMLRL1Norm mlr = new GurobiMLRL1Norm(designMatrix, responseVector, 50);
        double[] solution = mlr.solve();
        System.out.println("solution: " + MiscUtils.arrayToString(solution));
    }
}
