package taxonomy;

import java.util.ArrayList;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import util.SamplerUtils;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class GaussianTreeBuilder extends AbstractTaxonomyBuilder {

    protected double alpha;
    protected double mean;
    protected double variance;

    public GaussianTreeBuilder(int[][] labels, ArrayList<String> labVoc,
            double alpha, double mean, double var) {
        super(labels, labVoc);
        this.alpha = alpha;
        this.mean = mean;
        this.variance = var;
    }
    
    @Override
    public String getName(){
        return "gauss";
    }

    @Override
    public void buildTree() {
    }

    class Sampler {

        int maxIter = 10;
        int numLabels;
        int numDocuments;
        Restaurant<Group, Integer, SparseVector> restaurant;
        ArrayList<Integer>[] labelDocs;
        ArrayList<Integer> labelIndices;
        ArrayList<Integer> docIndices;
        int[] z;
        Group emptyGroup;
        double dpAlpha;

        public Sampler(
                ArrayList<Integer>[] labelDocs,
                ArrayList<Integer> labelIndices,
                ArrayList<Integer> docIndices,
                int numDocs,
                double dpAlpha) {
            this.labelDocs = labelDocs;
            this.labelIndices = labelIndices;
            this.docIndices = docIndices;
            this.numDocuments = numDocs;
            this.numLabels = this.labelIndices.size();
            this.dpAlpha = dpAlpha;
            this.emptyGroup = createNewGroup(Restaurant.EMPTY_TABLE_INDEX);
        }

        ArrayList<ArrayList<Integer>> getPartitions() {
            ArrayList<ArrayList<Integer>> partitions = new ArrayList<ArrayList<Integer>>();
            for (Group group : this.restaurant.getTables()) {
                partitions.add(group.getCustomers());
            }
            return partitions;
        }

        void initialize() {
            System.out.println("Initializing ...");

            this.z = new int[labelIndices.size()];
            this.restaurant = new Restaurant<Group, Integer, SparseVector>();

            // initialize assignment: each with a table
            for (int ii = 0; ii < labelIndices.size(); ii++) {
                int labIdx = labelIndices.get(ii);

                Group group = createNewGroup(restaurant.getNextTableIndex());
                restaurant.addTable(group);
                addLabelToGroup(group, labIdx);
                z[ii] = group.getIndex();
            }

            validate("initialize");
            System.out.println("--- Done initialization.");
            System.out.println("--- " + getCurrentState());
        }

        final Group createNewGroup(int groupIdx) {
            SparseVector weights = new SparseVector();
            for (int d : docIndices) {
                double weight = SamplerUtils.getGaussian(mean, variance);
                weights.set(d, weight);
            }
            Group group = new Group(groupIdx, weights);
            return group;
        }

        // do we need to aggregate all observations
        void removeLabelFromGroup(Group group, int labIdx) {
            restaurant.removeCustomerFromTable(labIdx, group.getIndex());
            if (group.isEmpty()) {
                restaurant.removeTable(group.getIndex());
            }
        }

        void addLabelToGroup(Group group, int labIdx) {
            restaurant.addCustomerToTable(labIdx, group.getIndex());
        }

        void iterate() {
            System.out.println("Iterating ...");

            for (int iter = 0; iter < maxIter; iter++) {
                System.out.println("--- iter " + iter);

                for (int ii = 0; ii < labelIndices.size(); ii++) {
                    int lIdx = labelIndices.get(ii);

                    // remove the current assignment
                    removeLabelFromGroup(restaurant.getTable(z[ii]), lIdx);

                    // sample
                    int sampleGroup = sampleZ(lIdx);
                    if (sampleGroup == Restaurant.EMPTY_TABLE_INDEX) {
                        Group group = createNewGroup(restaurant.getNextTableIndex());
                        restaurant.addTable(group);
                        sampleGroup = group.getIndex();
                    }
                    z[ii] = sampleGroup;

                    // add the current assignment
                    addLabelToGroup(restaurant.getTable(z[ii]), lIdx);
                }

                updateParameters();

                System.out.println("--- " + getCurrentState());
                validate("iter " + iter);
            }
        }

        int sampleZ(int labIdx) {
            ArrayList<Integer> groupIndices = new ArrayList<Integer>();
            ArrayList<Double> logprobs = new ArrayList<Double>();
            for (Group group : restaurant.getTables()) {
                groupIndices.add(group.getIndex());

                double lp = Math.log(group.getNumCustomers())
                        + getDotProduct(group, this.labelDocs[labIdx]);
                logprobs.add(lp);
            }

            groupIndices.add(Restaurant.EMPTY_TABLE_INDEX);
            double lp = Math.log(dpAlpha)
                    + getDotProduct(emptyGroup, this.labelDocs[labIdx]);
            logprobs.add(lp);

            int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
            return groupIndices.get(sampledIdx);
        }

        double getDotProduct(Group group, ArrayList<Integer> docs) {
            double val = 0.0;
            for (int d : docs) {
                val += group.getContent().get(d);
            }
            return val;
        }

        void updateParameters() {
            throw new RuntimeException("Not supported");
        }

        String getCurrentState() {
            StringBuilder str = new StringBuilder();
            str.append("# groups: ").append(restaurant.getNumTables()).append("\n");
            return str.toString();
        }

        void validate(String msg) {
            restaurant.validate(msg);
        }

        void printGroups() {
            StringBuilder str = new StringBuilder();
            for (Group group : restaurant.getTables()) {
                str.append("group ").append(group.getIndex())
                        .append(". # customers: ").append(group.getNumCustomers())
                        .append("\n");
                for (int labIdx : group.getCustomers()) {
                    str.append(">>> ").append(labIdx)
                            .append(": ").append(labelVocab.get(labIdx))
                            .append(" (").append(labelFreqs[labIdx]).append(")")
                            .append("\n");
                }
                str.append("\n");
            }
            System.out.println(str.toString());
        }
    }

    class Group extends FullTable<Integer, SparseVector> {

        public Group(int index, SparseVector content) {
            super(index, content);
        }
    }
}
