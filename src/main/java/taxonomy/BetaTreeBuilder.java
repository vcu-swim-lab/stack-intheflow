package taxonomy;

import java.util.ArrayList;
import java.util.HashMap;
import sampling.likelihood.DirMult;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import sampling.util.TreeNode;
import util.MiscUtils;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class BetaTreeBuilder extends AbstractTaxonomyBuilder {

    public static final int NEW_LABEL = -1;
    public static final int YES = 0;
    public static final int NO = 1;
    protected double alpha;
    protected double a;
    protected double b;
    // internal
    ArrayList<Integer>[] labelDocs;

    public BetaTreeBuilder() {
        super();
    }

    public BetaTreeBuilder(int[][] labels, ArrayList<String> labVoc,
            double alpha, double a, double b) {
        super(labels, labVoc);
        this.alpha = alpha;
        this.a = a;
        this.b = b;
    }

    @Override
    public String getName() {
        return "beta-" + MiscUtils.formatDouble(alpha)
                + "-" + MiscUtils.formatDouble(a)
                + "-" + MiscUtils.formatDouble(b);
    }

    @Override
    public void buildTree() {
        System.out.println("Building tree in " + BetaTreeBuilder.class);

        labelDocs = new ArrayList[labelVocab.size()];
        for (int ll = 0; ll < labelDocs.length; ll++) {
            labelDocs[ll] = new ArrayList<Integer>();
        }

        for (int d = 0; d < labels.length; d++) {
            for (int l : labels[d]) {
                labelDocs[l].add(d);
            }
        }

        // all labels
        ArrayList<Integer> labelIndices = new ArrayList<Integer>();
        for (int l = 0; l < labelVocab.size(); l++) {
            labelIndices.add(l);
        }

        int rootIdx = this.labelVocab.size();
        this.treeRoot = new TreeNode<TreeNode, Integer>(0, 0, rootIdx, null);
        this.labelVocab.add("root");
        recursiveBuild(labelIndices, treeRoot);
    }

    private void recursiveBuild(ArrayList<Integer> labelIndices,
            TreeNode<TreeNode, Integer> node) {
        if (labelIndices.size() < 15) {
            for (int idx : labelIndices) {
                TreeNode<TreeNode, Integer> child = new TreeNode<TreeNode, Integer>(
                        node.getNextChildIndex(),
                        node.getLevel() + 1,
                        idx,
                        node);
                node.addChild(child.getIndex(), child);
            }
            return;
        }
        ArrayList<Integer> docIndices = new ArrayList<Integer>();
        for (int ll : labelIndices) {
            for (int d : labelDocs[ll]) {
                if (!docIndices.contains(d)) {
                    docIndices.add(d);
                }
            }
        }

        ArrayList<ArrayList<Integer>> partitions = partition(labelIndices,
                docIndices, alpha / (node.getLevel() + 1));

        for (ArrayList<Integer> partition : partitions) {
            Integer exemIdx = chooseExemplar(partition);
            TreeNode<TreeNode, Integer> exemplar;
            if (exemIdx == NEW_LABEL) {
                String newLabel = "label-" + labelVocab.size();
                exemplar = new TreeNode<TreeNode, Integer>(
                        node.getNextChildIndex(),
                        node.getLevel() + 1,
                        labelVocab.size(),
                        node);
                this.labelVocab.add(newLabel);
            } else {
                exemplar = new TreeNode<TreeNode, Integer>(
                        node.getNextChildIndex(),
                        node.getLevel() + 1,
                        exemIdx,
                        node);
            }
            node.addChild(exemplar.getIndex(), exemplar);
            partition.remove(exemIdx);
            recursiveBuild(partition, exemplar);
        }
    }

    private int chooseExemplar(ArrayList<Integer> labelIndices) {
        if (labelIndices.size() == 1) {
            return labelIndices.get(0);
        }
        return chooseExemplarByDocumentFrequency(labelIndices);
//        return NEW_LABEL;
    }

    private int chooseExemplarByDocumentFrequency(ArrayList<Integer> labelIndices) {
        int maxCount = -1;
        int idx = -1;
        for (int ii : labelIndices) {
            if (labelFreqs[ii] > maxCount) {
                maxCount = labelFreqs[ii];
                idx = ii;
            }
        }
        return idx;
    }

    private int chooseExemplarByCentrality(ArrayList<Integer> labelIndices) {
        throw new RuntimeException("To be implemented");
    }

    private ArrayList<ArrayList<Integer>> partition(
            ArrayList<Integer> labelIndices,
            ArrayList<Integer> docIndices,
            double dpAlpha) {

        System.out.println("Partitioning:"
                + " # labels: " + labelIndices.size()
                + ". # docs: " + docIndices.size());

        Sampler sampler = new Sampler(labelDocs,
                labelIndices,
                docIndices,
                labels.length,
                dpAlpha);
        sampler.initialize();
        sampler.iterate();
        sampler.printGroups();
        return sampler.getPartitions();
    }

    class Sampler {

        int maxIter = 10;
        int numLabels;
        int numDocuments;
        Restaurant<Group, Integer, HashMap<Integer, DirMult>> restaurant;
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
            this.dpAlpha = dpAlpha;
            this.numLabels = this.labelIndices.size();

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
            this.restaurant = new Restaurant<Group, Integer, HashMap<Integer, DirMult>>();

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
            HashMap<Integer, DirMult> coins = new HashMap<Integer, DirMult>();
            for (int d : docIndices) {
                coins.put(d, new DirMult(new double[]{a, b}));
            }
            Group group = new Group(groupIdx, coins);
            return group;
        }

        void removeLabelFromGroup(Group group, int labIdx) {
            restaurant.removeCustomerFromTable(labIdx, group.getIndex());
            if (group.isEmpty()) {
                restaurant.removeTable(group.getIndex());
                return;
            }
            for (int d : docIndices) {
                if (this.labelDocs[labIdx].contains(d)) {
                    group.getContent().get(d).decrement(YES);
                } else {
                    group.getContent().get(d).decrement(NO);
                }
            }
        }

        void addLabelToGroup(Group group, int labIdx) {
            restaurant.addCustomerToTable(labIdx, group.getIndex());
            for (int d : docIndices) {
                if (this.labelDocs[labIdx].contains(d)) {
                    group.getContent().get(d).increment(YES);
                } else {
                    group.getContent().get(d).increment(NO);
                }
            }
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

                System.out.println("--- " + getCurrentState());
                validate("iter " + iter);
            }
        }

        double getLogLikelihood(Group group, ArrayList<Integer> obs) {
            double llh = 0.0;
            for (int d : docIndices) {
                if (obs.contains(d)) {
                    llh += group.getContent().get(d).getLogLikelihood(YES);
                } else {
                    llh += group.getContent().get(d).getLogLikelihood(NO);
                }
            }
            return llh;
        }

        int sampleZ(int labIdx) {
            ArrayList<Integer> groupIndices = new ArrayList<Integer>();
            ArrayList<Double> logprobs = new ArrayList<Double>();
            for (Group group : restaurant.getTables()) {
                groupIndices.add(group.getIndex());

                double lp = Math.log(group.getNumCustomers())
                        + getLogLikelihood(group, this.labelDocs[labIdx]);
                logprobs.add(lp);
            }

            groupIndices.add(Restaurant.EMPTY_TABLE_INDEX);
            double lp = Math.log(dpAlpha)
                    + getLogLikelihood(emptyGroup, this.labelDocs[labIdx]);
            logprobs.add(lp);

            int sampledIdx = SamplerUtils.logMaxRescaleSample(logprobs);
            return groupIndices.get(sampledIdx);
        }

        String getCurrentState() {
            StringBuilder str = new StringBuilder();
            str.append("# groups: ").append(restaurant.getNumTables()).append("\n");
            double groupLlh = restaurant.getJointProbabilityAssignments(dpAlpha);
            double llh = 0.0;
            for (Group group : restaurant.getTables()) {
                for (DirMult dir : group.getContent().values()) {
                    llh += dir.getLogLikelihood();
                }
            }
            str.append("LLH: table: ").append(MiscUtils.formatDouble(groupLlh))
                    .append(". obs: ").append(MiscUtils.formatDouble(llh))
                    .append(". ").append(MiscUtils.formatDouble(groupLlh + llh))
                    .append("\n");
            return str.toString();
        }

        void validate(String msg) {
            restaurant.validate(msg);
            for (Group group : restaurant.getTables()) {
                for (DirMult swtch : group.getContent().values()) {
                    swtch.validate(msg);
                }
            }
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

    class Group extends FullTable<Integer, HashMap<Integer, DirMult>> {

        Group(int index, HashMap<Integer, DirMult> coins) {
            super(index, coins);
        }
    }
}
