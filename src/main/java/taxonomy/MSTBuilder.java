package taxonomy;

import graph.DirectedGraph;
import graph.EdmondsMST;
import graph.GraphEdge;
import graph.GraphNode;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import sampling.util.TreeNode;
import util.SparseVector;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class MSTBuilder extends AbstractTaxonomyBuilder {

    protected DirectedGraph<Integer> tree;
    protected GraphNode<Integer> root;

    public MSTBuilder(int[][] labels, ArrayList<String> labVoc) {
        super(labels, labVoc);
    }

    @Override
    public String getName() {
        return "mst";
    }

    @Override
    public void buildTree() {
        int L = getNumLabels();

        // create raw label graph
        DirectedGraph<Integer> labelGraph = new DirectedGraph<Integer>();

        // create label nodes
        GraphNode<Integer>[] graphNodes = new GraphNode[L + 1];
        for (int ll = 0; ll < L + 1; ll++) {
            graphNodes[ll] = new GraphNode<Integer>(ll);
        }
        root = graphNodes[L];

        SparseVector[] outWeights = new SparseVector[L];
        for (int ll = 0; ll < L; ll++) {
            outWeights[ll] = new SparseVector();
        }

        // pair frequencies
        for (int dd = 0; dd < labels.length; dd++) {
            int[] docLabels = labels[dd];
            for (int ii = 0; ii < docLabels.length; ii++) {
                for (int jj = 0; jj < docLabels.length; jj++) {
                    if (ii == jj) {
                        continue;
                    }
                    Double weight = outWeights[docLabels[jj]].get(docLabels[ii]);
                    if (weight == null) {
                        outWeights[docLabels[jj]].set(docLabels[ii], 1.0);
                    } else {
                        outWeights[docLabels[jj]].set(docLabels[ii], weight + 1.0);
                    }
                }
            }
        }

        // edges
        for (int l = 0; l < L; l++) {
            for (int ii : outWeights[l].getIndices()) {
                double weight = outWeights[l].get(ii) / labelFreqs[ii];
                GraphNode<Integer> source = graphNodes[l];
                GraphNode<Integer> target = graphNodes[ii];
                labelGraph.addEdge(source, target, -weight);
            }
        }

        // root's edges
        int maxLabelFreq = StatUtils.max(labelFreqs);
        for (int l = 0; l < L; l++) {
            double weight = (double) labelFreqs[l] / maxLabelFreq;
            labelGraph.addEdge(root, graphNodes[l], -weight);
        }

        EdmondsMST<Integer> dmst = new EdmondsMST<Integer>(root, labelGraph);
        this.tree = dmst.getMinimumSpanningTree();

        convertTree();
        this.labelVocab.add("root");
    }

    public void convertTree() {
        int L = labelVocab.size();
        this.treeRoot = new TreeNode<TreeNode, Integer>(0, 0, L, null);
        TreeNode<TreeNode, Integer>[] treeNodes = new TreeNode[L + 1];
        Queue<GraphNode<Integer>> queue = new LinkedList<GraphNode<Integer>>();
        queue.add(this.root);
        treeNodes[L] = this.treeRoot;
        while (!queue.isEmpty()) {
            GraphNode<Integer> mstNode = queue.poll();
            TreeNode<TreeNode, Integer> node = treeNodes[mstNode.getId()];
            
            if (tree.hasOutEdges(mstNode)) {
                for (GraphEdge edge : tree.getOutEdges(mstNode)) {
                    GraphNode<Integer> mstChild = edge.getTarget();
                    int labelIdx = mstChild.getId();
                    TreeNode<TreeNode, Integer> childNode =
                            new TreeNode<TreeNode, Integer>(
                            node.getNextChildIndex(),
                            node.getLevel() + 1, labelIdx,
                            node);
                    node.addChild(childNode.getIndex(), childNode);
                    treeNodes[labelIdx] = childNode;
                    queue.add(mstChild);
                }
            }
        }
    }
    
    @Override
    public String printTree() {
        StringBuilder str = new StringBuilder();
        Stack<TreeNode<TreeNode, Integer>> stack =
                new Stack<TreeNode<TreeNode, Integer>>();
        stack.add(treeRoot);
        while (!stack.isEmpty()) {
            TreeNode<TreeNode, Integer> node = stack.pop();
            for (int ii = 0; ii < node.getLevel(); ii++) {
                str.append("  ");
            }
            double weight = 0.0;
            if(!node.isRoot()) {
                
            }
            str.append(node.getContent())
                    .append("\t").append(labelVocab.get(node.getContent()))
                    .append("\n");
            for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                stack.add(child);
            }
        }
        return str.toString();
    }

    public DirectedGraph<Integer> getTree() {
        return this.tree;
    }

    public GraphNode<Integer> getRoot() {
        return this.root;
    }
}
