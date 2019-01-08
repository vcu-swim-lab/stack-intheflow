package taxonomy;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;
import sampling.util.TreeNode;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public abstract class AbstractTaxonomyBuilder {

    // inputs
    protected int[][] labels;
    protected ArrayList<String> labelVocab;
    // internal
    protected int[] labelFreqs;
    protected TreeNode<TreeNode, Integer> treeRoot;

    public AbstractTaxonomyBuilder() {
    }

    public AbstractTaxonomyBuilder(int[][] labels, ArrayList<String> labVoc) {
        this.labels = labels;
        this.labelVocab = labVoc;
        labelFreqs = new int[labelVocab.size()];
        for (int[] label : labels) {
            for (int ii = 0; ii < label.length; ii++) {
                labelFreqs[label[ii]]++;
            }
        }
    }

    public abstract String getName();

    public abstract void buildTree();

    public void setLabelVocab(ArrayList<String> labVoc) {
        this.labelVocab = labVoc;
    }

    public void setLabels(int[][] labs) {
        this.labels = labs;
    }

    public TreeNode<TreeNode, Integer> getTreeRoot() {
        return this.treeRoot;
    }

    public int getNumLabels() {
        return this.labelVocab.size();
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
    }

    /**
     * Get the document frequency of each label
     *
     * @return An array of label's document frequency
     */
    protected int[] getLabelFrequencies() {
        return labelFreqs;
    }

    /**
     * Get the document frequency of each label pair
     *
     * @return Document frequency of each label pair
     */
    protected HashMap<String, Integer> getLabelPairFrequencies() {
        HashMap<String, Integer> pairFreqs = new HashMap<String, Integer>();
        for (int[] docLabels : labels) {
            for (int ii = 0; ii < docLabels.length; ii++) {
                for (int jj = 0; jj < docLabels.length; jj++) {
                    if (ii == jj) {
                        continue;
                    }
                    String pair = docLabels[ii] + "-" + docLabels[jj];
                    Integer count = pairFreqs.get(pair);
                    if (count == null) {
                        pairFreqs.put(pair, 1);
                    } else {
                        pairFreqs.put(pair, count + 1);
                    }
                }
            }
        }
        return pairFreqs;
    }

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
            str.append(node.getContent())
                    .append("\t").append(labelVocab.get(node.getContent()))
                    .append("\n");
            for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                stack.add(child);
            }
        }
        return str.toString();
    }

    public void outputTree(File filepath) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            Stack<TreeNode<TreeNode, Integer>> stack = new Stack<TreeNode<TreeNode, Integer>>();
            stack.add(treeRoot);
            while (!stack.isEmpty()) {
                TreeNode<TreeNode, Integer> node = stack.pop();
                for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                    stack.add(child);
                }
                writer.write(node.getPathString()
                        + "\t" + node.getContent()
                        + "\t" + labelVocab.get(node.getContent())
                        + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing tree to "
                    + filepath);
        }
    }

    public void outputTreeTemp(File filepath) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            Stack<TreeNode<TreeNode, Integer>> stack = new Stack<TreeNode<TreeNode, Integer>>();
            stack.add(treeRoot);
            while (!stack.isEmpty()) {
                TreeNode<TreeNode, Integer> node = stack.pop();
                for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                    stack.add(child);
                }
                for (int ii = 0; ii < node.getLevel(); ii++) {
                    writer.write(" ");
                }
                writer.write(node.getPathString()
                        + "\t" + node.getContent()
                        + "\t" + labelVocab.get(node.getContent())
                        + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing tree to "
                    + filepath);
        }
    }

    public void outputLabelVocab(File filepath) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            for (String element : labelVocab) {
                writer.write(element + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing label vocab to "
                    + filepath);
        }
    }

    public void inputLabelVocab(File filepath) {
        try {
            this.labelVocab = new ArrayList<String>();
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            String line;
            while ((line = reader.readLine()) != null) {
                this.labelVocab.add(line);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing label vocab from "
                    + filepath);
        }
    }

    public void inputTree(File filepath) {
        try {
            HashMap<String, TreeNode<TreeNode, Integer>> nodeMap =
                    new HashMap<String, TreeNode<TreeNode, Integer>>();
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            String line;
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                String pathStr = sline[0];
                int labelIdx = Integer.parseInt(sline[1]);

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                TreeNode<TreeNode, Integer> parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }

                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;
                TreeNode<TreeNode, Integer> node =
                        new TreeNode<TreeNode, Integer>(nodeIndex,
                        nodeLevel, labelIdx, parent);

                if (node.getLevel() == 0) {
                    this.treeRoot = node;
                }

                if (parent != null) {
                    parent.addChild(node.getIndex(), node);
                }

                nodeMap.put(pathStr, node);
            }
            reader.close();

            Stack<TreeNode<TreeNode, Integer>> stack = new Stack<TreeNode<TreeNode, Integer>>();
            stack.add(this.treeRoot);
            while (!stack.isEmpty()) {
                TreeNode<TreeNode, Integer> node = stack.pop();
                if (!node.isLeaf()) {
                    node.fillInactiveChildIndices();
                    for (TreeNode<TreeNode, Integer> child : node.getChildren()) {
                        stack.add(child);
                    }
                }
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing tree from "
                    + filepath);
        }
    }
}
