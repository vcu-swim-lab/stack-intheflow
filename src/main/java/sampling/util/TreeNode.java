package sampling.util;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.SortedSet;
import java.util.Stack;
import java.util.TreeSet;

/**
 * Implementation of a generic node in a tree which has a single parent node, a
 * set of (unbounded) children nodes.
 *
 * The content of the node can be any object.
 *
 * @author vietan
 * @param <N>
 * @param <C>
 */
public class TreeNode<N extends TreeNode, C>
        implements Comparable<TreeNode<N, C>>, Serializable {

    private static final long serialVersionUID = 1123581321L;
    public static final int ROOT_PARENT_INDEX = -1;
    protected int index;
    protected int level;
    protected C content;
    protected N parent;
    protected SortedSet<Integer> inactiveChildren; // indices for reuse
    protected HashMap<Integer, N> children;

    public TreeNode(int index, int level, C content, N parent) {
        this.index = index;
        this.level = level;
        this.content = content;
        this.parent = parent;

        this.inactiveChildren = new TreeSet<Integer>();
        this.children = new HashMap<Integer, N>();
    }

    public void setIndex(int idx) {
        this.index = idx;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    /**
     * Fill in the inactive indices (after all children nodes are loaded)
     */
    public void fillInactiveChildIndices() {
        int maxChildIndex = -1;
        for (TreeNode child : this.getChildren()) {
            if (child.getIndex() > maxChildIndex) {
                maxChildIndex = child.getIndex();
            }
        }

        this.inactiveChildren = new TreeSet<Integer>();
        for (int i = 0; i < maxChildIndex; i++) {
            if (!hasChild(i)) {
                this.inactiveChildren.add(i);
            }
        }
    }

    public void removeAllChilren() {
        this.children = new HashMap<Integer, N>();
        this.inactiveChildren = new TreeSet<Integer>();
    }

    public N getChild(int index) {
        return this.children.get(index);
    }

    public N addChild(int childIndex, N child) {
        if (this.children.containsKey(childIndex)) {
            throw new RuntimeException("Child node " + childIndex + " has already existed. "
                    + this.toString());
        }

        // remove this index from the inactive set
        if (this.inactiveChildren.contains(childIndex)) {
            this.inactiveChildren.remove(childIndex);
        }

        child.index = childIndex;
        this.children.put(childIndex, child);
        return child;
    }

    /**
     * Remove a child node. After the removal, the index will be added to the
     * inactive set for reuse
     *
     * @param childIndex The index of the child node to be removed
     */
    public void removeChild(int childIndex) {
        if (!this.hasChild(childIndex)) {
            throw new RuntimeException("Child " + childIndex + " does not exist. "
                    + "In node " + this.toString());
        }
        this.children.remove(childIndex);
        this.inactiveChildren.add(childIndex);
    }

    /**
     * Get the next available child index
     * @return 
     */
    public int getNextChildIndex() {
        if (this.inactiveChildren.isEmpty()) {
            return this.children.size();
        }
        return this.inactiveChildren.first();
    }

    /**
     * Return the unique path string for each node in the tree
     * @return 
     */
    public String getPathString() {
        if (this.isRoot()) {
            return Integer.toString(this.index);
        } else {
            return this.parent.getPathString() + ":" + this.index;
        }
    }

    public int[] getPathIndex() {
        int[] pathIndex = new int[this.level + 1];
        getPathIndex(this, pathIndex);
        return pathIndex;
    }

    private void getPathIndex(TreeNode<N, C> curNode, int[] pathIndex) {
        if (curNode == null) {
            return;
        }
        pathIndex[curNode.getLevel()] = curNode.getIndex();
        getPathIndex(curNode.getParent(), pathIndex);
    }

    public boolean isRoot() {
        return this.parent == null;
    }

    public boolean isLeaf() {
        return this.children.isEmpty();
    }

    public boolean hasChild(int childIndex) {
        return this.children.containsKey(childIndex);
    }

    public int getNumChildren() {
        return this.children.size();
    }

    public Collection<N> getChildren() {
        return this.children.values();
    }

    public int getLevel() {
        return this.level;
    }

    public int getIndex() {
        return this.index;
    }

    public void setParent(N p) {
        this.parent = p;
    }

    public N getParent() {
        return this.parent;
    }

    public int getParentIndex() {
        if (parent == null) {
            return ROOT_PARENT_INDEX;
        }
        return this.parent.getIndex();
    }

    public C getContent() {
        return this.content;
    }

    public void setContent(C content) {
        this.content = content;
    }

    public String printSubtreeStructure() {
        HashMap<Integer, Integer> levelNodeCounts = new HashMap<Integer, Integer>();
        int maxLevel = 0;
        Stack<TreeNode<N, C>> stack = new Stack<TreeNode<N, C>>();
        stack.add(this);
        while (!stack.isEmpty()) {
            TreeNode<N, C> node = stack.pop();
            int nodeLevel = node.getLevel();
            if (nodeLevel > maxLevel) {
                maxLevel = nodeLevel;
            }

            Integer count = levelNodeCounts.get(nodeLevel);
            if (count == null) {
                levelNodeCounts.put(nodeLevel, 1);
            } else {
                levelNodeCounts.put(nodeLevel, count + 1);
            }

            for (N child : node.getChildren()) {
                stack.add(child);
            }
        }

        StringBuilder str = new StringBuilder();
        for (int l = 0; l <= maxLevel; l++) {
            str.append(l).append("(").append(levelNodeCounts.get(l)).append(")\t");
        }
        return str.toString();
    }

    @Override
    public int hashCode() {
        String hashCodeStr = getPathString();
        return hashCodeStr.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if ((obj == null) || (this.getClass() != obj.getClass())) {
            return false;
        }
        TreeNode<N, C> r = (TreeNode<N, C>) (obj);
        return r.getPathString().equals(this.getPathString());
    }

    @Override
    public int compareTo(TreeNode<N, C> r) {
        if (this.level == r.level) {
            if (this.parent.equals(r.parent)) {
                return this.index - r.index;
            } else {
                return this.parent.compareTo(r.parent);
            }
        } else {
            return this.getLevel() - r.getLevel();
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", #ch = ").append(getChildren().size())
                .append("]");
        return str.toString();
    }
}
