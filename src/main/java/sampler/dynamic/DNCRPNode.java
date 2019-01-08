/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sampler.dynamic;

import java.util.HashMap;
import sampling.likelihood.LogisticNormal;
import sampling.util.TreeNode;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public class DNCRPNode extends TreeNode<DNCRPNode, LogisticNormal> {

    public static final int PSEUDO_CHILD_INDEX = -1;
    public static final String EMPTY_NODE_PATH = "*";
    private DNCRPNode preNode;
    private DNCRPNode posNode;
    private DNCRPNode pseudoChildNode;
    private int numActualCustomers; // M
    private double numPseudoCustomers; // S

    public DNCRPNode(int index, int level,
            LogisticNormal content,
            DNCRPNode parent,
            DNCRPNode preNode,
            DNCRPNode posNode) {
        super(index, level, content, parent);
        this.preNode = preNode;
        this.posNode = posNode;
        this.numActualCustomers = 0;
        this.numPseudoCustomers = 0.0;
    }

    public String getPreNodePathString() {
        if (this.preNode == null) {
            return EMPTY_NODE_PATH;
        } else {
            return this.preNode.getPathString();
        }
    }

    public String getPosNodePathString() {
        if (this.posNode == null) {
            return EMPTY_NODE_PATH;
        } else {
            return this.posNode.getPathString();
        }
    }

    public void updateNumeratorLogProduct() {
        throw new RuntimeException("to be implemented");
    }

    public void updateDenominatorLogProduct() {
        throw new RuntimeException("to be implemented");
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[")
                .append(getPathString())
                .append(", #ac: ").append(numActualCustomers)
                .append(", #pc: ").append(MiscUtils.formatDouble(numPseudoCustomers))
                .append(", #obs: ").append(content.getCountSum())
                .append("]")
                .append(" [")
                .append(preNode == null ? "*" : preNode.getPathString())
                .append("]")
                .append(" [")
                .append(posNode == null ? "*" : posNode.getPathString())
                .append("]");
        return str.toString();
    }

    /**
     * TODO: redundant. check fillInactiveChildIndices
     */
//    public void updateInactiveChildIndices(){
//        int maxChildIndex = -1;
//        for(DNCRPNode child : this.getChildren()){
//            if(child.getIndex() > maxChildIndex)
//                maxChildIndex = child.getIndex();
//        }
//        
//        this.inactiveChildren = new TreeSet<Integer>();
//        for(int i=0; i<=maxChildIndex; i++){
//            if(!isChild(i))
//                this.inactiveChildren.add(i);
//        }
//    }
    public void createPseudoChildNode() {
        this.pseudoChildNode = new DNCRPNode(PSEUDO_CHILD_INDEX, this.level + 1,
                null, this, null, null);
    }

    public void addObservations(HashMap<Integer, Integer> obsCounts) {
        for (int obs : obsCounts.keySet()) {
            this.content.changeCount(obs, obsCounts.get(obs));
        }
    }

    public void removeObservations(HashMap<Integer, Integer> obsCounts) {
        for (int obs : obsCounts.keySet()) {
            this.content.changeCount(obs, -obsCounts.get(obs));
        }
    }

    public void changeNumCustomers(int delta) {
        this.numActualCustomers += delta;
    }

    public void incrementNumCustomers() {
        this.numActualCustomers++;
    }

    public void decrementNumCustomers() {
        this.numActualCustomers--;
    }

    public int getNumActualCustomers() {
        return this.numActualCustomers;
    }

    public DNCRPNode getPseudoChildNode() {
        return this.pseudoChildNode;
    }

    public DNCRPNode getPreNode() {
        return preNode;
    }

    public void setPreNode(DNCRPNode preNode) {
        this.preNode = preNode;
    }

    public DNCRPNode getPosNode() {
        return posNode;
    }

    public void setPosNode(DNCRPNode posNode) {
        this.posNode = posNode;
    }

    public double getNumPseudoCustomers() {
        return numPseudoCustomers;
    }

    public void setNumPseudoCustomers(double pseudoCount) {
        this.numPseudoCustomers = pseudoCount;
    }
}
