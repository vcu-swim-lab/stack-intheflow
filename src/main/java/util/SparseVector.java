package util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class SparseVector implements Serializable {

    private static final long serialVersionUID = 1123581321L;
    private HashMap<Integer, Double> values;
    private int dim;

    public SparseVector() {
        this.values = new HashMap<Integer, Double>();
    }

    public SparseVector(int dim) {
        this.values = new HashMap<Integer, Double>();
        this.dim = dim;
    }

    public SparseVector(SparseVector other) {
        this.values = new HashMap<>();
        for (int key : other.getIndices()) {
            this.values.put(key, other.get(key));
        }
        this.dim = other.getDimension();
    }

    public SparseVector(double[] vec) {
        this.values = new HashMap<>();
        this.dim = vec.length;
        for (int ii = 0; ii < vec.length; ii++) {
            if (vec[ii] != 0) {
                this.values.put(ii, vec[ii]);
            }
        }
    }

    public void setDimension(int dim) {
        this.dim = dim;
    }

    public int getDimension() {
        return this.dim;
    }

    public void scale(double scalar) {
        for (int kk : getIndices()) {
            values.put(kk, get(kk) * scalar);
        }
    }

    public void reset() {
        this.values = new HashMap<Integer, Double>();
    }

    public double[] dense() {
        double[] vec = new double[dim];
        for (int idx : getIndices()) {
            vec[idx] = get(idx);
        }
        return vec;
    }

    public boolean isEmpty() {
        return this.values.isEmpty();
    }

    public int size() {
        return this.values.size();
    }

    public void normalize() {
        double sum = this.sum();
        for (int idx : getIndices()) {
            double normVal = this.get(idx) / sum;
            this.values.put(idx, normVal);
        }
    }

    public double sum() {
        double sum = 0.0;
        for (double val : values.values()) {
            sum += val;
        }
        return sum;
    }

    public void remove(int index) {
        this.values.remove(index);
    }

    public double get(int index) {
        if (containsIndex(index)) {
            return this.values.get(index);
        }
        return 0.0;
    }

    public void set(int index, Double value) {
        this.values.put(index, value);
    }

    public void change(int index, double delta) {
        if (!this.containsIndex(index)) {
            this.set(index, delta);
        } else {
            double val = this.get(index) + delta;
            this.set(index, val);
        }
    }

    public boolean containsIndex(int idx) {
        return this.values.containsKey(idx);
    }

    public Set<Integer> getIndices() {
        return this.values.keySet();
    }

    public ArrayList<Integer> getSortedIndices() {
        ArrayList<Integer> sortedIndices = new ArrayList<Integer>();
        for (int idx : getIndices()) {
            sortedIndices.add(idx);
        }
        Collections.sort(sortedIndices);
        return sortedIndices;
    }

    /**
     * Add another sparse vector to this vector
     *
     * @param other The other sparse vector
     */
    public void add(SparseVector other) {
        for (int idx : other.getIndices()) {
            double otherVal = other.get(idx);
            Double thisVal = this.get(idx);
            if (thisVal == null) {
                this.set(idx, otherVal);
            } else {
                this.set(idx, thisVal + otherVal);
            }
        }
    }

    /**
     * Divide each element by a constant
     *
     * @param c The constant
     */
    public void divide(double c) {
        if (c == 0) {
            throw new RuntimeException("Dividing 0");
        }
        for (int idx : this.getIndices()) {
            this.set(idx, this.get(idx) / c);
        }
    }

    public void multiply(double c) {
        for (int idx : this.getIndices()) {
            this.set(idx, this.get(idx) * c);
        }
    }

    public double getL2Norm() {
        if (this.values.isEmpty()) {
            return 0.0;
        }
        double sumSquare = 0.0;
        for (double val : this.values.values()) {
            sumSquare += val * val;
        }
        return Math.sqrt(sumSquare);
    }

    public double dotProduct(SparseVector other) {
        double sum = 0.0;
        for (int idx : this.getIndices()) {
            Double otherVal = other.get(idx);
            if (otherVal == null) {
                continue;
            }
            sum += this.get(idx) * otherVal;
        }
        return sum;
    }

    public double dotProduct(double[] other) {
        double sum = 0.0;
        for (int idx : this.getIndices()) {
            sum += this.get(idx) * other[idx];
        }
        return sum;
    }

    public double cosineSimilarity(SparseVector other) {
        if (this.isEmpty() || other.isEmpty()) {
            return 0.0;
        }
        double thisL2Norm = this.getL2Norm();
        double thatL2Norm = other.getL2Norm();
        double dotProd = this.dotProduct(other);
        double cosine = dotProd / (thisL2Norm * thatL2Norm);
        return cosine;
    }

    public double cosineSimilarity(double[] other) {
        if (this.isEmpty()) {
            return 0.0;
        }
        double thisL2Norm = this.getL2Norm();
        double thatL2Norm = StatUtils.getL2Norm(other);
        double dotProb = this.dotProduct(other);
        double cosine = dotProb / (thisL2Norm * thatL2Norm);
        return cosine;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(Integer.toString(this.values.size()));
        for (int idx : this.getIndices()) {
            str.append(" ").append(idx).append(":").append(this.get(idx));
        }
        return str.toString();
    }

    public static SparseVector parseString(String str) {
        SparseVector vector = new SparseVector();
        String[] sstr = str.split(" ");
        for (int ii = 1; ii < sstr.length; ii++) {
            String[] se = sstr[ii].split(":");
            vector.set(Integer.parseInt(se[0]), Double.parseDouble(se[1]));
        }
        return vector;
    }

    public ArrayList<RankingItem<Integer>> getSortedList() {
        ArrayList<RankingItem<Integer>> sortedList = new ArrayList<RankingItem<Integer>>();
        for (int key : this.getIndices()) {
            sortedList.add(new RankingItem<Integer>(key, this.values.get(key)));
        }
        Collections.sort(sortedList);
        return sortedList;
    }

    public static String output(SparseVector vector) {
        StringBuilder str = new StringBuilder();
        str.append(Integer.toString(vector.dim));
        for (int key : vector.getIndices()) {
            str.append("\t").append(key).append(",").append(vector.get(key));
        }
        return str.toString();
    }

    public static SparseVector input(String str) {
        SparseVector vector = new SparseVector();
        String[] sstr = str.split("\t");
        int d = Integer.parseInt(sstr[0]);
        vector.setDimension(d);
        for (int ii = 1; ii < sstr.length; ii++) {
            String[] ss = sstr[ii].split(",");
            int key = Integer.parseInt(ss[0]);
            double val = Double.parseDouble(ss[1]);
            vector.set(key, val);
        }
        return vector;
    }

    public static SparseVector concatenate(ArrayList<SparseVector> vectors) {
        SparseVector concVec = null;
        for (SparseVector vec : vectors) {
            if (concVec == null) {
                concVec = vec;
            } else {
                int curDim = concVec.getDimension();
                int newDim = vec.getDimension();
                for (int jj : vec.getIndices()) {
                    concVec.set(jj + curDim, vec.get(jj));
                }
                concVec.setDimension(curDim + newDim);
            }
        }
        return concVec;
    }

    public void concatenate(SparseVector vector) {
        for (int jj : vector.getIndices()) {
            double val = vector.get(jj);
            this.set(jj + dim, val);
        }
        this.setDimension(dim + vector.getDimension());
    }
}
