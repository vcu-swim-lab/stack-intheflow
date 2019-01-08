package sampling.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class SparseCount implements Cloneable, Serializable {

    private static final long serialVersionUID = 1123581321L;
    private HashMap<Integer, Integer> counts;
    private int countSum;

    public SparseCount() {
        this.counts = new HashMap<Integer, Integer>();
        this.countSum = 0;
    }

    public SparseCount(SparseCount other) {
        this.counts = new HashMap<>();
        for (int key : other.getIndices()) {
            this.counts.put(key, other.getCount(key));
        }
        this.countSum = other.getCountSum();
    }

    @Override
    public SparseCount clone() throws CloneNotSupportedException {
        SparseCount sc = (SparseCount) super.clone();
        sc.counts = (HashMap<Integer, Integer>) this.counts.clone();
        return sc;
    }

    public double dotprod(double[] w) {
        double dp = 0.0;
        for (int idx : this.getIndices()) {
            dp += this.getCount(idx) * w[idx];
        }
        return dp;
    }

    public int size() {
        return this.counts.size();
    }

    public void remove(int idx) {
        if (!this.containsIndex(idx)) {
            throw new RuntimeException("Index " + idx + " not found");
        }
        this.setCount(idx, 0);
    }

    public HashMap<Integer, Integer> getObservations() {
        return this.counts;
    }

    public void setCount(int observation, int count) {
        if (count < 0) {
            throw new RuntimeException("Setting a negative count. " + count);
        }
        int curCount = this.getCount(observation);
        this.counts.put(observation, count);
        this.countSum += count - curCount;
        if (count == 0) {
            this.counts.remove(observation);
        }

        if (counts.get(observation) != null && this.counts.get(observation) < 0) {
            throw new RuntimeException("Negative count for observation " + observation
                    + ". count = " + this.counts.get(observation));
        }
        if (countSum < 0) {
            throw new RuntimeException("Negative count sumze " + countSum);
        }
    }

    public ArrayList<Integer> getSortedIndices() {
        ArrayList<Integer> sortedIndices = new ArrayList<Integer>();
        for (int ii : getIndices()) {
            sortedIndices.add(ii);
        }
        Collections.sort(sortedIndices);
        return sortedIndices;
    }

    public Set<Integer> getIndices() {
        return this.counts.keySet();
    }

    public boolean containsIndex(int idx) {
        return this.counts.containsKey(idx);
    }

    public int getCountSum() {
        return this.countSum;
    }

    public int getCount(int observation) {
        Integer count = this.counts.get(observation);
        if (count == null) {
            return 0;
        } else {
            return count;
        }
    }

    public void changeCount(int observation, int delta) {
        int count = getCount(observation);
        this.setCount(observation, count + delta);
    }

    public void increment(int observation) {
        Integer count = this.counts.get(observation);
        if (count == null) {
            this.counts.put(observation, 1);
        } else {
            this.counts.put(observation, count + 1);
        }
        this.countSum++;
    }

    public void decrement(int observation) {
        Integer count = this.counts.get(observation);
        if (count == null) {
            for (Integer obs : this.counts.keySet()) {
                System.out.println(obs + ": " + this.counts.get(obs));
            }
            throw new RuntimeException("Removing observation that does not exist " + observation);
        }
        if (count == 1) {
            this.counts.remove(observation);
        } else {
            this.counts.put(observation, count - 1);
        }
        this.countSum--;

        if (counts.get(observation) != null && this.counts.get(observation) < 0) {
            throw new RuntimeException("Negative count for observation " + observation
                    + ". count = " + this.counts.get(observation));
        }
        if (countSum < 0) {
            throw new RuntimeException("Negative count sumze " + countSum);
        }
    }

    public boolean isEmpty() {
        return this.countSum == 0;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (int obs : this.getIndices()) {
            str.append(obs).append(":").append(getCount(obs)).append(" ");
        }
        return str.toString();
    }

    public void validate(String msg) {
        if (this.countSum < 0) {
            throw new RuntimeException(msg + ". Negative countSum");
        }

        int totalCount = 0;
        for (int obs : this.counts.keySet()) {
            totalCount += this.counts.get(obs);
        }
        if (totalCount != this.countSum) {
            throw new RuntimeException(msg + ". Total counts mismatched. " + totalCount + " vs. " + countSum);
        }
    }

    public void add(SparseCount other) {
        for (int key : other.getIndices()) {
            this.changeCount(key, other.getCount(key));
        }
    }

    public static SparseCount add(SparseCount sc1, SparseCount sc2) {
        SparseCount sc = new SparseCount();
        for (int key1 : sc1.getIndices()) {
            sc.changeCount(key1, sc1.getCount(key1) + sc2.getCount(key1));
        }
        for (int key2 : sc2.getIndices()) {
            if (sc1.containsIndex(key2)) {
                continue;
            }
            sc.changeCount(key2, sc1.getCount(key2) + sc2.getCount(key2));
        }
        return sc;
    }

    public static String output(SparseCount sc) {
        StringBuilder str = new StringBuilder();
        for (int obs : sc.counts.keySet()) {
            str.append(obs).append(":").append(sc.counts.get(obs)).append("\t");
        }
        return str.toString();
    }

    public static SparseCount input(String line) {
        SparseCount sp = new SparseCount();
        if (!line.isEmpty()) {
            String[] sline = line.trim().split("\t");
            for (String obsCount : sline) {
                String[] parse = obsCount.split(":");
                int obs = Integer.parseInt(parse[0]);
                int count = Integer.parseInt(parse[1]);
                sp.changeCount(obs, count);
            }
        }
        return sp;
    }
}
