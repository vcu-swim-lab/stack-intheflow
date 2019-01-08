package core.crossvalidation;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 *
 * @author vietan
 */
public class CrossValidation<I, T extends Instance<I>> {

    public static Random rand = new Random(1123581321);
    private final String folder;
    private final String name;
    private ArrayList<T> instanceList;
    private Fold<I, T>[] folds;

    public CrossValidation(String folder, String name, ArrayList<T> instList) {
        this.folder = folder;
        this.name = name;
        this.instanceList = instList;
    }

    public String getName() {
        return this.name;
    }

    public String getFolderPath() {
        return new File(this.folder, this.name).getAbsolutePath();
    }

    public Fold<I, T>[] getFolds() {
        return this.folds;
    }

    public Fold<I, T> getFold(int i) {
        return this.folds[i];
    }

    public int getNumFolds() {
        return this.folds.length;
    }

    public void outputFolds() {
        for (Fold<I, T> fold : folds) {
            fold.outputFold();
        }
    }

    public void inputFolds(int numFolds) {
        this.folds = new Fold[numFolds];
        for (int i = 0; i < numFolds; i++) {
            this.folds[i] = new Fold(i, getFolderPath(), this.instanceList);
            this.folds[i].inputFold();
        }
    }

    public T getInstance(int index) {
        return this.instanceList.get(index);
    }

    public ArrayList<T> getInstanceList() {
        return this.instanceList;
    }

    public int getNumInstances() {
        return this.instanceList.size();
    }

    public void stratify(ArrayList<Integer> groupIds, int numFolds, double trToDevRatio) {
        this.folds = new Fold[numFolds];

        // group instance according to the group id
        HashMap<Integer, ArrayList<Integer>> groupedInstanceMap = new HashMap<Integer, ArrayList<Integer>>();
        for (int i = 0; i < instanceList.size(); i++) {
            int groupId = groupIds.get(i);
            ArrayList<Integer> instancesInGroup = groupedInstanceMap.get(groupId);
            if (instancesInGroup == null) {
                instancesInGroup = new ArrayList<Integer>();
            }
            instancesInGroup.add(i);
            groupedInstanceMap.put(groupId, instancesInGroup);
        }

        // split
        for (int i = 0; i < numFolds; i++) {
            ArrayList<Integer> trSet = new ArrayList<Integer>();
            ArrayList<Integer> deSet = new ArrayList<Integer>();
            ArrayList<Integer> teSet = new ArrayList<Integer>();

            for (int groupId : groupedInstanceMap.keySet()) {
                ArrayList<Integer> instancesInGroup = groupedInstanceMap.get(groupId);
                int numInstancesInGroup = instancesInGroup.size();

                for (int j = 0; j < numInstancesInGroup; j++) {
                    int instId = instancesInGroup.get(j);
                    if (j >= i * numInstancesInGroup / numFolds && j < (i + 1) * numInstancesInGroup / numFolds) {
                        teSet.add(instId);
                    } else {
                        if (rand.nextDouble() < trToDevRatio) {
                            trSet.add(instId);
                        } else {
                            deSet.add(instId);
                        }
                    }
                }
            }
            this.folds[i] = new Fold(i, this.getFolderPath(), instanceList, trSet, deSet, teSet);
        }
    }
}
