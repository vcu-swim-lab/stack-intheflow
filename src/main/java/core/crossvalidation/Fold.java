package core.crossvalidation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class Fold<I, T extends Instance<I>> {

    public static final int TRAIN = 0;
    public static final int DEV = 1;
    public static final int TEST = 2;
    public static final String TrainingExt = ".tr";
    public static final String DevelopExt = ".de";
    public static final String TestExt = ".te";
    private final int index;
    private final String folder;
    private ArrayList<T> instanceList;
    private ArrayList<Integer> trainingInstances;
    private ArrayList<Integer> developmentInstances;
    private ArrayList<Integer> testingInstances;

    public Fold(int idx, String folder) {
        this.index = idx;
        this.folder = folder;
    }

    public Fold(int idx, String folder,
            ArrayList<T> instList) {
        this.index = idx;
        this.folder = folder;
        this.instanceList = instList;
    }

    public Fold(int idx, String folder,
            ArrayList<T> instList,
            ArrayList<Integer> trList,
            ArrayList<Integer> devList,
            ArrayList<Integer> teList) {
        this.index = idx;
        this.folder = folder;
        this.instanceList = instList;
        this.trainingInstances = trList;
        this.developmentInstances = devList;
        this.testingInstances = teList;
    }

    public void outputFold(File folder) {
        try {
            // file format: <instance_index>\t<instance_id>\n
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(folder, "fold-" + index + TrainingExt));
            writer.write(trainingInstances.size() + "\n");
            for (int trInst : trainingInstances) {
                writer.write(trInst
                        + "\t" + instanceList.get(trInst).getId()
                        + "\n");
            }
            writer.close();

            writer = IOUtils.getBufferedWriter(new File(folder, "fold-" + index + DevelopExt));
            writer.write(developmentInstances.size() + "\n");
            for (int deInst : developmentInstances) {
                writer.write(deInst
                        + "\t" + instanceList.get(deInst).getId()
                        + "\n");
            }
            writer.close();

            writer = IOUtils.getBufferedWriter(new File(folder, "fold-" + index + TestExt));
            writer.write(testingInstances.size() + "\n");
            for (int teInst : testingInstances) {
                writer.write(teInst
                        + "\t" + instanceList.get(teInst).getId()
                        + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void outputFold() {
        File foldFolder = new File(this.getFoldFolderPath());
        IOUtils.createFolder(foldFolder);
        this.outputFold(foldFolder);
    }

    public void inputFold(File folder) {
        try {
            String line;
            BufferedReader reader = IOUtils.getBufferedReader(new File(folder, "fold-" + index + TrainingExt));
            this.trainingInstances = new ArrayList<Integer>();
            reader.readLine(); // first line showing # of instance
            while ((line = reader.readLine()) != null) {
                this.trainingInstances.add(Integer.parseInt(line.split("\t")[0]));
            }
            reader.close();

            reader = IOUtils.getBufferedReader(new File(folder, "fold-" + index + DevelopExt));
            this.developmentInstances = new ArrayList<Integer>();
            reader.readLine(); // first line showing # of instance
            while ((line = reader.readLine()) != null) {
                this.developmentInstances.add(Integer.parseInt(line.split("\t")[0]));
            }
            reader.close();

            reader = IOUtils.getBufferedReader(new File(folder, "fold-" + index + TestExt));
            this.testingInstances = new ArrayList<Integer>();
            reader.readLine(); // first line showing # of instance
            while ((line = reader.readLine()) != null) {
                this.testingInstances.add(Integer.parseInt(line.split("\t")[0]));
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void inputFold() {
        File foldFolder = new File(this.getFoldFolderPath());
        IOUtils.createFolder(foldFolder);
        this.inputFold(foldFolder);
    }

    public String getFoldFolderPath() {
        return new File(getFolder(), getFoldName()).getAbsolutePath();
    }

    public String getFoldName() {
        return "fold-" + index;
    }

    public String getFolder() {
        return folder;
    }

    public int getIndex() {
        return index;
    }

    public ArrayList<Integer> getDevelopmentInstances() {
        return developmentInstances;
    }

    public void setDevelopmentInstances(ArrayList<Integer> developmentInstances) {
        this.developmentInstances = developmentInstances;
    }

    public T getInstance(int index) {
        return this.instanceList.get(index);
    }

    public ArrayList<T> getInstanceList() {
        return instanceList;
    }

    public void setInstanceList(ArrayList<T> instanceList) {
        this.instanceList = instanceList;
    }

    public ArrayList<Integer> getTestingInstances() {
        return testingInstances;
    }

    public void setTestingInstances(ArrayList<Integer> testingInstances) {
        this.testingInstances = testingInstances;
    }

    public ArrayList<Integer> getTrainingInstances() {
        return trainingInstances;
    }

    public void setTrainingInstances(ArrayList<Integer> trainingInstances) {
        this.trainingInstances = trainingInstances;
    }

    public int getNumTrainingInstances() {
        return this.trainingInstances.size();
    }

    public int getNumDevelopmentInstances() {
        return this.developmentInstances.size();
    }

    public int getNumTestingInstances() {
        return this.testingInstances.size();
    }
}
