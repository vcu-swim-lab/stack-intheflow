package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import util.IOUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public class AbstractLinearModel extends AbstractModel {

    public static final String MODEL_FILE = "model";
    protected double[] weights;
    protected boolean quiet;

    public AbstractLinearModel(String basename) {
        super(basename);
    }

    public double[] getWeights() {
        return this.weights;
    }

    public void setQuiet(boolean q) {
        this.quiet = q;
    }

    public void outputWeights(File file, ArrayList<String> features) {
        System.out.println("Outputing ranked weights to " + file);
        if (features.size() != weights.length) {
            throw new RuntimeException("Mismatch");
        }
        try {
            ArrayList<RankingItem<String>> rankWeights = new ArrayList<>();
            for (int k = 0; k < weights.length; k++) {
                rankWeights.add(new RankingItem<String>(features.get(k), weights[k]));
            }
            Collections.sort(rankWeights);
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write("Feature\tWeight\n");
            for (RankingItem<String> item : rankWeights) {
                writer.write(item.getObject() + "\t" + item.getPrimaryValue() + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    @Override
    public void output(File file) {
        System.out.println("Outputing model to " + file);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(weights.length + "\n");
            for (int ii = 0; ii < weights.length; ii++) {
                writer.write(weights[ii] + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    @Override
    public void input(File file) {
        System.out.println("Inputing model from " + file);
        try {
            BufferedReader reader = IOUtils.getBufferedReader(file);
            int V = Integer.parseInt(reader.readLine());
            this.weights = new double[V];
            for (int ii = 0; ii < V; ii++) {
                this.weights[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading from " + file);
        }
    }
}
