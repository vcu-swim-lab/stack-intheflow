package util;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 *
 * @author vietan
 */
public class IOUtils {

    public static double[] inputArray(File inputFile) {
        double[] a = null;
        try {
            BufferedReader reader = getBufferedReader(inputFile);
            int nrow = Integer.parseInt(reader.readLine());
            a = new double[nrow];
            for (int ii = 0; ii < nrow; ii++) {
                a[ii] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + inputFile);
        }
        return a;
    }

    public static void outputArray(File outputFile, double[] a) {
        try {
            BufferedWriter writer = getBufferedWriter(outputFile);
            writer.write(a.length + "\n");
            for (int ii = 0; ii < a.length; ii++) {
                writer.write(a[ii] + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public static double[][] input2DArray(File inputFile) {
        double[][] m = null;
        try {
            BufferedReader reader = getBufferedReader(inputFile);
            int nrow = Integer.parseInt(reader.readLine());
            m = new double[nrow][];
            for (int ii = 0; ii < nrow; ii++) {
                String[] sline = reader.readLine().split("\t");
                int ncol = Integer.parseInt(sline[0]);
                m[ii] = new double[ncol];
                for (int jj = 0; jj < ncol; jj++) {
                    m[ii][jj] = Double.parseDouble(sline[jj + 1]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + inputFile);
        }
        return m;
    }

    public static void output2DArray(File outputFile, double[][] m) {
        try {
            BufferedWriter writer = getBufferedWriter(outputFile);
            writer.write(m.length + "\n");
            for (double[] row : m) {
                writer.write(Integer.toString(row.length));
                for (int jj = 0; jj < row.length; jj++) {
                    writer.write("\t" + row[jj]);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public static void outputLibSVM(File outputFile, SparseVector[] features, int[][] labels) {
        System.out.println("Outputing LIBSVM-formatted data to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int ii = 0; ii < features.length; ii++) {
                if (labels[ii].length == 0) {
                    continue;
                }
                // labels
                for (int jj = 0; jj < labels[ii].length - 1; jj++) {
                    writer.write(labels[ii][jj] + ",");
                }
                writer.write(Integer.toString(labels[ii][labels[ii].length - 1]) + " ");

                // features
                for (int idx : features[ii].getSortedIndices()) {
                    double featureVal = features[ii].get(idx);
                    if (Math.abs(featureVal) < 10E-6) {
                        continue;
                    }
                    writer.write(" " + idx + ":" + features[ii].get(idx));
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing "
                    + "LIBSVM-formatted data to " + outputFile);
        }
    }

    public static void outputPerplexities(File outputFile, ArrayList<Double> perplexities) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Average-Perplexity\t" + StatUtils.mean(perplexities) + "\n");
            writer.write("Min-Perplexity\t" + StatUtils.min(perplexities) + "\n");
            writer.write("Max-Perplexity\t" + StatUtils.max(perplexities) + "\n");
            writer.write("Median-Perplexity\t" + StatUtils.median(perplexities) + "\n");
            for (Double perplexitie : perplexities) {
                writer.write(perplexitie + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing perplexity"
                    + " results to " + outputFile);
        }
    }

    public static void outputTopicCoherences(File outputFile, ArrayList<Double> topicCoherences) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Average-Coherence\t" + StatUtils.mean(topicCoherences) + "\n");
            writer.write("Min-Coherence\t" + StatUtils.min(topicCoherences) + "\n");
            writer.write("Max-Coherence\t" + StatUtils.max(topicCoherences) + "\n");
            writer.write("Median-Coherence\t" + StatUtils.median(topicCoherences) + "\n");
            for (Double topicCoherence : topicCoherences) {
                writer.write(topicCoherence + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing perplexity"
                    + " results to " + outputFile);
        }
    }

    public static String getAbsolutePath(File folder, String filename) {
        return new File(folder, filename).getAbsolutePath();
    }

    public static double inputPerplexity(File inputFile) {
        return inputPerplexity(inputFile.getAbsolutePath());
    }

    public static double inputPerplexity(String inputFile) {
        double ppx = 0;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            ppx = Double.parseDouble(reader.readLine());
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing " + inputFile);
        }
        return ppx;
    }

    public static void outputPerplexity(File outputFile, double perplexity) {
        outputPerplexity(outputFile.getAbsolutePath(), perplexity);
    }

    public static void outputPerplexity(String outputFile, double perplexity) {
        System.out.println("Outputing perplexity to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(perplexity + "\n");
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing " + outputFile);
        }
    }

    public static ArrayList<String> loadVocab(String filepath) throws Exception {
        ArrayList<String> voc = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        while ((line = reader.readLine()) != null) {
            voc.add(line);
        }
        reader.close();
        return voc;
    }

    public static int[][] loadLDACFile(String filepath) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(filepath);

        ArrayList<int[]> wordList = new ArrayList<int[]>();
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split(" ");

            int numTypes = Integer.parseInt(sline[0]);
            int[] types = new int[numTypes];
            int[] counts = new int[numTypes];

            int numTokens = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                String[] entry = sline[ii + 1].split(":");
                int count = Integer.parseInt(entry[1]);
                int id = Integer.parseInt(entry[0]);
                numTokens += count;
                types[ii] = id;
                counts[ii] = count;
            }

            int[] gibbsString = new int[numTokens];
            int index = 0;
            for (int ii = 0; ii < numTypes; ++ii) {
                for (int jj = 0; jj < counts[ii]; ++jj) {
                    gibbsString[index++] = types[ii];
                }
            }
            wordList.add(gibbsString);
        }
        reader.close();
        int[][] words = wordList.toArray(new int[wordList.size()][]);
        return words;
    }

    public static ZipOutputStream getZipOutputStream(String outptuFile) throws Exception {
        File f = new File(outptuFile);
        ZipOutputStream out = new ZipOutputStream(new FileOutputStream(f));
        return out;
    }

    public static ZipInputStream getZipInputStream(String inputFile) throws Exception {
        File f = new File(inputFile);
        ZipInputStream in = new ZipInputStream(new FileInputStream(f));
        return in;
    }

    public static BufferedReader getBufferedReader(String filepath)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(filepath), "UTF-8"));
        return in;
    }

    public static BufferedReader getBufferedReader(File file)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        return in;
    }

    public static BufferedReader getBufferedReader(String zipFilePath, String zipEntry) throws Exception {
        ZipFile zipFile = new ZipFile(zipFilePath);
        ZipEntry modelEntry = zipFile.getEntry(zipEntry);
        return getBufferedReader(zipFile, modelEntry);
    }

    public static BufferedReader getBufferedReader(ZipFile zipFile, ZipEntry modelEntry)
            throws FileNotFoundException, IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        return reader;
    }

    public static BufferedWriter getBufferedWriter(String filepath)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath), "UTF-8"));
        return out;
    }

    public static BufferedWriter getBufferedWriter(File file)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
        return out;
    }

    public static BufferedWriter getBufferedWriterAppend(File file)
            throws FileNotFoundException, UnsupportedEncodingException {
        return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "UTF-8"));
    }

    public static BufferedWriter getBufferedWriterAppend(String filepath)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath, true), "UTF-8"));
        return out;
    }

    /**
     * Create a folder if it does not exist
     */
    public static void createFolder(String dir) {
        try {
            File folder = new File(dir);
            createFolder(folder);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating folder " + dir);
        }
    }

    public static void createFolder(File dir) {
        try {
            if (!dir.exists()) {
                dir.mkdirs();
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating folder " + dir);
        }
    }

    /**
     * Method that makes an empty folder. If the folder does not exist, create
     * it.
     *
     * @param dir the String indicates the directory to the folder
     */
    public static void makeEmptyFolder(String dir) {
        try {
            File folder = new File(dir);
            if (!folder.exists()) {
                folder.mkdirs();
            } else {
                deleteFolderContent(dir);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    /**
     * Delete a file
     *
     * @param filepath The directory of the file to be deleted
     */
    public static void deleteFile(String filepath) {
        try {
            File aFile = new File(filepath);
            if (aFile.exists()) {
                aFile.delete();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    /**
     * Method that deletes all the files in a given folder
     *
     * @param dir Directory of the folder to be deleted
     */
    public static void deleteFolderContent(String dir) {
        try {
            File folder = new File(dir);
            if (folder.isDirectory()) {
                String[] children = folder.list();
                if (children != null) {
                    for (String child : children) {
                        File tempF = new File(dir, child);
                        tempF.delete();
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    /**
     * Method that returns all the file names in a folder
     *
     * @param dir Directory of the folder
     * @return A String array consists of all the file names in that folder
     */
    public static String[] getFilesFromFolder(String dir) {
        String[] subFolderName = null;
        try {
            File folder = new File(dir);
            if (folder.isDirectory()) {
                subFolderName = folder.list();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
        return subFolderName;
    }

    /**
     * Method that returns the file name without the extension
     *
     * @param oriFilename A file name with extension (eg: filename.ext)
     * @return The file name without the extension (eg: filename)
     */
    public static String removeExtension(String oriFilename) {
        int dotAt = oriFilename.lastIndexOf(".");
        if (dotAt > 0) {
            return oriFilename.substring(0, dotAt);
        } else {
            return oriFilename;
        }
    }

    /**
     * Return the file name from a given file path
     *
     * @param filepath The given file path
     * @return The file name
     */
    public static String getFilename(String filepath) {
        return new File(filepath).getName();
    }

    /**
     * Copy files from one folder to another
     */
    public static void copyFile(String sourceFile, String destinationFile) {
        try {
            File f1 = new File(sourceFile);
            File f2 = new File(destinationFile);
            InputStream in = new FileInputStream(f1);

            //For Overwrite the file.
            OutputStream out = new FileOutputStream(f2);

            byte[] buf = new byte[1024];
            int len;
            while ((len = in.read(buf)) > 0) {
                out.write(buf, 0, len);
            }
            in.close();
            out.close();
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
            System.exit(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Output top words for each topic
     *
     * @param topicWordDistr 2D array containing topical word distributions
     * @param vocab List of tokens in the vocabulary
     * @param numTopWord Number of top words to output
     * @param file The output file
     */
    public static void outputTopWords(double[][] topicWordDistr,
            ArrayList<String> vocab,
            int numTopWord,
            File file) throws Exception {
        outputTopWords(topicWordDistr, vocab, numTopWord, file.getAbsolutePath());
    }

    /**
     * Output top words for each topic
     *
     * @param topicWordDistr 2D array containing topical word distributions
     * @param vocab List of tokens in the vocabulary
     * @param numTopWord Number of top words to output
     * @param filepath Path to the output file
     */
    public static void outputTopWords(double[][] topicWordDistr,
            ArrayList<String> vocab,
            int numTopWord,
            String filepath) throws Exception {

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < topicWordDistr.length; t++) {
            // sort words
            double[] bs = topicWordDistr[t];
            ArrayList<RankingItem<Integer>> rankedWords = new ArrayList<RankingItem<Integer>>();
            for (int i = 0; i < bs.length; i++) {
                rankedWords.add(new RankingItem<Integer>(i, bs[i]));
            }
            Collections.sort(rankedWords);

            // output top words
            writer.write("Topic " + (t + 1));
            for (int i = 0; i < Math.min(numTopWord, vocab.size()); i++) {
                writer.write("\t" + vocab.get(rankedWords.get(i).getObject()));
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    /**
     * Output top words for each topic
     *
     * @param topicWordDistr array list containing topical word distributions
     * @param vocab List of tokens in the vocabulary
     * @param numTopWord Number of top words to output
     * @param filepath Path to the output file
     */
    public static void outputTopWords(ArrayList<double[]> topicWordDistr, ArrayList<String> vocab,
            int numTopWord, String filepath) throws Exception {

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < topicWordDistr.size(); t++) {
            // sort words
            double[] bs = topicWordDistr.get(t);
            ArrayList<RankingItem<Integer>> rankedWords = new ArrayList<RankingItem<Integer>>();
            for (int i = 0; i < bs.length; i++) {
                rankedWords.add(new RankingItem<Integer>(i, bs[i]));
            }
            Collections.sort(rankedWords);

            // output top words
            writer.write("Topic " + (t + 1));
            for (int i = 0; i < Math.min(numTopWord, vocab.size()); i++) {
                writer.write("\t" + vocab.get(rankedWords.get(i).getObject()));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public static void outputLogLikelihoods(ArrayList<Double> logLhoods, String filepath)
            throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < logLhoods.size(); i++) {
            writer.write(i + "\t" + logLhoods.get(i) + "\n");
        }
        writer.close();
    }

    public static ArrayList<RankingItem<String>> getSortedVocab(double[] distr, ArrayList<String> vocab) {
        if (distr.length != vocab.size()) {
            throw new RuntimeException("In IOUtils: dimensions mismatched. "
                    + distr.length + " vs. " + vocab.size());
        }
        ArrayList<RankingItem<String>> sortedVocab = new ArrayList<RankingItem<String>>();
        for (int i = 0; i < distr.length; i++) {
            sortedVocab.add(new RankingItem<String>(vocab.get(i), distr[i]));
        }
        Collections.sort(sortedVocab);
        return sortedVocab;
    }

    public static void outputTopWordsWithProbs(double[][] topicWordDistr, ArrayList<String> vocab,
            int numTopWord, String filepath) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < topicWordDistr.length; t++) {
            // sort words
            double[] bs = topicWordDistr[t];
            ArrayList<RankingItem<Integer>> rankedWords = new ArrayList<RankingItem<Integer>>();
            for (int i = 0; i < bs.length; i++) {
                rankedWords.add(new RankingItem<Integer>(i, bs[i]));
            }
            Collections.sort(rankedWords);

            // output top words
            writer.write("Topic " + (t + 1));
            double cumm_prob = 0;
            for (int i = 0; i < Math.min(numTopWord, vocab.size()); i++) {
                cumm_prob += rankedWords.get(i).getPrimaryValue();
                writer.write("\t" + vocab.get(rankedWords.get(i).getObject())
                        + ", " + rankedWords.get(i).getPrimaryValue()
                        + ", " + cumm_prob);
            }
            writer.write("\n");
        }
        writer.close();
    }

    /**
     * Output top words for each topic with indices
     *
     * @param topicIndices List of topic indices
     * @param topicWordDistr 2D array containing topical word distributions
     * @param vocab List of tokens in the vocabulary
     * @param numTopWord Number of top words to output
     * @param filepath Path to the output file
     */
    public static void outputTopWords(ArrayList<Integer> topicIndices,
            double[][] topicWordDistr, ArrayList<String> vocab,
            int numTopWord, String filepath) throws Exception {

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int t = 0; t < topicWordDistr.length; t++) {
            // sort words
            double[] bs = topicWordDistr[t];
            ArrayList<RankingItem<Integer>> rankedWords = new ArrayList<RankingItem<Integer>>();
            for (int i = 0; i < bs.length; i++) {
                rankedWords.add(new RankingItem<Integer>(i, bs[i]));
            }
            Collections.sort(rankedWords);

            // output top words
            writer.write("Topic " + topicIndices.get(t));
            for (int i = 0; i < Math.min(numTopWord, vocab.size()); i++) {
                writer.write("\t" + vocab.get(rankedWords.get(i).getObject()));
            }
            writer.write("\n");
        }
        writer.close();
    }

    /**
     * Output latent variable values
     *
     * @param distrs 2D array containing the variable values
     * @param filepath Path to the output file
     */
    public static void outputDistributions(double[][] distrs, String filepath)
            throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        StringBuilder line;
        for (double[] var : distrs) {
            line = new StringBuilder();
            for (double v : var) {
                line.append(Double.toString(v)).append(" ");
            }
            writer.write(line.toString().trim() + "\n");
        }
        writer.close();
    }

    /**
     * Input latent variable values
     *
     * @param filepath Path to the input file
     */
    public static double[][] inputDistributions(String filepath)
            throws Exception {
        ArrayList<double[]> distr_list = new ArrayList<double[]>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split(" ");
            double[] distr = new double[sline.length];
            for (int i = 0; i < distr.length; i++) {
                distr[i] = Double.parseDouble(sline[i]);
            }
            distr_list.add(distr);
        }
        reader.close();

        double[][] distrs = new double[distr_list.size()][];
        for (int i = 0; i < distrs.length; i++) {
            distrs[i] = distr_list.get(i);
        }
        return distrs;
    }

    public static void outputDistribution(double[] distr, String filepath)
            throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (double d : distr) {
            writer.write(d + " ");
        }
        writer.close();
    }

    public static double[] inputDistribution(String filepath) throws Exception {
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String[] sline = reader.readLine().split(" ");
        reader.close();
        double[] distr = new double[sline.length];
        for (int i = 0; i < distr.length; i++) {
            distr[i] = Double.parseDouble(sline[i]);
        }
        return distr;
    }

    /**
     * Output latent variable assignments
     */
    public static void outputLatentVariableAssignment(int[][] var, String filepath)
            throws Exception {
        StringBuilder outputLine;
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int[] var_line : var) {
            if (var_line.length == 0) {
                writer.write("\n");
            } else {
                outputLine = new StringBuilder();
                outputLine.append(Integer.toString(var_line.length)).append("\t");
                for (int v : var_line) {
                    outputLine.append(Integer.toString(v)).append(" ");
                }
                writer.write(outputLine.toString().trim() + "\n");
            }
        }
        writer.close();
    }

    /**
     * Input latent variable assignments
     */
    public static int[][] inputLatentVariableAssignment(String filepath)
            throws Exception {
        ArrayList<int[]> list = new ArrayList<int[]>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            if (line.equals("")) {
                list.add(new int[0]);
                continue;
            }

            sline = line.split("\t")[1].split(" ");
            int[] assignments = new int[sline.length];
            for (int i = 0; i < assignments.length; i++) {
                assignments[i] = Integer.parseInt(sline[i]);
            }
            list.add(assignments);
        }
        reader.close();

        int[][] latentVar = new int[list.size()][];
        for (int i = 0; i < latentVar.length; i++) {
            latentVar[i] = list.get(i);
        }
        return latentVar;
    }

    public static void outputLatentVariables(double[][] vars, String filepath)
            throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        StringBuilder line;
        for (double[] var : vars) {
            line = new StringBuilder();
            for (double v : var) {
                line.append(Double.toString(v)).append(" ");
            }
            writer.write(line.toString().trim() + "\n");
        }
        writer.close();
    }

    public static double[][] inputLatentVariables(String filepath)
            throws Exception {
        ArrayList<double[]> var_list = new ArrayList<double[]>();
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split(" ");
            double[] distr = new double[sline.length];
            for (int i = 0; i < distr.length; i++) {
                distr[i] = Double.parseDouble(sline[i]);
            }
            var_list.add(distr);
        }
        reader.close();

        double[][] vars = new double[var_list.size()][];
        for (int i = 0; i < vars.length; i++) {
            vars[i] = var_list.get(i);
        }
        return vars;
    }

    public static void metaSummarize(ArrayList<String> singleRunFilepaths, String outputFolderpath) throws Exception {
        BufferedReader reader;
        String line;
        String[] sline;
        HashMap<String, HashMap<String, ArrayList<Double>>> metaSummary
                = new HashMap<String, HashMap<String, ArrayList<Double>>>();
        ArrayList<String> measurementNames = new ArrayList<String>();
        ArrayList<String> modelNames = new ArrayList<String>();

        // input
        for (int j = 0; j < singleRunFilepaths.size(); j++) {
            String singleRunFilepath = singleRunFilepaths.get(j);
            reader = getBufferedReader(singleRunFilepath);

            // header - first line
            line = reader.readLine();
            sline = line.split("\t");
            if (metaSummary.isEmpty()) { // for the first file
                for (int i = 1; i < sline.length; i++) {
                    metaSummary.put(sline[i], new HashMap<String, ArrayList<Double>>());
                    measurementNames.add(sline[i]);
                }
            }

            // from 2nd line onwards
            while ((line = reader.readLine()) != null) {
                sline = line.split("\t");
                String modelName = sline[0];

                if (j == 0) {
                    modelNames.add(modelName);
                }

                for (int i = 1; i < sline.length; i++) {
                    double perfValue = Double.parseDouble(sline[i]);
                    String measurementName = measurementNames.get(i - 1);

                    HashMap<String, ArrayList<Double>> measurementTable
                            = metaSummary.get(measurementName);
                    ArrayList<Double> modelPerfList = measurementTable.get(modelName);
                    if (modelPerfList == null) {
                        modelPerfList = new ArrayList<Double>();
                    }
                    modelPerfList.add(perfValue);
                    measurementTable.put(modelName, modelPerfList);
                    metaSummary.put(measurementName, measurementTable);
                }
            }
            reader.close();
        }

        // output
        BufferedWriter writer;
        for (String measurement : metaSummary.keySet()) {
            writer = getBufferedWriter(outputFolderpath + measurement + ".txt");
            HashMap<String, ArrayList<Double>> measurementTable = metaSummary.get(measurement);

            // write header
            for (String modelName : modelNames) {
                writer.write(modelName + "\t");
            }
            writer.write("\n");

            // write contents
            for (int j = 0; j < measurementTable.get(modelNames.get(0)).size(); j++) {
                for (String modelName : modelNames) {
                    writer.write(measurementTable.get(modelName).get(j) + "\t");
                }
                writer.write("\n");
            }
            writer.close();
        }
    }
}
