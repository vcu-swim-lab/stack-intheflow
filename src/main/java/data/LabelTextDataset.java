package data;

import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.DataUtils;
import util.IOUtils;
import util.RankingItem;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 *
 * @author vietan
 */
public class LabelTextDataset extends TextDataset {

    public static final String labelVocabExt = ".lvoc";
    public static final String arffExt = ".arff";
    public static final String xmlExt = ".xml";
    protected ArrayList<ArrayList<String>> labelList;
    protected ArrayList<String> labelVocab;
    protected int[][] labels; // list of labels for each document (maybe duplicate)
    protected int[][] uniqueLabels;
    protected int maxLabelVocSize = Integer.MAX_VALUE;
    protected int minLabelDocFreq = 1;

    public LabelTextDataset(String name) {
        super(name);
    }

    public LabelTextDataset(String name, String folder) {
        super(name, folder);
    }

    public LabelTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public void setMaxLabelVocabSize(int L) {
        this.maxLabelVocSize = L;
    }

    public void setMinLabelDocFreq(int f) {
        this.minLabelDocFreq = f;
    }

    public int[][] getLabels() {
        return this.labels;
    }

    public int[] getSingleLabels() {
        int[] singleLabels = new int[labels.length];
        for (int dd = 0; dd < singleLabels.length; dd++) {
            singleLabels[dd] = labels[dd][0];
        }
        return singleLabels;
    }

    public int[][] getUniqueLabels() {
        if (uniqueLabels == null) {
            uniqueLabels = new int[labels.length][];
        }
        for (int dd = 0; dd < labels.length; dd++) {
            Set<Integer> labelSet = new HashSet<>();
            for (int ll : labels[dd]) {
                labelSet.add(ll);
            }
            uniqueLabels[dd] = new int[labelSet.size()];
            int count = 0;
            for (int ll : labelSet) {
                uniqueLabels[dd][count++] = ll;
            }
        }
        return uniqueLabels;
    }

    public ArrayList<String> getLabelVocab() {
        return this.labelVocab;
    }

    public void setLabelVocab(ArrayList<String> lVoc) {
        this.labelVocab = lVoc;
    }

    public ArrayList<ArrayList<String>> getLabelList() {
        return this.labelList;
    }

    public void setLabelList(ArrayList<ArrayList<String>> lblList) {
        this.labelList = lblList;
    }

    /**
     * Filter document labels. The remaining labels only come from a given set
     * of labels
     *
     * @param labVoc The given set of labels
     */
    public void filterLabels(ArrayList<String> labVoc) {
        int D = words.length;
        this.labelVocab = labVoc;

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = this.labelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
    }

    /**
     * Filter labels that do not meet the minimum frequency requirement.
     *
     * @param minLabelFreq Minimum frequency
     */
    public void filterLabelsByFrequency(int minLabelFreq) {
        int D = words.length;
        int L = labelVocab.size();
        int[] labelFreqs = new int[L];
        for (int dd = 0; dd < D; dd++) {
            for (int ii = 0; ii < labels[dd].length; ii++) {
                labelFreqs[labels[dd][ii]]++;
            }
        }

        ArrayList<String> filterLabelVocab = new ArrayList<String>();
        for (int ll = 0; ll < L; ll++) {
            if (labelFreqs[ll] > minLabelFreq) {
                filterLabelVocab.add(labelVocab.get(ll));
            }
        }
        Collections.sort(filterLabelVocab);

        int[][] filterLabels = new int[D][];
        for (int d = 0; d < D; d++) {
            ArrayList<Integer> docFilterLabels = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String label = labelVocab.get(labels[d][ii]);
                int filterLabelIndex = filterLabelVocab.indexOf(label);
                if (filterLabelIndex >= 0) {
                    docFilterLabels.add(filterLabelIndex);
                }
            }

            filterLabels[d] = new int[docFilterLabels.size()];
            for (int ii = 0; ii < docFilterLabels.size(); ii++) {
                filterLabels[d][ii] = docFilterLabels.get(ii);
            }
        }

        this.labels = filterLabels;
        this.labelVocab = filterLabelVocab;
    }

    /**
     * Set the label vocabulary of this dataset with a new vocabulary. This
     * requires updating the label indices for all documents.
     *
     * @param newLabelVoc The new label vocabulary
     */
    public void resetLabelVocab(ArrayList<String> newLabelVoc) {
        int[][] newLabels = new int[labels.length][];
        for (int d = 0; d < labels.length; d++) {
            ArrayList<Integer> newDocLabelIndices = new ArrayList<Integer>();
            for (int ii = 0; ii < labels[d].length; ii++) {
                String labelStr = labelVocab.get(labels[d][ii]);
                int idx = newLabelVoc.indexOf(labelStr);
                if (idx >= 0) {
                    newDocLabelIndices.add(idx);
                }
            }
            newLabels[d] = new int[newDocLabelIndices.size()];
            for (int ii = 0; ii < newLabels[d].length; ii++) {
                newLabels[d][ii] = newDocLabelIndices.get(ii);
            }
        }

        this.labels = newLabels;
        this.labelVocab = newLabelVoc;
    }

    public void loadLabels(File labelFile) throws Exception {
        loadLabels(labelFile.getAbsolutePath());
    }

    public void loadLabels(String labelFile) throws Exception {
        logln("--- Loading labels from " + labelFile);

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        HashMap<String, ArrayList<String>> docLabelMap = new HashMap<String, ArrayList<String>>();
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(labelFile);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];

            ArrayList<String> docLabels = new ArrayList<String>();
            for (int ii = 1; ii < sline.length; ii++) {
                docLabels.add(sline[ii]);
            }
            docLabelMap.put(docId, docLabels);
        }
        reader.close();

        this.labelList = new ArrayList<ArrayList<String>>();
        for (String docId : docIdList) {
            ArrayList<String> docLabels = docLabelMap.get(docId);
            this.labelList.add(docLabels);
        }
        logln("--- --- Loaded " + labelList.size() + " label instances");
    }

    @Override
    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);
        formatLabels(outputFolder);
        super.format(outputFolder);
    }

    public void formatLabels(String outputFolder) throws Exception {
        logln("Formatting labels ...");
        if (this.labelVocab == null) {
            createLabelVocab();
        }

        // output label vocab
        outputLabelVocab(outputFolder);

        // get label indices
        this.labels = new int[this.labelList.size()][];
        for (int ii = 0; ii < labels.length; ii++) {
            ArrayList<Integer> docLabels = new ArrayList<Integer>();
            for (int jj = 0; jj < labelList.get(ii).size(); jj++) {
                int labelIndex = labelVocab.indexOf(labelList.get(ii).get(jj));
                if (labelIndex >= 0) { // filter out labels not in label vocab
                    docLabels.add(labelIndex);
                }
            }

            this.labels[ii] = new int[docLabels.size()];
            for (int jj = 0; jj < labels[ii].length; jj++) {
                this.labels[ii][jj] = docLabels.get(jj);
            }
        }
    }

    /**
     * Output the list of unique labels
     *
     * @param outputFolder Output folder
     * @throws java.lang.Exception
     */
    protected void outputLabelVocab(String outputFolder) throws Exception {
        File labelVocFile = new File(outputFolder, formatFilename + labelVocabExt);
        logln("--- Outputing label vocab ... " + labelVocFile.getAbsolutePath());
        DataUtils.outputVocab(labelVocFile.getAbsolutePath(), this.labelVocab);
    }

    /**
     * Create label vocabulary
     *
     * @throws java.lang.Exception
     */
    public void createLabelVocab() throws Exception {
        logln("--- Creating label vocab ...");
        createLabelVocabByFrequency();
    }

    protected void createLabelVocabByFrequency() throws Exception {
        HashMap<String, Integer> labelFreqs = new HashMap<String, Integer>();
        for (ArrayList<String> ls : this.labelList) {
            for (String l : ls) {
                Integer count = labelFreqs.get(l);
                if (count == null) {
                    labelFreqs.put(l, 1);
                } else {
                    labelFreqs.put(l, count + 1);
                }
            }
        }

        ArrayList<RankingItem<String>> rankLabels = new ArrayList<RankingItem<String>>();
        for (String label : labelFreqs.keySet()) {
            int freq = labelFreqs.get(label);
            if (freq >= this.minLabelDocFreq) {
                rankLabels.add(new RankingItem<String>(label, labelFreqs.get(label)));
            }
        }
        Collections.sort(rankLabels);

        this.labelVocab = new ArrayList<String>();
        for (int k = 0; k < Math.min(this.maxLabelVocSize, rankLabels.size()); k++) {
            this.labelVocab.add(rankLabels.get(k).getObject());
        }
        Collections.sort(this.labelVocab);
    }

    @Override
    public void createCrossValidation(String cvFolder, int numFolds,
            double trToDevRatio) throws Exception {
        ArrayList<Instance<String>> instanceList = new ArrayList<Instance<String>>();
        ArrayList<Integer> groupIdList = new ArrayList<Integer>();
        for (String docId : this.docIdList) {
            instanceList.add(new Instance<String>(docId));
            groupIdList.add(0); // random, no stratified
        }

        String cvName = "";
        CrossValidation<String, Instance<String>> cv
                = new CrossValidation<String, Instance<String>>(
                        cvFolder,
                        cvName,
                        instanceList);

        cv.stratify(groupIdList, numFolds, trToDevRatio);
        cv.outputFolds();

        for (Fold<String, Instance<String>> fold : cv.getFolds()) {
            // processor
            CorpusProcessor cp = new CorpusProcessor(corpProc);

            // training data
            LabelTextDataset trainData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            trainData.setFormatFilename(fold.getFoldName() + Fold.TrainingExt);
            ArrayList<String> trDocIds = new ArrayList<String>();
            ArrayList<String> trDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> trLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumTrainingInstances(); ii++) {
                int idx = fold.getTrainingInstances().get(ii);
                trDocIds.add(this.docIdList.get(idx));
                trDocTexts.add(this.textList.get(idx));
                trLabelList.add(this.labelList.get(idx));
            }
            trainData.setTextData(trDocIds, trDocTexts);
            trainData.setLabelList(trLabelList);
            trainData.format(fold.getFoldFolderPath());

            // development data: process using vocab from training
            LabelTextDataset devData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            devData.setFormatFilename(fold.getFoldName() + Fold.DevelopExt);
            ArrayList<String> deDocIds = new ArrayList<String>();
            ArrayList<String> deDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> deLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumDevelopmentInstances(); ii++) {
                int idx = fold.getDevelopmentInstances().get(ii);
                deDocIds.add(this.docIdList.get(idx));
                deDocTexts.add(this.textList.get(idx));
                deLabelList.add(this.labelList.get(idx));
            }
            devData.setTextData(deDocIds, deDocTexts);
            devData.setLabelVocab(trainData.getLabelVocab());
            devData.setLabelList(deLabelList);
            devData.format(fold.getFoldFolderPath());

            // test data: process using vocab from training
            LabelTextDataset testData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            testData.setFormatFilename(fold.getFoldName() + Fold.TestExt);
            ArrayList<String> teDocIds = new ArrayList<String>();
            ArrayList<String> teDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> teLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumTestingInstances(); ii++) {
                int idx = fold.getTestingInstances().get(ii);
                teDocIds.add(this.docIdList.get(idx));
                teDocTexts.add(this.textList.get(idx));
                teLabelList.add(this.labelList.get(idx));
            }
            testData.setTextData(teDocIds, teDocTexts);
            testData.setLabelVocab(trainData.getLabelVocab());
            testData.setLabelList(teLabelList);
            testData.format(fold.getFoldFolderPath());
        }
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex));
            for (int label : labels[docIndex]) {
                infoWriter.write("\t" + label);
            }
            infoWriter.write("\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File file) throws Exception {
        logln("--- Reading document info from " + file);

        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<int[]> labelIndexList = new ArrayList<int[]>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            int[] labelIndices = new int[sline.length - 1];
            for (int ii = 0; ii < sline.length - 1; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 1]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.labels = new int[labelIndexList.size()][];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
        }
    }

    public void outputArffFile(File filepath) {
        if (verbose) {
            logln("Outputing to " + filepath);
        }

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (String word : wordVocab) {
            attributes.add(new Attribute("voc_" + word));
        }
        for (String label : labelVocab) {
            ArrayList<String> attVals = new ArrayList<String>();
            attVals.add("0");
            attVals.add("1");
            attributes.add(new Attribute("label_" + label, attVals));
        }

        Instances data = new Instances(name, attributes, 0);
        for (int dd = 0; dd < docIds.length; dd++) {
            double[] vals = new double[wordVocab.size() + labelVocab.size()];

            // words
            SparseCount count = new SparseCount();
            for (int w : words[dd]) {
                count.increment(w);
            }
            for (int idx : count.getIndices()) {
                vals[idx] = count.getCount(idx);
            }
            for (int ll : labels[dd]) {
                vals[ll + wordVocab.size()] = 1;
            }

            data.add(new SparseInstance(1.0, vals));
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            writer.write(data.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing ARFF file");
        }
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputLabelVocab(new File(fFolder, formatFilename + labelVocabExt));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading formatted data "
                    + "from " + fFolder);
        }
    }

    protected void inputLabelVocab(File file) throws Exception {
        labelVocab = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        while ((line = reader.readLine()) != null) {
            labelVocab.add(line);
        }
        reader.close();
    }

    /**
     * Load train/development/test data in a cross validation fold.
     *
     * @param fold The given fold
     * @return
     * @throws java.lang.Exception
     */
    public static LabelTextDataset[] loadCrossValidationFold(Fold fold) throws Exception {
        LabelTextDataset[] foldData = new LabelTextDataset[3];
        LabelTextDataset trainData = new LabelTextDataset(fold.getFoldName(), fold.getFolder());
        trainData.setFormatFilename(fold.getFoldName() + Fold.TrainingExt);
        trainData.loadFormattedData(fold.getFoldFolderPath());
        foldData[Fold.TRAIN] = trainData;

        if (new File(fold.getFoldFolderPath(), fold.getFoldName() + Fold.DevelopExt + wordVocabExt).exists()) {
            LabelTextDataset devData = new LabelTextDataset(fold.getFoldName(), fold.getFolder());
            devData.setFormatFilename(fold.getFoldName() + Fold.DevelopExt);
            devData.loadFormattedData(fold.getFoldFolderPath());
            foldData[Fold.DEV] = devData;
        }

        if (new File(fold.getFoldFolderPath(), fold.getFoldName() + Fold.TestExt + wordVocabExt).exists()) {
            LabelTextDataset testData = new LabelTextDataset(fold.getFoldName(), fold.getFolder());
            testData.setFormatFilename(fold.getFoldName() + Fold.TestExt);
            testData.loadFormattedData(fold.getFoldFolderPath());
            foldData[Fold.TEST] = testData;
        }

        return foldData;
    }

    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + LabelTextDataset.class.getName() + " -help";
    }

    public static void main(String[] args) {
        try {
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addDataDirectoryOptions();
            addOption("label-file", "Directory of the label file");
            addOption("min-label-freq", "Minimum label frequency");

            // text processing
            addCorpusProcessorOptions();

            // cross validation
            addCrossValidationOptions();

            addOption("run-mode", "Run mode");
            options.addOption("v", false, "Verbose");
            options.addOption("d", false, "Debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            String runMode = cmd.getOptionValue("run-mode");
            switch (runMode) {
                case "process":
                    process();
                    break;
                case "load":
                    load();
                    break;
                case "cross-validate":
                    crossValidate();
                    break;
                default:
                    throw new RuntimeException("Run mode " + runMode + " is not supported");
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            throw new RuntimeException();
        }
    }

    private static void crossValidate() throws Exception {
        CorpusProcessor corpProc = createCorpusProcessor();
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String labelFile = cmd.getOptionValue("label-file");

        int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        double trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
        String cvFolder = cmd.getOptionValue("cv-folder");
        IOUtils.createFolder(cvFolder);

        LabelTextDataset dataset = new LabelTextDataset(datasetName, datasetFolder,
                corpProc);
        dataset.setFormatFilename(formatFile);

        // load text data
        if (cmd.hasOption("file")) {
            dataset.loadTextDataFromFile(textInputData);
        } else {
            dataset.loadTextDataFromFolder(textInputData);
        }
        dataset.loadLabels(labelFile); // load response data
        dataset.createCrossValidation(cvFolder, numFolds, trToDevRatio);
    }

    private static void process() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String labelFile = cmd.getOptionValue("label-file");

        CorpusProcessor corpProc = createCorpusProcessor();
        LabelTextDataset dataset = new LabelTextDataset(datasetName, datasetFolder,
                corpProc);
        dataset.setFormatFilename(formatFile);

        // load text data
        File textPath = new File(textInputData);
        if (textPath.isFile()) {
            dataset.loadTextDataFromFile(textInputData);
        } else if (textPath.isDirectory()) {
            dataset.loadTextDataFromFolder(textInputData);
        } else {
            throw new RuntimeException(textInputData + " is neither a file nor a folder");
        }
        dataset.loadLabels(labelFile); // load response data
        dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder));
    }

    private static LabelTextDataset load() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        LabelTextDataset data = new LabelTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        return data;
    }
}
