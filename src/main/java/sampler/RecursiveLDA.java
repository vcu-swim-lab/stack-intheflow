package sampler;

import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import data.TextDataset;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampling.likelihood.DirMult;
import util.CLIUtils;
import util.IOUtils;
import util.MiscUtils;

/**
 * Implementation of a hierarchy of LDAs.
 *
 * @author vietan
 */
public class RecursiveLDA extends AbstractSampler {

    public static final int BACKGROUND = 0;
    private double[] alphas;
    private double[] betas;
    private int L; // number of levels
    private int[] Ks; // number of children per node at each level
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected double ratio = 1.0;
    protected int[][] words;
    protected int[][][] zs;
    private int numTokens;
    private RLDA rootLDA;

    public RLDA getRoot() {
        return this.rootLDA;
    }

    public int getNumLevels() {
        return this.L;
    }

    public void configure(String folder,
            int[][] words,
            int V, int[] Ks,
            double ratio,
            double[] alphas,
            double[] betas,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.words = words;

        this.Ks = Ks;
        this.V = V;
        this.D = this.words.length;
        this.L = this.Ks.length;

        this.ratio = ratio;

        this.alphas = alphas;
        this.betas = betas;

        this.hyperparams = new ArrayList<Double>();
        for (double alpha : alphas) {
            this.hyperparams.add(alpha);
        }
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.numTokens = 0;
        for (int d = 0; d < D; d++) {
            this.numTokens += words[d].length;
        }

        if (alphas.length != L) {
            throw new RuntimeException("Dimensions mismatch. "
                    + alphas.length + " vs. " + L);
        }
        if (betas.length != L) {
            throw new RuntimeException("Dimensions mismatch. "
                    + betas.length + " vs. " + L);
        }

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- # documents:\t" + D);
            logln("--- # topics:\t" + MiscUtils.arrayToString(Ks));
            logln("--- # tokens:\t" + numTokens);
            logln("--- vocab size:\t" + V);
            logln("--- ratio:\t" + ratio);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_RecursiveLDA"
                + "_K";
        for (int K : Ks) {
            this.name += "-" + K;
        }
        this.name += "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a";
        for (double alpha : alphas) {
            this.name += "-" + alpha;
        }
        this.name += "_b";
        for (double beta : betas) {
            this.name += "-" + beta;
        }
        this.name += "-r-" + this.ratio;
        this.name += "_opt-" + this.paramOptimized;
    }

    public boolean hasBackground() {
        return this.ratio != 1.0;
    }

    public RLDA[] getAssignedPath(int d, int n) {
        RLDA[] path = new RLDA[L - 1];
        RLDA parent = rootLDA;
        for (int l = 0; l < L - 1; l++) {
            path[l] = parent.getChildren().get(zs[l][d][n]);
            parent = path[l];
        }
        return path;
    }

    public RLDA getAssignedLeaf(int d, int n) {
        RLDA[] path = getAssignedPath(d, n);
        return path[path.length - 1];
    }

    public int[][][] getAssingments() {
        return this.zs;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        zs = new int[L][][];
        for (int l = 0; l < L; l++) {
            zs[l] = new int[D][];
            for (int d = 0; d < D; d++) {
                zs[l][d] = new int[words[d].length];
            }
        }

        boolean[][] valid = new boolean[D][];
        for (int d = 0; d < D; d++) {
            valid[d] = new boolean[words[d].length];
            Arrays.fill(valid[d], true);
        }
        rootLDA = new RLDA(0, 0, valid, null);

        if (debug) {
            validate("Initialized");
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }

        recursive(0, 0, rootLDA, null);
    }

    public void iterate(int[][] seededZs) {
        recursive(0, 0, rootLDA, seededZs);
    }

    private void recursive(int index, int level, RLDA rlda, int[][] seededZs) {
        if (verbose) {
            System.out.println();
            logln("Sampling LDA " + rlda.getPathString());
        }

        rlda.setVerbose(verbose);
        rlda.setDebug(debug);
        rlda.setLog(false);
        rlda.setReport(false);
        rlda.configure(null, words, V, Ks[level],
                alphas[level], betas[level],
                initState,
                paramOptimized,
                BURN_IN, MAX_ITER, LAG, REP_INTERVAL);
        if (hasBackground() && level == 0) {
            double prob = 1.0 / (Ks[level] - 1 + ratio);
            double[][] docTopicPrior = new double[D][Ks[level]];
            for (int d = 0; d < D; d++) {
                docTopicPrior[d][BACKGROUND] = ratio * prob;
                for (int k = 1; k < Ks[level]; k++) {
                    docTopicPrior[d][k] = prob;
                }
            }
            rlda.initialize(docTopicPrior, null);
        } else {
            rlda.initialize();
        }

        if (seededZs != null && level == 0) { // use seeded assignments for the 1st level LDA
            if (verbose) {
                logln("--- Using seeded assgnments ...");
            }
            for (int d = 0; d < D; d++) {
                System.arraycopy(seededZs[d], 0, zs[level][d], 0, words[d].length);
            }
        } else {
            rlda.iterate();
            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    if (rlda.getValid()[d][n]) {
                        zs[level][d][n] = rlda.z[d][n];
                    }
                }
            }

            // debug
//            try {
//                if (level == 0) {
//                    IOUtils.createFolder(getSamplerFolderPath());
//                    BufferedWriter writer = IOUtils.getBufferedWriter(new File(this.getSamplerFolderPath(), "asgn.txt"));
//                    writer.write(D + "\n");
//                    for (int d = 0; d < D; d++) {
//                        for (int n = 0; n < words[d].length; n++) {
//                            writer.write(rlda.z[d][n] + " ");
//                        }
//                        writer.write("\n");
//                    }
//                    writer.close();
//                }
//            } catch (Exception e) {
//                e.printStackTrace();
//                System.exit(1);
//            }
        }

        if (level++ == L - 1) {
            return;
        }

        for (int k = 0; k < Ks[level - 1]; k++) {
            // don't split the background topic
            if (level == 1 && hasBackground() && k == BACKGROUND) {
                continue;
            }

            boolean[][] subValid = new boolean[D][];
            for (int d = 0; d < D; d++) {
                subValid[d] = new boolean[words[d].length];
                Arrays.fill(subValid[d], false);
                for (int n = 0; n < words[d].length; n++) {
                    if (!rlda.getValid()[d][n]) {
                        continue;
                    }
                    if (level == 1 && hasBackground() && rlda.z[d][n] == BACKGROUND) {
                        continue;
                    }
                    if (rlda.z[d][n] == k) {
                        subValid[d][n] = true;
                    }
                }
            }

            RLDA subRLda = new RLDA(k, level, subValid, rlda);
            rlda.addChild(subRLda);
            recursive(index, level, subRLda, seededZs);
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        return 0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
    }

    @Override
    public void validate(String msg) {
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            StringBuilder modelStr = new StringBuilder();
            Stack<RLDA> stack = new Stack<RLDA>();
            stack.add(rootLDA);
            while (!stack.isEmpty()) {
                RLDA node = stack.pop();
                for (RLDA child : node.getChildren()) {
                    stack.add(child);
                }
                modelStr.append(node.getPathString()).append("\n");
                for (int k = 0; k < Ks[node.getLevel()]; k++) {
                    modelStr.append(DirMult.output(node.topic_words[k])).append("\n");
                }
            }

            StringBuilder assignStr = new StringBuilder();
            for (int l = 0; l < L; l++) {
                assignStr.append(l).append("\n");
                for (int d = 0; d < D; d++) {
                    assignStr.append(d).append("\n");
                    for (int n = 0; n < words[d].length; n++) {
                        assignStr.append(zs[l][d][n]).append("\t");
                    }
                    assignStr.append("\n");
                }
            }

            // output to a compressed file
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        inputAssignments(filepath);

        boolean[][] valid = new boolean[D][];
        for (int d = 0; d < D; d++) {
            valid[d] = new boolean[words[d].length];
            Arrays.fill(valid[d], true);
        }
        this.rootLDA = new RLDA(0, 0, valid, null);
        Stack<RLDA> stack = new Stack<RLDA>();
        stack.add(rootLDA);

        while (!stack.isEmpty()) {
            RLDA node = stack.pop();
            node.setVerbose(false);
            int level = node.getLevel();
            node.configure(null, words, V, Ks[level],
                    alphas[level], betas[level],
                    initState,
                    paramOptimized,
                    BURN_IN, MAX_ITER, LAG, REP_INTERVAL);
            if (hasBackground() && level == 0) {
                double prob = 1.0 / (Ks[level] - 1 + ratio);
                double[][] docTopicPrior = new double[D][Ks[level]];
                for (int d = 0; d < D; d++) {
                    docTopicPrior[d][BACKGROUND] = ratio * prob;
                    for (int k = 1; k < Ks[level]; k++) {
                        docTopicPrior[d][k] = prob;
                    }
                }
                node.initializeModelStructure(null);
                node.initializeDataStructure(docTopicPrior);
            } else {
                node.initializeModelStructure(null);
                node.initializeDataStructure(null);
            }

            if (level == L - 1) {
                continue;
            }
            for (int k = 0; k < Ks[level]; k++) {
                RLDA child = new RLDA(k, level + 1, null, node);
                node.addChild(child);
                stack.add(child);
            }
        }

        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                int[] path = new int[L];
                for (int l = 0; l < L; l++) {
                    path[l] = zs[l][d][n];
                }
                assign(rootLDA, path, words[d][n]);
            }
        }

        if (debug) {
            validate("Input from " + filepath);
        }
    }

    private void assign(RLDA node, int[] path, int obs) {
        int level = node.getLevel();
        node.topic_words[path[level]].increment(obs);

        if (level < L - 1) {
            assign(node.getChildren().get(path[level]), path, obs);
        }
    }

    void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath + "\n");
        }

        zs = new int[L][][];
        for (int l = 0; l < L; l++) {
            zs[l] = new int[D][];
            for (int d = 0; d < D; d++) {
                zs[l][d] = new int[words[d].length];
            }
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            String line;
            String[] sline;
            while ((line = reader.readLine()) != null) {
                int l = Integer.parseInt(line);
                for (int d = 0; d < D; d++) {
                    int docIdx = Integer.parseInt(reader.readLine());
                    if (docIdx != d) {
                        throw new RuntimeException("Mismatch. " + d + " vs. " + docIdx);
                    }

                    // if this document is empty
                    line = reader.readLine().trim();
                    if (line.isEmpty()) {
                        zs[l][d] = new int[0];
                    } else {
                        sline = line.split("\t");
                        if (sline.length != words[d].length) {
                            throw new RuntimeException("Mismatch. "
                                    + sline.length
                                    + " vs. " + words[d].length
                                    + ". in document " + d);
                        }
                        for (int n = 0; n < words[d].length; n++) {
                            zs[l][d][n] = Integer.parseInt(sline[n]);
                        }
                    }
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading assignments from "
                    + zipFilepath);
        }
    }

    void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath + "\n");
        }

        try {
            // initialize
            rootLDA = new RLDA(0, 0, null, null);

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);

            String line;
            HashMap<String, RLDA> nodeMap = new HashMap<String, RLDA>();
            while ((line = reader.readLine()) != null) {
                String pathStr = line;

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                RLDA parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }

                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;
                RLDA node = new RLDA(nodeIndex, nodeLevel, null, parent);

                DirMult[] topics = new DirMult[Ks[nodeLevel]];
                for (int k = 0; k < Ks[nodeLevel]; k++) {
                    topics[k] = DirMult.input(reader.readLine());
                }
                node.topic_words = topics;

                if (node.getLevel() == 0) {
                    rootLDA = node;
                }

                if (parent != null) {
                    parent.getChildren().add(node.getIndex(), node);
                }

                nodeMap.put(pathStr, node);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing topics to file " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            if (hasBackground()) {
                double[] bgTopic = rootLDA.getTopics()[BACKGROUND].getDistribution();
                String[] bgWords = getTopWords(bgTopic, numTopWords);
                writer.write("[Background: " + rootLDA.getTopics()[BACKGROUND].getCountSum() + "]");
                for (String tw : bgWords) {
                    writer.write(" " + tw);
                }
                writer.write("\n");
            }

            Stack<RLDA> stack = new Stack<RLDA>();
            stack.add(rootLDA);

            while (!stack.isEmpty()) {
                RLDA node = stack.pop();
                for (RLDA child : node.getChildren()) {
                    stack.add(child);
                }

                int level = node.getLevel();
                if (node.getParent() != null) {
                    double[] parentTopics = node.getParent().getTopics()[node.getIndex()].getDistribution();
                    String[] parentTopWords = getTopWords(parentTopics, numTopWords);
                    for (int l = 0; l < level; l++) {
                        writer.write("\t");
                    }
                    writer.write("[" + node.getPathString()
                            + ": " + node.getParent().getTopics()[node.getIndex()].getCountSum() + "]");
                    for (String tw : parentTopWords) {
                        writer.write(" " + tw);
                    }
                    writer.write("\n");
                }

                if (node.getChildren().isEmpty()) {
                    DirMult[] topics = node.getTopics();
                    for (int k = 0; k < topics.length; k++) {
                        String[] topWords = getTopWords(topics[k].getDistribution(), numTopWords);
                        for (int l = 0; l < level + 1; l++) {
                            writer.write("\t");
                        }
                        writer.write("[" + node.getPathString() + ":" + k
                                + ":" + topics[k].getCountSum() + "]");
                        for (String tw : topWords) {
                            writer.write(" " + tw);
                        }
                        writer.write("\n");
                    }
                }
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            System.exit(1);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/segan.jar " + RecursiveLDA.class.getName() + " -help";
    }

    public static void run(String[] args) throws Exception {
        // create the command line parser
        parser = new BasicParser();

        // create the Options
        options = new Options();

        // directories
        addOption("dataset", "Dataset");
        addOption("output", "Output folder");
        addOption("data-folder", "Processed data folder");
        addOption("format-folder", "Folder holding formatted data");
        addOption("format-file", "Format file name");

        // sampling configurations
        addOption("burnIn", "Burn-in");
        addOption("maxIter", "Maximum number of iterations");
        addOption("sampleLag", "Sample lag");
        addOption("report", "Report interval");

        // model parameters
        addOption("K", "Number of topics");
        addOption("numTopwords", "Number of top words per topic");

        // model hyperparameters
        addOption("alpha", "Hyperparameter of the symmetric Dirichlet prior "
                + "for topic distributions");
        addOption("beta", "Hyperparameter of the symmetric Dirichlet prior "
                + "for word distributions");

        options.addOption("paramOpt", false, "Whether hyperparameter "
                + "optimization using slice sampling is performed");
        options.addOption("v", false, "verbose");
        options.addOption("d", false, "debug");
        options.addOption("help", false, "Help");

        cmd = parser.parse(options, args);
        if (cmd.hasOption("help")) {
            CLIUtils.printHelp("java -cp dist/segan.jar main.RunLDA -help", options);
            return;
        }

        // data 
        String datasetName = CLIUtils.getStringArgument(cmd, "dataset", "amazon-data");
        String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "demo");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
        String outputFolder = CLIUtils.getStringArgument(cmd, "output", "demo/"
                + datasetName + "/" + formatFolder + "-model");
        int numTopWords = CLIUtils.getIntegerArgument(cmd, "numTopwords", 20);
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        // sampler
        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 5);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 10);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        boolean paramOpt = cmd.hasOption("paramOpt");
        boolean verbose = cmd.hasOption("v");
        boolean debug = cmd.hasOption("d");

        if (verbose) {
            System.out.println("Loading data ...");
        }
        TextDataset dataset = new TextDataset(datasetName, datasetFolder);
        dataset.setFormatFilename(formatFile);
        dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder));
        dataset.prepareTopicCoherence(numTopWords);

        int V = dataset.getWordVocab().size();
        InitialState initState = InitialState.RANDOM;

        if (verbose) {
            System.out.println("Running Recursive LDA ...");
        }
        RecursiveLDA sampler = new RecursiveLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setWordVocab(dataset.getWordVocab());

        int[] Ks = {10, 3};
        double[] alphas = {0.1, 0.1};
        double[] betas = {0.1, 0.1};
        double ratio = 1000;

        sampler.configure(outputFolder, dataset.getWords(),
                V, Ks, ratio, alphas, betas, initState, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);

        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
//        sampler.sample();
//        sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
//        sampler.outputState(new File(samplerFolder, "model.zip"));
        sampler.inputState(new File(samplerFolder, "model.zip"));
        sampler.outputState(new File(samplerFolder, "model1.zip"));
    }
}
