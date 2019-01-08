package sampler.supervised.multiscale;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizer;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import sampler.LDA;
import sampler.supervised.objective.GaussianIndLinearRegObjective;
import sampling.likelihood.DirMult;
import sampling.util.Restaurant;
import sampling.util.SparseCount;
import sampling.util.FullTable;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.MimnoTopicCoherence;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class MSHDPSampler extends AbstractSampler {

    public static final int PSEUDO_INDEX = -1;
    public static final int ALPHA_GLOBAL = 0;
    public static final int ALPHA_LOCAL = 1;
    public static final int BETA = 2;
    public static final int MU = 3;
    public static final int SIGMA_GLOBAL = 4;
    public static final int SIGMA_LOCAL = 5;
    public static final int RHO = 6;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K; // initial number of tables
    protected int[][][] words;  // [D] x [Td] x [Ndt]: words
    protected double[][] responses; // [D] x [Td]
    protected int[][][] z; // local table index
    protected SparseCount[][] turnCounts;
    private Restaurant<SHDPDish, SHDPTable, DirMult> globalRestaurant;
    private Restaurant<SHDPTable, String, SHDPDish>[] localRestaurants;
    private GaussianIndLinearRegObjective optimizable;
    private Optimizer optimizer;
    private int totalNumObservations = 0;
    private double[] uniform;
    private DirMult emptyDirMultModel;
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;
    private int numConverged;

    public void configure(String folder,
            int[][][] words, double[][] responses,
            int V,
            double alpha_global, double alpha_local, double beta,
            double mu, double sigma_global, double sigma_local, double rho,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;
        this.responses = responses;

        this.V = V;
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_global);
        this.hyperparams.add(alpha_local);
        this.hyperparams.add(beta);
        this.hyperparams.add(mu);
        this.hyperparams.add(sigma_global);
        this.hyperparams.add(sigma_local);
        this.hyperparams.add(rho);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.setName();

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        int numTs = 0;
        for (int d = 0; d < D; d++) {
            numTs += words[d].length;
            for (int t = 0; t < words[d].length; t++) {
                totalNumObservations += this.words[d][t].length;
            }
        }
        logln("--- D = " + D);
        logln("--- V = " + V);
        logln("--- Total Ts = " + numTs);
        logln("--- # observations = " + totalNumObservations);

        if (!debug) {
            System.err.close();
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_M-SHDP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_ag-").append(formatter.format(hyperparams.get(ALPHA_GLOBAL)))
                .append("_al-").append(formatter.format(hyperparams.get(ALPHA_LOCAL)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_m-").append(formatter.format(hyperparams.get(MU)))
                .append("_sg-").append(formatter.format(hyperparams.get(SIGMA_GLOBAL)))
                .append("_sl-").append(formatter.format(hyperparams.get(SIGMA_LOCAL)))
                .append("_r-").append(formatter.format(hyperparams.get(RHO)));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    public void setK(int K) {
        this.K = K;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;

        initializeModelStructure();

        initializeDataStructure();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- --- Done initializing. \n" + getCurrentState());
        }
    }

    protected void initializeModelStructure() {
        this.globalRestaurant = new Restaurant<SHDPDish, SHDPTable, DirMult>();

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<SHDPTable, String, SHDPDish>();
        }

        emptyDirMultModel = new DirMult(V, hyperparams.get(BETA), uniform);
    }

    protected void initializeDataStructure() {
        z = new int[D][][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length][];
            for (int t = 0; t < words[d].length; t++) {
                z[d][t] = new int[words[d][t].length];
            }
        }

        turnCounts = new SparseCount[D][];
        for (int d = 0; d < D; d++) {
            turnCounts[d] = new SparseCount[words[d].length];
            for (int t = 0; t < words[d].length; t++) {
                turnCounts[d][t] = new SparseCount();
            }
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case PRESET:
                this.initializePresetAssignments();
                break;

            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    private void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        if (K == 0) // this is not set
        {
            K = 50;
        }
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        int[][] flattenWords = new int[D][];
        for (int d = 0; d < D; d++) {
            int numDocTokens = 0;
            for (int t = 0; t < words[d].length; t++) {
                numDocTokens += words[d][t].length;
            }

            int count = 0;
            flattenWords[d] = new int[numDocTokens];
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    flattenWords[d][count++] = words[d][t][n];
                }
            }
        }

        lda.configure(null, flattenWords, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            String ldaFile = this.folder + "lda-init-" + K + ".txt";
            File ldaZFile = new File(ldaFile);
            if (ldaZFile.exists()) {
                ldaZ = inputLDAInitialization(ldaFile);
            } else {
                lda.sample();
                ldaZ = lda.getZ();
                outputLDAInitialization(ldaFile, ldaZ);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(this.folder, "lda-topwords.txt"), 15);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        setLog(true);

        // initialize assignments
        for (int d = 0; d < D; d++) {
            // create tables
            for (int k = 0; k < K; k++) {
                SHDPTable table = new SHDPTable(k, null, d, Double.NaN);
                this.localRestaurants[d].addTable(table);
            }

            int count = 0;
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    z[d][t][n] = ldaZ[d][count++];
                    localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
                    turnCounts[d][t].increment(z[d][t][n]);
                }
            }

            // assign tables with global nodes
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }
                this.sampleDishForTable(d, table.getIndex(), !REMOVE, !OBSERVED);
            }

            // remove empty table
            for (int tIndex : emptyTables) {
                this.localRestaurants[d].removeTable(tIndex);
            }
        }

        // debug
        logln("After assignment initialization\n"
                + getCurrentState() + "\n");

        // optimize
        for (int d = 0; d < D; d++) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                double mean = SamplerUtils.getGaussian(table.getContent().getMean(), hyperparams.get(SIGMA_LOCAL));
                table.setEta(mean);
            }

            optimize(d);
        }

        // update parameters of dishes in the global restaurant
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            updateDishParameters(dish);
        }
    }

    private int[][] inputLDAInitialization(String filepath) {
        if (verbose) {
            logln("--- --- LDA init file found. Loading from " + filepath);
        }

        int[][] ldaZ = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            int numDocs = Integer.parseInt(reader.readLine());
            ldaZ = new int[numDocs][];
            for (int d = 0; d < numDocs; d++) {
                String[] sline = reader.readLine().split("\t")[1].split(" ");
                ldaZ[d] = new int[sline.length];
                for (int n = 0; n < ldaZ[d].length; n++) {
                    ldaZ[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return ldaZ;
    }

    private void outputLDAInitialization(String filepath, int[][] z) {
        if (verbose) {
            logln("--- --- Outputing LDA init state to file " + filepath);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
            writer.write(z.length + "\n");
            for (int d = 0; d < z.length; d++) {
                writer.write(z[d].length + "\t");
                for (int n = 0; n < z[d].length; n++) {
                    writer.write(z[d][n] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();

        try {
            if (report) {
                IOUtils.createFolder(this.folder + this.getSamplerFolder() + ReportFolder);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens: " + numTokenAssignmentsChange
                            + "\t # tables: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                } else {
                    logln("--- Sampling. Iter " + iter
                            + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                            + "\t # tokens change: " + numTokenAssignmentsChange
                            + "\t # tables change: " + numTableAssignmentsChange
                            + "\t # converge: " + numConverged
                            + "\n" + getCurrentState());
                }
            }

            numTableAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;
            numConverged = 0;

            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        this.sampleTableForToken(d, t, n, REMOVE, OBSERVED, ADD, EXTEND);
                    }
                }

                for (SHDPTable table : this.localRestaurants[d].getTables()) {
                    this.sampleDishForTable(d, table.getIndex(), REMOVE, OBSERVED);
                }

                // optimize regression parameters of local restaurants
                optimize(d);
            }

            // update parameters of dishes in the global restaurant
            for (SHDPDish dish : this.globalRestaurant.getTables()) {
                updateDishParameters(dish);
            }

            if (verbose) {
                double[][] trPredResponses = getRegressionValues();
                RegressionEvaluation eval = new RegressionEvaluation(
                        MiscUtils.flatten2DArray(responses),
                        MiscUtils.flatten2DArray(trPredResponses));
                eval.computeCorrelationCoefficient();
                eval.computeMeanSquareError();
                eval.computeRSquared();
                ArrayList<Measurement> measurements = eval.getMeasurements();
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
            }

            if (iter >= BURN_IN) {
                if (paramOptimized) {
                    if (verbose) {
                        logln("--- --- Slice sampling ...");
                    }

                    sliceSample();
                    this.sampledParams.add(this.cloneHyperparameters());

                    if (verbose) {
                        logln("--- ---- " + MiscUtils.listToString(hyperparams));
                    }
                }
            }

            if (debug) {
                this.validate("Iteration " + iter);
            }
            if (verbose) {
                System.out.println();
            }

            // store model
            if (report && iter >= BURN_IN && iter % LAG == 0) {
                outputState(this.folder + this.getSamplerFolder() + ReportFolder + "iter-" + iter + ".zip");
            }
        }

        outputState(this.folder + this.getSamplerFolder() + "final.zip");

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }

        try {
            if (paramOptimized && log) {
                this.outputSampledHyperparameters(this.folder + this.getSamplerFolder() + "hyperparameters.txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Create a brand new dish
     */
    private SHDPDish createDish() {
        int newDishIndex = globalRestaurant.getNextTableIndex();
        DirMult dm = new DirMult(V, hyperparams.get(BETA), uniform);
        double dishEta = SamplerUtils.getGaussian(hyperparams.get(MU), hyperparams.get(SIGMA_GLOBAL));
        SHDPDish newDish = new SHDPDish(newDishIndex, dm, dishEta);
        globalRestaurant.addTable(newDish);
        return newDish;
    }

    /**
     * Remove a customer from a table
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param n The customer
     */
    private void removeCustomerFromTable(int d, int tableIndex, int t, int n) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);
        SHDPDish dish = table.getContent();

        this.localRestaurants[d].removeCustomerFromTable(getTokenId(t, n), tableIndex);
        dish.getContent().decrement(words[d][t][n]);

        if (table.isEmpty()) {
            removeTableFromDish(d, tableIndex, null);
            this.localRestaurants[d].removeTable(tableIndex);
        }
    }

    /**
     * Remove a table from a dish
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     */
    private void removeTableFromDish(int d, int tableIndex, HashMap<Integer, Integer> observations) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);
        SHDPDish dish = table.getContent();

        // remove observations from dish
        if (observations != null) {
            removeObservations(dish, observations);
        }

        // remove table from dish
        this.globalRestaurant.removeCustomerFromTable(table, dish.getIndex());

        // if the dish is empty, remove it
        if (dish.isEmpty()) {
            this.globalRestaurant.removeTable(dish.getIndex());
        }
    }

    private void removeObservations(SHDPDish dish, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            dish.getContent().changeCount(obs, -observations.get(obs));
        }
    }

    private void addObservations(SHDPDish dish, HashMap<Integer, Integer> observations) {
        for (int obs : observations.keySet()) {
            dish.getContent().changeCount(obs, observations.get(obs));
        }
    }

    /**
     * Sample a local table for a token
     *
     * @param d The group index
     * @param t The document index
     * @param n The token index
     * @param remove Whether the current assignment should be removed
     * @param resObserved Whether the response variable is observed
     * @param add Whether the new assignment should be add to the structure
     */
    private void sampleTableForToken(int d, int t, int n,
            boolean remove, boolean resObserved, boolean add, boolean extend) {
        int curObs = words[d][t][n];
        int curTableIndex = z[d][t][n];

        if (remove) {
            removeCustomerFromTable(d, curTableIndex, t, n);
            turnCounts[d][t].decrement(curTableIndex);
        }

        double weightedSum = 0.0;
        int numTokens = turnCounts[d][t].getCountSum() + 1;
        if (resObserved) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                weightedSum += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
            }
        }

//        logln("d = " + d + ". t = " + t + ". n = " + n 
//                + ". numTokens = " + numTokens
//                + ". sparseCount = " + turnCounts.getCountSum()
//                + ". true count = " + words[d][t].length
//                + ". weighted sum = " + weightedSum
//                );

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        // for existing tables
        for (SHDPTable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());
            double logprior = Math.log(table.getNumCustomers());
            double wordLlh = table.getContent().getContent().getLogLikelihood(curObs);
            double logprob = logprior + wordLlh;

            if (resObserved) {
                double mean = (weightedSum + table.getEta()) / numTokens;
                double resLlh = StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(hyperparams.get(RHO)));
                logprob += resLlh;

                // debug
//                logln((logprobs.size()-1)
//                        + "\t" + d + ":" + t + ":" + n + ":" + words[d][t][n]
//                        + "\t lp: " + MiscUtils.formatDouble(Math.log(table.getNumCustomers()))
//                        + "\t wllh: " + MiscUtils.formatDouble(table.getContent().getContent().getLogLikelihood(curObs))
//                        + "\t rllh: " + MiscUtils.formatDouble(resLlh)
//                        + "\t" + MiscUtils.formatDouble(logprob)
//                        + "\t" + table.toString()
//                        + "\t" + table.getContent().toString());
            }

            logprobs.add(logprob);
        }

        HashMap<Integer, Double> dishLogPriors = null;
        HashMap<Integer, Double> dishLogLikelihoods = null;
        HashMap<Integer, Double> dishResLogLikelihoods = null;
        if (extend) { // in test time, only use the learnt structure, don't create new table
            // for new tables
            tableIndices.add(PSEUDO_INDEX);
            dishLogPriors = getDishLogPriors();
            dishLogLikelihoods = getDishLogLikelihoods(curObs);

            if (dishLogPriors.size() != dishLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of dishes mismatch");
            }

            dishResLogLikelihoods = new HashMap<Integer, Double>();
            if (resObserved) {
                dishResLogLikelihoods = getDishResponseLogLikelihoodsNewTable(
                        responses[d][t], weightedSum, numTokens);

                if (dishLogPriors.size() != dishResLogLikelihoods.size()) {
                    throw new RuntimeException("Numbers of dishes mismatch");
                }
            }

            double marginalLogLikelihood = 0.0;
            for (int dishIndex : dishLogPriors.keySet()) {
                double lp = dishLogPriors.get(dishIndex) + dishLogLikelihoods.get(dishIndex);

                if (resObserved) {
                    lp += dishResLogLikelihoods.get(dishIndex);
                }

                if (marginalLogLikelihood == 0.0) {
                    marginalLogLikelihood = lp;
                } else {
                    marginalLogLikelihood = SamplerUtils.logAdd(marginalLogLikelihood, lp);
                }
            }

            double logprob = Math.log(hyperparams.get(ALPHA_LOCAL))
                    + marginalLogLikelihood;
            logprobs.add(logprob);

            // debug
//            logln((logprobs.size()-1)
//                    + "\t" + d + ":" + t + ":" + n + ":" + words[d][t][n]
//                    + "\t lp: " + MiscUtils.formatDouble(Math.log(hyperparams.get(ALPHA_LOCAL)))
//                    + "\t marginal: " + MiscUtils.formatDouble(marginalLogLikelihood)
//                    + "\t" + MiscUtils.formatDouble(logprob)
//                    );
        }
        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int tableIndex = tableIndices.get(sampledIndex);

        // debug
//        logln("---> index = " + sampledIndex + ". " + tableIndex + "\n\n");

        if (curTableIndex != tableIndex) {
            numTokenAssignmentsChange++;
        }

        SHDPTable table;
        if (tableIndex == PSEUDO_INDEX) {
            // sample dish
            int sampledDish = sampleDish(dishLogPriors, dishLogLikelihoods, dishResLogLikelihoods, !OBSERVED);
            SHDPDish dish;
            if (sampledDish == PSEUDO_INDEX) {
                dish = createDish();
            } else {
                dish = globalRestaurant.getTable(sampledDish);
            }

            // create a new table
            int newTableIndex = localRestaurants[d].getNextTableIndex();
            double tempTableRegParam = 0.0; // should this be sampled?
            table = new SHDPTable(newTableIndex, dish, d, tempTableRegParam);

            globalRestaurant.addCustomerToTable(table, dish.getIndex());
            localRestaurants[d].addTable(table);
        } else {
            table = this.localRestaurants[d].getTable(tableIndex);
        }

        // update
        z[d][t][n] = table.getIndex();
        turnCounts[d][t].increment(z[d][t][n]);

        if (add) {
            localRestaurants[d].addCustomerToTable(getTokenId(t, n), z[d][t][n]);
            table.getContent().getContent().increment(curObs);
        }
    }

    /**
     * Sample a dish for a table
     *
     * @param d The restaurant (group) index
     * @param tableIndex The table index
     * @param remove Whether the current assignment should be removed
     * @param resObserved Whether the response variable is observed
     */
    private void sampleDishForTable(int d, int tableIndex, boolean remove, boolean resObserved) {
        SHDPTable table = this.localRestaurants[d].getTable(tableIndex);

        // current observations assigned to this table
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        for (String c : table.getCustomers()) {
            int[] parsedId = parseTokenId(c);
            int type = words[d][parsedId[0]][parsedId[1]];
            Integer count = observations.get(type);
            if (count == null) {
                observations.put(type, 1);
            } else {
                observations.put(type, count + 1);
            }
        }

        // in case of the first table assignment
        if (globalRestaurant.isEmpty()) {
            SHDPDish dish = createDish();
            table.setContent(dish);
            addObservations(dish, observations);
            globalRestaurant.addCustomerToTable(table, dish.getIndex());
            return;
        }

        int curDish = PSEUDO_INDEX;
        if (table.getContent() != null) {
            curDish = table.getContent().getIndex();
        }

        if (remove) {
            removeTableFromDish(d, tableIndex, observations);
        }

        HashMap<Integer, Double> dishLogPriors = getDishLogPriors();
        HashMap<Integer, Double> dishLogLikelihoods = getDishLogLikelihoods(observations);

        if (dishLogPriors.size() != dishLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of dishes mismatch");
        }

        HashMap<Integer, Double> dishResLogLikelihoods = new HashMap<Integer, Double>();
        if (resObserved) {
            dishResLogLikelihoods = getDishResponseLogLikelihoodsExistingTable(table.getEta());

            if (dishLogPriors.size() != dishResLogLikelihoods.size()) {
                throw new RuntimeException("Numbers of dishes mismatch");
            }
        }

        int sampledDish = sampleDish(dishLogPriors, dishLogLikelihoods, dishResLogLikelihoods, resObserved);
        if (curDish != sampledDish) {
            numTableAssignmentsChange++;
        }

        SHDPDish dish;
        if (sampledDish == PSEUDO_INDEX) {
            dish = createDish();
        } else {
            dish = globalRestaurant.getTable(sampledDish);
        }

        // update
        table.setContent(dish);
        globalRestaurant.addCustomerToTable(table, dish.getIndex());
        addObservations(dish, observations);
    }

    /**
     * Sample a dish using precomputed probabilities
     *
     * @param dishLogPriors Log priors
     * @param dishLogLikelihoods Word log likelihoods
     * @param dishLogLikelihoods Response variable log likelihoods
     * @param resObserved Whether the response variable is observed
     */
    private int sampleDish(HashMap<Integer, Double> dishLogPriors,
            HashMap<Integer, Double> dishLogLikelihoods,
            HashMap<Integer, Double> dishResLogLikelihoods,
            boolean resObserved) {
        ArrayList<Integer> dishIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        for (int dishIndex : dishLogPriors.keySet()) {
            dishIndices.add(dishIndex);
            double logprob = dishLogPriors.get(dishIndex) + dishLogLikelihoods.get(dishIndex);

            if (resObserved) {
                logprob += dishResLogLikelihoods.get(dishIndex);
            }

            logprobs.add(logprob);
        }
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);

        if (sampledIndex == logprobs.size()) {
            for (int dishIndex : dishLogPriors.keySet()) {
                logln(dishIndex
                        + "\tlog prior: " + MiscUtils.formatDouble(dishLogPriors.get(dishIndex))
                        + "\tlog likelihood: " + MiscUtils.formatDouble(dishLogLikelihoods.get(dishIndex))
                        + "\ttotal = " + MiscUtils.formatDouble(dishLogPriors.get(dishIndex) + dishLogLikelihoods.get(dishIndex)));
            }
            throw new RuntimeException("Out-of-bound sampling");
        }

        return dishIndices.get(sampledIndex);
    }

    /**
     * Optimize the regression parameters at local restaurant's tables
     *
     * @param d The local restaurant index
     */
    private void optimize(int d) {
        int numTables = this.localRestaurants[d].getNumTables();

        double[] regParams = new double[numTables];
        double[] priorMeans = new double[numTables];
        double[] priorVars = new double[numTables];
        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        int count = 0;
        for (SHDPTable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());
            regParams[count] = table.getEta();
            priorMeans[count] = table.getContent().getMean();
            priorVars[count] = Math.sqrt(hyperparams.get(SIGMA_LOCAL));
            count++;
        }

        int Td = words[d].length;
        double[][] designMatrix = new double[Td][numTables];
        for (int t = 0; t < Td; t++) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                int idx = tableIndices.indexOf(table.getIndex());
                designMatrix[t][idx] = (double) turnCounts[d][t].getCount(table.getIndex())
                        / turnCounts[d][t].getCountSum();
            }
        }

        // debug
//        logln("Optimizing " + d + ". # instances: " + responses[d].length + ". # parameters: " + numTables);
//        double sse = 0.0;
//        for(int t=0; t<words[d].length; t++){
//            double dotProd = StatisticsUtils.dotProduct(designMatrix[t], regParams);
//            sse += Math.pow(dotProd - responses[d][t], 2);
//            logln("*** t = " + t 
//                    + ". dotprod = " + MiscUtils.formatDouble(dotProd)
//                    + ". response = " + MiscUtils.formatDouble(responses[d][t])
//                    + ". " + MiscUtils.formatDouble(dotProd));
//        }
//        logln(">>> Old SSE = " + sse);

        this.optimizable = new GaussianIndLinearRegObjective(
                regParams, designMatrix, responses[d],
                hyperparams.get(RHO), priorMeans, priorVars);
        this.optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            // This exception may be thrown if L-BFGS
            //  cannot step in the current direction.
            // This condition does not necessarily mean that
            //  the optimizer has failed, but it doesn't want
            //  to claim to have succeeded... 
            // do nothing
        }

        if (converged) // if the optimization converges
        {
            numConverged++;
        }

        // update regression parameters
        for (int i = 0; i < regParams.length; i++) {
            // debug
//            logln("--- " + tableIndices.get(i)
//                    + ". prior mean: " + MiscUtils.formatDouble(priorMeans[i])
//                    + ". prior std: " + MiscUtils.formatDouble(priorVars[i])
//                    + ". old eta: " + MiscUtils.formatDouble(regParams[i])
//                    + ". new eta: " + MiscUtils.formatDouble(optimizable.getParameter(i))
//                    );

            regParams[i] = optimizable.getParameter(i);
        }

        // debug
//        sse = 0.0;
//        for(int t=0; t<words[d].length; t++){
//            double dotProd = StatisticsUtils.dotProduct(designMatrix[t], regParams);
//            sse += Math.pow(dotProd - responses[d][t], 2);
//        }
//        logln(">>> New SSE = " + sse);
//        System.out.println("\n");

        // upadte tables' etas
        for (int i = 0; i < regParams.length; i++) {
            int tableIndex = tableIndices.get(i);
            this.localRestaurants[d].getTable(tableIndex).setEta(regParams[i]);
        }
    }

    private void updateDishParameters(SHDPDish dish) {
        double priorVar = hyperparams.get(SIGMA_GLOBAL);
        double priorMean = hyperparams.get(MU);

        double newPriorVar = priorVar * hyperparams.get(SIGMA_LOCAL)
                / (dish.getNumCustomers() * priorVar + hyperparams.get(SIGMA_LOCAL));
        double sumEtas = 0.0;
        for (SHDPTable table : dish.getCustomers()) {
            sumEtas += table.getEta();
        }
        double newPriorMean = newPriorVar * sumEtas / hyperparams.get(SIGMA_LOCAL)
                + newPriorVar * priorMean / priorVar;

        double newMean = SamplerUtils.getGaussian(newPriorMean, newPriorVar);
        dish.setMean(newMean);
    }

    private HashMap<Integer, Double> getDishResponseLogLikelihoodsExistingTable(double eta) {
        HashMap<Integer, Double> resLlhs = new HashMap<Integer, Double>();

        // for existing dishes
        for (SHDPDish dish : globalRestaurant.getTables()) {
            double mean = dish.getMean();
            double var = hyperparams.get(SIGMA_LOCAL);
            double resLlh = StatUtils.logNormalProbability(eta, mean, Math.sqrt(var));
            resLlhs.put(dish.getIndex(), resLlh);
        }

        // for new dish
        double mean = hyperparams.get(MU);
        double var = hyperparams.get(SIGMA_GLOBAL) + hyperparams.get(SIGMA_LOCAL);
        double resLlh = StatUtils.logNormalProbability(eta, mean, Math.sqrt(var));
        resLlhs.put(PSEUDO_INDEX, resLlh);

        return resLlhs;
    }

    private HashMap<Integer, Double> getDishResponseLogLikelihoodsNewTable(
            double response, double weightedSum, double tokenCount) {
        HashMap<Integer, Double> resLlhs = new HashMap<Integer, Double>();

        double tokenCountSquare = tokenCount * tokenCount;

        // for existing dishes
        for (SHDPDish dish : globalRestaurant.getTables()) {
            double mean = (weightedSum + dish.getMean()) / tokenCount;
            double var = hyperparams.get(SIGMA_LOCAL) / tokenCountSquare + hyperparams.get(RHO);
            double resLlh = StatUtils.logNormalProbability(response, mean, Math.sqrt(var));
            resLlhs.put(dish.getIndex(), resLlh);
        }

        // for new dish
        double mean = (weightedSum + hyperparams.get(MU)) / tokenCount;
        double var = (hyperparams.get(SIGMA_GLOBAL) + hyperparams.get(SIGMA_LOCAL)) / tokenCountSquare
                + hyperparams.get(RHO);
        double resLlh = StatUtils.logNormalProbability(response, mean, Math.sqrt(var));
        resLlhs.put(PSEUDO_INDEX, resLlh);

        return resLlhs;
    }

    private HashMap<Integer, Double> getDishLogPriors() {
        HashMap<Integer, Double> dishLogPriors = new HashMap<Integer, Double>();
        double normalizer = Math.log(this.globalRestaurant.getTotalNumCustomers()
                + hyperparams.get(ALPHA_GLOBAL));
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogPriors.put(dish.getIndex(), Math.log(dish.getNumCustomers()) - normalizer);
        }
        dishLogPriors.put(PSEUDO_INDEX, Math.log(hyperparams.get(ALPHA_GLOBAL)) - normalizer);
        return dishLogPriors;
    }

    private HashMap<Integer, Double> getDishLogLikelihoods(int observation) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observation));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyDirMultModel.getLogLikelihood(observation));
        return dishLogLikelihoods;
    }

    private HashMap<Integer, Double> getDishLogLikelihoods(HashMap<Integer, Integer> observations) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observations));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyDirMultModel.getLogLikelihood(observations));
        return dishLogLikelihoods;
    }

    private int[] parseTokenId(String tokenId) {
        String[] parse = tokenId.split("_");
        int[] parsedTokenId = new int[parse.length];
        for (int i = 0; i < parsedTokenId.length; i++) {
            parsedTokenId[i] = Integer.parseInt(parse[i]);
        }
        return parsedTokenId;
    }

    private String getTokenId(int turnIndex, int tokenIndex) {
        return turnIndex + "_" + tokenIndex;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(">>> >>> # dishes: ").append(globalRestaurant.getNumTables()).append("\n");

        int[] numTables = new int[D];
        for (int d = 0; d < D; d++) {
            numTables[d] = this.localRestaurants[d].getNumTables();
        }
        str.append(">>> >>> # tables: avg: ").append(StatUtils.mean(numTables))
                .append(". min: ").append(StatUtils.min(numTables))
                .append(". max: ").append(StatUtils.max(numTables));
        str.append("\n");
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double obsLlh = 0.0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood();
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(hyperparams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA_LOCAL));
        }

        double tableRegLlh = 0.0;
        for (int d = 0; d < D; d++) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                tableRegLlh += StatUtils.logNormalProbability(table.getEta(),
                        table.getContent().getMean(), Math.sqrt(hyperparams.get(SIGMA_LOCAL)));
            }
        }

        double dishRegLlh = 0.0;
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishRegLlh += StatUtils.logNormalProbability(dish.getMean(),
                    hyperparams.get(MU), Math.sqrt(hyperparams.get(SIGMA_GLOBAL)));
        }

        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < responses[d].length; t++) {
                double mean = 0.0;
                for (SHDPTable table : localRestaurants[d].getTables()) {
                    mean += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                mean /= words[d][t].length;
                resLlh += StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(hyperparams.get(RHO)));
            }
        }

        if (verbose) {
            logln("*** obs llh: " + MiscUtils.formatDouble(obsLlh)
                    + ". res llh: " + MiscUtils.formatDouble(resLlh)
                    + ". assignments: " + MiscUtils.formatDouble(assignLp)
                    + ". global reg: " + MiscUtils.formatDouble(dishRegLlh)
                    + ". local reg: " + MiscUtils.formatDouble(tableRegLlh));
        }

        return obsLlh + assignLp + tableRegLlh + dishRegLlh + resLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        double obsLlh = 0.0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood(tParams.get(BETA), uniform);
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(tParams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(tParams.get(ALPHA_LOCAL));
        }

        double tableRegLlh = 0.0;
        for (int d = 0; d < D; d++) {
            for (SHDPTable table : this.localRestaurants[d].getTables()) {
                tableRegLlh += StatUtils.logNormalProbability(table.getEta(),
                        table.getContent().getMean(), Math.sqrt(tParams.get(SIGMA_LOCAL)));
            }
        }

        double dishRegLlh = 0.0;
        for (SHDPDish dish : this.globalRestaurant.getTables()) {
            dishRegLlh += StatUtils.logNormalProbability(dish.getMean(), tParams.get(MU), Math.sqrt(tParams.get(SIGMA_GLOBAL)));
        }

        double resLlh = 0.0;
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < responses[d].length; t++) {
                double mean = 0.0;
                for (SHDPTable table : localRestaurants[d].getTables()) {
                    mean += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                mean /= words[d][t].length;
                resLlh += StatUtils.logNormalProbability(responses[d][t], mean, Math.sqrt(tParams.get(RHO)));
            }
        }

        return obsLlh + assignLp + tableRegLlh + dishRegLlh + resLlh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        for (SHDPDish dish : globalRestaurant.getTables()) {
            dish.getContent().setConcentration(newParams.get(BETA));
        }

        this.hyperparams = new ArrayList<Double>();
        for (double param : newParams) {
            this.hyperparams.add(param);
        }
    }

    @Override
    public void validate(String msg) {
        globalRestaurant.validate(msg);
        for (int d = 0; d < D; d++) {
            localRestaurants[d].validate(msg);
        }

        for (int d = 0; d < D; d++) {
            for (SHDPTable table : localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    throw new RuntimeException(msg + ". Empty table. " + table.toString());
                }
            }
        }

        for (SHDPDish dish : globalRestaurant.getTables()) {
            if (dish.isEmpty() || dish.getContent().getCountSum() == 0) {
                throw new RuntimeException(msg + ". Empty dish. " + dish.toString()
                        + ". tables: " + dish.getCustomers().toString());
            }
        }

        int totalObs = 0;
        for (SHDPDish dish : globalRestaurant.getTables()) {
            int dishNumObs = dish.getContent().getCountSum();
            int tableNumObs = 0;
            for (SHDPTable table : dish.getCustomers()) {
                tableNumObs += table.getNumCustomers();
            }

            if (dishNumObs != tableNumObs) {
                throw new RuntimeException(msg + ". Numbers of observations mismatch. "
                        + dishNumObs + " vs. " + tableNumObs);
            }

            totalObs += dishNumObs;
        }

        if (totalObs != totalNumObservations) {
            throw new RuntimeException(msg + ". Total numbers of observations mismatch. "
                    + totalObs + " vs. " + totalNumObservations);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        try {
            StringBuilder modelStr = new StringBuilder();

            // local restaurants
            for (int d = 0; d < D; d++) {
                modelStr.append(d)
                        .append("\t").append(localRestaurants[d].getNumTables())
                        .append("\n");
                for (SHDPTable table : localRestaurants[d].getTables()) {
                    modelStr.append(table.getIndex())
                            .append("\t").append(table.getEta());
                    for (String customer : table.getCustomers()) {
                        modelStr.append("\t").append(customer);
                    }
                    modelStr.append("\n");
                }
            }

            // global restaurants
            modelStr.append(globalRestaurant.getNumTables()).append("\n");
            for (SHDPDish dish : globalRestaurant.getTables()) {
                modelStr.append(dish.getIndex()).append("\n");
                modelStr.append(dish.getMean()).append("\n");
                modelStr.append(DirMult.output(dish.getContent())).append("\n");
                for (SHDPTable table : dish.getCustomers()) {
                    modelStr.append(table.getTableId()).append("\t");
                }
                modelStr.append("\n");
            }

            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        assignStr.append(d)
                                .append(":").append(t)
                                .append(":").append(n)
                                .append("\t").append(z[d][t][n])
                                .append("\n");
                    }
                }
            }

            // output to a compressed file
            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ZipOutputStream writer = IOUtils.getZipOutputStream(filepath);

            ZipEntry modelEntry = new ZipEntry(filename + ModelFileExt);
            writer.putNextEntry(modelEntry);
            byte[] data = modelStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            ZipEntry assignEntry = new ZipEntry(filename + AssignmentFileExt);
            writer.putNextEntry(assignEntry);
            data = assignStr.toString().getBytes();
            writer.write(data, 0, data.length);
            writer.closeEntry();

            writer.close();
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
        try {
            inputModel(filepath);

            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        validate("Done reading state from " + filepath);
    }

    /**
     * Load the model from a compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputModel(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        // initialize
        this.initializeModelStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + ModelFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));
        HashMap<String, SHDPTable> tableMap = new HashMap<String, SHDPTable>();

        // local restaurants
        for (int d = 0; d < D; d++) {
            int numTables = Integer.parseInt(reader.readLine().split("\t")[1]);
            for (int i = 0; i < numTables; i++) {
                String[] sline = reader.readLine().split("\t");
                int tableIndex = Integer.parseInt(sline[0]);
                double eta = Double.parseDouble(sline[1]);

                SHDPTable table = new SHDPTable(tableIndex, null, d, eta);
                this.localRestaurants[d].addTable(table);
                for (int j = 2; j < sline.length; j++) {
                    this.localRestaurants[d].addCustomerToTable(sline[j], tableIndex);
                }
                tableMap.put(table.getTableId(), table);
            }
        }

        // global restaurant
        int numTables = Integer.parseInt(reader.readLine());
        for (int i = 0; i < numTables; i++) {
            int dishIndex = Integer.parseInt(reader.readLine());
            double dishMean = Double.parseDouble(reader.readLine());
            DirMult dmm = DirMult.input(reader.readLine());
            SHDPDish dish = new SHDPDish(dishIndex, dmm, dishMean);
            globalRestaurant.addTable(dish);

            String[] sline = reader.readLine().split("\t");
            for (int j = 0; j < sline.length; j++) {
                SHDPTable table = tableMap.get(sline[j]);
                globalRestaurant.addCustomerToTable(table, dishIndex);
                table.setContent(dish);
            }
        }

        // update inactive tables
        globalRestaurant.fillInactiveTableIndices();
        for (int d = 0; d < D; d++) {
            localRestaurants[d].fillInactiveTableIndices();
        }

        reader.close();
    }

    /**
     * Load the assignments of the training data from the compressed state file
     *
     * @param zipFilepath Path to the compressed state file (.zip)
     */
    private void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        // initialize
        this.initializeDataStructure();

        String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));

        ZipFile zipFile = new ZipFile(zipFilepath);
        ZipEntry modelEntry = zipFile.getEntry(filename + AssignmentFileExt);
        BufferedReader reader = new BufferedReader(new InputStreamReader(zipFile.getInputStream(modelEntry), "UTF-8"));

        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    String[] sline = reader.readLine().split("\t");
                    if (!sline[0].equals(d + ":" + t + ":" + n)) {
                        throw new RuntimeException("Mismatch");
                    }
                    z[d][t][n] = Integer.parseInt(sline[1]);
                    turnCounts[d][t].increment(z[d][t][n]);
                }
            }
        }
        reader.close();
    }

    public void outputTopicTopWords(String outputFile, int numWords)
            throws Exception {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            System.out.println("Outputing top words to file " + outputFile);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (SHDPDish dish : globalRestaurant.getTables()) {
            String[] topWords = getTopWords(dish.getContent().getDistribution(), numWords);
            writer.write("[" + dish.getIndex()
                    + ", " + dish.getNumCustomers()
                    + ", " + MiscUtils.formatDouble(dish.getMean())
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");
        }
        writer.close();
    }

    public void outputTopicCoherence(
            String filepath,
            MimnoTopicCoherence topicCoherence) throws Exception {
        if (verbose) {
            System.out.println("Outputing topic coherence to file " + filepath);
        }

        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (SHDPDish dish : globalRestaurant.getTables()) {
            double[] distribution = dish.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(dish.getIndex()
                    + "\t" + dish.getNumCustomers()
                    + "\t" + dish.getContent().getCountSum()
                    + "\t" + MiscUtils.formatDouble(score));
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void diagnose(String outputFile) throws Exception {
        if (verbose) {
            System.out.println("Diagnosing to file " + outputFile);
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (SHDPDish dish : globalRestaurant.getTables()) {
            String[] topWords = getTopWords(dish.getContent().getDistribution(), 15);
            writer.write("[" + dish.getIndex()
                    + ", " + dish.getNumCustomers()
                    + ", " + dish.getContent().getCountSum()
                    + ", " + MiscUtils.formatDouble(dish.getMean())
                    + "]");
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n\n");

            for (SHDPTable table : dish.getCustomers()) {
                StringBuilder str = new StringBuilder();
                str.append("\t")
                        .append("[").append(table.restIndex)
                        .append("-").append(table.getIndex())
                        .append(", ").append(table.getNumCustomers())
                        .append(", ").append(MiscUtils.formatDouble(table.getEta()))
                        .append("]\t");

                int[] counts = new int[V];
                for (String customer : table.getCustomers()) {
                    int[] parsedId = parseTokenId(customer);
                    counts[words[table.restIndex][parsedId[0]][parsedId[1]]]++;
                }
                ArrayList<RankingItem<Integer>> rankList = new ArrayList<RankingItem<Integer>>();
                for (int v = 0; v < V; v++) {
                    rankList.add(new RankingItem<Integer>(v, counts[v]));
                }
                Collections.sort(rankList);

                for (int i = 0; i < 15; i++) {
                    str.append(wordVocab.get(rankList.get(i).getObject())).append(":")
                            .append(rankList.get(i).getPrimaryValue())
                            .append(", ");
                }
                writer.write(str.toString() + "\n\n");
            }
        }
        writer.close();
    }

    // debug
    public void test(String filepath, int[][][] trWords, double[][] trResponses) {
        try {
            inputState(filepath);
            double[][] predResponses = getRegressionValues();
            RegressionEvaluation eval = new RegressionEvaluation(
                    MiscUtils.flatten2DArray(responses),
                    MiscUtils.flatten2DArray(predResponses));
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();
            for (Measurement measurement : measurements) {
                logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
            }
            System.out.println();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public double[][] getRegressionValues() {
        double[][] regValues = new double[D][];
        for (int d = 0; d < D; d++) {
            regValues[d] = new double[responses[d].length];
            for (int t = 0; t < responses[d].length; t++) {
                double sum = 0.0;
                for (SHDPTable table : localRestaurants[d].getTables()) {
                    sum += table.getEta() * turnCounts[d][t].getCount(table.getIndex());
                }
                regValues[d][t] = sum / words[d][t].length;

                // debug
//                logln("d = " + d + ". t = " + t 
//                        + ". count = " + words[d][t].length
//                        + " vs. " + turnCounts[d][t].getCountSum()
//                        + ". true: " + MiscUtils.formatDouble(responses[d][t])
//                        + ". pred: " + MiscUtils.formatDouble(regValues[d][t]))
//                        ;
//                for(SHDPTable table : localRestaurants[d].getTables()){
//                    if(turnCounts[d][t].getCount(table.getIndex()) > 0)
//                        logln("--- table " + table.getIndex()
//                            + ". " + MiscUtils.formatDouble(table.getEta())
//                            + ". " + table.getNumCustomers()
//                            + ". " + turnCounts[d][t].getCount(table.getIndex())
//                            );
//                }
//                System.out.println();
            }
        }
        return regValues;
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
    public double[][] regressExistingGroups(int[][][] newWords, double[][] newResponses, String filepath) throws Exception {
        String reportFolderpath = this.folder + this.getSamplerFolder() + ReportFolder;
        File reportFolder = new File(reportFolderpath);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist");
        }
        String[] filenames = reportFolder.list();

        ArrayList<double[][]> predResponsesList = new ArrayList<double[][]>();
        BufferedWriter writer = IOUtils.getBufferedWriter(filepath);
        for (int i = 0; i < filenames.length; i++) {
            String filename = filenames[i];
            double[][] predResponses = regressExistingGroups(reportFolderpath
                    + filename, newWords, newResponses);
            predResponsesList.add(predResponses);

            RegressionEvaluation eval = new RegressionEvaluation(
                    MiscUtils.flatten2DArray(responses),
                    MiscUtils.flatten2DArray(predResponses));
            eval.computeCorrelationCoefficient();
            eval.computeMeanSquareError();
            eval.computeRSquared();
            ArrayList<Measurement> measurements = eval.getMeasurements();

            // output results
            if (i == 0) {
                writer.write("Model");
                for (Measurement measurement : measurements) {
                    writer.write("\t" + measurement.getName());
                }
                writer.write("\n");
            }
            writer.write(filename);
            for (Measurement measurement : measurements) {
                writer.write("\t" + measurement.getValue());
            }
            writer.write("\n");

            if (verbose) {
                logln("Model from " + reportFolderpath + filename);
                for (Measurement measurement : measurements) {
                    logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                }
                System.out.println();

            }
        }
        writer.close();

        // average predicted response over different models
        double[][] finalPredResponses = new double[D][];
        for (int d = 0; d < D; d++) {
            finalPredResponses[d] = new double[words[d].length];
            for (int t = 0; t < finalPredResponses[d].length; t++) {
                double sum = 0.0;
                for (int i = 0; i < predResponsesList.size(); i++) {
                    sum += predResponsesList.get(i)[d][t];
                }
                finalPredResponses[d][t] = sum / predResponsesList.size();
            }
        }
        return finalPredResponses;
    }

    /**
     * Perform regression on test documents in the same groups as in the
     * training data using a specific model.
     *
     * @param newWords New documents
     * @param newResponses The true new responses. This is used to evaluate the
     * predicted values.
     */
    private double[][] regressExistingGroups(String stateFile, int[][][] newWords, double[][] newResponses) {
        if (newWords.length != D) {
            throw new RuntimeException("Number of test documents does not match");
        }

        if (verbose) {
            logln("Perform regression using model from " + stateFile);
        }

        try {
            inputModel(stateFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        words = newWords;
        responses = newResponses; // for evaluation
        D = words.length;

        // initialize structure
        initializeDataStructure();

        // initialize assignments
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < words[d].length; t++) {
                for (int n = 0; n < words[d][t].length; n++) {
                    sampleTableForToken(d, t, n, !REMOVE, !OBSERVED, ADD, !EXTEND);
                }
            }
        }

        // iterate
        ArrayList<double[][]> predResponsesList = new ArrayList<double[][]>();
        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    for (int n = 0; n < words[d][t].length; n++) {
                        sampleTableForToken(d, t, n, REMOVE, !OBSERVED, ADD, !EXTEND);
                    }
                }
            }

            if (iter >= BURN_IN && iter % LAG == 0) {
                double[][] predResponses = getRegressionValues();
                predResponsesList.add(predResponses);

                if (verbose) {
                    logln("state file: " + stateFile
                            + ". iter = " + iter
                            + ". llh = " + getLogLikelihood());

                    RegressionEvaluation eval = new RegressionEvaluation(
                            MiscUtils.flatten2DArray(responses),
                            MiscUtils.flatten2DArray(predResponses));
                    eval.computeCorrelationCoefficient();
                    eval.computeMeanSquareError();
                    eval.computeRSquared();
                    ArrayList<Measurement> measurements = eval.getMeasurements();
                    for (Measurement measurement : measurements) {
                        logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
                    }
                    System.out.println();
                }
            }
        }

        // averaging prediction responses over time
        double[][] finalPredResponses = new double[D][];
        for (int d = 0; d < D; d++) {
            finalPredResponses[d] = new double[words[d].length];
            for (int t = 0; t < finalPredResponses[d].length; t++) {
                double sum = 0.0;
                for (int i = 0; i < predResponsesList.size(); i++) {
                    sum += predResponsesList.get(i)[d][t];
                }
                finalPredResponses[d][t] = sum / predResponsesList.size();
            }
        }
        return finalPredResponses;
    }

    public void regressNewDocuments(int[][][] newWords) {
    }

    class SHDPDish extends FullTable<SHDPTable, DirMult> {

        private double mean;

        public SHDPDish(int index, DirMult content, double mean) {
            super(index, content);
            this.mean = mean;
        }

        public double getMean() {
            return mean;
        }

        public void setMean(double mean) {
            this.mean = mean;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append(index)
                    .append(". #c: ").append(getNumCustomers())
                    .append(". #o: ").append(content.getCountSum())
                    .append(". mean: ").append(MiscUtils.formatDouble(mean));
            return str.toString();
        }
    }

    class SHDPTable extends FullTable<String, SHDPDish> {

        int restIndex;
        private double eta;

        public SHDPTable(int index, SHDPDish dish, int restIndex, double eta) {
            super(index, dish);
            this.restIndex = restIndex;
            this.eta = eta;
        }

        public String getTableId() {
            return restIndex + ":" + index;
        }

        public double getEta() {
            return eta;
        }

        public void setEta(double eta) {
            this.eta = eta;
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append(restIndex).append("-").append(index)
                    .append(". #c = ").append(getNumCustomers())
                    .append(". -> ").append(content.getIndex());
            return str.toString();
        }
    }
}
