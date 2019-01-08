package sampler;

import core.AbstractSampler;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import sampling.likelihood.DirMult;
import sampling.util.FullTable;
import sampling.util.Restaurant;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.evaluation.MimnoTopicCoherence;


/**
 *
 * @author vietan
 */
public class HDP extends AbstractSampler {

    public static final int PSEUDO_INDEX = -1;
    public static final int ALPHA_GLOBAL = 0;
    public static final int ALPHA_LOCAL = 1;
    public static final int BETA = 2;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int K;
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z; // local table index
    private Restaurant<HDPDish, HDPTable, DirMult> globalRestaurant;
    private Restaurant<HDPTable, Integer, HDPDish>[] localRestaurants;
    private double[] uniform;
    private int totalNumObservations = 0;
    private HDPDish emptyDish;
    private int numTokenAssignmentsChange;
    private int numTableAssignmentsChange;

    public void configure(String folder,
            int[][] words,
            int V,
            double alpha_global, double alpha_local, double beta,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;

        this.words = words;

        this.V = V;
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha_global);
        this.hyperparams.add(alpha_local);
        this.hyperparams.add(beta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.uniform = new double[V];
        for (int v = 0; v < V; v++) {
            this.uniform[v] = 1.0 / V;
        }

        this.setName();

        if (verbose) {
            for (int d = 0; d < D; d++) {
                totalNumObservations += this.words[d].length;
            }
            logln("--- D = " + D);
            logln("--- V = " + V);
            logln("--- # observations = " + totalNumObservations);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_HDP")
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_ag-").append(formatter.format(hyperparams.get(ALPHA_GLOBAL)))
                .append("_al-").append(formatter.format(hyperparams.get(ALPHA_LOCAL)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)));
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

        initializeHierarchies();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- --- Done initializing. \n" + getCurrentState());
        }
    }

    protected void initializeHierarchies() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

        this.globalRestaurant = new Restaurant<HDPDish, HDPTable, DirMult>();

        this.localRestaurants = new Restaurant[D];
        for (int d = 0; d < D; d++) {
            this.localRestaurants[d] = new Restaurant<HDPTable, Integer, HDPDish>();
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        DirMult emptyModel = new DirMult(V, hyperparams.get(BETA), uniform);
        this.emptyDish = new HDPDish(PSEUDO_INDEX, emptyModel);
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
        lda.configure(null, words, V, K, lda_alpha, lda_beta, initState,
                paramOptimized, lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);
        lda.sample();

        for (int d = 0; d < D; d++) {
            // create tables
            for (int k = 0; k < K; k++) {
                HDPTable table = new HDPTable(k, null, d);
                this.localRestaurants[d].addTable(table);
            }

            // add customers to tables
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = lda.z[d][n];
                this.localRestaurants[d].addCustomerToTable(n, z[d][n]);
            }

            // assign table to dish
            ArrayList<Integer> emptyTables = new ArrayList<Integer>();
            for (HDPTable table : this.localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    emptyTables.add(table.getIndex());
                    continue;
                }

                sampleDishForTable(d, table.getIndex(), !REMOVE);
            }

            // remove empty tables
            for (int tabIndex : emptyTables) {
                this.localRestaurants[d].removeTable(tabIndex);
            }
        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        this.logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose) {
                String str = new String();
                if (iter < BURN_IN) {
                    str += "--- Burning in. Iter " + iter;
                } else {
                    str += "--- Sampling. Iter " + iter;
                }
                str += "\t llh = " + loglikelihood
                        + "\t #tokens change: " + numTokenAssignmentsChange
                        + "\t #tables change: " + numTableAssignmentsChange
                        + "\n" + getCurrentState();
                logln(str);
            }

            numTableAssignmentsChange = 0;
            numTokenAssignmentsChange = 0;

            for (int d = 0; d < D; d++) {
                for (int n = 0; n < words[d].length; n++) {
                    this.sampleTableForToken(d, n, REMOVE);
                }

                for (HDPTable table : this.localRestaurants[d].getTables()) {
                    this.sampleDishForTable(d, table.getIndex(), REMOVE);
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
            System.out.println();
        }
    }

    /**
     * Create a brand new dish
     */
    private HDPDish createDish() {
        int newDishIndex = globalRestaurant.getNextTableIndex();
        DirMult dm = new DirMult(V, hyperparams.get(BETA), uniform);
        HDPDish newDish = new HDPDish(newDishIndex, dm);
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
    private void removeCustomerFromTable(int d, int tableIndex, int n) {
        HDPTable table = this.localRestaurants[d].getTable(tableIndex);
        HDPDish dish = table.getContent();

        this.localRestaurants[d].removeCustomerFromTable(n, tableIndex);
        dish.getContent().decrement(words[d][n]);

        if (table.isEmpty()) {
            removeTableFromDish(d, tableIndex);
            this.localRestaurants[d].removeTable(tableIndex);
        }
    }

    /**
     * Remove a table from a dish
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     */
    private void removeTableFromDish(int d, int tableIndex) {
        HDPTable table = this.localRestaurants[d].getTable(tableIndex);
        HDPDish dish = table.getContent();

        // remove observations from dish
        for (int n : table.getCustomers()) {
            dish.getContent().decrement(words[d][n]);
        }

        // remove table from dish
        this.globalRestaurant.removeCustomerFromTable(table, dish.getIndex());

        // if the dish is empty, remove it
        if (dish.isEmpty()) {
            this.globalRestaurant.removeTable(dish.getIndex());
        }
    }

    /**
     * Sample a dish for a table
     *
     * @param d The restaurant index
     * @param tableIndex The table index
     * @param remove Whether the current table assignment should be removed
     */
    private void sampleDishForTable(int d, int tableIndex, boolean remove) {
        HDPTable table = localRestaurants[d].getTable(tableIndex);

        // current observations assigned to this table
        HashMap<Integer, Integer> observations = new HashMap<Integer, Integer>();
        for (int c : table.getCustomers()) {
            Integer count = observations.get(words[d][c]);
            if (count == null) {
                observations.put(words[d][c], 1);
            } else {
                observations.put(words[d][c], count + 1);
            }
        }

//        boolean condition = d == 170 && tableIndex == 1 && remove;

        // if this is the first assignment (during initialization), create the 
        // first dish and assign to it
        if (globalRestaurant.isEmpty()) {
            HDPDish dish = createDish();
            table.setContent(dish);
            globalRestaurant.addCustomerToTable(table, dish.getIndex());
            for (int obs : observations.keySet()) {
                dish.getContent().changeCount(obs, observations.get(obs));
            }
            return;
        }

        int curDish = PSEUDO_INDEX;
        if (table.getContent() != null) {
            curDish = table.getContent().getIndex();
        }

//        if(condition){
//            System.out.println("Before removing");
//            System.out.println("table: " + table.toString());
//            System.out.println("dish: " + table.getContent().toString());
//            for(HDPDish dish : globalRestaurant.getTables())
//                System.out.println("--- " + dish.toString());
//            System.out.println();
//        }

        if (remove) {
            removeTableFromDish(d, tableIndex);
        }

//        if(condition){
//            System.out.println("After removing");
//            System.out.println("table: " + table.toString());
//            System.out.println("dish: " + table.getContent().toString());
//            for(HDPDish dish : globalRestaurant.getTables())
//                System.out.println("--- " + dish.toString());
//            System.out.println();
//        }

        HashMap<Integer, Double> dishLogPriors = getDishLogPriors();
        HashMap<Integer, Double> dishLogLikelihoods = getDishLogLikelihoods(observations);

        if (dishLogPriors.size() != dishLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of dishes mismatch");
        }

        int sampledDish = sampleDish(dishLogPriors, dishLogLikelihoods);
        if (curDish != sampledDish) {
            numTableAssignmentsChange++;
        }

        HDPDish dish;
        if (sampledDish == PSEUDO_INDEX) {
            dish = createDish();
        } else {
            dish = globalRestaurant.getTable(sampledDish);
        }

        table.setContent(dish);
        globalRestaurant.addCustomerToTable(table, dish.getIndex());
        for (int obs : observations.keySet()) {
            dish.getContent().changeCount(obs, observations.get(obs));
        }

//        if(condition){
//            System.out.println("After updating");
//            System.out.println("table: " + table.toString());
//            System.out.println("dish: " + table.getContent().toString());
//            for(HDPDish di : globalRestaurant.getTables())
//                System.out.println("--- " + di.toString());
//            System.out.println();
//        }

        if (remove) {
            validate("sample dish. d = " + d + ". table index = " + tableIndex);
        }
    }

    /**
     * Sample a dish given precomputed log priors and log likelihoods
     *
     * @param dishLogPriors Precomputed log priors
     * @param dishLogLikelihoods Precomputed log likelihoods
     */
    private int sampleDish(
            HashMap<Integer, Double> dishLogPriors,
            HashMap<Integer, Double> dishLogLikelihoods) {
        ArrayList<Integer> dishIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        for (int dishIndex : dishLogPriors.keySet()) {
            dishIndices.add(dishIndex);
            double logprob = dishLogPriors.get(dishIndex) + dishLogLikelihoods.get(dishIndex);
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
     * Sample a table for a token
     *
     * @param d The restaurant index
     * @param n The token index
     * @param remove Whether the current token assignment should be removed
     */
    private void sampleTableForToken(int d, int n, boolean remove) {
        int curObs = words[d][n];
        int curTableIndex = z[d][n];

        if (remove) {
            removeCustomerFromTable(d, curTableIndex, n);
        }

        ArrayList<Integer> tableIndices = new ArrayList<Integer>();
        ArrayList<Double> logprobs = new ArrayList<Double>();

        double normalizer = Math.log(this.localRestaurants[d].getTotalNumCustomers()
                + hyperparams.get(ALPHA_LOCAL));

        // for existing tables
        for (HDPTable table : this.localRestaurants[d].getTables()) {
            tableIndices.add(table.getIndex());
            double logprob = Math.log(table.getNumCustomers()) - normalizer
                    + table.getContent().getContent().getLogLikelihood(curObs);
            logprobs.add(logprob);
        }

        // for new tables
        tableIndices.add(PSEUDO_INDEX);
        HashMap<Integer, Double> dishLogPriors = getDishLogPriors();
        HashMap<Integer, Double> dishLogLikelihoods = getDishLogLikelihoods(curObs);

        if (dishLogPriors.size() != dishLogLikelihoods.size()) {
            throw new RuntimeException("Numbers of dishes mismatch");
        }

        double marginalLogLikelihood = 0.0;
        for (int dishIndex : dishLogPriors.keySet()) {
            double lp = dishLogPriors.get(dishIndex) + dishLogLikelihoods.get(dishIndex);
            if (marginalLogLikelihood == 0.0) {
                marginalLogLikelihood = lp;
            } else {
                marginalLogLikelihood = SamplerUtils.logAdd(marginalLogLikelihood, lp);
            }
        }
        double logprob = Math.log(hyperparams.get(ALPHA_LOCAL)) - normalizer
                + marginalLogLikelihood;
        logprobs.add(logprob);

        // sample
        int sampledIndex = SamplerUtils.logMaxRescaleSample(logprobs);
        int tableIndex = tableIndices.get(sampledIndex);

        if (curTableIndex != tableIndex) {
            numTokenAssignmentsChange++;
        }

        HDPTable table;
        if (tableIndex == PSEUDO_INDEX) {
            // sample dish
            int sampledDish = sampleDish(dishLogPriors, dishLogLikelihoods);
            HDPDish dish;
            if (sampledDish == PSEUDO_INDEX) {
                dish = createDish();
            } else {
                dish = globalRestaurant.getTable(sampledDish);
            }

            // create a new table
            int newTableIndex = localRestaurants[d].getNextTableIndex();
            table = new HDPTable(newTableIndex, dish, d);

            globalRestaurant.addCustomerToTable(table, dish.getIndex());
            localRestaurants[d].addTable(table);
        } else {
            table = this.localRestaurants[d].getTable(tableIndex);
        }

        // update
        z[d][n] = table.getIndex();
        this.localRestaurants[d].addCustomerToTable(n, z[d][n]);
        table.getContent().getContent().increment(curObs);

        if (remove) {
            validate("sample table: d = " + d + ". n = " + n);
        }
    }

    private HashMap<Integer, Double> getDishLogPriors() {
        HashMap<Integer, Double> dishLogPriors = new HashMap<Integer, Double>();
        double normalizer = Math.log(this.globalRestaurant.getTotalNumCustomers()
                + hyperparams.get(ALPHA_GLOBAL));
        for (HDPDish dish : this.globalRestaurant.getTables()) {
            dishLogPriors.put(dish.getIndex(), Math.log(dish.getNumCustomers()) - normalizer);
        }
        dishLogPriors.put(PSEUDO_INDEX, Math.log(hyperparams.get(ALPHA_GLOBAL)) - normalizer);
        return dishLogPriors;
    }

    private HashMap<Integer, Double> getDishLogLikelihoods(int observation) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (HDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observation));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyDish.getContent().getLogLikelihood(observation));
        return dishLogLikelihoods;
    }

    private HashMap<Integer, Double> getDishLogLikelihoods(HashMap<Integer, Integer> observations) {
        HashMap<Integer, Double> dishLogLikelihoods = new HashMap<Integer, Double>();
        for (HDPDish dish : this.globalRestaurant.getTables()) {
            dishLogLikelihoods.put(dish.getIndex(), dish.getContent().getLogLikelihood(observations));
        }
        dishLogLikelihoods.put(PSEUDO_INDEX, emptyDish.getContent().getLogLikelihood(observations));
        return dishLogLikelihoods;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append("# topics: ").append(globalRestaurant.getNumTables()).append("\t");
        for (HDPDish dish : globalRestaurant.getTables()) {
            str.append(dish.getIndex()).append(" (").append(dish.getNumCustomers())
                    .append(", ").append(dish.getContent().getCountSum())
                    .append("); ");
        }
        return str.toString();
    }

    @Override
    public double getLogLikelihood() {
        double obsLlh = 0.0;
        for (HDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood();
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(hyperparams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(hyperparams.get(ALPHA_LOCAL));
        }

        if (verbose) {
            logln("*** obs: " + MiscUtils.formatDouble(obsLlh)
                    + ". assignments: " + MiscUtils.formatDouble(assignLp));
        }

        return obsLlh + assignLp;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> tParams) {
        double obsLlh = 0.0;
        for (HDPDish dish : globalRestaurant.getTables()) {
            obsLlh += dish.getContent().getLogLikelihood(tParams.get(BETA), uniform);
        }

        double assignLp = globalRestaurant.getJointProbabilityAssignments(tParams.get(ALPHA_GLOBAL));
        for (int d = 0; d < D; d++) {
            assignLp += localRestaurants[d].getJointProbabilityAssignments(tParams.get(ALPHA_LOCAL));
        }
        return obsLlh + assignLp;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        for (HDPDish dish : globalRestaurant.getTables()) {
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
            for (HDPTable table : localRestaurants[d].getTables()) {
                if (table.isEmpty()) {
                    throw new RuntimeException(msg + ". Empty table. " + table.toString());
                }
            }
        }

        for (HDPDish dish : globalRestaurant.getTables()) {
            if (dish.isEmpty() || dish.getContent().getCountSum() == 0) {
                throw new RuntimeException(msg + ". Empty dish. " + dish.toString()
                        + ". tables: " + dish.getCustomers().toString());
            }
        }

        int totalObs = 0;
        for (HDPDish dish : globalRestaurant.getTables()) {
            int dishNumObs = dish.getContent().getCountSum();
            int tableNumObs = 0;
            for (HDPTable table : dish.getCustomers()) {
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
        throw new RuntimeException("This function is not supported at the moment");
    }

    @Override
    public void inputState(String filepath) {
        throw new RuntimeException("This function is not supported at the moment");
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
        for (HDPDish dish : globalRestaurant.getTables()) {
            String[] topWords = getTopWords(dish.getContent().getDistribution(), numWords);
            writer.write("Topic " + dish.getIndex());
            for (String topWord : topWords) {
                writer.write("\t" + topWord);
            }
            writer.write("\n");
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
        for (HDPDish dish : globalRestaurant.getTables()) {
            double[] distribution = dish.getContent().getDistribution();
            int[] topic = SamplerUtils.getSortedTopic(distribution);
            double score = topicCoherence.getCoherenceScore(topic);
            writer.write(dish.getIndex()
                    + "\t" + dish.getNumCustomers()
                    + "\t" + dish.getContent().getCountSum()
                    + "\t" + score);
            for (int i = 0; i < topicCoherence.getNumTokens(); i++) {
                writer.write("\t" + this.wordVocab.get(topic[i]));
            }
            writer.write("\n");
        }
        writer.close();
    }
}
class HDPDish extends FullTable<HDPTable, DirMult> {

    public HDPDish(int index, DirMult content) {
        super(index, content);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(index)
                .append(". #c = ").append(getNumCustomers())
                .append(". #o = ").append(content.getCountSum());
        return str.toString();
    }
}

class HDPTable extends FullTable<Integer, HDPDish> {

    int restIndex;

    public HDPTable(int index, HDPDish dish, int restIndex) {
        super(index, dish);
        this.restIndex = restIndex;
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
