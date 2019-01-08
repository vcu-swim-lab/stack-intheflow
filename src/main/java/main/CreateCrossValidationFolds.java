package main;

import core.AbstractRunner;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.RegressionDocumentInstance;
import data.ResponseTextDataset;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class CreateCrossValidationFolds extends AbstractRunner {

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            addOption("dataset", "Dataset name");
            addOption("data-folder", "Processed data folder");
            addOption("output", "Output folder");
            addOption("format-folder", "Formatted data folder");
            addOption("num-classes", "Number of classes that the response"
                    + " variable are discretized into to perform stratified"
                    + " sampling. Default 1.");
            addOption("num-folds", "Number of folds. Default 5.");
            addOption("tr2dev-ratio", "Training-to-development ratio. Default 0.8.");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' "
                        + "main.CreateCrossValidationFolds -help", options);
                return;
            }

            stratifiedSampling();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void stratifiedSampling() {
        try {
            System.out.println("\nLoading formatted data ...");
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("data-folder");
            String formatFolder = cmd.getOptionValue("format-folder");
            ResponseTextDataset dataset = new ResponseTextDataset(datasetName, datasetFolder);
            dataset.loadFormattedData(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());

            ArrayList<RegressionDocumentInstance> instanceList = new ArrayList<RegressionDocumentInstance>();
            for (int d = 0; d < dataset.getDocIds().length; d++) {
                instanceList.add(new RegressionDocumentInstance(
                        dataset.getDocIds()[d],
                        dataset.getWords()[d],
                        dataset.getResponses()[d]));
            }

            String outputFolder = cmd.getOptionValue("output");
            IOUtils.createFolder(outputFolder);

            String cvName = "";
            CrossValidation<String, RegressionDocumentInstance> cv =
                    new CrossValidation<String, RegressionDocumentInstance>(
                    outputFolder,
                    cvName,
                    instanceList);

            int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
            double trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
            int numClasses = CLIUtils.getIntegerArgument(cmd, "num-classes", 1);

            // create groupIdList based on the response variable
            ArrayList<Integer> groupIdList = StatUtils.discretize(dataset.getResponses(), numClasses);

            System.out.println("\nStratified sampling ... " + outputFolder);
            cv.stratify(groupIdList, numFolds, trToDevRatio);
            cv.outputFolds();
            for (Fold<String, RegressionDocumentInstance> fold : cv.getFolds()) {
                outputLexicalSVMLightData(fold);
            }
            System.out.println("--- Cross validation data are written to " + outputFolder);
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.CreateCrossValidationFolds -help", options);
        }
    }

    private static void outputLexicalSVMLightData(
            Fold<String, RegressionDocumentInstance> fold) throws Exception {
        String featureType = "lexical";
        BufferedWriter writer = IOUtils.getBufferedWriter(new File(fold.getFolder(),
                "fold-" + fold.getIndex() + "-" + featureType + Fold.TrainingExt).getAbsoluteFile());
        for (int idx : fold.getTrainingInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();

        writer = IOUtils.getBufferedWriter(new File(fold.getFolder(),
                "fold-" + fold.getIndex() + "-" + featureType + Fold.DevelopExt).getAbsoluteFile());
        for (int idx : fold.getDevelopmentInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();

        writer = IOUtils.getBufferedWriter(new File(fold.getFolder(),
                "fold-" + fold.getIndex() + "-" + featureType + Fold.TestExt).getAbsoluteFile());
        for (int idx : fold.getTestingInstances()) {
            RegressionDocumentInstance inst = fold.getInstance(idx);
            writer.write(inst.getFullVocabSVMLigthString() + "\n");
        }
        writer.close();
    }
}
