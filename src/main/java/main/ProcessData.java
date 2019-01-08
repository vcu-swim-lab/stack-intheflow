/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import core.AbstractRunner;
import data.CorpusProcessor;
import data.LabelResponseTextDataset;
import data.LabelTextDataset;
import data.ResponseTextDataset;
import data.TextDataset;
import java.io.File;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;

/**
 *
 * @author vietan
 */
public class ProcessData extends AbstractRunner {

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            addOption("dataset", "Dataset");
            addOption("data-folder", "Folder that stores the processed data");
            addOption("text-data", "Directory of the text data");
            addOption("response-file", "Directory of the response file");
            addOption("label-file", "Directory of the label file");
            addOption("format-folder", "Folder that stores formatted data");
            addOption("format-file", "Formatted file name");

            addOption("u", "The minimum count of raw unigrams");
            addOption("b", "The minimum count of raw bigrams");
            addOption("bs", "The minimum score of bigrams");
            addOption("V", "Maximum vocab size");
            addOption("min-tf", "Term frequency minimum cutoff");
            addOption("max-tf", "Term frequency maximum cutoff");
            addOption("min-df", "Document frequency minimum cutoff");
            addOption("max-df", "Document frequency maximum cutoff");
            addOption("min-doc-length", "Document minimum length");

            addOption("L", "Maximum label vocab size");
            addOption("min-label-df", "Minimum count of raw labels");

            options.addOption("s", false, "Whether stopwords are filtered");
            options.addOption("l", false, "Whether lemmatization is performed");
            options.addOption("file", false, "Whether the text input data is stored in a file or a folder");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp("java -cp 'dist/segan.jar:dist/lib/*' main.ProcessData -help", options);
                return;
            }

            processData();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void processData() {
        System.out.println("\nProcessing data ...");

        try {
            String datasetName = cmd.getOptionValue("dataset");
            String datasetFolder = cmd.getOptionValue("data-folder");
            String textInputData = cmd.getOptionValue("text-data");
            String formatFolder = cmd.getOptionValue("format-folder");
            String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

            int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 0);
            int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 0);
            double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
            int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
            int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 0);
            int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
            int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 0);
            int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
            int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 1);

            boolean stopwordFilter = cmd.hasOption("s");
            boolean lemmatization = cmd.hasOption("l");

            CorpusProcessor corpProc = new CorpusProcessor(
                    unigramCountCutoff,
                    bigramCountCutoff,
                    bigramScoreCutoff,
                    maxVocabSize,
                    vocTermFreqMinCutoff,
                    vocTermFreqMaxCutoff,
                    vocDocFreqMinCutoff,
                    vocDocFreqMaxCutoff,
                    docTypeCountCutoff,
                    stopwordFilter,
                    lemmatization);

            if (cmd.hasOption("response-file") && cmd.hasOption("label-file")) {
                String responseFile = cmd.getOptionValue("response-file");
                String labelFile = cmd.getOptionValue("label-file");
                LabelResponseTextDataset dataset =
                        new LabelResponseTextDataset(datasetName, datasetFolder, corpProc);
                dataset.setFormatFilename(formatFile);

                if (cmd.hasOption("L")) {
                    dataset.setMaxLabelVocabSize(Integer.parseInt(cmd.getOptionValue("L")));
                }
                if (cmd.hasOption("min-label-df")) {
                    dataset.setMinLabelDocFreq(Integer.parseInt(cmd.getOptionValue("min-label-df")));
                }

                // load text data
                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }

                dataset.loadResponses(responseFile); // load response data
                dataset.loadLabels(labelFile); // load labels
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            } else if (cmd.hasOption("response-file")) {
                String responseFile = cmd.getOptionValue("response-file");
                ResponseTextDataset dataset = new ResponseTextDataset(datasetName, datasetFolder, corpProc);
                dataset.setFormatFilename(formatFile);

                // load text data
                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.loadResponses(responseFile); // load response data
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            } else if (cmd.hasOption("label-file")) {
                String labelFile = cmd.getOptionValue("label-file");
                LabelTextDataset dataset = new LabelTextDataset(datasetName, datasetFolder, corpProc);
                dataset.setFormatFilename(formatFile);
                if (cmd.hasOption("L")) {
                    dataset.setMaxLabelVocabSize(Integer.parseInt(cmd.getOptionValue("L")));
                }
                if (cmd.hasOption("min-label-df")) {
                    dataset.setMinLabelDocFreq(Integer.parseInt(cmd.getOptionValue("min-label-df")));
                }

                // load text data
                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.loadLabels(labelFile);
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            } else {
                TextDataset dataset = new TextDataset(datasetName, datasetFolder, corpProc);
                dataset.setFormatFilename(formatFile);

                if (cmd.hasOption("file")) {
                    dataset.loadTextDataFromFile(textInputData);
                } else {
                    dataset.loadTextDataFromFolder(textInputData);
                }
                dataset.format(new File(dataset.getDatasetFolderPath(), formatFolder).getAbsolutePath());
            }
        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp("java -cp dist/segan.jar main.ProcessData -help", options);
        }
    }
}