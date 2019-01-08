package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class LabelResponseTextDataset extends LabelTextDataset {

    protected double[] responses;

    public LabelResponseTextDataset(String name, String folder) {
        super(name, folder);
    }

    public LabelResponseTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public double[] getResponses() {
        return this.responses;
    }

    public void loadResponses(String responseFilepath) throws Exception {
        logln("--- Loading response from file " + responseFilepath);

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        this.responses = new double[this.docIdList.size()];
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(responseFilepath);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];
            double docResponse = Double.parseDouble(sline[1]);
            int index = this.docIdList.indexOf(docId);
            this.responses[index] = docResponse;
        }
        reader.close();
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        logln("--- Outputing document info ... " + outputFile);

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex));
            infoWriter.write("\t" + this.responses[docIndex]);
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
        ArrayList<Double> responseList = new ArrayList<Double>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            responseList.add(Double.parseDouble(sline[1]));
            int[] labelIndices = new int[sline.length - 2];
            for (int ii = 0; ii < sline.length - 2; ii++) {
                labelIndices[ii] = Integer.parseInt(sline[ii + 2]);
            }
            labelIndexList.add(labelIndices);
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.labels = new int[labelIndexList.size()][];
        this.responses = new double[labelIndexList.size()];
        for (int ii = 0; ii < this.labels.length; ii++) {
            this.labels[ii] = labelIndexList.get(ii);
            this.responses[ii] = responseList.get(ii);
        }
    }
}
