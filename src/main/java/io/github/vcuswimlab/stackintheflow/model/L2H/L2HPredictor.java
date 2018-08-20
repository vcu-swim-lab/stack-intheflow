package io.github.vcuswimlab.stackintheflow.model.L2H;

import java.io.*;
import java.util.*;
import core.AbstractSampler;
import data.TextDataset;
import data.LabelTextDataset;
import sampling.likelihood.DirMult;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import taxonomy.MSTBuilder;
import util.IOUtils;

public class L2HPredictor {
	// L2H Parameters
	private static final String datasetName = "SOBigrams6";
	private static final String formatFolder = "/SOBigrams6/";
	private static final String formatFile = "SOBigrams6";
	private static final String outputFolder = "/SOBigrams6_out";
	private static final String stateFile = outputFolder + "/PRESET_L2H_K-391_B-250_M-500_L-25_opt-false_MAXIMAL-10-1000-90-10-mst-false-falsereport/iter-500.zip";
	private static final int numTopWords = 20;
	private static final int minLabelFreq = 100;
	private static final int burnIn = 250;
	private static final int modelIterations = 500;
	private static final int sampleLag = 25;
	private static final int alpha = 10;
	private static final int beta = 1000;
	private static final int a0 = 90;
	private static final int b0 = 10;

	private static int[] toIntArray(List<Integer> list){
	  int[] ret = new int[list.size()];
	  for(int i = 0;i < ret.length;i++)
		ret[i] = list.get(i);
	  return ret;
	}

	private static class TextDataLoader extends TextDataset {		
		public TextDataLoader(String dataset, String fileName) throws FileNotFoundException, IOException, Exception {
			super(dataset);
			File testFile = new File(fileName);
			inputTextData(testFile);
		}
		
		public int[][] getWords() {
			return words;
		}
	}

	private static double[][] computeInitPredictions(int[][] newWords, int numUniqTags) {
        double[][] initPredictions = new double[newWords.length][];
        double p = 1./ (double) numUniqTags;
        for (int i=0; i<newWords.length; i++) {
            initPredictions[i] = new double[numUniqTags];
            for (int j=0; j<numUniqTags; j++) {
                initPredictions[i][j] = p;
            }
        }
        return initPredictions;
    }

	private Map<String, Integer> nameToIdMap;
	private CustomL2H testSampler;
	private LabelTextDataset data;

	public L2HPredictor() throws IOException {
		nameToIdMap = new HashMap<>();
		BufferedReader br = new BufferedReader(new InputStreamReader(this.getClass().getClassLoader().getResourceAsStream(formatFolder+formatFile+".wvoc")));
		int maxId = -1;
		try {
			String line;
			int intId = 0;
			while ((line = br.readLine()) != null) {
				nameToIdMap.put(line, intId);
				intId++;
			}
		} finally {
			br.close();
		}

		data = new LabelTextDataset(datasetName);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(formatFolder);
        data.filterLabelsByFrequency(minLabelFreq);
		data.prepareTopicCoherence(numTopWords);

		testSampler = new CustomL2H();
		testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure("./output", data.getWordVocab().size(), alpha, beta, a0, b0,
				new MSTBuilder(data.getLabels(), data.getLabelVocab()), false, false, AbstractSampler.InitialState.PRESET, PathAssumption.MAXIMAL, false, burnIn, modelIterations, sampleLag, 1);
        testSampler.setTestConfigurations(burnIn, modelIterations, sampleLag);
	}

	public void computeL2HPredictions(String fileToPredict, String outputFileName, String tempFileName) throws Exception {
		BufferedReader br;
		br = new BufferedReader(new InputStreamReader(this.getClass().getClassLoader().getResourceAsStream(fileToPredict)));
		Map<Integer, Integer> wordOccurrenceMap = new HashMap<>();
		try {
			String line;
			while ((line = br.readLine()) != null) {
				String[] parts = line.split("\\s+");
				for(int i = 1; i < parts.length; i++) {
					String name = parts[i-1] + " " + parts[i];
					if(nameToIdMap.containsKey(name)) {
						int index = nameToIdMap.get(name);
						if(wordOccurrenceMap.containsKey(index)) {
							wordOccurrenceMap.put(index, wordOccurrenceMap.get(index)+1);
						} else {
							wordOccurrenceMap.put(index, 1);
						}
					}
				}
			}
		} finally {
			br.close();
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(tempFileName));
		try {
			bw.write(wordOccurrenceMap.size() + " ");
			for(Map.Entry<Integer,Integer> e : wordOccurrenceMap.entrySet()) {
				bw.write(e.getKey() + ":" + e.getValue() + " ");
			}
		} finally {
			bw.close();
		}
		int[][] newWords = new TextDataLoader("dummyDataset",tempFileName).getWords();
		double[][] initPredictions = computeInitPredictions(newWords, data.getLabelVocab().size());

		try {
            testSampler.sampleNewDocuments(stateFile, newWords, outputFileName, initPredictions, 20);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
	}

	public List<TagPrediction> computeMostLikelyTags(String predictionsFileName, int topN) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(this.getClass().getClassLoader().getResourceAsStream(predictionsFileName)));
		
		TagPrediction[] tagProbabilities; 
		try {
			br.readLine();
			String line = br.readLine();
			String[] parts = line.split("\\s+");
			tagProbabilities = new TagPrediction[parts.length-1];
			for(int i = 0; i < parts.length-1; i++) {
				tagProbabilities[i] = new TagPrediction(data.getLabelVocab().get(i),Double.parseDouble(parts[i]));
			}
		} finally {
			br.close();
		}

		if(topN <= 0 || topN > tagProbabilities.length) {
			topN = tagProbabilities.length;
		}

		List<TagPrediction> result = new ArrayList<>();
		Arrays.sort(tagProbabilities);
		for(int i = 0; i < topN; i++) {
			result.add(tagProbabilities[i]);
		}
		return result;
	}
}
