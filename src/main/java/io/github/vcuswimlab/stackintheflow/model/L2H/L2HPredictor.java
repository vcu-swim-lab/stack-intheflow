package io.github.vcuswimlab.stackintheflow.model.L2H;

import java.io.*;
import java.util.*;
import java.nio.file.Paths;
import core.AbstractSampler;
import data.LabelTextDataset;
import sampling.likelihood.DirMult;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import taxonomy.MSTBuilder;
import util.IOUtils;

public class L2HPredictor {
	// L2H Parameters
	private static final String datasetName = "SOBigrams6";
	private static final String formatFolder = "SOBigrams6";
	private static final String formatFile = "SOBigrams6";
	private static final String outputFolder = "SOBigrams6_out";
	private static final String stateFile = Paths
			.get(outputFolder, "PRESET_L2H_K-391_B-250_M-500_L-25_opt-false_MAXIMAL-10-1000-90-10-mst-false-false",
					"report", "iter-500.zip")
			.toString();
	private static final int numTopWords = 20;
	private static final int minLabelFreq = 100;
	private static final int burnIn = 250;
	private static final int modelIterations = 500;
	private static final int sampleLag = 25;
	private static final int alpha = 10;
	private static final int beta = 1000;
	private static final int a0 = 90;
	private static final int b0 = 10;

	private static int[] toIntArray(List<Integer> list) {
		int[] ret = new int[list.size()];
		for (int i = 0; i < ret.length; i++)
			ret[i] = list.get(i);
		return ret;
	}

	private static class TextDataLoader {
		private int[][] words;

		public TextDataLoader(InputStream stream) throws Exception {
			words = inputFormattedTextData(stream);
		}

		public int[][] getWords() {
			return words;
		}

		protected int[][] inputFormattedTextData(InputStream stream) throws Exception {
			BufferedReader reader = new BufferedReader(new InputStreamReader(stream));

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
			int[][] wds = wordList.toArray(new int[wordList.size()][]);
			return wds;
		}
	}

	private static double[][] computeInitPredictions(int[][] newWords, int numUniqTags) {
		double[][] initPredictions = new double[newWords.length][];
		double p = 1. / (double) numUniqTags;
		for (int i = 0; i < newWords.length; i++) {
			initPredictions[i] = new double[numUniqTags];
			for (int j = 0; j < numUniqTags; j++) {
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
		BufferedReader br = new BufferedReader(new InputStreamReader(this.getClass().getClassLoader()
				.getResourceAsStream(Paths.get(formatFolder, formatFile + ".wvoc").toString())));
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
				new MSTBuilder(data.getLabels(), data.getLabelVocab()), false, false,
				AbstractSampler.InitialState.PRESET, PathAssumption.MAXIMAL, false, burnIn, modelIterations, sampleLag,
				1);
		testSampler.setTestConfigurations(burnIn, modelIterations, sampleLag);
	}

	public String computeL2HPredictions(String fileToPredict) throws Exception {
		BufferedReader br;
		br = new BufferedReader(
				new InputStreamReader(this.getClass().getClassLoader().getResourceAsStream(fileToPredict)));
		Map<Integer, Integer> wordOccurrenceMap = new HashMap<>();
		try {
			String line;
			while ((line = br.readLine()) != null) {
				String[] parts = line.split("\\s+");
				for (int i = 1; i < parts.length; i++) {
					String name = parts[i - 1] + " " + parts[i];
					if (nameToIdMap.containsKey(name)) {
						int index = nameToIdMap.get(name);
						if (wordOccurrenceMap.containsKey(index)) {
							wordOccurrenceMap.put(index, wordOccurrenceMap.get(index) + 1);
						} else {
							wordOccurrenceMap.put(index, 1);
						}
					}
				}
			}
		} finally {
			br.close();
		}

		StringBuilder builder = new StringBuilder();
		builder.append(wordOccurrenceMap.size() + " ");
		for (Map.Entry<Integer, Integer> e : wordOccurrenceMap.entrySet()) {
			builder.append(e.getKey() + ":" + e.getValue() + " ");
		}

		int[][] newWords = new TextDataLoader(new ByteArrayInputStream(builder.toString().getBytes())).getWords();
		double[][] initPredictions = computeInitPredictions(newWords, data.getLabelVocab().size());

		try {
			return testSampler.sampleNewDocuments(stateFile, newWords, null, initPredictions, 20);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}

	public List<TagPrediction> computeMostLikelyTags(InputStream stream, int topN) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(stream));

		TagPrediction[] tagProbabilities;
		try {
			br.readLine();
			String line = br.readLine();
			String[] parts = line.split("\\s+");
			tagProbabilities = new TagPrediction[parts.length - 1];
			for (int i = 0; i < parts.length - 1; i++) {
				tagProbabilities[i] = new TagPrediction(data.getLabelVocab().get(i), Double.parseDouble(parts[i]));
			}
		} finally {
			br.close();
		}

		if (topN <= 0 || topN > tagProbabilities.length) {
			topN = tagProbabilities.length;
		}

		List<TagPrediction> result = new ArrayList<>();
		Arrays.sort(tagProbabilities);
		for (int i = 0; i < topN; i++) {
			result.add(tagProbabilities[i]);
		}
		return result;
	}
}
