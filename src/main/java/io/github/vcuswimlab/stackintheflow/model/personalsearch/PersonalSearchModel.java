package io.github.vcuswimlab.stackintheflow.model.personalsearch;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.tags.TagStatComponent;
import io.github.vcuswimlab.stackintheflow.model.Question;
import io.github.vcuswimlab.stackintheflow.model.L2H.L2HPredictor;
import io.github.vcuswimlab.stackintheflow.model.L2H.TagPrediction;

import java.util.*;
import java.util.stream.Collectors;
import java.io.*;

/**
 * Created by chase on 6/13/17.
 */
public class PersonalSearchModel {
    private Map<String, Integer> userStatMap;
    private TagStatComponent tagStatComponent;
	private L2HPredictor l2hPredictor;

    public PersonalSearchModel(TagStatComponent tagStatComponent, Map<String, Integer> userStateMap) {
        this.tagStatComponent = tagStatComponent;
        this.userStatMap = userStateMap;
		try {
			l2hPredictor = new L2HPredictor();
		} catch(IOException e) {
			// L2H could not be loaded, so don't add initial predictions
			l2hPredictor = null;
		}
    }

	public boolean createInitialTagPredictions(String filePath) {
		if(l2hPredictor == null) {
			return false;
		}
		try {
			String rawPredictions = l2hPredictor.computeL2HPredictions(filePath);
			int maxTags = 5;
			int countAmount = 5;
			List<TagPrediction> tagPredictions = l2hPredictor.computeMostLikelyTags(new ByteArrayInputStream(rawPredictions.getBytes()), maxTags);
			List<String> tagToIncrement = new ArrayList<>();
			for(int i = 0; i < maxTags; i++) {
				int tagAmt = (int)(tagPredictions.get(i).getProbability()*countAmount + 0.5);
				if(tagAmt != 0) {
					tagToIncrement.clear();
					tagToIncrement.add(tagPredictions.get(i).getName());
					increaseTags(tagToIncrement, tagAmt);
				}
			}
		} catch(Exception e) {
			return false;
		}

		return true;
	}

    public void increaseTags(Collection<String> tags) {
        increaseTags(tags, 1);
    }

    public void increaseTags(Collection<String> tags, int amount) {
        tags.forEach(tag -> userStatMap.put(tag, userStatMap.getOrDefault(tag, 0) + amount));
    }

    public List<Question> rankQuestionList(List<Question> initialList) {

        List<QuestionRank> questionRankList = new ArrayList<>();
        for (int i = 0; i < initialList.size(); i++) {
            questionRankList.add(new QuestionRank(initialList.get(i), i));
        }

        generateWeightedRank(questionRankList);
        for (int i = 0; i < questionRankList.size(); i++) {
            questionRankList.get(i).weightedRank = i;
        }

        generateFinalRank(questionRankList);

        return questionRankList.stream().map(questionRank -> questionRank.question).collect(Collectors.toList());
    }

    private void generateWeightedRank(List<QuestionRank> initialList) {
        Collections.sort(initialList, (a, b) -> Double.compare(calculateRawScore(b.question), calculateRawScore(a.question)));
    }

    private void generateFinalRank(List<QuestionRank> weightedList) {
        Collections.sort(weightedList, Comparator.comparingDouble(QuestionRank::getAverageRank));
    }

    private double calculateRawScore(Question question) {
        List<String> tags = question.getTags();

        if (tags.isEmpty()) {
            return 0;
        } else {
            double score = 0;
            for (String tag : tags) {

                // If this is a tag the user has clicked on
                if (userStatMap.containsKey(tag)) {

                    Optional<Stat> statOptional = tagStatComponent.getTagStat(tag);

                    // This tag is in the dump
                    if (statOptional.isPresent()) {

                        double clickFrequency = userStatMap.containsKey(tag) ? 1 + Math.log10(userStatMap.get(tag)) : 0;
                        score += statOptional.get().getIdf() * clickFrequency;
                    }
                }
            }

            return score / tags.size();
        }
    }

    private class QuestionRank {

        final int INITIAL_RANK;
        Question question;
        int weightedRank;

        public QuestionRank(Question question, int INITIAL_RANK) {
            this.question = question;
            this.INITIAL_RANK = INITIAL_RANK;
        }

        public double getAverageRank() {
            return (INITIAL_RANK + weightedRank) / 2;
        }
    }
}
