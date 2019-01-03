package io.github.vcuswimlab.stackintheflow.model.L2H;

public class TagPrediction implements Comparable<TagPrediction> {
	private String name;
	private double probability;

	public TagPrediction(String name, double probability) {
		this.name = name;
		this.probability = probability;
	}

	@Override
	public int compareTo(TagPrediction other) {
		return Double.compare(other.probability, probability);
	}

	public String getName() {
		return name;
	}

	public double getProbability() {
		return probability;
	}
}
