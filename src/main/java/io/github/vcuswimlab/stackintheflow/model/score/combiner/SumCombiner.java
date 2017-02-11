package io.github.vcuswimlab.stackintheflow.model.score.combiner;

import io.github.vcuswimlab.stackintheflow.model.score.Scorer;

import java.util.Collection;

/**
 * Created by Chase on 2/11/2017.
 */
public class SumCombiner extends AbstractCombiner {

    public SumCombiner(Collection<Scorer> scorers) {
        super(scorers);
    }

    @Override
    public double generateCumulativeScore(String term) {
        return scorers.stream().mapToDouble(s -> s.score(term)).sum();
    }
}
