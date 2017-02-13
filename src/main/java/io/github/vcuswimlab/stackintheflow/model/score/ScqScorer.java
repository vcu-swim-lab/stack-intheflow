package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.TermStat;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;

import java.util.Optional;

/**
 * Created by Chase on 2/11/2017.
 */
public class ScqScorer extends AbstractScorer {

    public ScqScorer(TermStatComponent statComponent) {
        super(statComponent);
    }

    @Override
    public double score(String term) {

        Optional<TermStat> termStatOptional = statComponent.getTermStat(term);

        if (termStatOptional.isPresent()) {
            TermStat termStat = termStatOptional.get();

            return (1 + Math.log10(termStat.getCtf())) * termStat.getIdf();
        }

        return 0;
    }
}
