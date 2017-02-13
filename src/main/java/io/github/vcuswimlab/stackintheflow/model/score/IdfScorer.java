package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.TermStat;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;

import java.util.Optional;

/**
 * Created by chase on 2/13/17.
 */
public class IdfScorer extends AbstractScorer {

    public IdfScorer(TermStatComponent statComponent) {
        super(statComponent);
    }

    @Override
    public double score(String term) {

        Optional<TermStat> termStatOptional = statComponent.getTermStat(term);

        if (termStatOptional.isPresent()) {
            TermStat termStat = termStatOptional.get();

            return termStat.getIdf();
        }

        return 0;
    }
}
