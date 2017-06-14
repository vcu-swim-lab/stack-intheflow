package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;

import java.util.Optional;

/**
 * Created by chase on 2/13/17.
 */
public class VarScorer extends AbstractScorer {

    public VarScorer(TermStatComponent statComponent) {
        super(statComponent);
    }

    @Override
    public double score(String term) {

        Optional<Stat> termStatOptional = statComponent.getTermStat(term);

        if (termStatOptional.isPresent()) {
            Stat termStat = termStatOptional.get();

            return (1 + Math.log10(termStat.getCtf())) * termStat.getIdf() / statComponent.getDocCount();
        }

        return 0;
    }
}
