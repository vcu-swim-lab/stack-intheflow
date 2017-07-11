package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;

/**
 * Created by Chase on 2/11/2017.
 */
public abstract class AbstractScorer implements Scorer {

    protected TermStatComponent statComponent;

    public AbstractScorer(TermStatComponent statComponent) {
        this.statComponent = statComponent;
    }

}
