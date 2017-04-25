package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.TermStat;
import io.github.vcuswimlab.stackintheflow.controller.component.TermStatComponent;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.Optional;

import static org.mockito.Mockito.mock;

/**
 * Created by chase on 4/25/17.
 */
public class VarScorerTest {

    private static TermStatComponent termStatComponent;

    private VarScorer varScorer;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        termStatComponent = mock(TermStatComponent.class);
        Mockito.when(termStatComponent.getTermStat("term1"))
                .thenReturn(Optional.of(new TermStat(5, 5, 5, 5)));
        Mockito.when(termStatComponent.getTermStat("term2"))
                .thenReturn(Optional.empty());
    }

    @Before
    public void setUp() throws Exception {
        varScorer = new VarScorer(termStatComponent);
    }

    @Test
    public void score() throws Exception {
        System.out.println(varScorer.score("term1"));
    }

}