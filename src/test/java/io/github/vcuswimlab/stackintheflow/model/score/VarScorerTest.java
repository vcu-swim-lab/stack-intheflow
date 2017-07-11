package io.github.vcuswimlab.stackintheflow.model.score;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.Optional;

import static org.junit.Assert.assertEquals;
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
                .thenReturn(Optional.of(new Stat(25, 20, 5, 5)));
        Mockito.when(termStatComponent.getTermStat("term2"))
                .thenReturn(Optional.empty());
        Mockito.when(termStatComponent.getTermCount()).thenReturn(200L);
        Mockito.when(termStatComponent.getDocCount()).thenReturn(300L);
    }

    @Before
    public void setUp() throws Exception {
        varScorer = new VarScorer(termStatComponent);
    }

    @Test
    public void testScore() throws Exception {
        assertEquals(0.0399, varScorer.score("term1"), 0.001);
    }

    @Test
    public void testScoreEmpty() throws Exception {
        assertEquals(0.0, varScorer.score("term2"), 0.001);
    }
}