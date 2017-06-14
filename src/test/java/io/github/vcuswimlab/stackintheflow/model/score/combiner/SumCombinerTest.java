package io.github.vcuswimlab.stackintheflow.model.score.combiner;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.terms.TermStatComponent;
import io.github.vcuswimlab.stackintheflow.model.score.IctfScorer;
import jersey.repackaged.com.google.common.collect.ImmutableList;
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
public class SumCombinerTest {

    private static TermStatComponent termStatComponent;
    private SumCombiner sumCombiner;

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
        sumCombiner = new SumCombiner(ImmutableList.of(new IctfScorer(termStatComponent),
                new IctfScorer(termStatComponent),
                new IctfScorer(termStatComponent)));
    }

    @Test
    public void testGenerateCumulativeScore() throws Exception {
        assertEquals(15.0, sumCombiner.generateCumulativeScore("term1"), 0.001);
    }

}