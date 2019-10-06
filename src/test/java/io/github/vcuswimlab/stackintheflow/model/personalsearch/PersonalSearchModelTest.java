package io.github.vcuswimlab.stackintheflow.model.personalsearch;

import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.tags.TagStatComponent;
import io.github.vcuswimlab.stackintheflow.model.Question;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PersonalSearchModelTest {
    private TagStatComponent tagStatComponent;

    Map<String, Integer> userStateMap = new HashMap<>();
    private PersonalSearchModel searchModel;

    @Before
    public void setup() {
        tagStatComponent = mock(TagStatComponent.class);
        when(tagStatComponent.getTagStat("test")).thenReturn(Optional.of(new Stat(42, 42, 42.42, 42.42)));
        this.searchModel = new PersonalSearchModel(tagStatComponent, userStateMap);
    }

    @Test
    public void increaseTags() {
        this.searchModel.increaseTags(Collections.singletonList("test"));
        assertTrue(this.searchModel.getUserStatMap().containsKey("test"));
        assertEquals(this.searchModel.getUserStatMap().size(), 1);
    }

    @Test
    public void increaseTagsUsermapNewKey() {
        this.searchModel.increaseTags(Collections.singletonList("test"), 42);
        assertTrue(this.searchModel.getUserStatMap().containsKey("test"));
        assertEquals(this.searchModel.getUserStatMap().get("test"), Optional.of(42).get());
    }

    @Test
    public void increaseTagsUsermapHasKey() {
        this.searchModel.increaseTags(Collections.singletonList("test"));
        this.searchModel.increaseTags(Collections.singletonList("test"), 42);
        assertTrue(this.searchModel.getUserStatMap().containsKey("test"));
        assertEquals(this.searchModel.getUserStatMap().get("test"), Optional.of(43).get());
    }

    @Test
    public void rankQuestionList() {
        List<Question> questions = new ArrayList<>();
//        questions.add(new Question(null, null, null, null, null));
        questions.add(new Question(Arrays.asList("test", "test1", "test3"), "test", "test", "test", "test"));
        questions.add(new Question(Collections.emptyList(), "test2", "test2", "test2", "test2"));
        questions.add(new Question(Arrays.asList("", "test"), "test1", "test2", "test3", "test3"));


        List<Question> ranked = this.searchModel.rankQuestionList(questions);
        assertEquals("test", ranked.get(0).getTitle());
        assertEquals("test2", ranked.get(1).getTitle());
        assertEquals("test3", ranked.get(2).getTitle());
    }

    @Test
    public void rankQuestionListUserStatMapPreexisting() {
        List<Question> questions = new ArrayList<>();
//        questions.add(new Question());
        questions.add(new Question(Arrays.asList("test", "test1", "test3"), "test", "test", "test", "test"));
        questions.add(new Question(Collections.emptyList(), "test2", "test2", "test2", "test2"));
        questions.add(new Question(Arrays.asList("", "test"), "test1", "test2", "test3", "test3"));
        this.searchModel.increaseTags(Collections.singletonList("test"));

        List<Question> ranked = this.searchModel.rankQuestionList(questions);
        assertEquals("test", ranked.get(0).getTitle());
        assertEquals("test3", ranked.get(1).getTitle());
        assertEquals("test2", ranked.get(2).getTitle());
    }
}