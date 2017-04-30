package io.github.vcuswimlab.stackintheflow.model;

import jersey.repackaged.com.google.common.collect.ImmutableList;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by chase on 4/25/17.
 */
public class QuestionTest {

    private Question question;

    @Before
    public void setUp() throws Exception {
        question = new Question(ImmutableList.of("Tag1", "Tag2"),
                "myBody",
                "myExcerpt",
                "myTitle",
                "myLink");
    }

    @Test
    public void testFixNulls() throws Exception {

        Question nullQuestion = new Question();
        nullQuestion.fixNulls();

        assertTrue(nullQuestion.getTags().isEmpty());
        assertEquals("", nullQuestion.getBody());
        assertEquals("", nullQuestion.getExcerpt());
        assertEquals("", nullQuestion.getTitle());
        assertEquals("http://www.stackoverflow.com/", nullQuestion.getLink());
    }

    @Test
    public void testGetTags() throws Exception {
        assertEquals(ImmutableList.of("Tag1", "Tag2"), question.getTags());
    }

    @Test
    public void testSetTags() throws Exception {
        question.setTags(ImmutableList.of("Tag3", "Tag4"));
        assertEquals(ImmutableList.of("Tag3", "Tag4"), question.getTags());
    }

    @Test
    public void testGetBody() throws Exception {
        assertEquals("myBody", question.getBody());
    }

    @Test
    public void testSetBody() throws Exception {
        question.setBody("myOtherBody");
        assertEquals("myOtherBody", question.getBody());
    }

    @Test
    public void testGetExcerpt() throws Exception {
        assertEquals("myExcerpt", question.getExcerpt());
    }

    @Test
    public void testSetExcerpt() throws Exception {
        question.setExcerpt("myOtherExcerpt");
        assertEquals("myOtherExcerpt", question.getExcerpt());
    }

    @Test
    public void testGetTitle() throws Exception {
        assertEquals("myTitle", question.getTitle());
    }

    @Test
    public void testSetTitle() throws Exception {
        question.setTitle("myOtherTitle");
        assertEquals("myOtherTitle", question.getTitle());
    }

    @Test
    public void testGetLink() throws Exception {
        assertEquals("myLink", question.getLink());
    }

    @Test
    public void testSetLink() throws Exception {
        question.setLink("myOtherLink");
        assertEquals("myOtherLink", question.getLink());
    }

    @Test
    public void testIsExpanded() throws Exception {
        assertFalse(question.isExpanded());
    }

    @Test
    public void testToggleExpanded() throws Exception {
        assertFalse(question.isExpanded());
        question.toggleExpanded();
        assertTrue(question.isExpanded());
        question.toggleExpanded();
        assertFalse(question.isExpanded());
    }

    @Test
    public void testToString() throws Exception {
        assertEquals("Question{tags=[Tag1, Tag2], body='myBody', excerpt='myExcerpt', " +
                        "title='myTitle', link=myLink}",
                question.toString());
    }

    @Test
    public void testGetTagsAsFormattedString() throws Exception {
        assertEquals("[Tag1] [Tag2] ", question.getTagsAsFormattedString());
    }

    @Test
    public void testGetTagsAsFormattedStringNull() throws Exception {
        question.setTags(null);
        assertEquals("", question.getTagsAsFormattedString());
    }
}