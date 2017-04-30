package io.github.vcuswimlab.stackintheflow.model;

import jersey.repackaged.com.google.common.collect.ImmutableList;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by chase on 4/25/17.
 */
public class JerseyResponseTest {

    private JerseyResponse jerseyResponse;
    private List<Question> items = ImmutableList.of(new Question(ImmutableList.of("Tag1", "Tag2"),
                    "myBody",
                    "myExcerpt",
                    "myTitle",
                    "myLink"),
            new Question(ImmutableList.of("Tag3", "Tag4"),
                    "myOtherBody",
                    "myOtherExcerpt",
                    "myOtherTitle",
                    "myOtherLink"));

    @Before
    public void setUp() throws Exception {
        jerseyResponse = new JerseyResponse(items, true, 100, 10);
    }

    @Test
    public void testEmptyConstructor() {
        jerseyResponse = new JerseyResponse();

        assertNull(jerseyResponse.getItems());
        assertFalse(jerseyResponse.isHas_more());
        assertEquals(0, jerseyResponse.getQuota_max());
        assertEquals(0, jerseyResponse.getQuota_remaining());
    }

    @Test
    public void testGetItems() throws Exception {
        assertEquals(items, jerseyResponse.getItems());
    }

    @Test
    public void testSetItems() throws Exception {
        jerseyResponse.setItems(new ArrayList<>());
        assertTrue(jerseyResponse.getItems().isEmpty());
    }

    @Test
    public void testIsHas_more() throws Exception {
        assertTrue(jerseyResponse.isHas_more());
    }

    @Test
    public void testSetHas_more() throws Exception {
        jerseyResponse.setHas_more(false);
        assertFalse(jerseyResponse.isHas_more());
    }

    @Test
    public void testGetQuota_max() throws Exception {
        assertEquals(100, jerseyResponse.getQuota_max());
    }

    @Test
    public void testSetQuota_max() throws Exception {
        jerseyResponse.setQuota_max(50);
        assertEquals(50, jerseyResponse.getQuota_max());
    }

    @Test
    public void testGetQuota_remaining() throws Exception {
        assertEquals(10, jerseyResponse.getQuota_remaining());
    }

    @Test
    public void testSetQuota_remaining() throws Exception {
        jerseyResponse.setQuota_remaining(5);
        assertEquals(5, jerseyResponse.getQuota_remaining());
    }

    @Test
    public void testToString() throws Exception {
        assertEquals("JerseyResponse{items=[Question{tags=[Tag1, Tag2], body='myBody', " +
                "excerpt='myExcerpt', title='myTitle', link=myLink}, " +
                "Question{tags=[Tag3, Tag4], body='myOtherBody', excerpt='myOtherExcerpt', " +
                "title='myOtherTitle', link=myOtherLink}], has_more=true, quota_max=100, " +
                "quota_remaining=10}", jerseyResponse.toString());
    }

}