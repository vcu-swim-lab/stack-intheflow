package io.github.vcuswimlab.stackintheflow.model;

import jersey.repackaged.com.google.common.collect.ImmutableMap;
import jersey.repackaged.com.google.common.collect.ImmutableSet;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by chase on 4/25/17.
 */
public class QueryTest {

    private Query query;

    @Before
    public void setUp() throws Exception {
        query = new Query("mySite");
    }

    @Test
    public void testEmptyConstructor() {
        query = new Query();
        assertTrue(query.getComponentMap().isEmpty());
    }

    @Test
    public void testSetComponent() throws Exception {
        assertTrue(query.set(Query.Component.Q, "myQ") != null);
        assertEquals("myQ", query.get(Query.Component.Q));
    }

    @Test
    public void testSetString() throws Exception {
        assertTrue(query.set("q", "myQ") != null);
        assertEquals("myQ", query.get(Query.Component.Q));
    }

    @Test
    public void get() throws Exception {
        assertEquals("mySite", query.get(Query.Component.SITE));
    }

    @Test
    public void getComponents() throws Exception {
        assertEquals(ImmutableSet.of("site"), query.getComponents());
    }

    @Test
    public void getComponentMap() throws Exception {
        assertEquals(ImmutableMap.of("site", "mySite"), query.getComponentMap());
    }

}