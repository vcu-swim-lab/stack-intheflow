package io.github.vcuswimlab.stackintheflow.model;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Jeet on 10/18/2016.
 */
public class Query {
    private Map<Component, String> components;

    public enum Component {
        Q,
        TAGGED,

        PAGE,
        PAGESIZE,
        FROMDATE,
        TODATE,
        MIN,
        MAX,
        ANSWERS,
        BODY,
        NOTTAGGED,
        TITLE,
        USER,
        URL,
        VIEWS,

        ORDER,
        SORT,
        ACCEPTED,
        CLOSED,
        MIGRATED,
        NOTICE,
        WIKI,
    }

    public Query(String q) {
        this(new HashMap<>(), q);
    }

    public Query(Map<Component, String> components, String q) {
        this.components = components;
        set(Component.Q,q);
    }

    public Query set(Component component, String value) {
        components.put(component, value);
        return this;
    }

    public String get(String name) {
        return components.get(name);
    }
}
