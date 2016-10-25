package io.github.vcuswimlab.stackintheflow.model;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Jeet on 10/18/2016.
 */
public class Query {
    private Map<String, String> components;

    public Query(String q) {
        this(new HashMap<>(), q);
    }

    public Query(Map<String, String> components, String q) {
        this.components = components;
        set("q",q);
    }

    public Query set(String componentName, String value) {
        components.put(componentName, value);
        return this;
    }

    public String get(String name) {
        return components.get(name);
    }
}
