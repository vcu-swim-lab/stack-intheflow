package io.github.vcuswimlab.stackintheflow.model;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by Jeet on 10/18/2016.
 */
public class Query {
    private Map<String, String> components;

    public enum Component {

        Q("q"),
        TAGGED("tagged"),

        PAGE("page"),
        PAGESIZE("pagesize"),
        FROMDATE("fromdate"),
        TODATE("todate"),
        MIN("min"),
        MAX("max"),
        ANSWERS("answers"),
        BODY("body"),
        NOTTAGGED("nottagged"),
        TITLE("title"),
        USER("user"),
        URL("url"),
        VIEWS("views"),

        ORDER("order"),
        SORT("sort"),
        ACCEPTED("accepted"),
        CLOSED("closed"),
        MIGRATED("migrated"),
        NOTICE("notice"),
        WIKI("wiki"),

        SITE("site"),
        FILTER("filter");

        private final String label;

        Component(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public Query() {
        components = new HashMap<>();
    }

    public Query(String site) {
        this(new HashMap<>(), site);
    }

    public Query(Map<String, String> components, String site) {
        this.components = components;
        set(Component.SITE, site);
    }

    public Query set(Component component, String value) {
        components.put(component.toString(), value);
        return this;
    }

    public Query set(String component, String value) {
        components.put(component, value);
        return this;
    }

    public String get(Component component) {
        return components.get(component.toString());
    }

    public Set<String> getComponents() {
        return components.keySet();
    }

    public Map<String, String> getComponentMap() {
        return Collections.unmodifiableMap(components);
    }
}
