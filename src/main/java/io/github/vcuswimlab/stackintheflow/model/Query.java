package io.github.vcuswimlab.stackintheflow.model;

import java.util.HashMap;
import java.util.Map;

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

        ORDER("oder"),
        SORT("sort"),
        ACCEPTED("accepted"),
        CLOSED("closed"),
        MIGRATED("migrated"),
        NOTICE("notice"),
        WIKI("wiki");

        private final String label;

        Component(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public Query(String q) {
        this(new HashMap<>(), q);
    }

    public Query(Map<String, String> components, String q) {
        this.components = components;
        set(Component.Q,q);
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

    public String get(String compoment) {
        return components.get(compoment);
    }
}
