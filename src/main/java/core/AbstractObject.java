package core;

import java.util.HashMap;
import java.util.Set;

/**
 *
 * @author vietan
 */
public abstract class AbstractObject<I> {

    protected final I id;
    protected HashMap<String, String> properties;

    public AbstractObject(I id) {
        this.id = id;
        this.properties = new HashMap<String, String>();
    }

    public I getId() {
        return this.id;
    }

    public void addProperty(String propName, String propValue) {
        if (this.properties.containsKey(propName)) {
            if (!this.properties.get(propName).equals(propValue)) {
                System.out.println("[WARNING] Adding to existing property"
                        + ". object id = " + id
                        + ". property = " + propName
                        + ". current value = " + properties.get(propName)
                        + ". new value = " + propValue);
            }
        }
        this.properties.put(propName, propValue);
    }

    public String getProperty(String propName) {
        return this.properties.get(propName);
    }

    public boolean hasProperty(String propName) {
        return this.properties.containsKey(propName);
    }

    public Set<String> getPropertyNames() {
        return this.properties.keySet();
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("ID ").append(id).append("\n");
        for (String attr : getPropertyNames()) {
            str.append(">>> ").append(attr).append(": ").append(getProperty(attr));
        }
        return str.toString();
    }
}
