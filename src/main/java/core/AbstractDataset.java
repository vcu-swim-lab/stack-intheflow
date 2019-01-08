package core;

/**
 *
 * @author vietan
 */
public abstract class AbstractDataset extends AbstractRunner {
    
    public static final String TRAIN_PREFIX = "tr_";
    public static final String DEVELOPE_PREFIX = "de_";
    public static final String TEST_PREFIX = "te_";

    protected final String name;

    public AbstractDataset(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public String getFolder() {
        return this.name + "/";
    }
}
