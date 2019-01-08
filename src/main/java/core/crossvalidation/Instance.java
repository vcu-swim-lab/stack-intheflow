package core.crossvalidation;

/**
 *
 * @author vietan
 */
public class Instance<I> {

    private final I id;

    public Instance(I id) {
        this.id = id;
    }

    public I getId() {
        return this.id;
    }
    
    @Override
    public String toString() {
        return this.id.toString();
    }
}
