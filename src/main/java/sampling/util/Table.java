package sampling.util;

/**
 * Implementation of a table in the Chinese restaurant process, in which only
 * the number of customers is stored. If the actual customers are needed, use
 * FullTable instead.
 *
 * @author vietan
 */
public class Table<M> {

    protected final int index;
    protected M content;
    protected int numCustomers;

    public Table(int idx, M content) {
        this.index = idx;
        this.content = content;
        this.numCustomers = 0;
    }

    public int getIndex() {
        return this.index;
    }

    public M getContent() {
        return this.content;
    }

    public int getNumCustomers() {
        return this.numCustomers;
    }

    public void changeNumCustomers(int delta) {
        this.numCustomers += delta;
    }

    public void incrementNumCustomers() {
        this.numCustomers++;
    }

    public void decrementNumCustomers() {
        this.numCustomers--;
    }

    public void validate(String msg) {
        if (this.numCustomers < 0) {
            throw new RuntimeException("Negative nunmber of customers. " + numCustomers);
        }
    }
}
