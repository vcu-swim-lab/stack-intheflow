package sampling.util;

import java.util.ArrayList;

/**
 * Implementation of a table in the Chinese restaurant process metaphor. This
 * explicitly stores the IDs of all customers sitting at this table.
 *
 * @author vietan
 */
public class FullTable<C, M> {

    protected final int index;
    protected ArrayList<C> customers;
    protected M content;

    public FullTable(int index, M content) {
        this.index = index;
        this.content = content;
        this.customers = new ArrayList<C>();
    }

    public ArrayList<C> getCustomers() {
        return this.customers;
    }

    public boolean isEmpty() {
        return this.customers.isEmpty();
    }

    public int getNumCustomers() {
        return this.customers.size();
    }

    /**
     * Remove a customer from this table. This should only be used through the
     * Restaurant object since this will change the total number of customers
     * stored in the restaurant containing this table.
     *
     * @param customer The customer to be removed
     */
    protected void removeCustomer(C customer) {
        if (!this.customers.contains(customer)) {
            throw new RuntimeException("This table does not contain the given customer. "
                    + this.toString());
        }
        this.customers.remove(customer);
    }

    /**
     * Add a customer to this table. This should only be used through the
     * Restaurant object since this will change the total number of customers
     * stored in the restaurant containing this table.
     *
     * @param customer The customer to be added
     */
    protected void addCustomer(C customer) {
        this.customers.add(customer);
    }

    public M getContent() {
        return content;
    }

    public void setContent(M content) {
        this.content = content;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("Table ").append(index);
        str.append(". # customers = ").append(getNumCustomers());
        return str.toString();
    }
}