package sampling.util;

import java.util.Collection;
import java.util.HashMap;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 *
 * @author vietan
 */
public class Restaurant<T extends FullTable<C, M>, C, M> {

    public static final int EMPTY_TABLE_INDEX = -1;
    private HashMap<Integer, T> activeTables;
    private SortedSet<Integer> inactiveTables;
    private int totalNumCustomers;

    public Restaurant() {
        this.totalNumCustomers = 0;
        this.activeTables = new HashMap<Integer, T>();
        this.inactiveTables = new TreeSet<Integer>();
    }

    public boolean isEmpty() {
        return this.activeTables.isEmpty();
    }

    public void fillInactiveTableIndices() {
        int maxTableIndex = -1;
        for (T table : this.getTables()) {
            if (table.getIndex() > maxTableIndex) {
                maxTableIndex = table.getIndex();
            }
        }

        this.inactiveTables = new TreeSet<Integer>();
        for (int i = 0; i < maxTableIndex; i++) {
            if (!isActive(i)) {
                this.inactiveTables.add(i);
            }
        }
    }

    /**
     * Add a customer to a table in this restaurant. This will also increment
     * the number of total customers in this restaurant.
     *
     * @param customer The customer to be added
     * @param tableIndex The table index
     */
    public void addCustomerToTable(C customer, int tableIndex) {
        this.totalNumCustomers++;
        this.getTable(tableIndex).addCustomer(customer);
    }

    /**
     * Remove a customer from a table in this restaurant. This will also
     * decrement the number of total customers in this restaurant.
     *
     * @param customer The customer to be removed
     * @param tableIndex The table index
     */
    public void removeCustomerFromTable(C customer, int tableIndex) {
        this.totalNumCustomers--;
        this.getTable(tableIndex).removeCustomer(customer);
    }

    /**
     * Return the next available table index
     */
    public int getNextTableIndex() {
        int newTableIndex;
        if (this.inactiveTables.isEmpty()) {
            newTableIndex = this.activeTables.size();
        } else {
            newTableIndex = this.inactiveTables.first();
        }
        return newTableIndex;
    }

    /**
     * Return a table
     *
     * @param index The table index
     */
    public T getTable(int index) {
        return this.activeTables.get(index);
    }

    /**
     * Remove an existing table
     *
     * @param tableIndex Table index
     */
    public void removeTable(int tableIndex) {
        if (!this.activeTables.containsKey(tableIndex)) {
            throw new RuntimeException("Removing table that does not exist. " + tableIndex);
        }
        this.totalNumCustomers -= this.getTable(tableIndex).getNumCustomers();
        this.inactiveTables.add(tableIndex);
        this.activeTables.remove(tableIndex);
    }

    /**
     * Add a new table
     *
     * @param table The new table
     */
    public void addTable(T table) {
        int tableIndex = table.getIndex();
        if (this.activeTables.containsKey(tableIndex)) {
            throw new RuntimeException("Exception while creating new table. Table "
                    + tableIndex + " already exists.");
        }
        if (inactiveTables.contains(tableIndex)) {
            this.inactiveTables.remove(tableIndex);
        }
        this.activeTables.put(tableIndex, table);
    }

    /**
     * Return the total number of customers sitting in this restaurant
     */
    public int getTotalNumCustomers() {
        return this.totalNumCustomers;
    }

    /**
     * Get the number of active tables
     */
    public int getNumTables() {
        return this.activeTables.size();
    }

    /**
     * Check whether this restaurant contains a given table
     *
     * @param tableIndex The table index
     */
    public boolean isActive(int tableIndex) {
        return this.activeTables.containsKey(tableIndex);
    }

    /**
     * Return the set of active tables
     */
    public Collection<T> getTables() {
        return this.activeTables.values();
    }

    public double getJointProbabilityAssignments(double alpha) {
        double llh = this.getNumTables() * Math.log(alpha);
        for (FullTable<C, M> table : this.activeTables.values()) {
            for (int n = 1; n < table.getNumCustomers(); n++) {
                llh += Math.log(n);
            }
        }
        for (int x = 1; x <= this.totalNumCustomers; x++) {
            llh -= Math.log(x - 1 + alpha);
        }
        return llh;
    }

    public void validate(String msg) {
        int tnc = 0;
        for (FullTable<C, M> table : this.getTables()) {
            tnc += table.getNumCustomers();
        }
        if (tnc != this.totalNumCustomers) {
            throw new RuntimeException(msg + ": Total number of customers mismatched. "
                    + tnc + " vs. " + this.totalNumCustomers);
        }

        int maxTableIndex = -1;
        for (T table : this.getTables()) {
            if (table.getIndex() > maxTableIndex) {
                maxTableIndex = table.getIndex();
            }
        }

        for (int i = 0; i < maxTableIndex; i++) {
            if (!isActive(i) && !inactiveTables.contains(i)) {
                throw new RuntimeException(msg + ". Inactive index has not been updated."
                        + ". Table " + toString()
                        + ". index " + i + " is neither active nor inactive");
            }
        }
    }
}
