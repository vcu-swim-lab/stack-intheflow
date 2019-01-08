package util.evaluation;

/**
 *
 * @author vietan
 */
public class Measurement {

    private final String name;
    private double value;

    public Measurement(String name, double value) {
        this.name = name;
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return this.name + "\t" + this.value;
    }
}
