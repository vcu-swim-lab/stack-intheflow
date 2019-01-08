package util;

/**
 *
 * @author Viet-An Nguyen
 */
public class RankingItem<A> implements Comparable<RankingItem<A>> {

    private A object;
    private double primaryValue;
    private double secondaryValue;
    private int ranking;

    /**
     * Creates a new instance of Ranking
     */
    public RankingItem(A obj, double value) {
        this.object = obj;
        this.primaryValue = value;
    }

    public RankingItem(A obj, double primary, double secondary) {
        this.object = obj;
        this.primaryValue = primary;
        this.secondaryValue = secondary;
    }

    public A getObject() {
        return object;
    }

    public int getRankingOrder() {
        return this.ranking;
    }

    public void setRankingOrder(int ranking) {
        this.ranking = ranking;
    }

    public double getPrimaryValue() {
        return primaryValue;
    }

    public void setPrimaryValue(double v) {
        this.primaryValue = v;
    }

    public double getSecondaryValue() {
        return secondaryValue;
    }

    public void setSecondaryValue(double secondaryValue) {
        this.secondaryValue = secondaryValue;
    }

    // implement compareTo method of the Comparable interface to facilitate sorting
    @Override
    public int compareTo(RankingItem r) {
        if (this.primaryValue != r.primaryValue) {
            return -(Double.compare(this.getPrimaryValue(), r.getPrimaryValue()));
        } else {
            return -(Double.compare(this.getSecondaryValue(), r.getSecondaryValue()));
        }
    }

    @Override
    public int hashCode() {
        return this.object.hashCode();
    }

    // override equals method to facilitate entry searching
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if ((obj == null) || (this.getClass() != obj.getClass())) {
            return false;
        }
        RankingItem r = (RankingItem) (obj);

        return (this.object.equals(r.getObject()));
    }

    @Override
    public String toString() {
        return (this.object + "\t" + Double.toString(primaryValue));
    }
}