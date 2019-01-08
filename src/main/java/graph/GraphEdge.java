package graph;

/**
 *
 * @author vietan
 */
public class GraphEdge implements Comparable<GraphEdge> {

    private final GraphNode source;
    private final GraphNode target;
    private double weight;

    public GraphEdge(GraphNode s, GraphNode t) {
        this(s, t, 0.0);
    }

    public GraphEdge(GraphNode s, GraphNode t, double weight) {
        this.source = s;
        this.target = t;
        this.weight = weight;
    }

    public GraphEdge reverse() {
        return new GraphEdge(target, source, weight);
    }

    public GraphNode getSource() {
        return this.source;
    }

    public GraphNode getTarget() {
        return this.target;
    }

    public void setWeight(double w) {
        this.weight = w;
    }

    public double getWeight() {
        return this.weight;
    }

    @Override
    public int compareTo(GraphEdge otherEdge) {
        return Double.compare(this.weight, otherEdge.weight);
    }

    @Override
    public boolean equals(Object otherObject) {
        // Not strictly necessary, but often a good optimization
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof GraphEdge)) {
            return false;
        }
        GraphEdge otherA = (GraphEdge) otherObject;
        return this.source.equals(otherA.source)
                && this.target.equals(otherA.target);
    }

    @Override
    public int hashCode() {
        return (source.hashCode() + target.hashCode());
    }

    @Override
    public String toString() {
        return source.toString()
                + " -> " + target.toString()
                + " (" + weight + ")";
    }
}
