package graph;

/**
 *
 * @author vietan
 */
public class GraphNode<C> {

    private final C id;

    public GraphNode(C id) {
        this.id = id;
    }

    public C getId() {
        return this.id;
    }

    @Override
    public boolean equals(Object otherObject) {
        // Not strictly necessary, but often a good optimization
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof GraphNode)) {
            return false;
        }
        GraphNode otherA = (GraphNode) otherObject;
        return this.id.equals(otherA.getId());
    }

    @Override
    public int hashCode() {
        return this.id.hashCode();
    }

    @Override
    public String toString() {
        return this.id.toString();
    }
}
