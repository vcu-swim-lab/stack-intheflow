package graph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class DirectedGraph<C> {

    private Map<GraphNode<C>, List<GraphEdge>> outEdges;
    private Map<GraphNode<C>, List<GraphEdge>> inEdges;

    public DirectedGraph() {
        this.outEdges = new HashMap<GraphNode<C>, List<GraphEdge>>();
        this.inEdges = new HashMap<GraphNode<C>, List<GraphEdge>>();
    }

    public void addEdge(GraphNode<C> source, GraphNode<C> target, double weight) {
        GraphEdge edge = new GraphEdge(source, target, weight);
        this.addEdge(edge);
    }

    public void addEdge(GraphEdge edge) {
        GraphNode<C> source = edge.getSource();
        List<GraphEdge> outList;
        if (!outEdges.containsKey(source)) {
            outList = new ArrayList<GraphEdge>();
            outEdges.put(source, outList);
        } else {
            outList = outEdges.get(source);
        }
        outList.add(edge);

        GraphNode<C> target = edge.getTarget();
        List<GraphEdge> inList;
        if (!inEdges.containsKey(target)) {
            inList = new ArrayList<GraphEdge>();
            inEdges.put(target, inList);
        } else {
            inList = inEdges.get(target);
        }
        inList.add(edge);
    }

    public void removeEdge(GraphEdge edge) {
        GraphNode<C> source = edge.getSource();
        List<GraphEdge> outList = outEdges.get(source);
        outList.remove(edge);
        if (outList.isEmpty()) {
            outEdges.remove(source);
        }

        GraphNode<C> target = edge.getTarget();
        List<GraphEdge> inList = inEdges.get(target);
        inList.remove(edge);
        if (inList.isEmpty()) {
            inEdges.remove(target);
        }
    }

    public boolean hasOutEdges(GraphNode<C> node) {
        return this.outEdges.get(node) != null;
    }

    public boolean hasInEdges(GraphNode<C> node) {
        return this.inEdges.get(node) != null;
    }

    public List<GraphEdge> getOutEdges(GraphNode<C> node) {
        return this.outEdges.get(node);
    }

    public List<GraphEdge> getInEdges(GraphNode<C> node) {
        return this.inEdges.get(node);
    }

    public Set<GraphNode<C>> getSourceNodeSet() {
        return outEdges.keySet();
    }

    public Set<GraphNode<C>> getTargetNodeSet() {
        return inEdges.keySet();
    }

    public Collection<GraphEdge> getAllEdges() {
        List<GraphEdge> edges = new ArrayList<GraphEdge>();
        for (List<GraphEdge> e : outEdges.values()) {
            edges.addAll(e);
        }
        return edges;
    }

    public void clear() {
        if (this.outEdges != null) {
            outEdges.clear();
        }
        if (this.inEdges != null) {
            inEdges.clear();
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (GraphEdge edge : this.getAllEdges()) {
            str.append(edge.toString()).append("\n");
        }
        return str.toString();
    }
}
