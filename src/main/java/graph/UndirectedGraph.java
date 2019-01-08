package graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class UndirectedGraph<C> {

    private Map<GraphNode<C>, List<GraphEdge>> adjacencyList;

    public UndirectedGraph() {
        this.adjacencyList = new HashMap<GraphNode<C>, List<GraphEdge>>();
    }

    public List<GraphEdge> getEdges(GraphNode<C> node) {
        return this.adjacencyList.get(node);
    }

    public Set<GraphNode<C>> getNodes() {
        return this.adjacencyList.keySet();
    }

    public int getNumNodes() {
        return this.adjacencyList.size();
    }

    public void addEdge(GraphEdge edge) {
        this.addAdjacentNode(edge.getSource(), edge);
        this.addAdjacentNode(edge.getTarget(), edge);
    }

    private void addAdjacentNode(GraphNode<C> node, GraphEdge edge) {
        List<GraphEdge> nodeAdj = adjacencyList.get(node);
        if (nodeAdj == null) {
            nodeAdj = new ArrayList<GraphEdge>();
        }
        nodeAdj.add(edge);
        adjacencyList.put(node, nodeAdj);
    }

    public void clear() {
        this.adjacencyList.clear();
    }
}
