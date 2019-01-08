package graph;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class PrimMST<C> {

    private GraphNode<C> root;
    private UndirectedGraph<C> graph;
    private Set<GraphNode<C>> visited;

    public PrimMST(GraphNode<C> root, UndirectedGraph<C> graph) {
        this.root = root;
        this.graph = graph;
        this.visited = new HashSet<GraphNode<C>>();
    }

    public DirectedGraph<C> getMinimumSpanningTree() {
        DirectedGraph<C> mst = new DirectedGraph<C>();

        PriorityQueue<GraphEdge> frontTier = new PriorityQueue<GraphEdge>();
        this.visited.add(root);
        for (GraphEdge edge : graph.getEdges(root)) {
            frontTier.add(edge);
        }

        while (visited.size() < graph.getNumNodes()) {
            GraphEdge maxConnector = null;
            boolean found = false;
            while (!found) {
                maxConnector = frontTier.poll();
                GraphNode<C> source = maxConnector.getSource();
                GraphNode<C> target = maxConnector.getTarget();

                if (visited.contains(source) && visited.contains(target)) {
                    continue;
                } else if (!visited.contains(source) && !visited.contains(target)) {
                    throw new RuntimeException("Unconnected edge: " + maxConnector);
                } else {
                    found = true;
                    if (visited.contains(source)) {
                        visited.add(target);
                        for (GraphEdge edge : graph.getEdges(target)) {
                            if (edge.getTarget().equals(target)) {
                                frontTier.add(edge.reverse());
                            } else {
                                frontTier.add(edge);
                            }
                        }
                    } else {
                        maxConnector = maxConnector.reverse();
                        visited.add(source);
                        for (GraphEdge edge : graph.getEdges(source)) {
                            if (edge.getTarget().equals(source)) {
                                frontTier.add(edge.reverse());
                            } else {
                                frontTier.add(edge);
                            }
                        }
                    }
                }
            }
            mst.addEdge(maxConnector);
        }
        return mst;
    }

    public static void main(String[] args) {
        UndirectedGraph<Integer> graph = new UndirectedGraph<Integer>();
        GraphNode<Integer>[] nodes = new GraphNode[5];
        for (int ii = 0; ii < nodes.length; ii++) {
            nodes[ii] = new GraphNode<Integer>(ii);
        }

        graph.addEdge(new GraphEdge(nodes[0], nodes[1], -10));
        graph.addEdge(new GraphEdge(nodes[0], nodes[2], -11));
        graph.addEdge(new GraphEdge(nodes[1], nodes[2], -1));
        graph.addEdge(new GraphEdge(nodes[1], nodes[3], -12));
        graph.addEdge(new GraphEdge(nodes[1], nodes[4], -13));
        graph.addEdge(new GraphEdge(nodes[2], nodes[4], -4));

        PrimMST<Integer> prim = new PrimMST<Integer>(nodes[0], graph);
        DirectedGraph<Integer> mst = prim.getMinimumSpanningTree();
        System.out.println(mst.toString());
    }
}
