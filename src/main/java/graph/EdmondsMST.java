package graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import sampling.util.TreeNode;
import util.RankingItem;

/**
 * Implement Edmonds' algorithm to find minimum spanning tree in a directed
 * graph
 *
 * @author vietan
 */
public class EdmondsMST<C> {

    private GraphNode<C> root;
    private DirectedGraph<C> graph;

    public EdmondsMST(GraphNode<C> root, DirectedGraph<C> graph) {
        this.root = root;
        this.graph = graph;
    }

    public TreeNode<TreeNode, C> createTree(DirectedGraph<C> graphTree) {
        HashMap<C, TreeNode<TreeNode, C>> nodeMap = new HashMap<C, TreeNode<TreeNode, C>>();
        TreeNode<TreeNode, C> treeRoot = new TreeNode<TreeNode, C>(0, 0, root.getId(), null);
        nodeMap.put(root.getId(), treeRoot);

        Queue<GraphNode<C>> queue = new LinkedList<GraphNode<C>>();
        queue.add(root);
        while (!queue.isEmpty()) {
            GraphNode<C> node = queue.poll();
            TreeNode<TreeNode, C> treeNode = nodeMap.get(node.getId());

            if (!graphTree.hasOutEdges(node)) {
                continue;
            }
            for (GraphEdge edge : graphTree.getOutEdges(node)) {
                GraphNode<C> graphNode = edge.getTarget();
                TreeNode<TreeNode, C> childNode = new TreeNode<TreeNode, C>(
                        treeNode.getNextChildIndex(),
                        treeNode.getLevel() + 1,
                        graphNode.getId(), treeRoot);
                treeNode.addChild(childNode.getIndex(), childNode);
                queue.add(graphNode);
                nodeMap.put(childNode.getContent(), childNode);
            }
        }
        return treeRoot;
    }

    public DirectedGraph<C> getMinimumSpanningTree() {
        // remove in-edges of the root
        List<GraphEdge> rootInEdges = graph.getInEdges(root);
        if (rootInEdges != null) {
            for (GraphEdge rootInEdge : rootInEdges) {
                graph.removeEdge(rootInEdge);
            }
        }

        // create a graph such that for each node, only the incoming edge with
        // smallest weight is selected
        DirectedGraph<C> minGraph = new DirectedGraph<C>();
        for (GraphNode<C> node : graph.getTargetNodeSet()) {
            List<GraphEdge> nodeInEdges = graph.getInEdges(node);
            if (nodeInEdges.isEmpty()) {
                continue;
            }
            Collections.sort(nodeInEdges);
            minGraph.addEdge(nodeInEdges.get(0));
        }

        // get all cycles in the min-graph
        List<List<GraphEdge>> cycles = getCycles(minGraph);

        // break cycle if necessary
        for (List<GraphEdge> cycle : cycles) {
            breakCycle(cycle, graph, minGraph);
        }

        return minGraph;
    }

    /**
     * Check whether a given edge is a part of a cycle in a directed graph
     *
     * @param edge The edge
     * @param adjList The adjacency list representing the graph
     */
    private boolean isPartOfACycle(GraphEdge edge, DirectedGraph<C> adjList) {
        GraphNode<C> node = edge.getSource();
        Set<GraphEdge> visitedEdges = new HashSet<GraphEdge>();

        while (adjList.getInEdges(node) != null) {
            GraphEdge tempEdge = adjList.getInEdges(node).get(0);
            GraphNode<C> newNode = tempEdge.getSource();

            if (newNode.equals(edge.getTarget())) {
                return true;
            } else if (visitedEdges.contains(tempEdge)) {
                break;
            } else {
                visitedEdges.add(tempEdge);
                node = newNode;
            }
        }
        return false;
    }

    /**
     * List all cycles in a directed graph
     *
     * @param adjList The adjacency list representing the graph
     */
    private List<List<GraphEdge>> getCycles(DirectedGraph<C> adjList) {
        List<List<GraphEdge>> cycles = new ArrayList<List<GraphEdge>>();
        Set<GraphEdge> visitedEdges = new HashSet<GraphEdge>();

        for (GraphEdge candEdge : adjList.getAllEdges()) {
            if (visitedEdges.contains(candEdge)) {
                continue;
            }
            visitedEdges.add(candEdge);

            List<GraphEdge> cycle = new ArrayList<GraphEdge>();
            boolean isCycle = false;
            GraphNode<C> node = candEdge.getSource();

            while (adjList.getInEdges(node) != null) {
                GraphEdge edge = adjList.getInEdges(node).get(0);
                cycle.add(edge);
                GraphNode<C> newNode = edge.getSource();
                if (newNode.equals(candEdge.getTarget())) {
                    isCycle = true;
                    break;
                } else if (visitedEdges.contains(edge)
                        || cycle.contains(edge)) {
                    break;
                } else {
                    node = newNode;
                }
            }

            if (isCycle) {
                cycle.add(candEdge);
                cycles.add(cycle);
                for (GraphEdge edge : cycle) {
                    visitedEdges.add(edge);
                }
            }
        }
        return cycles;
    }

    /**
     * Break a cycle in subgraph. This is done by first removing an edge in the
     * cycle and replace it with another edge in the full graph.
     *
     * @param cycle The cycle
     * @param fullGraph The full graph
     * @param minGraph A subgraph of the full graph
     */
    private void breakCycle(
            List<GraphEdge> cycle,
            DirectedGraph<C> fullGraph,
            DirectedGraph<C> minGraph) {

        // edge inside the cycle having smallest weight
        Collections.sort(cycle);
        GraphEdge minInternalEdge = cycle.get(0);

        // all nodes in the cycle
        List<GraphNode<C>> cycleNodes = new ArrayList<GraphNode<C>>();
        for (GraphEdge edge : cycle) {
            cycleNodes.add(edge.getSource());
        }

        // all edges that are incoming edge of any node in the cycle
        List<GraphEdge> cycleAllInEdges = new ArrayList<GraphEdge>();
        for (GraphNode<C> node : cycleNodes) {
            for (GraphEdge e : fullGraph.getInEdges(node)) {
                if (!cycleNodes.contains(e.getSource())) {
                    cycleAllInEdges.add(e);
                }
            }
        }

        // rank external edges according to their weight differences
        ArrayList<RankingItem<GraphEdge>> rankExternalEdges
                = new ArrayList<RankingItem<GraphEdge>>();
        for (GraphEdge e : cycleAllInEdges) {
            GraphEdge ie = (GraphEdge) minGraph.getInEdges(e.getTarget()).get(0);
            double w = e.getWeight()
                    - (ie.getWeight() - minInternalEdge.getWeight());
            rankExternalEdges.add(new RankingItem<GraphEdge>(e, w));
        }
        Collections.sort(rankExternalEdges);

        // replace edges to break cycle
        for (int ii = rankExternalEdges.size() - 1; ii >= 0; ii--) {
            GraphEdge candEdge = rankExternalEdges.get(ii).getObject();

            if (isPartOfACycle(candEdge, minGraph)) // if this edge creates new cycle, ignore
            {
                continue;
            }

            GraphNode<C> target = candEdge.getTarget();
            GraphEdge removingEdge = minGraph.getInEdges(target).get(0);
            minGraph.removeEdge(removingEdge);
            minGraph.addEdge(candEdge);
            break;
        }
    }

    public static void main(String[] args) {
        GraphNode<Integer>[] nodes = new GraphNode[5];
        for (int ii = 0; ii < nodes.length; ii++) {
            nodes[ii] = new GraphNode<Integer>(ii);
        }
        DirectedGraph<Integer> graph = new DirectedGraph<Integer>();
        graph.addEdge(nodes[0], nodes[1], 0.95);
        graph.addEdge(nodes[0], nodes[2], 0.8);
        graph.addEdge(nodes[0], nodes[3], 0.9);
        graph.addEdge(nodes[0], nodes[4], 0.3);

        graph.addEdge(nodes[1], nodes[2], 0.3);
        graph.addEdge(nodes[2], nodes[1], 0.5);

        graph.addEdge(nodes[2], nodes[3], 0.1);
        graph.addEdge(nodes[3], nodes[2], 0.05);

        graph.addEdge(nodes[3], nodes[4], 0.4);
        graph.addEdge(nodes[4], nodes[3], 0.7);

        EdmondsMST<Integer> dmst = new EdmondsMST<Integer>(nodes[0], graph);
        DirectedGraph<Integer> tree = dmst.getMinimumSpanningTree();
        for (GraphEdge edge : tree.getAllEdges()) {
            System.out.println(edge.toString());
        }
    }
}
