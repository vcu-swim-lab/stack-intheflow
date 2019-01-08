package graph;

import java.util.HashMap;
import java.util.PriorityQueue;

/**
 *
 * @author vietan
 */
public class KruskalMST<C> {

    private UndirectedGraph<C> graph;

    public KruskalMST(UndirectedGraph<C> graph) {
        this.graph = graph;
    }

    public DirectedGraph<C> getMinimumSpanningTree() {
        DirectedGraph<C> mst = new DirectedGraph<C>();

        HashMap<GraphNode<C>, Tree<C>> treeMap = new HashMap<GraphNode<C>, Tree<C>>();
        int count = 0;
        for (GraphNode<C> node : graph.getNodes()) {
            Tree<C> tree = new Tree<C>(count++);
            treeMap.put(node, tree);
        }

        PriorityQueue<GraphEdge> rankEdges = new PriorityQueue<GraphEdge>();
        for (GraphNode<C> node : graph.getNodes()) {
            for (GraphEdge edge : graph.getEdges(node)) {
                rankEdges.add(edge);
            }
        }

        // debug
        System.out.println("# ranked edges: " + rankEdges.size());
        int numTrees = treeMap.size();

        while (!rankEdges.isEmpty() && numTrees > 1) {
            GraphEdge edge = rankEdges.poll();
            GraphNode<C> source = edge.getSource();
            GraphNode<C> target = edge.getTarget();

            Tree<C> sourceTree = treeMap.get(source);
            Tree<C> targetTree = treeMap.get(target);

            // debug
            System.out.println(edge.toString()
                    + ". " + treeMap.values().size()
                    + ". source: " + sourceTree.id + " (" + sourceTree.tree.getNumNodes() + ") "
                    + ". target: " + targetTree.id + " (" + targetTree.tree.getNumNodes() + ") ");

            if (sourceTree.equals(targetTree)) {
                continue;
            }

            // merge tree
            System.out.println("Merging trees ...");
            if (sourceTree.tree.getNumNodes() > targetTree.tree.getNumNodes()) {
                sourceTree.tree.addEdge(edge);
                for (GraphNode<C> n : targetTree.tree.getNodes()) {
                    for (GraphEdge e : targetTree.tree.getEdges(n)) {
                        sourceTree.tree.addEdge(e);
                    }
                }
                treeMap.put(target, sourceTree);
                for (GraphNode<C> n : targetTree.tree.getNodes()) {
                    treeMap.put(n, sourceTree);
                }
                targetTree.tree.clear();
            } else {
                targetTree.tree.addEdge(edge);
                for (GraphNode<C> n : sourceTree.tree.getNodes()) {
                    for (GraphEdge e : sourceTree.tree.getEdges(n)) {
                        targetTree.tree.addEdge(e);
                    }
                }
                treeMap.put(source, targetTree);
                for (GraphNode<C> n : sourceTree.tree.getNodes()) {
                    treeMap.put(n, targetTree);
                }
                sourceTree.tree.clear();
            }

            numTrees--;

            // debug
            System.out.println(">>> " + numTrees
                    + ". " + treeMap.values().size()
                    + ". source: " + treeMap.get(source).id
                    + " (" + treeMap.get(source).tree.getNumNodes() + ") "
                    + ". target: " + treeMap.get(target).id
                    + " (" + treeMap.get(target).tree.getNumNodes() + ") ");
            System.out.println();
        }

        return mst;
    }

    class Tree<C> {

        int id;
        UndirectedGraph<C> tree;

        public Tree(int id) {
            this.id = id;
            this.tree = new UndirectedGraph<C>();
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
            Tree otherA = (Tree) otherObject;
            return this.id == otherA.id;
        }

        @Override
        public int hashCode() {
            return Integer.valueOf(id).hashCode();
        }
    }
}
