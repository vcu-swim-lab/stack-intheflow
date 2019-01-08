package util;

/**
 *
 * @author NGUYEN Viet-An
 */
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class RankingItemList<A> {

    private final ArrayList<RankingItem<A>> rankingItems;

    public RankingItemList() {
        this.rankingItems = new ArrayList<RankingItem<A>>();
    }

    public RankingItemList(ArrayList<RankingItem<A>> rankingItems) {
        this.rankingItems = rankingItems;
    }

    public void addRankingItem(A item, double value) {
        this.rankingItems.add(new RankingItem<A>(item, value));
    }

    public void addRankingItem(RankingItem<A> rankingItem) {
        this.rankingItems.add(rankingItem);
    }

    public RankingItem<A> get(A item) {
        for (RankingItem<A> cur_rank_item : this.rankingItems) {
            if (cur_rank_item.getObject().equals(item)) {
                return cur_rank_item;
            }
        }
        return null;
    }

    public void sortAscending() {
        Collections.sort(rankingItems);
        Collections.reverse(rankingItems);

        int ranking = 0;
        double preValue = Float.MIN_VALUE;
        for (RankingItem rankingItem : rankingItems) {
            double curValue = rankingItem.getPrimaryValue();
            if (curValue > preValue) {
                ranking++;
                preValue = curValue;
            }
            rankingItem.setRankingOrder(ranking);
        }
    }

    public void sortDescending() {
        Collections.sort(rankingItems);
        int ranking = 0;
        double preValue = Float.MAX_VALUE;
        for (RankingItem rankingItem : rankingItems) {
            double curValue = rankingItem.getPrimaryValue();
            if (curValue < preValue) {
                ranking++;
                preValue = curValue;
            }
            rankingItem.setRankingOrder(ranking);
        }
    }

    public ArrayList<RankingItem<A>> getRankingItems() {
        return this.rankingItems;
    }

    public RankingItem<A> getRankingItem(int index) {
        return this.rankingItems.get(index);
    }

    public HashMap<A, RankingItem<A>> getRankingTable() {
        HashMap<A, RankingItem<A>> rankingTable = new HashMap<A, RankingItem<A>>();
        for (RankingItem<A> rankingItem : this.rankingItems) {
            A item = rankingItem.getObject();
            rankingTable.put(item, rankingItem);
        }
        return rankingTable;
    }

    public HashMap<A, Integer> getRankingOrders() {
        HashMap<A, Integer> rankingOrders = new HashMap<A, Integer>();
        for (RankingItem<A> rankingItem : this.rankingItems) {
            A rankingObject = rankingItem.getObject();
            int rankingOrder = rankingItem.getRankingOrder();
            rankingOrders.put(rankingObject, rankingOrder);
        }
        return rankingOrders;
    }

    public HashMap<Integer, Set<A>> getGroupByOrder() {
        HashMap<Integer, Set<A>> groupByOrder = new HashMap<Integer, Set<A>>();

        HashMap<A, Integer> rankingOrders = getRankingOrders();
        for (A item : rankingOrders.keySet()) {
            Integer order = rankingOrders.get(item);

            Set<A> group = groupByOrder.get(order);
            if (group == null) {
                group = new HashSet<A>();
            }
            group.add(item);
            groupByOrder.put(order, group);
        }

        return groupByOrder;
    }

    public int size() {
        return this.rankingItems.size();
    }
}
