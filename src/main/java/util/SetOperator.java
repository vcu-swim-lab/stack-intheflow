package util;

import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class SetOperator<C> {

    public Set<C> intersect(Set<C> setOne, Set<C> setTwo) {
        Set<C> intersectSet = new HashSet<C>();
        for (C element : setOne) {
            if (setTwo.contains(element)) {
                intersectSet.add(element);
            }
        }
        return intersectSet;
    }

    public Set<C> union(Set<C> setOne, Set<C> setTwo) {
        Set<C> unionSet = new HashSet<C>();
        for (C e : setOne) {
            unionSet.add(e);
        }
        for (C e : setTwo) {
            unionSet.add(e);
        }
        return unionSet;
    }

    public Set<C> minus(Set<C> setOne, Set<C> setTwo) {
        Set<C> diffSet = new HashSet<C>();
        for (C e : setOne) {
            if (!setTwo.contains(e)) {
                diffSet.add(e);
            }
        }
        return diffSet;
    }
}
