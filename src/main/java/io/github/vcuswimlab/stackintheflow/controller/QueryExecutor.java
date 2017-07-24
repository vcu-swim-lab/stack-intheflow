package io.github.vcuswimlab.stackintheflow.controller;

import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Query;

/**
 * Created by Chase on 1/5/2017.
 */
public class QueryExecutor {

    private final static String filter = "!-MOiNm40F1U019gR)UUjNV-IQScciBJZ0";

    public static JerseyResponse executeQuery(String q) {
        return executeQuery(q, JerseyGet.SortType.RELEVANCE);
    }

    public static JerseyResponse executeQuery(String q, JerseyGet.SortType sortType) {
        return executeQuery(new Query("stackoverflow")
                .set(Query.Component.Q, q)
                .set(Query.Component.FILTER, filter)
                .set(Query.Component.SORT, sortType.toString()));
    }

    public static JerseyResponse executeQuery(Query q) {
        return executeQuery(q, JerseyGet.SearchType.ADVANCED);
    }

    public static JerseyResponse executeQuery(Query q, JerseyGet.SearchType searchType) {
        return JerseyGet.getInstance().executeQuery(q, searchType);
    }
}
