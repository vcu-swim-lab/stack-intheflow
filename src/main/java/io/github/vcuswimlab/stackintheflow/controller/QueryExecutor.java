package io.github.vcuswimlab.stackintheflow.controller;

import io.github.vcuswimlab.stackintheflow.model.JerseyGet;
import io.github.vcuswimlab.stackintheflow.model.JerseyResponse;
import io.github.vcuswimlab.stackintheflow.model.Query;

/**
 * Created by Chase on 1/5/2017.
 */
public class QueryExecutor {

    public static JerseyResponse executeQuery(Query q) {
        return JerseyGet.getInstance().executeQuery(q, JerseyGet.SearchType.ADVANCED);
    }

    public static JerseyResponse executeQuery(Query q, JerseyGet.SearchType searchType) {
        return JerseyGet.getInstance().executeQuery(q, searchType);
    }
}
