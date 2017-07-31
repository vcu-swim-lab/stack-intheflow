package io.github.vcuswimlab.stackintheflow.view;

import io.github.vcuswimlab.stackintheflow.model.JerseyGet;

/**
 * Created by stackintheflow on 6/26/17.
 */
public class JavaBridge {
    private SearchToolWindowGUI guiInstance;

    public JavaBridge(SearchToolWindowGUI guiInstance){
        this.guiInstance = guiInstance;
    }

    public void print(String msg){
        System.out.println("From Java: " + msg);
    }

    public void searchButtonClicked(String query, String tags, String searchMethod, boolean addToQueryHistory){
        guiInstance.executeQuery(query, tags, false, JerseyGet.SortType.valueOf(searchMethod), addToQueryHistory);
    }

    public void autoQuery(String query, String tags, boolean backoff, boolean addToQueryHistory){
        guiInstance.executeQuery(query, tags, backoff, JerseyGet.SortType.RELEVANCE, addToQueryHistory);
    }

    public void log(String message){
        guiInstance.log(message);
    }

    public void openInBrowser(String url){
        guiInstance.openBrowser(url);
    }

    public void debugBreakpoint(){
        //(String) guiInstance.engine.executeScript("document.documentElement.outerHTML")
        System.out.println();
    }
}
