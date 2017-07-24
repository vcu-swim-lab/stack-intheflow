package io.github.vcuswimlab.stackintheflow.view;

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

    public void searchButtonClicked(String query, String searchMethod){
        System.out.println("Search Box is clicked with query: " + query + "with method: " + searchMethod);
        guiInstance.executeQuery(query, false, searchMethod);
    }

    public void autoQuery(String query, boolean backoff){
        System.out.println("Auto query: " + query);
        System.out.println("Backoff: " + backoff);
        guiInstance.executeQuery(query, backoff, "RELEVANCE");
    }


    public void openInBrowser(String url){
        guiInstance.openBrowser(url);
    }

    public void debugBreakpoint(){
        //(String) guiInstance.engine.executeScript("document.documentElement.outerHTML")
        System.out.println();
    }
}
