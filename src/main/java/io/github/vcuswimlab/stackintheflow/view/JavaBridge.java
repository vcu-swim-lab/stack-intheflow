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

    public void searchButtonClicked(String query){
        System.out.println("Search Box is clicked with query: " + query);
        guiInstance.executeQuery(query, false);
    }

    public void addLinkListeners(){
        guiInstance.addLinkListeners();
    }
}
