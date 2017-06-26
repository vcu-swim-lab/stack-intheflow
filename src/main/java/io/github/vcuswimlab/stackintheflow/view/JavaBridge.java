package io.github.vcuswimlab.stackintheflow.view;

/**
 * Created by stackintheflow on 6/26/17.
 */
public class JavaBridge {
    public void print(String msg){
        System.out.println("From Java: " + msg);
    }

    public void searchButtonClicked(String query){
        System.out.println("Search Box is clicked with query: " + query);
    }
}
