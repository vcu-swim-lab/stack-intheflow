package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.application.TransactionGuard;
import com.intellij.openapi.options.ShowSettingsUtil;
import io.github.vcuswimlab.stackintheflow.model.JerseyGet;

import java.util.Arrays;
import java.util.List;

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
        guiInstance.executeQuery(query, tags, false, JerseyGet.SortType.valueOf(searchMethod), addToQueryHistory, "manual");
    }

    public void autoQuery(String query, String tags, boolean backoff, boolean addToQueryHistory, String reasoning){
        guiInstance.executeQuery(query, tags, backoff, JerseyGet.SortType.RELEVANCE, addToQueryHistory, reasoning);
    }

    public void log(String message){
        guiInstance.log(message);
    }

    public void openSettings() {
        TransactionGuard.getInstance().submitTransactionLater(() -> {
                },
                () -> ShowSettingsUtil.getInstance().showSettingsDialog(guiInstance.getProject(), "Stack-InTheFlow"));
    }

    public void updatePersonalSearchModel(String tagsString, int amount) {
        List<String> tagsList = Arrays.asList(tagsString.split(","));
        guiInstance.updateSearchModel(tagsList, amount);
    }

    public void openInBrowser(String url){
        guiInstance.openBrowser(url);
    }

    public void debugBreakpoint(){
        //(String) guiInstance.engine.executeScript("document.documentElement.outerHTML")
        System.out.println();
    }
}
