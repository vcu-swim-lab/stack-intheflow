package io.github.vcuswimlab.stackintheflow.model.erroranalysis;

import java.util.ArrayList;
import java.util.List;

/**
 * <h1>ErrorMessage</h1>
 * Created on: 6/30/2017
 *
 * @author Tyler John Haden
 */
public class ErrorMessage {
    private List<String> error;
    private List<String> warning;
    private List<String> information;

    public ErrorMessage() {
        error = new ArrayList<>();
        warning = new ArrayList<>();
        information = new ArrayList<>();
    }

    public String[] get(MessageType messageType){
        switch (messageType) {
            case ERROR:
                return error.toArray(new String[error.size()]);
            case WARNING:
                return warning.toArray(new String[warning.size()]);
            case INFORMATION:
                return information.toArray(new String[information.size()]);
        }
        throw new IllegalArgumentException("messageType is not recognized");
    }

    public void put(MessageType messageType, String text) {
        switch (messageType) {
            case ERROR:
                error.add(text);
                break;
            case WARNING:
                warning.add(text);
                break;
            case INFORMATION:
                information.add(text);
                break;
            default:
                throw new IllegalArgumentException("messageType is not recognized");
        }
    }

    public void put(MessageType messageType, List<String> texts) {
        texts.forEach(t -> put(messageType, t));
    }

    public enum MessageType{
        ERROR,
        WARNING,
        INFORMATION
    }
}
