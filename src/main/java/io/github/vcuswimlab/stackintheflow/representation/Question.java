package io.github.vcuswimlab.stackintheflow.representation;

import org.jetbrains.annotations.NotNull;

import java.net.URL;
import java.util.List;

/**
 * Created by batman on 10/11/16.
 */
public class Question {
    //TODO: Add more appropriate fields as necessary
    private String name;
    private String body;
    private URL link;
    private List<String> tags;

    public Question(@NotNull String name, @NotNull String body, @NotNull URL link, @NotNull List<String> tags) {
        this.name = name;
        this.body = body;
        this.link = link;
        this.tags = tags;
    }

    public String getName() {
        return name;
    }

    public String getBody() {
        return body;
    }

    public URL getLink() {
        return link;
    }

    public List<String> getTags() {
        return tags;
    }
}
