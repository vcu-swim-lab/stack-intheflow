package io.github.vcuswimlab.stackintheflow.model;

import javax.xml.bind.annotation.XmlRootElement;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by batman on 10/11/16.
 */
@XmlRootElement
public class Question {

    private List<String> tags;
    private String body;
    private String excerpt;
    private String title;
    private String link;
    private boolean isExpanded;

    public Question() {
    }

    public Question(List<String> tags, String body, String excerpt, String title, String link) {
        this.tags = tags;
        this.body = body;
        this.excerpt = excerpt;
        this.title = title;
        this.link = link;
        this.isExpanded = false;
    }

    public void fixNulls() {
        if(tags == null) {
            tags = new ArrayList<String>();
        }
        if(body == null) {
            body = "";
        }
        if(excerpt == null) {
            excerpt = "";
        }
        if(title == null) {
            title = "";
        }
        if(link == null) {
            link = "http://www.stackoverflow.com/";
        }
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    public String getExcerpt() {
        return excerpt;
    }

    public void setExcerpt(String excerpt) {
        this.excerpt = excerpt;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getLink() {
        return link;
    }

    public void setLink(String link) {
        this.link = link;
    }

    public boolean isExpanded() {
        return this.isExpanded;
    }

    public void toggleExpanded() {
        this.isExpanded = !this.isExpanded;
    }

    @Override
    public String toString() {
        return "Question{" +
                "tags=" + tags +
                ", body='" + body + '\'' +
                ", excerpt='" + excerpt + '\'' +
                ", title='" + title + '\'' +
                ", link=" + link +
                '}';
    }

    public String getTagsAsFormattedString() {
        if(tags == null) {
            return "";
        }

        StringBuilder out = new StringBuilder();
        for(String str : tags) {
            out.append("[" + str + "] ");
        }
        return out.toString();
    }
}
