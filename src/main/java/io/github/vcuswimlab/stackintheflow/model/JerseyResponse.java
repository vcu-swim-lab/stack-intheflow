package io.github.vcuswimlab.stackintheflow.model;

import javax.xml.bind.annotation.XmlRootElement;
import java.util.List;

@XmlRootElement
public class JerseyResponse {

    private List<Question> items;
    private boolean has_more;
    private int quota_max;
    private int quota_remaining;

    public JerseyResponse() {
    }

    public JerseyResponse(List<Question> items, boolean has_more, int quota_max, int quota_remaining) {
        this.items = items;
        this.has_more = has_more;
        this.quota_max = quota_max;
        this.quota_remaining = quota_remaining;
    }

    public List<Question> getItems() {
        return items;
    }

    public void setItems(List<Question> items) {
        this.items = items;
    }

    public boolean isHas_more() {
        return has_more;
    }

    public void setHas_more(boolean has_more) {
        this.has_more = has_more;
    }

    public int getQuota_max() {
        return quota_max;
    }

    public void setQuota_max(int quota_max) {
        this.quota_max = quota_max;
    }

    public int getQuota_remaining() {
        return quota_remaining;
    }

    public void setQuota_remaining(int quota_remaining) {
        this.quota_remaining = quota_remaining;
    }

    @Override
    public String toString() {
        return "JerseyResponse{" +
                "items=" + items +
                ", has_more=" + has_more +
                ", quota_max=" + quota_max +
                ", quota_remaining=" + quota_remaining +
                '}';
    }
}
