package io.github.vcuswimlab.stackintheflow.model;

import org.glassfish.jersey.client.ClientProperties;
import org.glassfish.jersey.client.filter.EncodingFilter;
import org.glassfish.jersey.message.GZipEncoder;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import java.util.Map;


/**
 * Created by Jeet on 11/1/2016.
 */
public class JerseyGet {

    private static final String SEARCH_URL = "https://api.stackexchange.com/2.2/search";
    private static final String KEY_PARAM = "key";
    private static final String DEV_KEY = "4ZsC*xim)NbV1IbL5Z2xEg((";
    private static final String ENCODING_TYPE = "gzip";
    private static JerseyGet instance = null;
    private Client client;
    private WebTarget webTarget;

    protected JerseyGet() {
        client = ClientBuilder.newClient()
        .register(EncodingFilter.class)
        .register(GZipEncoder.class)
        .property(ClientProperties.USE_ENCODING, ENCODING_TYPE);
        webTarget = client.target(SEARCH_URL);
    }

    public static JerseyGet getInstance() {
        if (instance == null) {
            instance = new JerseyGet();
        }
        return instance;
    }

    public JerseyResponse executeQuery(Query query) {
        return executeQuery(query, SearchType.NORMAL);
    }

    public JerseyResponse executeQuery(Query query, SearchType type) {
        WebTarget target = webTarget.path(type.toString());

        for (Map.Entry<String, String> entry : query.getComponentMap().entrySet()) {
            target = target.queryParam(entry.getKey(), entry.getValue().replaceAll("[\\{\\}]", "").trim());
        }

        //This is the dev key
        target = target.queryParam(KEY_PARAM, DEV_KEY);
        Invocation.Builder builder = target.request(MediaType.APPLICATION_JSON_TYPE).acceptEncoding(ENCODING_TYPE);

        return builder.get().readEntity(JerseyResponse.class);
    }

    public enum SearchType {

        NORMAL(""),
        SIMILAR("similar"),
        EXCERPTS("excerpts"),
        ADVANCED("advanced");

        private final String label;

        SearchType(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public enum SortType {

        RELEVANCE("relevance"),
        ACTIVITY("activity"),
        CREATION("creation"),
        VOTES("votes");

        private final String label;

        SortType(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

}


