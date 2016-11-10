package io.github.vcuswimlab.stackintheflow.model;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.List;


/**
 * Created by Jeet on 11/1/2016.
 */
public class JerseyGet {

    private Client client;
    private WebTarget webTarget;

    public JerseyGet() {
        client = ClientBuilder.newClient();
        webTarget = client.target("https://api.stackexchange.com/2.2/search");
    }

    /*public static void main(String[] args) {
        try{
            Client client = ClientBuilder.newClient();
            Response response = client.target("http://cs.stackexchange.com/questions/65655/is-every-np-hard-problem-computable").request("application/json").get();
            String message = response.readEntity(String.class);

            System.out.println(String.format("message is %s", message));
        }
        catch (IndexOutOfBoundsException e) {
            return;
        }
    }*/

    public List<Question> executeQuery(Query query) {
        return executeQuery(query, SearchType.NORMAL);
    }

    public List<Question> executeQuery(Query query, SearchType type) {
        WebTarget target = webTarget.path(type.toString()).queryParam("q", query.get(Query.Component.Q));
        Invocation.Builder builder = target.request(MediaType.APPLICATION_JSON_TYPE);

        Response response = builder.get();
        System.out.println(response.readEntity(String.class));
        return null;
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

}


