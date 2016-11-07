package io.github.vcuswimlab.stackintheflow.model;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.core.Response;


/**
 * Created by Jeet on 11/1/2016.
 */
public class JerseyGet {

    public static void main(String[] args) {
        try{
            Client client = ClientBuilder.newClient();
            Response response = client.target("http://cs.stackexchange.com/questions/65655/is-every-np-hard-problem-computable").request("application/json").get();
            String message = response.readEntity(String.class);

            System.out.println(String.format("message is %s", message));
        }
        catch (IndexOutOfBoundsException e) {
            return;
        }
    }

}


