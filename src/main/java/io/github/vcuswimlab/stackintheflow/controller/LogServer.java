package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.ide.util.PropertiesComponent;
import org.glassfish.jersey.client.authentication.HttpAuthenticationFeature;

import javax.ws.rs.client.*;
import javax.ws.rs.core.Response;
import java.io.File;
import java.util.Scanner;
import java.util.UUID;

/**
 * Created by Ryan on 7/27/2017.
 */
public class LogServer {

    private Client client;
    private WebTarget target;
    private Response response;
    private Invocation.Builder builder;
    private HttpAuthenticationFeature feature;

    public LogServer(){
        this.client = ClientBuilder.newClient();
        this.feature = HttpAuthenticationFeature.basic("vcuswim", "SITF1234analytics");
        client.register(feature);
        this.target = client.target("http://104.131.188.205:31311");
    }

    public void LogToServer(File file){
        try{
            StringBuilder foo = new StringBuilder();
            Scanner scanner = new Scanner(file);
            foo = foo.append("{" + "\"" + getUUID() + "\"" + ": [");
            while(scanner.hasNextLine()){
                foo = foo.append(scanner.nextLine());
                if(scanner.hasNextLine()){
                    foo = foo.append(", ");
                }

            }
            foo = foo.append("]}");
            builder = target.request();
            response = builder.put(Entity.json(foo.toString()));
            response.close();
            scanner.close();

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public String getUUID() {

        String uuid;
        PropertiesComponent c = PropertiesComponent.getInstance();
        uuid = c.getValue("uuid");

        if (uuid == null){
            String save = UUID.randomUUID().toString().replace("-", "");
            c.setValue("uuid", save);
            uuid = c.getValue("uuid");
        }

        return uuid;
    }

}
