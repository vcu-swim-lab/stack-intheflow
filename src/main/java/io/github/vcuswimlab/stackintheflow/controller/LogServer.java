package io.github.vcuswimlab.stackintheflow.controller;

import org.glassfish.jersey.client.authentication.HttpAuthenticationFeature;
import org.jasypt.encryption.pbe.StandardPBEStringEncryptor;
import org.jasypt.properties.EncryptableProperties;

import javax.ws.rs.client.*;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.InputStream;
import java.util.Properties;
import java.util.Scanner;


/**
 * Created by Ryan on 7/27/2017.
 */
public class LogServer {

    private Client client;
    private WebTarget target;
    private Response response;
    private Invocation.Builder builder;
    private HttpAuthenticationFeature feature;
    private StandardPBEStringEncryptor encryptor;
    private Properties props;

    public LogServer(){
        this.client = ClientBuilder.newClient();
        this.encryptor = new StandardPBEStringEncryptor();
        encryptor.setPassword("jasypt");
        this.props = new EncryptableProperties(encryptor);
        getPropFile();
        this.feature = HttpAuthenticationFeature.basic(props.getProperty("logstash.username"), props.getProperty("logstash.password"));
        client.register(feature);
        this.target = client.target(props.getProperty("logstash.url"));
    }

    public void LogToServer(File file){
        try{
            StringBuilder sb = new StringBuilder();
            Scanner scanner = new Scanner(file);
            sb = sb.append("{" + "\"" + "Logs" + "\"" + ": [");
            while(scanner.hasNextLine()){
                sb = sb.append(scanner.nextLine());
                if(scanner.hasNextLine()){
                    sb = sb.append(", ");
                }

            }
            sb = sb.append("]}");
            builder = target.request();
            response = builder.put(Entity.json(sb.toString()));
            response.close();
            scanner.close();

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    public void getPropFile(){

        try {

            InputStream inputStream = getClass().getResourceAsStream("/logstash.properties");
            props.load(inputStream);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }



}
