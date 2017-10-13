package io.github.vcuswimlab.stackintheflow.controller;

import org.glassfish.jersey.client.ClientProperties;
import org.glassfish.jersey.client.authentication.HttpAuthenticationFeature;
import org.jasypt.encryption.pbe.StandardPBEStringEncryptor;
import org.jasypt.properties.EncryptableProperties;

import javax.ws.rs.client.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Properties;
import java.util.Scanner;


/**
 * Created by Ryan on 7/27/2017.
 */
public class LogServer {

    private WebTarget target;
    private Properties props;

    public LogServer(){
        Client client = ClientBuilder.newClient();
        client.property(ClientProperties.CONNECT_TIMEOUT, 1000);
        client.property(ClientProperties.READ_TIMEOUT, 1000);
        StandardPBEStringEncryptor encryptor = new StandardPBEStringEncryptor();
        encryptor.setPassword("jasypt");
        this.props = new EncryptableProperties(encryptor);
        getPropFile();
        HttpAuthenticationFeature feature = HttpAuthenticationFeature.basic(props.getProperty("logstash.username"), props.getProperty("logstash.password"));
        client.register(feature);
        this.target = client.target(props.getProperty("logstash.url"));
    }

    public void LogToServer(File file){
        try{
            StringBuilder sb = new StringBuilder();
            Scanner scanner = new Scanner(file);
            sb.append("{" + "\"" + "Logs" + "\"" + ": [");
            while(scanner.hasNextLine()){
                sb.append(scanner.nextLine());
                if(scanner.hasNextLine()){
                    sb.append(", ");
                }
            }
            scanner.close();
            sb.append("]}");
            AsyncInvoker invoker = target.request().async();
            invoker.put(Entity.json(sb.toString()));
        } catch (FileNotFoundException e) {
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
