package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.openapi.application.PathManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;

/**
 * Created by Ryan on 7/21/2017.
 */
public class Logging {

    private Project p;
    private int identifier;

    public Logging(){
        this.p = ProjectManager.getInstance().getOpenProjects()[0];
        this.identifier = p.getName().hashCode();
    }

    public void info(String info){

        File file = new File(PathManager.getLogPath() + "/logfile.log");
        //System.out.println(file.length());
        if (file.length() > 500){
            LogServer logServer = new LogServer();
            logServer.LogToServer(file);
            file.delete();
        }

        String message = "{" + "\"" + "ProjectCode" + "\"" + ":" +"\"" + identifier + "\"" + ", " + info;

        try {
            Logger logger = LogManager.getLogger("ROLLING_FILE_APPENDER");
            logger.info(message);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

}
