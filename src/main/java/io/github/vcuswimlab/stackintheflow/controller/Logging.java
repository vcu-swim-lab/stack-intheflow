package io.github.vcuswimlab.stackintheflow.controller;

import com.intellij.ide.util.PropertiesComponent;
import com.intellij.openapi.application.PathManager;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.util.UUID;

/**
 * Created by Ryan on 7/21/2017.
 */
public class Logging {

    private Project p;
    private int identifier;
    private PersistSettingsComponent persistSettingsComponent;

    public Logging(){
        this.p = ProjectManager.getInstance().getOpenProjects()[0];
        this.identifier = p.getName().hashCode();
        this.persistSettingsComponent = ServiceManager.getService(PersistSettingsComponent.class);
    }

    public void info(String info) {

        Boolean isEnabled = persistSettingsComponent.loggingEnabled();

        if (isEnabled == true) {

            File file = new File(PathManager.getLogPath() + "/logfile.log");
            if (file.length() > 1000) {
                LogServer logServer = new LogServer();
                logServer.LogToServer(file);
                file.delete();
            }

            String message = "{" + "\"" + "User" + "\"" + ":" + "\"" + getUUID() + "\"" + ", " + "\"" + "ProjectCode" + "\"" + ":" + "\"" + identifier + "\"" + ", " + "\"" + "Environment" + "\"" + ":" + "\"" + "InsertEnvironmentHere" + "\"" + ", " + info;

            try {
                Logger logger = LogManager.getLogger("ROLLING_FILE_APPENDER");
                logger.info(message);
                LogManager.shutdown();
            } catch (Exception e) {
                e.printStackTrace();
            }
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
