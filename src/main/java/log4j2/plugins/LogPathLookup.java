package log4j2.plugins;

import com.intellij.openapi.application.PathManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginValue;
import org.apache.logging.log4j.core.lookup.StrLookup;


/**
 * Created by Ryan on 7/6/2017.
 */

//Grabs the intellij specific log path for the operating system.

@Plugin(name = "LOG_PATH", category = StrLookup.CATEGORY)
public class LogPathLookup implements StrLookup{

    @Override
    public String lookup(String key) {

        return PathManager.getLogPath();
    }

    @Override
    public String lookup(LogEvent event, String key) {
        return PathManager.getLogPath();
    }
}