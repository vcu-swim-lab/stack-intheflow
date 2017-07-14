package io.github.vcuswimlab.stackintheflow.model.logging;

import com.intellij.openapi.application.PathManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by Ryan on 7/14/2017.
 */
public class LogTest {

    private Logger logger = LogManager.getLogger("ROLLING_FILE_APPENDER");

    @Test
    public void logTest(){

        //Checks if file exists

        System.out.println(PathManager.getLogPath());
        logger.info("HELLOWORLD");
        File file = new File(PathManager.getLogPath() + "/logfile.log");
        assertTrue(file != null);

        //Check if file has content

        Scanner scanner = new Scanner(PathManager.getLogPath() + "/logfile.log");
        List<String> list=new ArrayList<>();
        while(scanner.hasNextLine()){
            list.add(scanner.nextLine());

        }

        if(list.isEmpty()){

            assertTrue(false);
        }else{
            assertTrue(true);
        }


    }

}
