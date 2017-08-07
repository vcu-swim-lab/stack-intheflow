package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.compiler.*;
import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.components.ServiceManager;
import com.intellij.openapi.project.Project;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.model.erroranalysis.ErrorMessageParser;
import io.github.vcuswimlab.stackintheflow.model.erroranalysis.ErrorMessage;
import org.jetbrains.annotations.NotNull;

import javax.tools.Tool;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chase on 2/23/17.
 */
public class CompilerListenerComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.CompilerListenerComponent";

    private final Project project;
    private MessageBusConnection connection;

    public CompilerListenerComponent(Project project) {
        this.project = project;
    }

    @Override
    public void initComponent() {
        // Subscribe to compiler output
        if (project != null) {
            connection = project.getMessageBus().connect();
            connection.subscribe(CompilerTopics.COMPILATION_STATUS, new CompilationStatusListener() {
                @Override
                public void compilationFinished(boolean aborted, int errors, int warnings, CompileContext compileContext) {
                    if(ServiceManager.getService(PersistSettingsComponent.class).compileErrorEnabled()) {
                        if(compileContext.getMessages(CompilerMessageCategory.ERROR).length != 0){
                            ErrorMessage messages = new ErrorMessage();

                            messages.put(ErrorMessage.MessageType.ERROR, Arrays.stream(compileContext.getMessages(CompilerMessageCategory.ERROR)).map(CompilerMessage::getMessage).collect(Collectors.toList()));
                            messages.put(ErrorMessage.MessageType.WARNING, Arrays.stream(compileContext.getMessages(CompilerMessageCategory.WARNING)).map(CompilerMessage::getMessage).collect(Collectors.toList()));
                            messages.put(ErrorMessage.MessageType.INFORMATION, Arrays.stream(compileContext.getMessages(CompilerMessageCategory.INFORMATION)).map(CompilerMessage::getMessage).collect(Collectors.toList()));

                            // Let the parser class handle all data mining
                            List<String> parsedMessages = ErrorMessageParser.parseCompilerError(messages, project);

                            // Send the results to be displayed on the console
                            //project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().setConsoleError(parsedMessages);

                            project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().errorQuery(parsedMessages, false, "compiler");
                        }
                    }
                }
            });
        }
    }

    @Override
    public void disposeComponent() {
        connection.disconnect();
    }

    @NotNull
    @Override
    public String getComponentName() {
        return COMPONENT_ID;
    }

    @Override
    public void projectOpened() {

    }

    @Override
    public void projectClosed() {

    }
}
