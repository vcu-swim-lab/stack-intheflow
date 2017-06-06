package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.compiler.*;
import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.project.Project;
import com.intellij.util.messages.MessageBusConnection;
import io.github.vcuswimlab.stackintheflow.controller.error.ErrorMessageParser;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by chase on 2/23/17.
 */
public class CompilerListenerComponent implements ProjectComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.CompilerListenerComponent";

    private final Project project;
    private MessageBusConnection connection;

    // Categories of compiler messages that can be extracted
    // ERROR, WARNING, INFORMATION, STATISTICS
    private final List<CompilerMessageCategory> messageCategories = Arrays.asList(CompilerMessageCategory.values());

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
                    // Mapping from compiler category name, "ERROR", to list of messages, ["Caused by: ...", "..."]
                    Map<String, List<String>> compilerMessages = messageCategories.parallelStream().collect(
                            Collectors.toMap(CompilerMessageCategory::name, c ->
                                    Arrays.stream(compileContext.getMessages(c)).map(
                                            CompilerMessage::getMessage).collect(Collectors.toList())));

                    // Let the parser class handle all data mining
                    List<String> consoleDisplayItems = ErrorMessageParser.parseCompilerError(compilerMessages, project);

                    // Send the results to be displayed on the console
                    project.getComponent(ToolWindowComponent.class).getSearchToolWindowGUI().setConsoleError(consoleDisplayItems);
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
