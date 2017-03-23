package io.github.vcuswimlab.stackintheflow.controller.component;

import com.intellij.openapi.compiler.*;
import com.intellij.openapi.components.ProjectComponent;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.controller.ErrorMessageParser;
import io.github.vcuswimlab.stackintheflow.view.SearchToolWindowFactory;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chase on 2/23/17.
 */
public class CompilerListenerComponent implements ProjectComponent {

    private final Project project;

    private List<String> compilerMessages;

    public CompilerListenerComponent(Project project) {
        this.project = project;
    }


    @Override
    public void initComponent() {
        // Subscribe to compiler output
        if (project != null) {
            project.getMessageBus().connect().subscribe(CompilerTopics.COMPILATION_STATUS, new CompilationStatusListener() {
                @Override
                public void compilationFinished(boolean aborted, int errors, int warnings, CompileContext compileContext) {
                    CompilerMessage[] messages = compileContext.getMessages(CompilerMessageCategory.ERROR);
                    compilerMessages = Arrays.stream(messages).map(e -> ErrorMessageParser.parseError(e.getMessage(), project)).collect(Collectors.toList());
                    SearchToolWindowFactory.getInstance().setConsoleError(compilerMessages);
                }
            });
        }
    }

    @Override
    public void disposeComponent() {

    }

    @NotNull
    @Override
    public String getComponentName() {
        return "CompilerListenerComponent";
    }

    @Override
    public void projectOpened() {

    }

    @Override
    public void projectClosed() {

    }
}
