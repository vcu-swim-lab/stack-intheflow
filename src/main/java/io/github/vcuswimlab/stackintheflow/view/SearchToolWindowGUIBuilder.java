package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;
import java.net.*;
import java.util.ArrayList;
import java.util.List;

public class SearchToolWindowGUIBuilder {
    private JPanel content;
    private PersonalSearchModel searchModel;
    private Project project;

    public SearchToolWindowGUIBuilder setContent(JPanel content) {
        this.content = content;
        return this;
    }

    public SearchToolWindowGUIBuilder setSearchModel(PersonalSearchModel searchModel) {
        this.searchModel = searchModel;
        return this;
    }

    public SearchToolWindowGUIBuilder setProject(Project project) {
        this.project = project;
        return this;
    }

    public SearchToolWindowGUI build() {

        JavaFXInstaller installer = new JavaFXInstaller();
        String installPath = JavaFXInstaller.INSTALL_URL;

        if (!installer.isAvailable()) {
            if (installer.installOpenJFXAndReport(content)) {
                List<URL> urls = new ArrayList<>();
                try {
                    urls.add(new URI("file", "", installPath + "/jre/lib/ext/jfxrt.jar", null).toURL());
                    urls.add(new URI("file", "", installPath + "/jre/lib/jfxswt.jar", null).toURL());
                    urls.add(new URI("file", "", installPath + "/jre/lib/*.dylib", null).toURL());
                } catch (URISyntaxException | MalformedURLException e) {
                }

                ClassLoader parent = SearchToolWindowGUIBuilder.class.getClassLoader();

                URLClassLoader urlClassLoader = new URLClassLoader(urls.toArray(new URL[urls.size()]), parent);

                try {
                    return (SearchToolWindowGUI) urlClassLoader.loadClass("io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI")
                            .getConstructor(JPanel.class, Project.class, PersonalSearchModel.class).newInstance(content, project, searchModel);
                } catch (ClassNotFoundException | IllegalAccessException | InstantiationException | NoSuchMethodException | InvocationTargetException e) {
                }
            }
        }

        return new SearchToolWindowGUI(
                content,
                project,
                searchModel
        );
    }

}
