package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.ide.plugins.cl.PluginClassLoader;
import com.intellij.openapi.project.Project;
import io.github.vcuswimlab.stackintheflow.model.personalsearch.PersonalSearchModel;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;

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

    private static MyClassLoader createClassLoader() {

        String installPath = JavaFXInstaller.getInstallationPath();

        final ArrayList<URL> urls = new ArrayList<>();
        try {
            urls.add(new URI("file", "", installPath + "/jre/lib/ext/jfxrt.jar", null).toURL());
            urls.add(new URI("file", "", installPath + "/jre/lib/jfxswt.jar", null).toURL());
            urls.add(new URI("file", "", installPath + "/jre/lib/*.dylib", null).toURL());
        } catch (Exception ignore) {
        }

        final ClassLoader parent = SearchToolWindowGUIBuilder.class.getClassLoader();
        if (parent instanceof PluginClassLoader) {
            urls.addAll(((PluginClassLoader) parent).getUrls());
        }

        return new MyClassLoader(urls.toArray(new URL[urls.size()]), parent);
    }

    public SearchToolWindowGUI build() {

        JavaFXInstaller installer = new JavaFXInstaller();

        if (true) { //!installer.isAvailable()
            if (installer.installOpenJFXAndReport(content)) {
                try {
                    Class searchToolWindowClass = createClassLoader().loadClass("io.github.vcuswimlab.stackintheflow.view.SearchToolWindowGUI");
                    return (SearchToolWindowGUI) searchToolWindowClass.getConstructor(JPanel.class, Project.class, PersonalSearchModel.class).newInstance(content, project, searchModel);

                } catch (ClassNotFoundException | IllegalAccessException | InstantiationException | NoSuchMethodException | InvocationTargetException e) {
                }
            } else {
                throw new UnsupportedOperationException("Unable to install JavaFX");
            }
        }

        return new SearchToolWindowGUI(
                content,
                project,
                searchModel
        );
    }

    private static class MyClassLoader extends URLClassLoader {
        public MyClassLoader(URL[] urls, ClassLoader classLoader) {
            super(urls, classLoader);
        }

        @Override
        protected Class<?> loadClass(String s, boolean b) throws ClassNotFoundException {

            if (s.startsWith("java.") || s.startsWith("javax.")) {
                return super.loadClass(s, b);
            }

            Class<?> loadedClass;

            try {
                loadedClass = findClass(s);
            } catch (ClassNotFoundException e) {
                return super.loadClass(s, b);
            }

            if (b) {
                resolveClass(loadedClass);
            }

            return loadedClass;
        }
    }
}
