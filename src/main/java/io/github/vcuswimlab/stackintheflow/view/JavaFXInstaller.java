package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.openapi.application.PathManager;
import com.intellij.openapi.util.Pair;
import com.intellij.openapi.vfs.VfsUtilCore;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.util.download.DownloadableFileDescription;
import com.intellij.util.download.DownloadableFileService;
import com.intellij.util.download.FileDownloader;
import com.intellij.util.io.ZipUtil;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class JavaFXInstaller {

    private static final String INSTALL_URL = "http://download.jetbrains.com/idea/open-jfx/javafx-sdk-overlay.zip";

    public boolean isAvailable() {
        try {
            return Class.forName("javafx.scene.web.WebView") != null;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    public static String getInstallationPath() {
        return PathManager.getConfigPath() + "/openjfx";
    }

    public boolean installOpenJFXAndReport(@NotNull JComponent parentComponent) {
        final DownloadableFileService fileService = DownloadableFileService.getInstance();
        final DownloadableFileDescription fileDescription = fileService.createFileDescription(INSTALL_URL, "javafx-sdk-overlay.zip");
        final FileDownloader downloader = fileService.createDownloader(Collections.singletonList(fileDescription), "OpenJFX");

        final List<Pair<VirtualFile, DownloadableFileDescription>> progress =
                downloader.downloadWithProgress(getInstallationPath(), null, parentComponent);

        if (progress == null) {
            return false;
        }

        boolean success = false;
        for (Pair<VirtualFile, DownloadableFileDescription> pair : progress) {
            if (!pair.getSecond().equals(fileDescription)) {
                continue;
            }
            final VirtualFile file = pair.getFirst();
            if (file == null) {
                continue;
            }

            final File archiveFile = VfsUtilCore.virtualToIoFile(file);
            try {
                ZipUtil.extract(archiveFile, new File(getInstallationPath()), null, true);
                success = true;
            } catch (IOException ignore) {
            }
        }

        return success;
    }
}
