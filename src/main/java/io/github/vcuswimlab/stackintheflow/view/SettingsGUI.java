package io.github.vcuswimlab.stackintheflow.view;

import com.intellij.util.PlatformUtils;
import io.github.vcuswimlab.stackintheflow.controller.component.PersistSettingsComponent.SettingKey;

import javax.swing.*;
import java.util.EnumMap;
import java.util.Map;

/**
 * <h1>SettingsGUI</h1>
 * Created on: 7/28/2017
 *
 * @author Tyler John Haden
 */
public class SettingsGUI {
    private JPanel content;
    private JCheckBox autoQueryCheckBox;
    private JCheckBox runtimeErrorCheckBox;
    private JCheckBox compileErrorCheckbox;
    private JCheckBox difficultyCheckBox;
    private JCheckBox loggingCheckBox;

    private static final String NO_SETTINGS_GUI_MESSAGE =
            "<html>" +
                    "Your platform does not currently support settings." +
                    "</html>";

    public JPanel build(Map<SettingKey, Boolean> settingsMap) {

        if (!PlatformUtils.isIntelliJ()) {
            return getFallbackGUI();
        }

        this.autoQueryCheckBox.setSelected(settingsMap.get(SettingKey.AUTO_QUERY));
        this.runtimeErrorCheckBox.setSelected(settingsMap.get(SettingKey.RUNTIME_ERROR));
        this.compileErrorCheckbox.setSelected(settingsMap.get(SettingKey.COMPILE_ERROR));
        this.difficultyCheckBox.setSelected(settingsMap.get(SettingKey.DIFFICULTY));
        this.loggingCheckBox.setSelected(settingsMap.get(SettingKey.LOGGING));

        if(!settingsMap.get(SettingKey.AUTO_QUERY)) {
            this.runtimeErrorCheckBox.setEnabled(false);
            this.compileErrorCheckbox.setEnabled(false);
            this.difficultyCheckBox.setEnabled(false);
        }

        autoQueryCheckBox.addChangeListener(e -> {
            if (((JCheckBox)e.getSource()).isSelected()) {
                this.runtimeErrorCheckBox.setEnabled(true);
                this.compileErrorCheckbox.setEnabled(true);
                this.difficultyCheckBox.setEnabled(true);
            } else {
                this.runtimeErrorCheckBox.setEnabled(false);
                this.compileErrorCheckbox.setEnabled(false);
                this.difficultyCheckBox.setEnabled(false);
            }
        });

        return content;
    }

    public Map<SettingKey, Boolean> getGUIState() {
        Map<SettingKey, Boolean> guiState = new EnumMap<>(SettingKey.class);
        guiState.put(SettingKey.AUTO_QUERY, this.autoQueryCheckBox.isSelected());
        guiState.put(SettingKey.RUNTIME_ERROR, this.runtimeErrorCheckBox.isSelected());
        guiState.put(SettingKey.COMPILE_ERROR, this.compileErrorCheckbox.isSelected());
        guiState.put(SettingKey.DIFFICULTY, this.difficultyCheckBox.isSelected());
        guiState.put(SettingKey.LOGGING, this.loggingCheckBox.isSelected());

        return guiState;
    }

    private JPanel getFallbackGUI() {
        JPanel noSettingsPanel = new JPanel();
        JEditorPane noSettingsPane = new JEditorPane();
        noSettingsPane.setContentType("text/html");
        noSettingsPane.setEditable(false);
        noSettingsPane.setOpaque(false);
        noSettingsPane.setText(NO_SETTINGS_GUI_MESSAGE);
        noSettingsPanel.add(noSettingsPane);
        return noSettingsPanel;
    }
}
