package io.github.vcuswimlab.stackintheflow.controller.error.mocks;

import com.intellij.execution.ui.ConsoleViewContentType;
import com.intellij.openapi.editor.markup.TextAttributes;

/**
 * <h1>ConsoleViewContentTypeMock</h1>
 * Created on: 7/5/2017
 *
 * @author Tyler John Haden
 */
public class ConsoleViewContentTypeMock extends ConsoleViewContentType {
    String name;

    ConsoleViewContentTypeMock ERROR_OUTPUT = new ConsoleViewContentTypeMock("ERROR_OUTPUT", null);
    ConsoleViewContentTypeMock LOG_WARNING_OUTPUT = new ConsoleViewContentTypeMock("LOG_WARNING_OUTPUT", null);
    ConsoleViewContentTypeMock SYSTEM_OUTPUT = new ConsoleViewContentTypeMock("SYSTEM_OUTPUT", null);

    public ConsoleViewContentTypeMock(String name, TextAttributes textAttributes) {
        super(name, textAttributes);
        this.name = name;
    }

    public boolean equals(ConsoleViewContentType cvct) {
        return this.name.equals(cvct.toString());
    }
}
