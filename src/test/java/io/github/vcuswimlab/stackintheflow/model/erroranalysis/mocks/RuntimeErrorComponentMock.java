package io.github.vcuswimlab.stackintheflow.model.erroranalysis.mocks;

import com.intellij.execution.filters.InputFilter;
import io.github.vcuswimlab.stackintheflow.controller.component.RuntimeErrorComponent;

/**
 * <h1>RuntimeErrorComponentMock</h1>
 * Created on: 7/4/2017
 *
 * @author Tyler John Haden
 */
public class RuntimeErrorComponentMock extends RuntimeErrorComponent{

    public void append(InputFilter console, String type, String line) {
        System.out.println("REComponentMock: " + type + ", " + line);
    }
}
