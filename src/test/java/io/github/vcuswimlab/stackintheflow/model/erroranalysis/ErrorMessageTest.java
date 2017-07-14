package io.github.vcuswimlab.stackintheflow.model.erroranalysis;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * <h1>ErrorMessageTest</h1>
 * Created on: 7/4/2017
 *
 * @author Tyler John Haden
 */
public class ErrorMessageTest {

    String[] error = new String[] {"Error message 1", "java.lang.StackOverflow"};
    String[] information = new String[] {"Information message 1", "1 java.lang.StackOverflow"};
    String[] warning = new String[] {"Warning message 1", "2 java.lang.StackOverflow"};

    @Test
    public void testMessage_AllTypes() {
        ErrorMessage message = new ErrorMessage();
        for(String e : error) {
            message.put(ErrorMessage.MessageType.ERROR, e);
        }
        for(String i : information) {
            message.put(ErrorMessage.MessageType.INFORMATION, i);
        }
        for(String w : warning) {
            message.put(ErrorMessage.MessageType.WARNING, w);
        }

        assertArrayEquals(error, message.get(ErrorMessage.MessageType.ERROR));
        assertArrayEquals(warning, message.get(ErrorMessage.MessageType.WARNING));
        assertArrayEquals(information, message.get(ErrorMessage.MessageType.INFORMATION));
    }

    @Test
    public void testMessage_AllTypes_Lists() {
        ErrorMessage message = new ErrorMessage();
        message.put(ErrorMessage.MessageType.ERROR, Arrays.asList(error));
        message.put(ErrorMessage.MessageType.WARNING, Arrays.asList(warning));
        message.put(ErrorMessage.MessageType.INFORMATION, Arrays.asList(information));

        assertArrayEquals(error, message.get(ErrorMessage.MessageType.ERROR));
        assertArrayEquals(warning, message.get(ErrorMessage.MessageType.WARNING));
        assertArrayEquals(information, message.get(ErrorMessage.MessageType.INFORMATION));
    }

    @Test
    public void testMessage_SomeTypes() {
        ErrorMessage message = new ErrorMessage();
        message.put(ErrorMessage.MessageType.ERROR, Arrays.asList(error));

        assertArrayEquals(error, message.get(ErrorMessage.MessageType.ERROR));
        assertArrayEquals(new String[0], message.get(ErrorMessage.MessageType.WARNING));
        assertArrayEquals(new String[0], message.get(ErrorMessage.MessageType.INFORMATION));
    }


}
