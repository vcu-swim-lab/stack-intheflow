package io.github.vcuswimlab.stackintheflow.controller.error;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * <h1>MessageTest</h1>
 * Created on: 7/4/2017
 *
 * @author Tyler John Haden
 */
public class MessageTest {

    String[] error = new String[] {"Error message 1", "java.lang.StackOverflow"};
    String[] information = new String[] {"Information message 1", "1 java.lang.StackOverflow"};
    String[] warning = new String[] {"Warning message 1", "2 java.lang.StackOverflow"};

    @Test
    public void testMessage_AllTypes() {
        Message message = new Message();
        for(String e : error) {
            message.put(Message.MessageType.ERROR, e);
        }
        for(String i : information) {
            message.put(Message.MessageType.INFORMATION, i);
        }
        for(String w : warning) {
            message.put(Message.MessageType.WARNING, w);
        }

        assertArrayEquals(error, message.get(Message.MessageType.ERROR));
        assertArrayEquals(warning, message.get(Message.MessageType.WARNING));
        assertArrayEquals(information, message.get(Message.MessageType.INFORMATION));
    }

    @Test
    public void testMessage_AllTypes_Lists() {
        Message message = new Message();
        message.put(Message.MessageType.ERROR, Arrays.asList(error));
        message.put(Message.MessageType.WARNING, Arrays.asList(warning));
        message.put(Message.MessageType.INFORMATION, Arrays.asList(information));

        assertArrayEquals(error, message.get(Message.MessageType.ERROR));
        assertArrayEquals(warning, message.get(Message.MessageType.WARNING));
        assertArrayEquals(information, message.get(Message.MessageType.INFORMATION));
    }

    @Test
    public void testMessage_SomeTypes() {
        Message message = new Message();
        message.put(Message.MessageType.ERROR, Arrays.asList(error));

        assertArrayEquals(error, message.get(Message.MessageType.ERROR));
        assertArrayEquals(new String[0], message.get(Message.MessageType.WARNING));
        assertArrayEquals(new String[0], message.get(Message.MessageType.INFORMATION));
    }


}
