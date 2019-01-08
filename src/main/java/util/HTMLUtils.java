package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author vietan
 */
public class HTMLUtils {

    public static final String CSS_URL = "http://argviz.umiacs.umd.edu/teaparty/framing.css";
    public static final String JS_URL = "http://argviz.umiacs.umd.edu/teaparty/framing.js";
    public static final String[] Colors8 = {"#2166ac", "#4393c3",
        "#92c5de", "#d1e5f0",
        "#fddbc7", "#f4a582",
        "#d6604d", "#b2182b"};
    public static final String WHITE = "#FFFFFF";
    public static final String BLACK = "#000000";

    public static void outputHTMLFile(File htmlFile, String body) {
        StringBuilder str = new StringBuilder();
        str.append("<!DOCTYPE html>\n<html>\n");
        str.append("<head>\n"); // header containing styles and javascript functions
        str.append("<meta charset=\"UTF-8\">\n");
        str.append("<link type=\"text/css\" rel=\"stylesheet\" "
                + "href=\"" + CSS_URL + "\">\n"); // style
        str.append("<script type=\"text/javascript\" src=\"").append(JS_URL)
                .append("\"></script>\n");
        str.append("<title>Agenda-setting and Framing in U.S. Congress</title>\n");
        str.append("</head>\n"); // end head
        str.append("<body>\n"); // body
        str.append(body).append("\n");
        str.append("</body>\n");
        str.append("</html>\n");
        try { // output to file
            BufferedWriter writer = IOUtils.getBufferedWriter(htmlFile);
            writer.write(str.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing HTML to "
                    + htmlFile);
        }
    }

    public static String getEmbeddedLink(String text, String url) {
        return "<a href=\"" + url + "\" target=\"_blank\">" + text + "</a>";
    }

    public static String getHTMLList(ArrayList<String> strList) {
        StringBuilder str = new StringBuilder();
        str.append("<ul>\n");
        for (String s : strList) {
            str.append("\t<li>").append(s).append("</li>\n");
        }
        str.append("</ul>\n");
        return str.toString();
    }

    public static String getColor(double value) {
        if (value < -2) {
            return Colors8[0];
        } else if (value < -1) {
            return Colors8[1];
        } else if (value < -.5) {
            return Colors8[2];
        } else if (value < -.1) {
            return Colors8[3];
        } else if (value < .1) {
            return WHITE;
        } else if (value < 0.5) {
            return Colors8[4];
        } else if (value < 1) {
            return Colors8[5];
        } else if (value < 2) {
            return Colors8[6];
        }
        return Colors8[7];
    }

    public static String getTextColor(String backgroundColor) {
        if (backgroundColor.equals(Colors8[0])
                || backgroundColor.equals(Colors8[7])) {
            return WHITE;
        }
        return BLACK;
    }
}
