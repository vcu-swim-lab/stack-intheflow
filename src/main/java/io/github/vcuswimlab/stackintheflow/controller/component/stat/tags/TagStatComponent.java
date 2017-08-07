package io.github.vcuswimlab.stackintheflow.controller.component.stat.tags;

import com.ctc.wstx.stax.WstxInputFactory;
import com.intellij.openapi.components.ApplicationComponent;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import org.jetbrains.annotations.NotNull;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Created by chase on 6/13/17.
 */
public class TagStatComponent implements ApplicationComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.TagStatComponent";
    private long tagCount;
    private long docCount;
    private Map<String, Stat> tagStatMap;

    public TagStatComponent() {
    }

    @Override
    public void initComponent() {
        tagStatMap = new HashMap<>();
        loadTagStats();
    }

    @Override
    public void disposeComponent() {
        // TODO: insert component disposal logic here
    }

    @Override
    @NotNull
    public String getComponentName() {
        return COMPONENT_ID;
    }

    public Optional<Stat> getTagStat(String tag) {
        return Optional.ofNullable(tagStatMap.get(tag));
    }

    public long getTagCount() {
        return tagCount;
    }

    public long getDocCount() {
        return docCount;
    }

    private void loadTagStats() {
        XMLInputFactory inputFactory = new WstxInputFactory();

        try {
            XMLEventReader eventReader = inputFactory.createXMLEventReader(
                    this.getClass().getClassLoader().getResourceAsStream("Tags.xml"));
            while (eventReader.hasNext()) {
                XMLEvent event = eventReader.nextEvent();
                if (event.getEventType() == XMLStreamConstants.START_ELEMENT) {
                    StartElement startElement = event.asStartElement();
                    switch (startElement.getName().getLocalPart()) {
                        case "Collection":
                            tagCount = Long.parseLong(startElement.getAttributeByName(QName.valueOf("tagCount")).getValue());
                            docCount = Long.parseLong(startElement.getAttributeByName(QName.valueOf("docCount")).getValue());
                            break;
                        case "Tag":
                            String name = startElement.getAttributeByName(QName.valueOf("name")).getValue();
                            long ctf = Long.parseLong(startElement.getAttributeByName(QName.valueOf("ctf")).getValue());
                            long df = Long.parseLong(startElement.getAttributeByName(QName.valueOf("ctf")).getValue());
                            double idf = Double.parseDouble(startElement.getAttributeByName(QName.valueOf("idf")).getValue());
                            double ictf = Double.parseDouble(startElement.getAttributeByName(QName.valueOf("ictf")).getValue());

                            tagStatMap.put(name, new Stat(ctf, df, idf, ictf));
                            break;
                    }
                }
            }
        } catch (XMLStreamException e) {
            e.printStackTrace();
        }
    }
}
