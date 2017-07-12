package io.github.vcuswimlab.stackintheflow.controller.component.stat.terms;

import com.ctc.wstx.stax.WstxInputFactory;
import com.intellij.openapi.components.ApplicationComponent;
import com.intellij.openapi.editor.Editor;
import io.github.vcuswimlab.stackintheflow.controller.AutoQueryGenerator;
import io.github.vcuswimlab.stackintheflow.controller.component.stat.Stat;
import io.github.vcuswimlab.stackintheflow.model.score.*;
import io.github.vcuswimlab.stackintheflow.model.score.combiner.SumCombiner;
import org.jetbrains.annotations.NotNull;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;
import java.util.*;

public class TermStatComponent implements ApplicationComponent {

    public static final String COMPONENT_ID = "StackInTheFlow.TermStatComponent";
    private long termCount;
    private long docCount;
    private Map<String, Stat> termStatMap;
    private Collection<Scorer> scorers;


    public TermStatComponent() {
    }

    @Override
    public void initComponent() {
        termStatMap = new HashMap<>();
        loadTermStats();
        scorers = Arrays.asList(new ScqScorer(this),
                new VarScorer(this),
                new IctfScorer(this),
                new IdfScorer(this));
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

    public Optional<Stat> getTermStat(String term) {
        return Optional.ofNullable(termStatMap.get(term));
    }

    public long getTermCount() {
        return termCount;
    }

    public long getDocCount() {
        return docCount;
    }

    public String generateQuery(Editor editor) {
        return AutoQueryGenerator.generateQuery(editor, new SumCombiner(scorers));
    }

    private void loadTermStats() {
        XMLInputFactory inputFactory = new WstxInputFactory();

        try {
            XMLEventReader eventReader = inputFactory.createXMLEventReader(
                    this.getClass().getClassLoader().getResourceAsStream("Terms.xml"));
            while (eventReader.hasNext()) {
                XMLEvent event = eventReader.nextEvent();
                if (event.getEventType() == XMLStreamConstants.START_ELEMENT) {
                    StartElement startElement = event.asStartElement();
                    switch (startElement.getName().getLocalPart()) {
                        case "Collection":
                            termCount = Long.parseLong(startElement.getAttributeByName(QName.valueOf("termCount")).getValue());
                            docCount = Long.parseLong(startElement.getAttributeByName(QName.valueOf("docCount")).getValue());
                            break;
                        case "Term":
                            String name = startElement.getAttributeByName(QName.valueOf("name")).getValue();
                            long ctf = Long.parseLong(startElement.getAttributeByName(QName.valueOf("ctf")).getValue());
                            long df = Long.parseLong(startElement.getAttributeByName(QName.valueOf("ctf")).getValue());
                            double idf = Double.parseDouble(startElement.getAttributeByName(QName.valueOf("idf")).getValue());
                            double ictf = Double.parseDouble(startElement.getAttributeByName(QName.valueOf("ictf")).getValue());

                            termStatMap.put(name, new Stat(ctf, df, idf, ictf));
                            break;
                    }
                }
            }
        } catch (XMLStreamException e) {
            e.printStackTrace();
        }
    }
}
