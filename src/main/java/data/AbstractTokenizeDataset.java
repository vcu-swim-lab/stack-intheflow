package data;

import core.AbstractDataset;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Set;
import main.GlobalConstants;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

/**
 *
 * @author vietan
 */
public abstract class AbstractTokenizeDataset extends AbstractDataset {

    public static final String wordVocabExt = ".wvoc";
    public static final String speakerVocabExt = ".svoc";
    public static final String numDocDataExt = ".dat";
    public static final String numSentDataExt = ".sent-dat";
    public static final String docIdExt = ".docid";
    public static final String docTextExt = ".text";
    public static final String docInfoExt = ".docinfo";
    protected String folder; // main folder of the dataset
    protected Set<String> stopwords;
    protected Tokenizer tokenizer;
    protected CorpusProcessor corpProc;
    protected String formatFilename;

    public AbstractTokenizeDataset(String name) {
        super(name);
        this.formatFilename = name;
    }

    public AbstractTokenizeDataset(
            String name,
            String folder) {
        super(name);
        this.folder = folder;
        this.formatFilename = name; // by default
        try {
            // initiate tokenizer
            InputStream tokenizeIn = new FileInputStream(GlobalConstants.TokenizerFilePath);
            TokenizerModel tokenizeModel = new TokenizerModel(tokenizeIn);
            this.tokenizer = new TokenizerME(tokenizeModel);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public AbstractTokenizeDataset(
            String name, // dataset name
            String folder, // dataset folder
            CorpusProcessor corpProc) {
        super(name);
        this.folder = folder;
        this.formatFilename = name; // by default
        this.corpProc = corpProc;

        try {
            // initiate tokenizer
            InputStream tokenizeIn = new FileInputStream(GlobalConstants.TokenizerFilePath);
            TokenizerModel tokenizeModel = new TokenizerModel(tokenizeIn);
            this.tokenizer = new TokenizerME(tokenizeModel);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void setCorpusProcessor(CorpusProcessor cp) {
        this.corpProc = cp;
    }

    public CorpusProcessor getCorpusProcessor() {
        return this.corpProc;
    }

    public void setFormatFilename(String fn) {
        this.formatFilename = fn;
    }

    public String getFormatFilename() {
        return this.formatFilename;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public String getFolder() {
        return this.folder;
    }

    public String getDatasetFolderPath() {
        return new File(this.folder, getName()).getAbsolutePath();
    }
}
