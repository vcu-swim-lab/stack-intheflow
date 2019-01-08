package regression;

import data.ResponseTextDataset;
import java.io.File;

/**
 * Interface for a regressor.
 *
 * @author vietan
 */
public interface Regressor<D extends ResponseTextDataset> {

    public String getName();
    
    public void train(D trainData);

    public void test(D testData);

    public void output(File outputFile);

    public void input(File inputFile);
}
