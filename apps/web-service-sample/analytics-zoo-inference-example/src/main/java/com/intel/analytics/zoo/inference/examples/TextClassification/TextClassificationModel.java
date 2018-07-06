package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.inference.examples.preprocessor.GloveTextProcessing;
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TextClassificationModel extends AbstractInferenceModel {
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    private GloveTextProcessing preprocessor = new GloveTextProcessing();

    public TextClassificationModel() {
        super();
    }

    public JTensor preProcess(String text) {

        JTensor input = preprocessor.preprocess(text);
        return input;
    }


}
