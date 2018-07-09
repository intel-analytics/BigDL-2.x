package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.inference.examples.preprocessor.GloveTextProcessing;
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    private int stopWordsCount, sequenceLength;

    public TextClassificationModel(int stopWordsCount, int sequenceLength) {
        this.stopWordsCount = stopWordsCount;
        this.sequenceLength = sequenceLength;
    }

    private GloveTextProcessing preprocessor = new GloveTextProcessing();

    public JTensor preProcess(String text) {
        JTensor input = preprocessor.preprocess(text, stopWordsCount, sequenceLength);
        return input;
    }
}