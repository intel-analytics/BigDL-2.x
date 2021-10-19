package com.intel.analytics.zoo.apps.textclassfication.inference;

import com.intel.analytics.zoo.apps.textclassfication.processing.JTextProcessing;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.io.File;
import java.util.Map;

public class TextProcessor extends JTextProcessing {
    private int stopWordsCount;
    private int sequenceLength;
    private Map<String, Integer> wordToIndexMap;

    public TextProcessor(int stopWordsCount, int sequenceLength, String embeddingFilePath) {
        this.stopWordsCount = stopWordsCount;
        this.sequenceLength = sequenceLength;
        this.wordToIndexMap = loadWordToIndexMap(new File(embeddingFilePath));
    }

    public JTensor preprocess(String text) {
        return preprocess(text, this.stopWordsCount, this.sequenceLength, this.wordToIndexMap);
    }
}
