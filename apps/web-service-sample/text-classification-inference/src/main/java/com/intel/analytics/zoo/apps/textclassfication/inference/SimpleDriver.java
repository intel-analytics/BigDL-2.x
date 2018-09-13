package com.intel.analytics.zoo.apps.textclassfication.inference;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.ArrayList;
import java.util.List;

public class SimpleDriver {

    public static void main(String[] args) {
        String embeddingFilePath = System.getProperty("EMBEDDING_FILE_PATH", "./glove.6B.300d.txt");
        String modelPath = System.getProperty("MODEL_PATH", "./models/text-classification.bigdl");
        TextClassificationModel textClassificationModel = new TextClassificationModel(10,10, 500, embeddingFilePath);
        textClassificationModel.load(modelPath);
        String[] texts = new String[]{"hello world", "o brother where art thou"};
        List< List<JTensor>> inputs = new ArrayList<List<JTensor>>();
        for(String text : texts) {
            List<JTensor> input = new ArrayList<JTensor>();
            input.add(textClassificationModel.preprocess(text));
            inputs.add(input);
        }
        List<List<JTensor>> results = textClassificationModel.predict(inputs);
        System.out.println(results);
    }
}
