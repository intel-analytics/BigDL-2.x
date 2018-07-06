package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import java.util.ArrayList;
import java.util.List;

public class TextClassificationSample {
    private String sampleText;
    public void setText(String text){
        sampleText = new String(text);
    }
    private String process(String text){
        String baseDir = System.getProperty("baseDir", "/home/yidiyang/workspace");
        String modelPath = System.getProperty("modelPath", baseDir + "/model/textClassification/textClassificationModel");
        TextClassificationModel model = new TextClassificationModel();
        long begin = System.currentTimeMillis();
        JTensor input = model.preProcess(sampleText);
        long end = System.currentTimeMillis();
        long processTime = end - begin;
        model.load(modelPath);

        begin = System.currentTimeMillis();
        List<JTensor> inputList = new ArrayList<>();
        inputList.add(input);
        List<List<JTensor>> result = model.predict(inputList);
        end = System.currentTimeMillis();
        long predictTime = end - begin;
        JTensor resultTensor = result.get(0).get(0);
        List<Float> resultDis = resultTensor.getData();

        int resultClass = 0;
        float maxProb = 0;
        for( int i = 0; i<resultDis.size();i++){
            if(resultDis.get(i)>=maxProb){
                resultClass = i;
                maxProb = resultDis.get(i);
            }
        }
        String answer = String.format("The predict class is:%s\nThe probability distribution is:%s \n#Process Time elapsed : %d, Predict Time elapsed: %d", Integer.toString(resultClass), resultDis.toString(),processTime,predictTime);
        return answer;
    }

    public String getResult(){
        return process(sampleText);
    }



}
