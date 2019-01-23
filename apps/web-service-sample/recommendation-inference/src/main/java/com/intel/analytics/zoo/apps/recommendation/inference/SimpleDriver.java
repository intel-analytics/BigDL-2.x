package com.intel.analytics.zoo.apps.recommendation.inference;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.ArrayList;
import java.util.List;

public class SimpleDriver {

    public static void main(String[] args) {

        String modelPath = System.getProperty("MODEL_PATH", "./models/ncf.bigdl");

        NueralCFModel rcm = new NueralCFModel();

        rcm.load(modelPath);

        List<UserItemPair> userItemPairs = new ArrayList<>();
        for(int i= 1; i < 10; i++){
           userItemPairs.add(new UserItemPair(i, i+1));
        }

        List<List<JTensor>> jts = rcm.preProcess(userItemPairs);

        List<List<JTensor>> finalResult = rcm.predict(jts);

        for(List<JTensor> fjt : finalResult){
            for(JTensor t: fjt){
                System.out.println(t);
            }
        }

    }

}
