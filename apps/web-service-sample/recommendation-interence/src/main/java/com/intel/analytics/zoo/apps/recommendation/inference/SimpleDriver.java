package com.intel.analytics.zoo.apps.recommendation.inference;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.ArrayList;
import java.util.List;

public class SimpleDriver {

    public static void main(String[] args) {

        String modelPath = System.getProperty("MODEL_PATH", "./models/ncf.bigdl");

        NueralCFModel rcm = new NueralCFModel();

        rcm.load(modelPath);

        List<Integer> userIds = new ArrayList<>();
        for(int i= 1; i < 10; i++){
            userIds.add(i);
        }

        List<Integer> itemIds = new ArrayList<>();
        for(int i=2; i < 11; i++){
            itemIds.add(i);
        }

        List<List<JTensor>> jts = new ArrayList<>();


        for(int i =0; i < userIds.size(); i++){
            List<JTensor> input = new ArrayList<JTensor>();
            input.add(new JTensor(new float[]{userIds.get(i), itemIds.get(i)}, new int[]{2}));
            jts.add(input);
        }

        List<List<JTensor>> finalResult = rcm.predict(jts);

        for(List<JTensor> fjt : finalResult){
            for(JTensor t: fjt){
                System.out.println(t);
            }
        }

    }

}
