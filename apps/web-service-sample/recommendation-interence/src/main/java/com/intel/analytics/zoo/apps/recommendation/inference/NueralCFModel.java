package com.intel.analytics.zoo.apps.recommendation.inference;

import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.ArrayList;
import java.util.List;

public class NueralCFModel extends AbstractInferenceModel {

    public NueralCFModel(){

    }

    public List<JTensor> preProcess(List<Integer> userIds, List<Integer> itemIDs){

        List<JTensor> input = new ArrayList<JTensor>();

        for(int i =0; i < userIds.size(); i++){
            input.add(new JTensor(new float[]{userIds.get(i), itemIDs.get(i)}, new int[]{2}));
        }

        return input;

    }
}

