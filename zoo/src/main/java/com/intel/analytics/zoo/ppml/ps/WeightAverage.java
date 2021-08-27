/*
 * Copyright 2021 The Analytic Zoo Authors
 *
 * Licensed under the Apache License,  Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,  software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.zoo.ppml.ps;


import com.intel.analytics.zoo.ppml.generated.FLProto.FloatTensor;
import com.intel.analytics.zoo.ppml.generated.FLProto.Table;
import com.intel.analytics.zoo.ppml.generated.FLProto.TableMetaData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WeightAverage extends Aggregator{

    protected static final Logger logger = LoggerFactory.getLogger(WeightAverage.class);
    protected String modelName = "averaged";

    public WeightAverage(int clientNum) {
        this.clientNum = clientNum;
    }

    /**
     * aggregate current temporary model weights and put updated model into storage
     */
    @Override
    public void aggregate() {
        // assumed that all the weights is correct !
        // sum and average, then save new weights to storage
        Map<String, FloatTensor> sumedDataMap = new HashMap<>();
        // sum
        // to do: concurrent hashmap
        for (Table model : trainMap.values()) {
            Map<String, FloatTensor> modelMap = model.getTableMap();
            for (String tensorName : modelMap.keySet()) {
                List<Integer> shapeList = modelMap.get(tensorName).getShapeList();
                List<Float> dataList = modelMap.get(tensorName).getTensorList();
                if (sumedDataMap.get(tensorName) == null) {
                    sumedDataMap.put(
                            tensorName,
                            FloatTensor.newBuilder()
                                    .addAllTensor(dataList)
                                    .addAllShape(shapeList)
                                    .build());
                } else {
                    List<Integer> shapeListAgg = sumedDataMap.get(tensorName).getShapeList();
                    List<Float> dataListAgg = sumedDataMap.get(tensorName).getTensorList();
                    List<Float> dataListSum = new ArrayList<>();
                    for (int i = 0; i < dataListAgg.size(); i++) {
                        Float temp = dataList.get(i) + dataListAgg.get(i);
                        dataListSum.add(temp);
                    }
                    FloatTensor FloatTensorAgg =
                            FloatTensor.newBuilder().addAllTensor(dataListSum).addAllShape(shapeListAgg).build();
                    sumedDataMap.put(tensorName, FloatTensorAgg);
                }
            }
        }
        // average
        Map<String, FloatTensor> averagedDataMap = new HashMap<>();
        for (String tensorName : sumedDataMap.keySet()) {
            List<Integer> shapeList = sumedDataMap.get(tensorName).getShapeList();
            List<Float> dataList = sumedDataMap.get(tensorName).getTensorList();
            List<Float> averagedDataList = new ArrayList<>();
            for (int i = 0; i < dataList.size(); i++) {
                averagedDataList.add(dataList.get(i) / clientNum);
            }
            FloatTensor averagedFloatTensor = FloatTensor.newBuilder().addAllTensor(averagedDataList).addAllShape(shapeList).build();
            averagedDataMap.put(tensorName, averagedFloatTensor);
        }

        int previousVersion = trainStorage.retrieveCurrentVersion();
        currentVersion = previousVersion + 1;
        TableMetaData metaData =
                TableMetaData.newBuilder().setName(modelName).setVersion(currentVersion).build();
        Table aggregatedModel =
                Table.newBuilder().setMetaData(metaData).putAllTable(averagedDataMap).build();

        trainMap.clear();
        trainStorage.save(currentVersion, aggregatedModel);
        logger.info("run aggregate successfully");
    }

    @Override
    public void initAgg() {}

    @Override
    public void aggEvaluate(boolean agg) {
        throw new IllegalArgumentException("Unimplemented method.");
    }

    @Override
    public void aggPredict() { throw new IllegalArgumentException("Unimplemented method.");}
}
