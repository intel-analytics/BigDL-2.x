/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.inference;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class AbstractInferenceModel extends InferenceModel implements Serializable {

  public AbstractInferenceModel() {
    super();
  }

  public AbstractInferenceModel(int concurrentNum) {
    super(concurrentNum);
  }

  public AbstractInferenceModel(boolean autoScalingEnabled, int concurrentNum) {
    super(autoScalingEnabled, concurrentNum);
  }

  public void loadBigDL(String modelPath) {
    doLoadBigDL(modelPath, null, true);
  }

  public void loadBigDL(String modelPath, String weightPath) {
    doLoadBigDL(modelPath, weightPath, true);
  }

  @Deprecated
  public void load(String modelPath) {
    doLoad(modelPath, null, true);
  }

  @Deprecated
  public void load(String modelPath, String weightPath) {
    doLoad(modelPath, weightPath, true);
  }

  public void loadCaffe(String modelPath) {
    doLoadCaffe(modelPath, null, true);
  }

  public void loadCaffe(String modelPath, String weightPath) {
    doLoadCaffe(modelPath, weightPath, true);
  }

  public void loadTensorflow(String modelPath, String modelType) {
    doLoadTensorflow(modelPath, modelType);
  }

  public void loadTensorflow(String modelPath, String modelType, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTensorflow(modelPath, modelType, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTensorflow(String modelPath, String modelType, String[] inputs, String[] outputs) {
    doLoadTensorflow(modelPath, modelType, inputs, outputs);
  }

  public void loadTensorflow(String modelPath, String modelType, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTensorflow(modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTensorflow(byte[] savedModelBytes, String modelType, String[] inputs, String[] outputs) {
    doLoadTensorflow(savedModelBytes, modelType, inputs, outputs);
  }

  public void loadTensorflow(byte[] savedModelBytes, String modelType, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTensorflow(savedModelBytes, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadPyTorch(String modelPath) {
    doLoadPyTorch(modelPath);
  }

  public void loadPyTorch(byte[] modelBytes) {
    doLoadPyTorch(modelBytes);
  }

  public void loadOpenVINO(String modelFilePath, String weightFilePath, int batchSize) {
    doLoadOpenVINO(modelFilePath, weightFilePath, batchSize);
  }

  public void loadOpenVINO(String modelFilePath, String weightFilePath) {
    doLoadOpenVINO(modelFilePath, weightFilePath, 0);
  }

  public void loadEncryptedOpenVINO(String modelFilePath, String weightFilePath, String secret, String salt, int batchSize) {
    doLoadEncryptedOpenVINO(modelFilePath, weightFilePath, secret, salt, batchSize);
  }

  public void loadEncryptedOpenVINO(String modelFilePath, String weightFilePath, String secret, String salt) {
    doLoadEncryptedOpenVINO(modelFilePath, weightFilePath, secret, salt, 0);
  }

  public void loadOpenVINO(byte[] modelBytes, byte[] weightBytes, int batchSize) {
    doLoadOpenVINO(modelBytes, weightBytes, batchSize);
  }

  public void loadOpenVINO(byte[] modelBytes, byte[] weightBytes) {
    doLoadOpenVINO(modelBytes, weightBytes, 0);
  }

  public void reload(String modelPath) {
    doReload(modelPath, null);
  }

  public void reload(String modelPath, String weightPath) {
    doReload(modelPath, weightPath);
  }

  public void release() {
    doRelease();
  }

  @Deprecated
  public List<Float> predict(List<Float> input, int... shape) {
    List<Integer> inputShape = new ArrayList<Integer>();
    for (int s : shape) {
      inputShape.add(s);
    }
    return doPredict(input, inputShape);
  }

  public List<List<JTensor>> predict(List<List<JTensor>> inputs) {
    return doPredict(inputs);
  }

  public List<List<JTensor>> predict(List<JTensor>[] inputs) {
    return predict(Arrays.asList(inputs));
  }

  @Override
  public String toString() {
    return super.toString();
  }
}
