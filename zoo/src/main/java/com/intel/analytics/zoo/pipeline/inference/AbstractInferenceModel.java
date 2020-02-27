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

  public void loadBigDL(String modelPath) { doLoadBigDL(modelPath, null, true); }

  public void loadBigDL(String modelPath, String weightPath) {
    doLoadBigDL(modelPath, weightPath, true);
  }

  public void loadCaffe(String modelPath) {
    doLoadCaffe(modelPath, null, true);
  }

  public void loadCaffe(String modelPath, String weightPath) {
    doLoadCaffe(modelPath, weightPath, true);
  }

  public void loadTFFrozen(String modelPath) {
    doLoadTFFrozen(modelPath);
  }

  public void loadTFFrozen(String modelPath, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTFFrozen(modelPath, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTFFrozen(String modelPath, String[] inputs, String[] outputs) {
    doLoadTFFrozen(modelPath, inputs, outputs);
  }

  public void loadTFFrozen(String modelPath, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTFFrozen(modelPath, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTFFrozen(byte[] forzenModelBytes, String[] inputs, String[] outputs) {
    doLoadTFFrozen(forzenModelBytes, inputs, outputs);
  }

  public void loadTFFrozen(byte[] forzenModelBytes, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads){
    doLoadTFFrozen(forzenModelBytes, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTFSaved(String modelPath, String[] inputs, String[] outputs) { doLoadTFSaved(modelPath, inputs, outputs);
  }

  public void loadTFSaved(String modelPath, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTFSaved(modelPath, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTFSaved(byte[] savedModelBytes, String[] inputs, String[] outputs) {
    doLoadTFSaved(savedModelBytes, inputs, outputs);
  }

  public void loadTFSaved(byte[] savedModelBytes, String[] inputs, String[] outputs, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTFSaved(savedModelBytes, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadPyTorch(String modelPath) {
    doLoadPyTorch(modelPath);
  }

  public void loadPyTorch(byte[] modelBytes) {
    doLoadPyTorch(modelBytes);
  }

  @Deprecated
  public void loadTF(String modelPath, String objectDetectionModelType) {
    doLoadTF(modelPath, objectDetectionModelType);
  }

  @Deprecated
  public void loadTF(String modelPath, String pipelineConfigFilePath, String extensionsConfigFilePath) {
    doLoadTF(modelPath, pipelineConfigFilePath, extensionsConfigFilePath);
  }

  @Deprecated
  public void loadTF(String modelPath, String objectDetectionModelType, String pipelineConfigFilePath, String extensionsConfigFilePath) {
    doLoadTF(modelPath, objectDetectionModelType, pipelineConfigFilePath, extensionsConfigFilePath);
  }

  @Deprecated
  public void loadTF(String modelPath, String imageClassificationModelType, String checkpointPath, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale) {
    doLoadTF(modelPath, imageClassificationModelType, checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale);
  }

  @Deprecated
  public void loadTF(byte[] modelBytes, String imageClassificationModelType, byte[] checkpointBytes, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale) {
    doLoadTF(modelBytes, imageClassificationModelType, checkpointBytes, inputShape, ifReverseInputChannels, meanValues, scale);
  }

  @Deprecated
  public void loadTF(String savedModelDir, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale, String input) {
    doLoadTF(savedModelDir, inputShape, ifReverseInputChannels, meanValues, scale, input);
  }

  @Deprecated
  public void loadTF(byte[] savedModelBytes, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale, String input) {
    doLoadTF(savedModelBytes, inputShape, ifReverseInputChannels, meanValues, scale, input);
  }

  @Deprecated
  public void loadTFAsCalibratedOpenVINO(String modelPath, String modelType, String checkpointPath, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale,
                                          String networkType, String validationFilePath, int subset, String opencvLibPath) {
    doLoadTFAsCalibratedOpenVINO(modelPath, modelType, checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale, networkType, validationFilePath, subset, opencvLibPath);
  }

  public void loadOpenVINO(String modelFilePath, String weightFilePath, int batchSize) {
    doLoadOpenVINO(modelFilePath, weightFilePath, batchSize);
  }

  public void loadOpenVINO(String modelFilePath, String weightFilePath) {
    doLoadOpenVINO(modelFilePath, weightFilePath, 0);
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

  public static void optimizeTF(String modelPath, String objectDetectionModelType, String pipelineConfigPath, String extensionsConfigPath, String outputDir) {
    InferenceModel.doOptimizeTF(modelPath, objectDetectionModelType, pipelineConfigPath, extensionsConfigPath, outputDir);
  }

  public static void optimizeTF(String modelPath, String imageClassificationModelType, String checkpointPath, int[] inputShape, boolean ifReverseInputChannels, float[] meanValues, float scale, String outputDir) {
    InferenceModel.doOptimizeTF(modelPath, imageClassificationModelType, checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale, outputDir);
  }

  public static void calibrateTF(String modelPath, String networkType, String validationFilePath, int subset, String opencvLibPath, String outputDir) {
    InferenceModel.doCalibrateTF(modelPath, networkType, validationFilePath, subset, opencvLibPath, outputDir);
  }
}
