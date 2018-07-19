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
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

public abstract class AbstractInferenceModel implements Serializable {
	private int supportedConcurrentNum = 1;
	protected LinkedBlockingQueue<FloatInferenceModel> modelQueue;

	public AbstractInferenceModel() {
	}

	public AbstractInferenceModel(int supportedConcurrentNum) {
		this.supportedConcurrentNum = supportedConcurrentNum;

	}

	public void load(String modelPath) {
		load(modelPath, null);
	}

	public void load(String modelPath, String weightPath) {
		modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
		for (int i = 0; i < supportedConcurrentNum; i++) {
			FloatInferenceModel model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
			modelQueue.offer(model);
		}
	}

	public void loadCaffe(String modelPath) {
		loadCaffe(modelPath, null);
	}

	public void loadCaffe(String modelPath, String weightPath) {
		modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
		for (int i = 0; i < supportedConcurrentNum; i++) {
			FloatInferenceModel model = InferenceModelFactory.loadFloatInferenceModelForCaffe(modelPath, weightPath);
			modelQueue.offer(model);
		}

	}

	public void loadTF(String modelPath) {
		loadTF(modelPath, 1, 1, true);
	}

	public void loadTF(String modelPath,
	                   int intraOpParallelismThreads,
	                   int interOpParallelismThreads,
	                   boolean usePerSessionThreads) {
		modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
		for (int i = 0; i < supportedConcurrentNum; i++) {
			FloatInferenceModel model = InferenceModelFactory.loadFloatInferenceModelForTF(modelPath,
					intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
			modelQueue.offer(model);
		}
	}

	public void reload(String modelPath) {
		load(modelPath, null);
	}

	public void reload(String modelPath, String weightPath) {
		modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
		for (int i = 0; i < supportedConcurrentNum; i++) {
			FloatInferenceModel model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
			modelQueue.offer(model);
		}

	}

	@Deprecated
	public List<Float> predict(List<Float> input, int... shape) throws InterruptedException {
		FloatInferenceModel model = null;
		try {
			model = modelQueue.take();
		} catch (InterruptedException e) {
			throw new InferenceRuntimeException("XXX", e);
		}
		List<Integer> inputShape = new ArrayList<Integer>();
		for (int s : shape) {
			inputShape.add(s);
		}
		List<Float> result = model.predict(input, inputShape);
		modelQueue.offer(model);
		return result;
	}

	public List<List<JTensor>> predict(List<JTensor> inputs) {
		FloatInferenceModel model = null;
		try {
			model = modelQueue.take();
		} catch (InterruptedException e) {
			throw new InferenceRuntimeException("XXX", e);
		}
		List<List<JTensor>> result = model.predict(inputs);
		modelQueue.offer(model);
		return result;
	}

}
