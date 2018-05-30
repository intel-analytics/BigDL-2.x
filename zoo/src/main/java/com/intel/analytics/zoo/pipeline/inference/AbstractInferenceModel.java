package com.intel.analytics.zoo.pipeline.inference;

import java.util.List;

public abstract class AbstractInferenceModel {
	private FloatInferenceModel model;

	public void load(String modelPath) {
		load(modelPath, null);
	}

	public void load(String modelPath, String weightPath) {
		this.model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
	}

	public void reload(String modelPath) {
		load(modelPath, null);
	}

	public void reload(String modelPath, String weightPath) {
		this.model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
	}

	public List<Float> predict(List<List<Float>> input) {
		return model.predict(input);
	}
}
