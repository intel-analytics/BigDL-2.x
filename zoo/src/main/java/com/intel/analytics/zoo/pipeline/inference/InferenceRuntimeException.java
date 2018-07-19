package com.intel.analytics.zoo.pipeline.inference;

/**
 * Created by xiaxue on 7/19/18.
 */
public class InferenceRuntimeException extends RuntimeException {
	public InferenceRuntimeException(String msg) {
		super(msg);
	}

	public InferenceRuntimeException(String msg, Throwable cause) {
		super(msg, cause);
	}
}
