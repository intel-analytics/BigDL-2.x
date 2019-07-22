package com.intel.analytics.zoo.pipeline.api.net;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Developer API for now, intentionally remove public visibility
 */
class PytorchModel {

    static native long loadModelNative(String modelPath);

    static native long loadLossNative(String lossPath);

    static native JTensor modelForwardNative(
            long nativeRef, boolean isTraining, float[] storage, int offset, int[] shape);

    static native JTensor modelBackwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    static native JTensor lossForwardNative(
            long nativeRef, float[] input_storage, int input_offset, int[] input_shape,
            float[] label_storage, int label_offset, int[] label_shape);

    static native JTensor lossBackwardNative(long nativeRef);

    static native float[] getGradientNative(long nativeRef);

    static native void updateWeightNative(long nativeRef, float[] storage);

    static native float[] getWeightNative(long nativeRef);

    static native void releaseModelNative(long nativeRef);

    static native void releaseLossNative(long nativeRef);

    static native int test();

}
