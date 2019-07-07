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

    static native long loadNative(String modelPath, String lossPath);

    static native JTensor forwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    static native JTensor backwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    static native float[] getGradientNative(long nativeRef);

    static native void updateWeightNative(long nativeRef, float[] storage);

    static native float[] getWeightNative(long nativeRef);

    static native void releaseNative(long nativeRef);

    static native int test();

}
