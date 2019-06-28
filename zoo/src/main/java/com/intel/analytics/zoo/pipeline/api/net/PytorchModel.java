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

    /**
     * sequential id in cpp: std::vector<std::shared_ptr<torch::jit::script::Module>> handles;
     */
    private long nativeRef;

    PytorchModel load(byte[] bytes, byte[] lossBytes) throws IOException {
        try {
            File tmpFile = File.createTempFile("TorchNet", "_pt");
            Files.write(Paths.get(tmpFile.toURI()), bytes);

            File lossFile = File.createTempFile("TorchNet", "_lpt");
            Files.write(Paths.get(lossFile.toURI()), lossBytes);

            this.load(tmpFile.getAbsolutePath(), lossFile.getAbsolutePath());

            FileUtils.deleteQuietly(tmpFile);
            FileUtils.deleteQuietly(lossFile);
        } catch (IOException io) {
            System.out.println("error during loading Torch model");
            throw io;
        }
        return this;
    }

    JTensor forward(float[] storage, int offset, int[] shape) {
        return forwardNative(this.nativeRef, storage, offset, shape);
    }

    JTensor backward(float[] storage, int offset, int[] shape) {
        return backwardNative(this.nativeRef, storage, offset, shape);
    }

    float[] getGradient() {
        return getGradientNative(this.nativeRef);
    }

    float[] getWeight() {
        return getWeightNative(this.nativeRef);
    }

    void updateWeight(float[] weights) {
        updateWeightNative(this.nativeRef, weights);
    }

    private void load(String modelPath, String lossPath) {
        this.nativeRef = loadNative(modelPath, lossPath);
    }

    protected void finalize() {
        releaseNative(this.nativeRef);
    }

    native long loadNative(String modelPath, String lossPath);

    native JTensor forwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    native JTensor backwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    native float[] getGradientNative(long nativeRef);

    native void updateWeightNative(long nativeRef, float[] storage);

    native float[] getWeightNative(long nativeRef);

    native void releaseNative(long nativeRef);


}
