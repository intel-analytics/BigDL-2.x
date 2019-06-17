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

    PytorchModel load(byte[] bytes) throws IOException {
        try {
            File tmpFile = File.createTempFile("TorchNet", "_pt");
            Files.write(Paths.get(tmpFile.toURI()), bytes);
            this.load(tmpFile.getAbsolutePath());
            FileUtils.deleteQuietly(tmpFile);
        } catch (IOException io) {
            System.out.println("error during loading Torch model");
            throw io;
        }
        return this;
    }

    JTensor forward(float[] storage, int offset, int[] shape) {
        return forwardNative(this.nativeRef, storage, offset, shape);
    }

    private void load(String path) {
        this.nativeRef = loadNative(path);
    }

    protected void release() {
        releaseNative(this.nativeRef);
    }

    native long loadNative(String path);

    native JTensor forwardNative(long nativeRef, float[] storage, int offset, int[] shape);

    native void releaseNative(long nativeRef);


}
