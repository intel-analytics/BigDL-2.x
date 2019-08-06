package com.intel.analytics.zoo.pipeline.api.net;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * Developer API for now, intentionally remove public visibility
 */
class PytorchModel {

    protected static boolean isLoaded = false;
    private static final String[] LIBS =
            new String[]{"libgomp-8bba0e50.so.1", "libc10.so", "libcaffe2.so",
                    "libtorch.so.1", "libpytorch-engine.so"};

    static {
        load();
    }

    protected static boolean isLoaded() {
        return isLoaded;
    }

    private static boolean load() {
        try {
            if (!isLoaded) {
                for(int i = 0; i < LIBS.length; i ++) {
                    System.out.println("loading " + LIBS[i]);
                    loadLib(LIBS[i]);
                }
                isLoaded = true;
            }
        } catch (Exception e) {
            isLoaded = false;
            e.printStackTrace();
            // TODO: Add an argument for user, continuing to run even if MKL load failed.
            throw new RuntimeException("Failed to load PMEM");
        }
        return isLoaded;
    }

    private static void loadLib(String libName) throws Exception {
        File tmpFile = extract(libName);
        try {
            System.load(tmpFile.getAbsolutePath());
        } finally {
            tmpFile.delete(); // delete so temp file after loaded
        }
    }

    // Extract so file from jar to a temp path
    private static File extract(String path) {
        try {
            URL url = PytorchModel.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find dynamic lib file in jar, path = " + path);
            }

            InputStream in = PytorchModel.class.getResourceAsStream("/" + path);
            File file = null;

            // Windows won't allow to change the dll name, so we keep the name
            // It's fine as windows is consider in a desktop env, so there won't multiple instance
            // produce the dynamic lib file
            if (System.getProperty("os.name").toLowerCase().contains("win")) {
                file = new File(System.getProperty("java.io.tmpdir") + File.separator + path);
            } else {
                file = createTempFile("dlNativeLoader", path);
            }

            ReadableByteChannel src = newChannel(in);
            FileChannel dest = new FileOutputStream(file).getChannel();
            dest.transferFrom(src, 0, Long.MAX_VALUE);
            dest.close();
            src.close();
            return file;
        } catch (Throwable e) {
            throw new Error("Can't extract dynamic lib file to /tmp dir.\n" + e);
        }
    }

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

}
